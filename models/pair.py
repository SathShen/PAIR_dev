"""
PAIR: Prompt-Aware Image-Point Reasoning

Temporal multimodal V1.

Supported task modes
--------------------
2d:
    Image T1 + Image T2 + prompt

3d:
    Point T1 + Point T2 + prompt

2d3d:
    Image T1 + Image T2 + Point T1 + Point T2 + prompt

The two temporal observations share the same:
    - Qwen3-VL vision backbone
    - PTv3 point encoder
    - PointAdapter

PAIR owns:
    - task routing
    - T1/T2 point encoding
    - <POINT> embedding injection
    - Qwen forward / generation orchestration
    - contextual T1/T2 hidden-state extraction

Current V1 limitation:
    batch size = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


SUPPORTED_TASK_MODES = ("2d", "3d", "2d3d")


@dataclass
class PAIROutput:
    # Temporal contextualized Qwen representations
    image_hidden_t1: Optional[torch.Tensor] = None
    image_hidden_t2: Optional[torch.Tensor] = None

    point_hidden_t1: Optional[torch.Tensor] = None
    point_hidden_t2: Optional[torch.Tensor] = None

    task_hidden: Optional[torch.Tensor] = None

    # Native Qwen LM output
    logits: Optional[torch.Tensor] = None

    generated_ids: Optional[torch.Tensor] = None
    generated_text: Optional[Any] = None

    # Dense PTv3 / token / spatial metadata
    aux: Optional[Dict[str, Any]] = None


class PAIRModel(nn.Module):
    """
    Top-level temporal PAIR model.

    Image T1 ----\
                  \
    Image T2 ------> Qwen3-VL Vision / LLM
                    /
    Point T1 -> PTv3 -> PointAdapter --\
                                        \
    Point T2 -> PTv3 -> PointAdapter ----> <POINT> injection
                                          /
    Prompt + <TASK> ---------------------/

    The same PTv3 and PointAdapter instances are reused for T1 and T2.
    """

    def __init__(
        self,
        qwen_backbone: nn.Module,
        point_encoder: Optional[nn.Module] = None,
        point_adapter: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.qwen_backbone = qwen_backbone
        self.point_encoder = point_encoder
        self.point_adapter = point_adapter

    # ==================================================================
    # Task routing
    # ==================================================================

    @staticmethod
    def _normalize_task_mode(task_mode: str) -> str:
        task_mode = task_mode.lower().strip()

        aliases = {
            "2d": "2d",
            "image": "2d",
            "image_only": "2d",

            "3d": "3d",
            "point": "3d",
            "point_only": "3d",

            "2d3d": "2d3d",
            "2d+3d": "2d3d",
            "image_point": "2d3d",
            "multimodal": "2d3d",
        }

        if task_mode not in aliases:
            raise ValueError(
                f"Unsupported task_mode={task_mode!r}. "
                f"Supported modes: {SUPPORTED_TASK_MODES}"
            )

        return aliases[task_mode]

    def _validate_inputs(
        self,
        *,
        task_mode: str,
        images_t1: Optional[Any],
        images_t2: Optional[Any],
        point_dict_t1: Optional[Dict[str, torch.Tensor]],
        point_dict_t2: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        if task_mode in ("2d", "2d3d"):
            if images_t1 is None or images_t2 is None:
                raise ValueError(
                    f"task_mode={task_mode!r} requires both "
                    "images_t1 and images_t2."
                )

        if task_mode in ("3d", "2d3d"):
            if point_dict_t1 is None or point_dict_t2 is None:
                raise ValueError(
                    f"task_mode={task_mode!r} requires both "
                    "point_dict_t1 and point_dict_t2."
                )

            if self.point_encoder is None:
                raise RuntimeError(
                    "3D input requested but point_encoder is not configured."
                )

            if self.point_adapter is None:
                raise RuntimeError(
                    "3D input requested but point_adapter is not configured."
                )

    # ==================================================================
    # Shared 3D branch
    # ==================================================================

    def encode_points(
        self,
        point_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        One temporal point cloud through the shared PTv3 + PointAdapter.
        """

        point_encoded = self.point_encoder(point_dict)
        point_tokens = self.point_adapter(point_encoded)

        return {
            "point_encoded": point_encoded,
            "point_tokens": point_tokens,
        }

    # ==================================================================
    # Point-token helpers
    # ==================================================================

    def _normalize_single_point_tokens(
        self,
        point_tokens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Normalize one time-step PointAdapter output to [N, D].
        """

        if point_tokens is None:
            return None

        if not torch.is_tensor(point_tokens):
            raise TypeError("point_tokens must be a torch.Tensor.")

        if point_tokens.ndim == 3:
            if point_tokens.shape[0] != 1:
                raise NotImplementedError(
                    "PAIR V1 supports batch size 1."
                )
            point_tokens = point_tokens[0]

        if point_tokens.ndim != 2:
            raise ValueError(
                "point_tokens must have shape [N, D] or [1, N, D], "
                f"got {tuple(point_tokens.shape)}."
            )

        if point_tokens.shape[-1] != self.qwen_backbone.hidden_size:
            raise ValueError(
                f"Point token hidden dimension must equal Qwen hidden "
                f"size {self.qwen_backbone.hidden_size}, "
                f"got {point_tokens.shape[-1]}."
            )

        return point_tokens

    def _concat_point_tokens(
        self,
        point_tokens_t1: Optional[torch.Tensor],
        point_tokens_t2: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Concatenate temporal point tokens in the same order used by the prompt:
            T1 first, then T2.

        Returns [1, N1+N2, D].
        """

        t1 = self._normalize_single_point_tokens(
            point_tokens_t1
        )
        t2 = self._normalize_single_point_tokens(
            point_tokens_t2
        )

        pieces = [
            x for x in (t1, t2)
            if x is not None
        ]

        if not pieces:
            return None

        return torch.cat(
            pieces,
            dim=0,
        ).unsqueeze(0)

    def _validate_point_layout(
        self,
        *,
        point_tokens: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> None:
        expected_count = point_tokens.shape[1]
        actual_count = int(point_mask.sum().item())

        if expected_count != actual_count:
            raise RuntimeError(
                f"Prepared {expected_count} external point tokens, "
                f"but Qwen prompt contains {actual_count} "
                "<POINT> placeholders."
            )

    def _make_point_injection_hook(
        self,
        *,
        point_tokens: torch.Tensor,
        point_mask: torch.Tensor,
        full_seq_len: int,
        stats: Dict[str, Any],
    ):
        """
        Inject concatenated T1/T2 point tokens into all <POINT> positions.

        Placeholder positions do not need to be globally contiguous:
        boolean indexing preserves their sequence order, which matches
        [T1 tokens, T2 tokens].
        """

        self._validate_point_layout(
            point_tokens=point_tokens,
            point_mask=point_mask,
        )

        hidden_size = self.qwen_backbone.hidden_size

        def hook(module, args, output):
            stats["calls"] += 1

            if (
                torch.is_tensor(output)
                and output.ndim == 3
                and output.shape[0] == 1
                and output.shape[1] == full_seq_len
                and output.shape[2] == hidden_size
            ):
                out = output.clone()

                mask = point_mask[0].to(out.device)

                out[0, mask] = point_tokens[0].to(
                    device=out.device,
                    dtype=out.dtype,
                )

                stats["replaced"] = True
                return out

            return output

        return hook

    # ==================================================================
    # Temporal hidden extraction
    # ==================================================================

    def _reshape_image_hidden(
        self,
        *,
        image_hidden: Optional[torch.Tensor],
        inputs: Dict[str, Any],
        grid_index: Optional[int],
    ) -> Optional[torch.Tensor]:
        """
        Recover one temporal image token set as [T, H, W, D].
        """

        if image_hidden is None or grid_index is None:
            return None

        if "image_grid_thw" not in inputs:
            return None

        grid = inputs["image_grid_thw"][grid_index]

        t = int(grid[0].item())
        h_patch = int(grid[1].item())
        w_patch = int(grid[2].item())

        merge = int(
            self.qwen_backbone.vision_spatial_merge_size
        )

        h_token = h_patch // merge
        w_token = w_patch // merge

        expected = t * h_token * w_token

        if image_hidden.shape[0] != expected:
            raise RuntimeError(
                f"Temporal image hidden count {image_hidden.shape[0]} "
                f"does not match expected {expected}."
            )

        return image_hidden.reshape(
            t,
            h_token,
            w_token,
            self.qwen_backbone.hidden_size,
        )

    def _extract_multimodal_hidden(
        self,
        *,
        last_hidden: torch.Tensor,
        prepared: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract T1/T2 image, T1/T2 point, and <TASK> hidden states.
        """

        device = last_hidden.device

        def to_device_mask(name: str):
            return prepared[name].to(device)

        image_mask_t1 = to_device_mask("image_mask_t1")
        image_mask_t2 = to_device_mask("image_mask_t2")

        point_mask_t1 = to_device_mask("point_mask_t1")
        point_mask_t2 = to_device_mask("point_mask_t2")

        task_mask = to_device_mask("task_mask")

        if int(task_mask.sum().item()) != 1:
            raise RuntimeError(
                "PAIR expects exactly one <TASK> token."
            )

        image_hidden_t1 = None
        image_hidden_t2 = None
        point_hidden_t1 = None
        point_hidden_t2 = None

        if int(image_mask_t1.sum().item()) > 0:
            image_hidden_t1 = last_hidden[0][
                image_mask_t1[0]
            ]

        if int(image_mask_t2.sum().item()) > 0:
            image_hidden_t2 = last_hidden[0][
                image_mask_t2[0]
            ]

        if int(point_mask_t1.sum().item()) > 0:
            point_hidden_t1 = last_hidden[0][
                point_mask_t1[0]
            ]

        if int(point_mask_t2.sum().item()) > 0:
            point_hidden_t2 = last_hidden[0][
                point_mask_t2[0]
            ]

        task_hidden = last_hidden[0][
            task_mask[0]
        ].reshape(
            1,
            self.qwen_backbone.hidden_size,
        )

        image_hidden_2d_t1 = self._reshape_image_hidden(
            image_hidden=image_hidden_t1,
            inputs=inputs,
            grid_index=prepared["image_grid_index_t1"],
        )

        image_hidden_2d_t2 = self._reshape_image_hidden(
            image_hidden=image_hidden_t2,
            inputs=inputs,
            grid_index=prepared["image_grid_index_t2"],
        )

        return {
            "image_hidden_t1": image_hidden_t1,
            "image_hidden_t2": image_hidden_t2,

            "point_hidden_t1": point_hidden_t1,
            "point_hidden_t2": point_hidden_t2,

            "task_hidden": task_hidden,

            "image_hidden_2d_t1": image_hidden_2d_t1,
            "image_hidden_2d_t2": image_hidden_2d_t2,
        }

    # ==================================================================
    # Qwen preparation
    # ==================================================================

    def _prepare_qwen(
        self,
        *,
        prompt: str,
        images_t1: Optional[Any],
        images_t2: Optional[Any],
        point_tokens_t1: Optional[torch.Tensor],
        point_tokens_t2: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        prepared = self.qwen_backbone.prepare_inputs(
            prompt=prompt,

            images_t1=images_t1,
            images_t2=images_t2,

            point_tokens_t1=point_tokens_t1,
            point_tokens_t2=point_tokens_t2,
        )

        model_device = self.qwen_backbone.model_device

        inputs = {
            key: (
                value.to(model_device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in prepared["inputs"].items()
        }

        point_tokens = self._concat_point_tokens(
            point_tokens_t1=point_tokens_t1,
            point_tokens_t2=point_tokens_t2,
        )

        return {
            **prepared,
            "inputs": inputs,
            "point_tokens": point_tokens,
        }

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        *,
        task_mode: str,
        prompt: str,

        images_t1: Optional[Any] = None,
        images_t2: Optional[Any] = None,

        point_dict_t1: Optional[Dict[str, torch.Tensor]] = None,
        point_dict_t2: Optional[Dict[str, torch.Tensor]] = None,

        return_logits: bool = True,
        return_hidden_states: bool = True,
        use_cache: bool = False,

        **qwen_kwargs,
    ) -> PAIROutput:
        task_mode = self._normalize_task_mode(task_mode)

        self._validate_inputs(
            task_mode=task_mode,

            images_t1=images_t1,
            images_t2=images_t2,

            point_dict_t1=point_dict_t1,
            point_dict_t2=point_dict_t2,
        )

        # --------------------------------------------------------------
        # Shared PTv3 + PointAdapter, applied separately to T1 / T2.
        # --------------------------------------------------------------
        point_encoded_t1 = None
        point_encoded_t2 = None

        point_tokens_t1 = None
        point_tokens_t2 = None

        if task_mode in ("3d", "2d3d"):
            out_t1 = self.encode_points(
                point_dict_t1
            )
            out_t2 = self.encode_points(
                point_dict_t2
            )

            point_encoded_t1 = out_t1["point_encoded"]
            point_encoded_t2 = out_t2["point_encoded"]

            point_tokens_t1 = out_t1["point_tokens"]
            point_tokens_t2 = out_t2["point_tokens"]

        # --------------------------------------------------------------
        # Native Qwen image + temporal prompt preparation.
        # --------------------------------------------------------------
        prepared = self._prepare_qwen(
            prompt=prompt,

            images_t1=(
                images_t1
                if task_mode in ("2d", "2d3d")
                else None
            ),
            images_t2=(
                images_t2
                if task_mode in ("2d", "2d3d")
                else None
            ),

            point_tokens_t1=(
                point_tokens_t1
                if task_mode in ("3d", "2d3d")
                else None
            ),
            point_tokens_t2=(
                point_tokens_t2
                if task_mode in ("3d", "2d3d")
                else None
            ),
        )

        inputs = prepared["inputs"]
        point_tokens = prepared["point_tokens"]
        point_mask = prepared["point_mask"]

        injection_stats = {
            "calls": 0,
            "replaced": False,
        }

        handle = None

        if point_tokens is not None:
            hook = self._make_point_injection_hook(
                point_tokens=point_tokens,
                point_mask=point_mask,
                full_seq_len=inputs["input_ids"].shape[1],
                stats=injection_stats,
            )

            handle = (
                self.qwen_backbone.model
                .get_input_embeddings()
                .register_forward_hook(hook)
            )

        try:
            qwen_outputs = self.qwen_backbone.model(
                **inputs,
                output_hidden_states=return_hidden_states,
                return_dict=True,
                use_cache=use_cache,
                **qwen_kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

        if (
            point_tokens is not None
            and not injection_stats["replaced"]
        ):
            raise RuntimeError(
                "Temporal point tokens were prepared but were not "
                "injected into Qwen."
            )

        image_hidden_t1 = None
        image_hidden_t2 = None
        point_hidden_t1 = None
        point_hidden_t2 = None
        task_hidden = None

        hidden_extra = {}

        if return_hidden_states:
            hidden = self._extract_multimodal_hidden(
                last_hidden=qwen_outputs.hidden_states[-1],
                prepared=prepared,
                inputs=inputs,
            )

            image_hidden_t1 = hidden["image_hidden_t1"]
            image_hidden_t2 = hidden["image_hidden_t2"]

            point_hidden_t1 = hidden["point_hidden_t1"]
            point_hidden_t2 = hidden["point_hidden_t2"]

            task_hidden = hidden["task_hidden"]

            hidden_extra = {
                "image_hidden_2d_t1": hidden[
                    "image_hidden_2d_t1"
                ],
                "image_hidden_2d_t2": hidden[
                    "image_hidden_2d_t2"
                ],
            }

        aux = {
            "task_mode": task_mode,

            "point_encoded_t1": point_encoded_t1,
            "point_encoded_t2": point_encoded_t2,

            "point_tokens_t1": point_tokens_t1,
            "point_tokens_t2": point_tokens_t2,
            "point_tokens_concat": point_tokens,

            "prompt_text": prepared["prompt_text"],

            "image_count_t1": prepared["image_count_t1"],
            "image_count_t2": prepared["image_count_t2"],

            "point_count_t1": prepared["point_count_t1"],
            "point_count_t2": prepared["point_count_t2"],

            "task_count": prepared["task_count"],

            "point_injection_calls": injection_stats["calls"],
            "point_injection_replaced": injection_stats["replaced"],

            **hidden_extra,
        }

        return PAIROutput(
            image_hidden_t1=image_hidden_t1,
            image_hidden_t2=image_hidden_t2,

            point_hidden_t1=point_hidden_t1,
            point_hidden_t2=point_hidden_t2,

            task_hidden=task_hidden,

            logits=(
                qwen_outputs.logits
                if return_logits
                else None
            ),

            aux=aux,
        )

    # ==================================================================
    # Generation
    # ==================================================================

    @torch.no_grad()
    def generate(
        self,
        *,
        task_mode: str,
        prompt: str,

        images_t1: Optional[Any] = None,
        images_t2: Optional[Any] = None,

        point_dict_t1: Optional[Dict[str, torch.Tensor]] = None,
        point_dict_t2: Optional[Dict[str, torch.Tensor]] = None,

        max_new_tokens: int = 64,
        do_sample: bool = False,

        **generate_kwargs,
    ) -> PAIROutput:
        task_mode = self._normalize_task_mode(task_mode)

        self._validate_inputs(
            task_mode=task_mode,

            images_t1=images_t1,
            images_t2=images_t2,

            point_dict_t1=point_dict_t1,
            point_dict_t2=point_dict_t2,
        )

        point_encoded_t1 = None
        point_encoded_t2 = None

        point_tokens_t1 = None
        point_tokens_t2 = None

        if task_mode in ("3d", "2d3d"):
            out_t1 = self.encode_points(
                point_dict_t1
            )
            out_t2 = self.encode_points(
                point_dict_t2
            )

            point_encoded_t1 = out_t1["point_encoded"]
            point_encoded_t2 = out_t2["point_encoded"]

            point_tokens_t1 = out_t1["point_tokens"]
            point_tokens_t2 = out_t2["point_tokens"]

        prepared = self._prepare_qwen(
            prompt=prompt,

            images_t1=(
                images_t1
                if task_mode in ("2d", "2d3d")
                else None
            ),
            images_t2=(
                images_t2
                if task_mode in ("2d", "2d3d")
                else None
            ),

            point_tokens_t1=(
                point_tokens_t1
                if task_mode in ("3d", "2d3d")
                else None
            ),
            point_tokens_t2=(
                point_tokens_t2
                if task_mode in ("3d", "2d3d")
                else None
            ),
        )

        inputs = prepared["inputs"]
        point_tokens = prepared["point_tokens"]
        point_mask = prepared["point_mask"]

        injection_stats = {
            "calls": 0,
            "replaced": False,
        }

        handle = None

        if point_tokens is not None:
            hook = self._make_point_injection_hook(
                point_tokens=point_tokens,
                point_mask=point_mask,
                full_seq_len=inputs["input_ids"].shape[1],
                stats=injection_stats,
            )

            handle = (
                self.qwen_backbone.model
                .get_input_embeddings()
                .register_forward_hook(hook)
            )

        try:
            generated_ids = self.qwen_backbone.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **generate_kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

        if (
            point_tokens is not None
            and not injection_stats["replaced"]
        ):
            raise RuntimeError(
                "Temporal point tokens were not injected "
                "during generation."
            )

        prompt_len = inputs["input_ids"].shape[1]

        new_token_ids = generated_ids[
            :,
            prompt_len:,
        ]

        generated_text = (
            self.qwen_backbone.processor.batch_decode(
                new_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

        generated_text_out = (
            generated_text[0]
            if len(generated_text) == 1
            else generated_text
        )

        return PAIROutput(
            generated_ids=generated_ids,
            generated_text=generated_text_out,

            aux={
                "task_mode": task_mode,

                "point_encoded_t1": point_encoded_t1,
                "point_encoded_t2": point_encoded_t2,

                "point_tokens_t1": point_tokens_t1,
                "point_tokens_t2": point_tokens_t2,
                "point_tokens_concat": point_tokens,

                "prompt_text": prepared["prompt_text"],

                "image_count_t1": prepared["image_count_t1"],
                "image_count_t2": prepared["image_count_t2"],

                "point_count_t1": prepared["point_count_t1"],
                "point_count_t2": prepared["point_count_t2"],

                "task_count": prepared["task_count"],

                "point_injection_calls": injection_stats["calls"],
                "point_injection_replaced": injection_stats["replaced"],
            },
        )


PAIR = PAIRModel


if __name__ == "__main__":
    print("pair.py temporal import scaffold OK")
    print("Supported task modes:", SUPPORTED_TASK_MODES)
