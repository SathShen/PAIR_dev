"""
PAIR: Prompt-Aware Image-Point Reasoning

Top-level multimodal model.

Responsibilities
----------------
1. Route 2D / 3D / 2D+3D inputs.
2. Run PTv3 + PointAdapter for the 3D branch.
3. Ask Qwen3VLBackbone to prepare Qwen prompt/image inputs.
4. Inject PointAdapter outputs into <POINT> placeholder embeddings.
5. Run the native Qwen3-VL model.
6. Extract contextualized image / point / <TASK> hidden states.
7. Preserve native text logits and generation.

V1 limitation:
    external point-token injection currently supports batch size 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


SUPPORTED_TASK_MODES = ("2d", "3d", "2d3d")


@dataclass
class PAIROutput:
    image_hidden: Optional[torch.Tensor] = None
    point_hidden: Optional[torch.Tensor] = None
    task_hidden: Optional[torch.Tensor] = None

    logits: Optional[torch.Tensor] = None

    generated_ids: Optional[torch.Tensor] = None
    generated_text: Optional[Any] = None

    aux: Optional[Dict[str, Any]] = None


class PAIRModel(nn.Module):
    """
    Top-level PAIR model.

    Image:
        image -> Qwen3-VL Vision -> native image tokens

    Point cloud:
        points -> PTv3 -> PointAdapter -> Qwen-width point tokens

    Fusion:
        PAIR replaces <POINT> placeholder embeddings with the real point
        tokens, then runs Qwen LLM.

    Output:
        contextualized image_hidden / point_hidden / task_hidden
        plus optional native LM logits / text generation.
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

    # ------------------------------------------------------------------
    # Task routing
    # ------------------------------------------------------------------

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
        task_mode: str,
        images: Optional[Any],
        point_dict: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        if task_mode in ("2d", "2d3d") and images is None:
            raise ValueError(
                f"task_mode={task_mode!r} requires image input."
            )

        if task_mode in ("3d", "2d3d") and point_dict is None:
            raise ValueError(
                f"task_mode={task_mode!r} requires point_dict."
            )

        if task_mode in ("3d", "2d3d"):
            if self.point_encoder is None:
                raise RuntimeError(
                    "3D input requested but point_encoder is not configured."
                )

            if self.point_adapter is None:
                raise RuntimeError(
                    "3D input requested but point_adapter is not configured."
                )

    # ------------------------------------------------------------------
    # 3D branch
    # ------------------------------------------------------------------

    def encode_points(
        self,
        point_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        point_encoded = self.point_encoder(point_dict)
        point_tokens = self.point_adapter(point_encoded)

        return {
            "point_encoded": point_encoded,
            "point_tokens": point_tokens,
        }

    # ------------------------------------------------------------------
    # Point-token injection
    # ------------------------------------------------------------------

    def _normalize_point_tokens(
        self,
        point_tokens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if point_tokens is None:
            return None

        if not torch.is_tensor(point_tokens):
            raise TypeError("point_tokens must be a torch.Tensor.")

        if point_tokens.ndim == 2:
            point_tokens = point_tokens.unsqueeze(0)

        if point_tokens.ndim != 3:
            raise ValueError(
                "point_tokens must have shape [N, D] or [1, N, D], "
                f"got {tuple(point_tokens.shape)}."
            )

        if point_tokens.shape[0] != 1:
            raise NotImplementedError(
                "PAIR V1 currently supports batch size 1 for point injection."
            )

        if point_tokens.shape[-1] != self.qwen_backbone.hidden_size:
            raise ValueError(
                "Point token hidden dimension does not match Qwen hidden size."
            )

        return point_tokens

    def _validate_point_layout(
        self,
        point_tokens: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> None:
        expected_count = point_tokens.shape[1]
        actual_count = int(point_mask.sum().item())

        if expected_count != actual_count:
            raise RuntimeError(
                f"PointAdapter produced {expected_count} tokens, "
                f"but Qwen prompt has {actual_count} <POINT> placeholders."
            )

        positions = torch.nonzero(
            point_mask[0],
            as_tuple=False,
        ).flatten()

        if positions.numel() == 0:
            raise RuntimeError(
                "Point tokens were supplied but no <POINT> placeholders exist."
            )

        expected_positions = torch.arange(
            positions[0],
            positions[0] + expected_count,
            device=positions.device,
            dtype=positions.dtype,
        )

        if not torch.equal(positions, expected_positions):
            raise RuntimeError(
                "<POINT> placeholders must be contiguous in PAIR V1."
            )

    def _make_point_injection_hook(
        self,
        *,
        point_tokens: torch.Tensor,
        point_mask: torch.Tensor,
        full_seq_len: int,
        stats: Dict[str, Any],
    ):
        point_tokens = self._normalize_point_tokens(point_tokens)

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

    # ------------------------------------------------------------------
    # Hidden extraction
    # ------------------------------------------------------------------

    def _extract_multimodal_hidden(
        self,
        *,
        last_hidden: torch.Tensor,
        image_mask: torch.Tensor,
        point_mask: torch.Tensor,
        task_mask: torch.Tensor,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        device = last_hidden.device

        image_mask = image_mask.to(device)
        point_mask = point_mask.to(device)
        task_mask = task_mask.to(device)

        if int(task_mask.sum().item()) != 1:
            raise RuntimeError("PAIR expects exactly one <TASK> token.")

        image_hidden = None
        point_hidden = None

        if int(image_mask.sum().item()) > 0:
            image_hidden = last_hidden[0][image_mask[0]]

        if int(point_mask.sum().item()) > 0:
            point_hidden = last_hidden[0][point_mask[0]]

        task_hidden = last_hidden[0][task_mask[0]].reshape(
            1,
            self.qwen_backbone.hidden_size,
        )

        aux = {}

        if image_hidden is not None and "image_grid_thw" in inputs:
            grid = inputs["image_grid_thw"][0]

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
                    "Image hidden token count does not match image_grid_thw."
                )

            aux["image_hidden_2d"] = image_hidden.reshape(
                t,
                h_token,
                w_token,
                self.qwen_backbone.hidden_size,
            )

            aux["image_grid_thw"] = (
                t,
                h_patch,
                w_patch,
            )

            aux["image_token_grid_thw"] = (
                t,
                h_token,
                w_token,
            )

        return {
            "image_hidden": image_hidden,
            "point_hidden": point_hidden,
            "task_hidden": task_hidden,
            "aux": aux,
        }

    # ------------------------------------------------------------------
    # Qwen input preparation
    # ------------------------------------------------------------------

    def _prepare_qwen(
        self,
        *,
        prompt: str,
        images: Optional[Any],
        point_tokens: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        point_tokens = self._normalize_point_tokens(point_tokens)

        prepared = self.qwen_backbone.prepare_inputs(
            prompt=prompt,
            images=images,
            point_tokens=point_tokens,
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

        return {
            **prepared,
            "inputs": inputs,
            "point_tokens": point_tokens,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        task_mode: str,
        prompt: str,
        images: Optional[Any] = None,
        point_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_logits: bool = True,
        return_hidden_states: bool = True,
        use_cache: bool = False,
        **qwen_kwargs,
    ) -> PAIROutput:
        task_mode = self._normalize_task_mode(task_mode)

        self._validate_inputs(
            task_mode=task_mode,
            images=images,
            point_dict=point_dict,
        )

        point_encoded = None
        point_tokens = None

        if task_mode in ("3d", "2d3d"):
            point_outputs = self.encode_points(point_dict)
            point_encoded = point_outputs["point_encoded"]
            point_tokens = point_outputs["point_tokens"]

        prepared = self._prepare_qwen(
            prompt=prompt,
            images=images if task_mode in ("2d", "2d3d") else None,
            point_tokens=point_tokens if task_mode in ("3d", "2d3d") else None,
        )

        inputs = prepared["inputs"]
        point_tokens = prepared["point_tokens"]

        image_mask = prepared["image_mask"]
        point_mask = prepared["point_mask"]
        task_mask = prepared["task_mask"]

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

        if point_tokens is not None and not injection_stats["replaced"]:
            raise RuntimeError(
                "Point tokens were provided but were not injected into Qwen."
            )

        image_hidden = None
        point_hidden = None
        task_hidden = None
        hidden_aux = {}

        if return_hidden_states:
            hidden = self._extract_multimodal_hidden(
                last_hidden=qwen_outputs.hidden_states[-1],
                image_mask=image_mask,
                point_mask=point_mask,
                task_mask=task_mask,
                inputs=inputs,
            )

            image_hidden = hidden["image_hidden"]
            point_hidden = hidden["point_hidden"]
            task_hidden = hidden["task_hidden"]
            hidden_aux = hidden["aux"]

        aux = {
            "task_mode": task_mode,
            "point_encoded": point_encoded,
            "point_tokens": point_tokens,
            "prompt_text": prepared["prompt_text"],
            "image_count": prepared["image_count"],
            "point_count": prepared["point_count"],
            "task_count": prepared["task_count"],
            "point_injection_calls": injection_stats["calls"],
            "point_injection_replaced": injection_stats["replaced"],
            **hidden_aux,
        }

        return PAIROutput(
            image_hidden=image_hidden,
            point_hidden=point_hidden,
            task_hidden=task_hidden,
            logits=qwen_outputs.logits if return_logits else None,
            aux=aux,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        *,
        task_mode: str,
        prompt: str,
        images: Optional[Any] = None,
        point_dict: Optional[Dict[str, torch.Tensor]] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> PAIROutput:
        task_mode = self._normalize_task_mode(task_mode)

        self._validate_inputs(
            task_mode=task_mode,
            images=images,
            point_dict=point_dict,
        )

        point_encoded = None
        point_tokens = None

        if task_mode in ("3d", "2d3d"):
            point_outputs = self.encode_points(point_dict)
            point_encoded = point_outputs["point_encoded"]
            point_tokens = point_outputs["point_tokens"]

        prepared = self._prepare_qwen(
            prompt=prompt,
            images=images if task_mode in ("2d", "2d3d") else None,
            point_tokens=point_tokens if task_mode in ("3d", "2d3d") else None,
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

        if point_tokens is not None and not injection_stats["replaced"]:
            raise RuntimeError(
                "Point tokens were provided but were not injected "
                "during generation."
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[:, prompt_len:]

        generated_text = self.qwen_backbone.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
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
                "point_encoded": point_encoded,
                "point_tokens": point_tokens,
                "prompt_text": prepared["prompt_text"],
                "image_count": prepared["image_count"],
                "point_count": prepared["point_count"],
                "task_count": prepared["task_count"],
                "point_injection_calls": injection_stats["calls"],
                "point_injection_replaced": injection_stats["replaced"],
            },
        )


PAIR = PAIRModel


if __name__ == "__main__":
    print("PAIR.py import scaffold OK")
    print("Supported task modes:", SUPPORTED_TASK_MODES)
