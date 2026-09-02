"""
Qwen3-VL backbone wrapper for PAIR.

This module is a cleaned, reusable version of the Qwen logic already
validated in the smoke tests:

    - native Qwen3-VL image path
    - <POINT> placeholder tokens
    - external point-token embedding injection
    - <TASK> token
    - contextualized image / point / task hidden extraction
    - native text generation

V1 intentionally keeps the same embedding-hook injection strategy that
already passed the joint PTv3-Qwen smoke test.

Current limitation:
    - batch size = 1 for external point-token injection

Later work can add:
    - multi-sample batches with variable point-token counts
    - T1/T2 multimodal packing
    - LoRA helpers
    - gradient checkpointing helpers
    - richer task-token routing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


@dataclass
class Qwen3VLBackboneOutput:
    image_hidden: Optional[torch.Tensor] = None
    point_hidden: Optional[torch.Tensor] = None
    task_hidden: Optional[torch.Tensor] = None

    logits: Optional[torch.Tensor] = None

    generated_ids: Optional[torch.Tensor] = None
    generated_text: Optional[Any] = None

    aux: Optional[Dict[str, Any]] = None


class Qwen3VLBackbone(nn.Module):
    """
    Qwen3-VL wrapper used by PAIR.

    Parameters
    ----------
    model_dir:
        Local Qwen3-VL checkpoint directory.

    dtype:
        Qwen model dtype. BF16 matches the validated smoke-test setup.

    device:
        Device used when device_map is None.

    device_map:
        Passed to Hugging Face from_pretrained().
        The validated setup uses "cuda".

    local_files_only:
        Keep True for the current local-checkpoint workflow.

    point_token:
        Placeholder token replaced by external point embeddings.

    task_token:
        Special token whose contextualized hidden state represents the
        current task / multimodal instruction.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        device_map: Optional[Union[str, Dict[str, Any]]] = "cuda",
        local_files_only: bool = True,
        point_token: str = "<POINT>",
        task_token: str = "<TASK>",
    ):
        super().__init__()

        self.model_dir = model_dir
        self.dtype = dtype
        self.device_name = str(device)
        self.device_map = device_map
        self.local_files_only = local_files_only

        self.point_token = point_token
        self.task_token = task_token

        # --------------------------------------------------------------
        # Processor / tokenizer
        # --------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
        )
        self.tokenizer = self.processor.tokenizer

        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.point_token,
                    self.task_token,
                ]
            }
        )

        self.point_token_id = self.tokenizer.convert_tokens_to_ids(
            self.point_token
        )
        self.task_token_id = self.tokenizer.convert_tokens_to_ids(
            self.task_token
        )

        # --------------------------------------------------------------
        # Qwen3-VL
        # --------------------------------------------------------------
        load_kwargs = {
            "dtype": dtype,
            "local_files_only": local_files_only,
        }

        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            **load_kwargs,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        if device_map is None:
            self.model.to(device)

        self.hidden_size = self.model.config.text_config.hidden_size
        self.image_token_id = self.model.config.image_token_id

        self.vision_spatial_merge_size = (
            self.model.config.vision_config.spatial_merge_size
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _normalize_point_tokens(
        self,
        point_tokens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Normalize external point tokens to [1, N, D].

        V1 deliberately supports batch size 1 only. This matches all
        validated smoke tests and avoids introducing untested variable-length
        batching logic before the base PAIR model is stable.
        """

        if point_tokens is None:
            return None

        if not torch.is_tensor(point_tokens):
            raise TypeError(
                "point_tokens must be a torch.Tensor or None."
            )

        if point_tokens.ndim == 2:
            point_tokens = point_tokens.unsqueeze(0)

        if point_tokens.ndim != 3:
            raise ValueError(
                "point_tokens must have shape [N, D] or [1, N, D]. "
                f"Got {tuple(point_tokens.shape)}."
            )

        if point_tokens.shape[0] != 1:
            raise NotImplementedError(
                "Qwen3VLBackbone V1 currently supports batch size 1 "
                "for external point-token injection."
            )

        if point_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"point_tokens hidden dim must equal Qwen hidden size "
                f"{self.hidden_size}, got {point_tokens.shape[-1]}."
            )

        return point_tokens

    def _build_user_text(
        self,
        prompt: str,
        num_point_tokens: int,
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError(
                "Qwen3VLBackbone V1 expects prompt to be a string."
            )

        if self.point_token in prompt:
            raise ValueError(
                f"Do not manually place {self.point_token} in prompt. "
                "PAIR inserts point placeholders automatically."
            )

        task_count = prompt.count(self.task_token)

        if task_count > 1:
            raise ValueError(
                f"Prompt contains {task_count} copies of {self.task_token}; "
                "exactly one task token is allowed."
            )

        chunks = [prompt.rstrip()]

        if num_point_tokens > 0:
            placeholders = self.point_token * num_point_tokens
            chunks.append(placeholders)

        if task_count == 0:
            chunks.append(self.task_token)

        return "\n".join(chunks)

    def _build_messages(
        self,
        prompt: str,
        images: Optional[Any],
        num_point_tokens: int,
    ):
        text = self._build_user_text(
            prompt=prompt,
            num_point_tokens=num_point_tokens,
        )

        content = []

        if images is not None:
            # V1: one image, same path used in the successful smoke tests.
            content.append(
                {
                    "type": "image",
                    "image": images,
                }
            )

        content.append(
            {
                "type": "text",
                "text": text,
            }
        )

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def prepare_inputs(
        self,
        *,
        prompt: str,
        images: Optional[Any] = None,
        point_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Build Qwen chat input and masks for image / point / task tokens.
        """

        point_tokens = self._normalize_point_tokens(point_tokens)

        num_point_tokens = (
            0 if point_tokens is None else point_tokens.shape[1]
        )

        messages = self._build_messages(
            prompt=prompt,
            images=images,
            num_point_tokens=num_point_tokens,
        )

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if images is not None:
            inputs = self.processor(
                text=[prompt_text],
                images=[images],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt",
            )

        input_ids = inputs["input_ids"]

        image_mask = input_ids == self.image_token_id
        point_mask = input_ids == self.point_token_id
        task_mask = input_ids == self.task_token_id

        image_count = int(image_mask.sum().item())
        point_count = int(point_mask.sum().item())
        task_count = int(task_mask.sum().item())

        if point_count != num_point_tokens:
            raise RuntimeError(
                f"Expected {num_point_tokens} {self.point_token} tokens, "
                f"but tokenizer produced {point_count}."
            )

        if task_count != 1:
            raise RuntimeError(
                f"Expected exactly one {self.task_token}, "
                f"but tokenizer produced {task_count}."
            )

        if images is None and image_count != 0:
            raise RuntimeError(
                "No image was provided but image tokens were produced."
            )

        # Ensure point placeholders are contiguous.
        if point_count > 0:
            point_positions = torch.nonzero(
                point_mask[0],
                as_tuple=False,
            ).flatten()

            expected = torch.arange(
                point_positions[0],
                point_positions[0] + point_count,
                dtype=point_positions.dtype,
            )

            if not torch.equal(point_positions.cpu(), expected.cpu()):
                raise RuntimeError(
                    f"{self.point_token} placeholders are not contiguous."
                )

        return {
            "prompt_text": prompt_text,
            "inputs": inputs,
            "point_tokens": point_tokens,
            "image_mask": image_mask,
            "point_mask": point_mask,
            "task_mask": task_mask,
            "image_count": image_count,
            "point_count": point_count,
            "task_count": task_count,
        }

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def _move_inputs_to_model_device(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        device = self.model_device

        return {
            key: (
                value.to(device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in inputs.items()
        }

    # ------------------------------------------------------------------
    # Point-token injection
    # ------------------------------------------------------------------

    def _make_point_injection_hook(
        self,
        *,
        point_tokens: Optional[torch.Tensor],
        point_mask: torch.Tensor,
        full_seq_len: int,
        stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Build the same forward-hook injection mechanism validated in the
        joint smoke test.

        It only modifies the full prompt embedding pass. During generate(),
        later autoregressive steps have shorter sequence lengths and are
        therefore left untouched.
        """

        if point_tokens is None:
            return None

        if stats is None:
            stats = {}

        stats.setdefault("calls", 0)
        stats.setdefault("replaced", False)

        hidden_size = self.hidden_size

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

                injected = point_tokens[0].to(
                    device=out.device,
                    dtype=out.dtype,
                )

                out[0, point_mask[0]] = injected

                stats["replaced"] = True
                return out

            return output

        return hook

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    def _extract_hidden_states(
        self,
        *,
        last_hidden: torch.Tensor,
        image_mask: torch.Tensor,
        point_mask: torch.Tensor,
        task_mask: torch.Tensor,
        inputs: Dict[str, Any],
    ) -> Dict[str, Optional[torch.Tensor]]:
        image_hidden = None
        point_hidden = None

        if int(image_mask.sum().item()) > 0:
            image_hidden = last_hidden[0][image_mask[0]]

        if int(point_mask.sum().item()) > 0:
            point_hidden = last_hidden[0][point_mask[0]]

        task_hidden = last_hidden[0][task_mask[0]]

        # Keep task hidden in [B, D] form for the top-level PAIR model.
        if task_hidden.ndim == 2 and task_hidden.shape[0] == 1:
            task_hidden_out = task_hidden
        else:
            task_hidden_out = task_hidden.reshape(1, -1)

        aux = {}

        # Recover 2D spatial image feature layout when possible.
        if (
            image_hidden is not None
            and "image_grid_thw" in inputs
        ):
            grid = inputs["image_grid_thw"][0]
            t, h_patch, w_patch = [
                int(x) for x in grid.tolist()
            ]

            merge = int(self.vision_spatial_merge_size)

            h_token = h_patch // merge
            w_token = w_patch // merge

            expected = t * h_token * w_token

            if image_hidden.shape[0] == expected:
                aux["image_hidden_2d"] = image_hidden.reshape(
                    t,
                    h_token,
                    w_token,
                    self.hidden_size,
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
            "task_hidden": task_hidden_out,
            "aux": aux,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        prompt: str,
        images: Optional[Any] = None,
        point_tokens: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        return_hidden_states: bool = True,
        use_cache: bool = False,
        **model_kwargs,
    ) -> Qwen3VLBackboneOutput:
        """
        Run Qwen3-VL with optional external point-token injection.

        This method keeps gradients enabled. It can therefore later be used
        for training the PointAdapter / LoRA / task head.
        """

        prepared = self.prepare_inputs(
            prompt=prompt,
            images=images,
            point_tokens=point_tokens,
        )

        inputs = self._move_inputs_to_model_device(
            prepared["inputs"]
        )

        image_mask = prepared["image_mask"].to(
            self.model_device
        )
        point_mask = prepared["point_mask"].to(
            self.model_device
        )
        task_mask = prepared["task_mask"].to(
            self.model_device
        )

        point_tokens = prepared["point_tokens"]

        full_seq_len = inputs["input_ids"].shape[1]

        hook_stats = {
            "calls": 0,
            "replaced": False,
        }

        hook = self._make_point_injection_hook(
            point_tokens=point_tokens,
            point_mask=point_mask,
            full_seq_len=full_seq_len,
            stats=hook_stats,
        )

        handle = None

        if hook is not None:
            handle = self.model.get_input_embeddings().register_forward_hook(
                hook
            )

        try:
            outputs = self.model(
                **inputs,
                output_hidden_states=return_hidden_states,
                return_dict=True,
                use_cache=use_cache,
                **model_kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

        if point_tokens is not None and not hook_stats["replaced"]:
            raise RuntimeError(
                "External point tokens were provided, but the Qwen "
                "embedding hook did not replace the <POINT> embeddings."
            )

        image_hidden = None
        point_hidden = None
        task_hidden = None
        hidden_aux = {}

        if return_hidden_states:
            last_hidden = outputs.hidden_states[-1]

            extracted = self._extract_hidden_states(
                last_hidden=last_hidden,
                image_mask=image_mask,
                point_mask=point_mask,
                task_mask=task_mask,
                inputs=inputs,
            )

            image_hidden = extracted["image_hidden"]
            point_hidden = extracted["point_hidden"]
            task_hidden = extracted["task_hidden"]
            hidden_aux = extracted["aux"]

        aux = {
            "prompt_text": prepared["prompt_text"],
            "image_count": prepared["image_count"],
            "point_count": prepared["point_count"],
            "task_count": prepared["task_count"],
            "point_injection_calls": hook_stats["calls"],
            "point_injection_replaced": hook_stats["replaced"],
            **hidden_aux,
        }

        return Qwen3VLBackboneOutput(
            image_hidden=image_hidden,
            point_hidden=point_hidden,
            task_hidden=task_hidden,
            logits=outputs.logits if return_logits else None,
            aux=aux,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        images: Optional[Any] = None,
        point_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> Qwen3VLBackboneOutput:
        """
        Native Qwen generation while injecting external point embeddings
        during the prompt/prefill pass.
        """

        prepared = self.prepare_inputs(
            prompt=prompt,
            images=images,
            point_tokens=point_tokens,
        )

        inputs = self._move_inputs_to_model_device(
            prepared["inputs"]
        )

        point_mask = prepared["point_mask"].to(
            self.model_device
        )

        point_tokens = prepared["point_tokens"]

        full_seq_len = inputs["input_ids"].shape[1]

        hook_stats = {
            "calls": 0,
            "replaced": False,
        }

        hook = self._make_point_injection_hook(
            point_tokens=point_tokens,
            point_mask=point_mask,
            full_seq_len=full_seq_len,
            stats=hook_stats,
        )

        handle = None

        if hook is not None:
            handle = self.model.get_input_embeddings().register_forward_hook(
                hook
            )

        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **generate_kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

        if point_tokens is not None and not hook_stats["replaced"]:
            raise RuntimeError(
                "External point tokens were provided, but generation "
                "did not inject them into the Qwen prompt embeddings."
            )

        prompt_len = inputs["input_ids"].shape[1]

        new_token_ids = generated_ids[:, prompt_len:]

        generated_text = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Keep a scalar string for the current batch-size-1 implementation.
        generated_text_out = (
            generated_text[0]
            if len(generated_text) == 1
            else generated_text
        )

        return Qwen3VLBackboneOutput(
            generated_ids=generated_ids,
            generated_text=generated_text_out,
            aux={
                "prompt_text": prepared["prompt_text"],
                "image_count": prepared["image_count"],
                "point_count": prepared["point_count"],
                "task_count": prepared["task_count"],
                "point_injection_calls": hook_stats["calls"],
                "point_injection_replaced": hook_stats["replaced"],
            },
        )


if __name__ == "__main__":
    print("qwen3vl_backbone.py import scaffold OK")
    print("Default checkpoint:", DEFAULT_MODEL_DIR)
