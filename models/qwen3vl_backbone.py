"""
Qwen3-VL backbone wrapper for PAIR.

This module ONLY owns the Qwen3-VL side:

    - processor / tokenizer
    - <POINT> and <TASK> registration
    - Qwen3-VL model loading
    - native image preprocessing
    - chat-template / placeholder construction
    - image / point / task masks

It does NOT own:
    - PTv3
    - PointAdapter
    - <POINT> embedding injection
    - hidden-state extraction
    - PAIR task routing
    - top-level generate orchestration

Those operations are handled by PAIR.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


class Qwen3VLBackbone(nn.Module):
    """Thin Qwen3-VL wrapper used by PAIR."""

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

        self.model.resize_token_embeddings(
            len(self.tokenizer)
        )

        if device_map is None:
            self.model.to(device)

        self.hidden_size = (
            self.model.config.text_config.hidden_size
        )

        self.image_token_id = (
            self.model.config.image_token_id
        )

        self.vision_spatial_merge_size = (
            self.model.config.vision_config.spatial_merge_size
        )

    # ------------------------------------------------------------------
    # Properties
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

    def _get_num_point_tokens(
        self,
        point_tokens: Optional[torch.Tensor],
    ) -> int:
        if point_tokens is None:
            return 0

        if not torch.is_tensor(point_tokens):
            raise TypeError(
                "point_tokens must be a torch.Tensor or None."
            )

        if point_tokens.ndim == 2:
            num_tokens, hidden_dim = point_tokens.shape

        elif point_tokens.ndim == 3:
            if point_tokens.shape[0] != 1:
                raise NotImplementedError(
                    "PAIR V1 currently supports batch size 1 "
                    "for external point tokens."
                )

            _, num_tokens, hidden_dim = point_tokens.shape

        else:
            raise ValueError(
                "point_tokens must have shape [N, D] or [1, N, D], "
                f"got {tuple(point_tokens.shape)}."
            )

        if hidden_dim != self.hidden_size:
            raise ValueError(
                f"point token dim must equal Qwen hidden size "
                f"{self.hidden_size}, got {hidden_dim}."
            )

        if num_tokens <= 0:
            raise ValueError(
                "point_tokens contains zero tokens."
            )

        return int(num_tokens)

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
                f"Do not manually include {self.point_token}; "
                "PAIR inserts placeholders automatically."
            )

        task_count = prompt.count(self.task_token)

        if task_count > 1:
            raise ValueError(
                f"Prompt contains {task_count} copies of "
                f"{self.task_token}; at most one is allowed."
            )

        chunks = [prompt.rstrip()]

        if num_point_tokens > 0:
            chunks.append(
                self.point_token * num_point_tokens
            )

        if task_count == 0:
            chunks.append(self.task_token)

        return "\n".join(chunks)

    def _build_messages(
        self,
        *,
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

    # ------------------------------------------------------------------
    # Input preparation only
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        *,
        prompt: str,
        images: Optional[Any] = None,
        point_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Build native Qwen inputs and modality masks.

        IMPORTANT:
        This function does NOT inject point embeddings.
        PAIR.py performs the actual <POINT> replacement.
        """

        num_point_tokens = self._get_num_point_tokens(
            point_tokens
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
                f"Expected {num_point_tokens} {self.point_token} "
                f"tokens, tokenizer produced {point_count}."
            )

        if task_count != 1:
            raise RuntimeError(
                f"Expected exactly one {self.task_token}, "
                f"tokenizer produced {task_count}."
            )

        if images is None and image_count != 0:
            raise RuntimeError(
                "No image was supplied but image tokens were produced."
            )

        if point_count > 0:
            positions = torch.nonzero(
                point_mask[0],
                as_tuple=False,
            ).flatten()

            expected = torch.arange(
                positions[0],
                positions[0] + point_count,
                dtype=positions.dtype,
                device=positions.device,
            )

            if not torch.equal(positions, expected):
                raise RuntimeError(
                    f"{self.point_token} placeholders are not contiguous."
                )

        return {
            "prompt_text": prompt_text,
            "inputs": inputs,
            "image_mask": image_mask,
            "point_mask": point_mask,
            "task_mask": task_mask,
            "image_count": image_count,
            "point_count": point_count,
            "task_count": task_count,
        }

    # ------------------------------------------------------------------
    # Trainability helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        self.model.requires_grad_(False)

    def unfreeze(self) -> None:
        self.model.requires_grad_(True)

    def parameter_count(self) -> int:
        return sum(
            p.numel()
            for p in self.model.parameters()
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )


if __name__ == "__main__":
    print("qwen3vl_backbone.py REFACTORED import OK")
    print("Default checkpoint:", DEFAULT_MODEL_DIR)
