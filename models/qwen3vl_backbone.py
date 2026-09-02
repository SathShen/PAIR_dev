"""
Qwen3-VL backbone wrapper for temporal PAIR.

This module owns only the Qwen3-VL side:

    - processor / tokenizer
    - <POINT> and <TASK> special-token registration
    - Qwen3-VL model loading
    - temporal T1/T2 prompt construction
    - native Qwen image preprocessing
    - T1/T2 image / point / task masks

It does NOT own:
    - PTv3
    - PointAdapter
    - <POINT> embedding injection
    - hidden-state extraction
    - PAIR task routing
    - top-level generation orchestration

Those operations are handled by pair.py.

V1 temporal layout
------------------
Time 1 image
Time 2 image
Time 1 point cloud: <POINT> x N1
Time 2 point cloud: <POINT> x N2
User instruction
<TASK>

Current limitation:
    batch size = 1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


class Qwen3VLBackbone(nn.Module):
    """Thin temporal Qwen3-VL wrapper used by PAIR."""

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

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ==================================================================
    # Point-token shape helper
    # ==================================================================

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

    # ==================================================================
    # Temporal prompt construction
    # ==================================================================

    def _validate_prompt(self, prompt: str) -> None:
        if not isinstance(prompt, str):
            raise TypeError(
                "Qwen3VLBackbone expects prompt to be a string."
            )

        if self.point_token in prompt:
            raise ValueError(
                f"Do not manually include {self.point_token}; "
                "PAIR inserts point placeholders automatically."
            )

        task_count = prompt.count(self.task_token)

        if task_count > 1:
            raise ValueError(
                f"Prompt contains {task_count} copies of "
                f"{self.task_token}; at most one is allowed."
            )

    def _build_messages(
        self,
        *,
        prompt: str,
        images_t1: Optional[Any],
        images_t2: Optional[Any],
        num_point_tokens_t1: int,
        num_point_tokens_t2: int,
    ):
        """
        Build one temporal user message.

        T1/T2 identity is made explicit in natural language rather than by
        creating separate special-token vocabularies.
        """

        self._validate_prompt(prompt)

        content = []

        if images_t1 is not None:
            content.append(
                {
                    "type": "text",
                    "text": "Time 1 image:",
                }
            )
            content.append(
                {
                    "type": "image",
                    "image": images_t1,
                }
            )

        if images_t2 is not None:
            content.append(
                {
                    "type": "text",
                    "text": "Time 2 image:",
                }
            )
            content.append(
                {
                    "type": "image",
                    "image": images_t2,
                }
            )

        if num_point_tokens_t1 > 0:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Time 1 point cloud:\n"
                        + self.point_token * num_point_tokens_t1
                    ),
                }
            )

        if num_point_tokens_t2 > 0:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Time 2 point cloud:\n"
                        + self.point_token * num_point_tokens_t2
                    ),
                }
            )

        task_count = prompt.count(self.task_token)

        task_text = prompt.rstrip()
        if task_count == 0:
            task_text += "\n" + self.task_token

        content.append(
            {
                "type": "text",
                "text": task_text,
            }
        )

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    # ==================================================================
    # Mask helpers
    # ==================================================================

    @staticmethod
    def _mask_from_positions(
        *,
        shape: torch.Size,
        positions: torch.Tensor,
        start: int,
        count: int,
    ) -> torch.Tensor:
        mask = torch.zeros(
            shape,
            dtype=torch.bool,
            device=positions.device,
        )

        if count == 0:
            return mask

        selected = positions[start:start + count]

        if selected.numel() != count:
            raise RuntimeError(
                f"Could not assign {count} token positions "
                f"starting at {start}."
            )

        mask[0, selected] = True
        return mask

    def _image_token_counts(
        self,
        image_grid_thw: Optional[torch.Tensor],
    ) -> List[int]:
        if image_grid_thw is None:
            return []

        counts = []
        merge = int(self.vision_spatial_merge_size)

        for row in image_grid_thw:
            t = int(row[0].item())
            h_patch = int(row[1].item())
            w_patch = int(row[2].item())

            if h_patch % merge != 0 or w_patch % merge != 0:
                raise RuntimeError(
                    "Qwen image grid is not divisible by "
                    "spatial_merge_size."
                )

            counts.append(
                t
                * (h_patch // merge)
                * (w_patch // merge)
            )

        return counts

    # ==================================================================
    # Input preparation
    # ==================================================================

    def prepare_inputs(
        self,
        *,
        prompt: str,
        images_t1: Optional[Any] = None,
        images_t2: Optional[Any] = None,
        point_tokens_t1: Optional[torch.Tensor] = None,
        point_tokens_t2: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Build native Qwen inputs and temporal modality masks.

        This function does NOT inject point embeddings.
        pair.py performs the actual <POINT> replacement.
        """

        n_point_t1 = self._get_num_point_tokens(
            point_tokens_t1
        )
        n_point_t2 = self._get_num_point_tokens(
            point_tokens_t2
        )

        messages = self._build_messages(
            prompt=prompt,
            images_t1=images_t1,
            images_t2=images_t2,
            num_point_tokens_t1=n_point_t1,
            num_point_tokens_t2=n_point_t2,
        )

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_list = []
        image_labels = []

        if images_t1 is not None:
            image_list.append(images_t1)
            image_labels.append("t1")

        if images_t2 is not None:
            image_list.append(images_t2)
            image_labels.append("t2")

        if image_list:
            inputs = self.processor(
                text=[prompt_text],
                images=image_list,
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

        if point_count != n_point_t1 + n_point_t2:
            raise RuntimeError(
                f"Expected {n_point_t1 + n_point_t2} "
                f"{self.point_token} tokens, tokenizer produced "
                f"{point_count}."
            )

        if task_count != 1:
            raise RuntimeError(
                f"Expected exactly one {self.task_token}, "
                f"tokenizer produced {task_count}."
            )

        # --------------------------------------------------------------
        # Split point mask into T1 / T2 by sequence order.
        # The two groups do NOT need to be globally contiguous because
        # the prompt contains a textual T2 label between them.
        # --------------------------------------------------------------
        point_positions = torch.nonzero(
            point_mask[0],
            as_tuple=False,
        ).flatten()

        point_mask_t1 = self._mask_from_positions(
            shape=point_mask.shape,
            positions=point_positions,
            start=0,
            count=n_point_t1,
        )

        point_mask_t2 = self._mask_from_positions(
            shape=point_mask.shape,
            positions=point_positions,
            start=n_point_t1,
            count=n_point_t2,
        )

        # --------------------------------------------------------------
        # Split image mask using Qwen image_grid_thw.
        # --------------------------------------------------------------
        image_grid_thw = inputs.get("image_grid_thw", None)
        image_counts_in_order = self._image_token_counts(
            image_grid_thw
        )

        if len(image_counts_in_order) != len(image_list):
            raise RuntimeError(
                "Number of Qwen image grids does not match the number "
                "of supplied temporal images."
            )

        if sum(image_counts_in_order) != image_count:
            raise RuntimeError(
                f"image_grid_thw implies {sum(image_counts_in_order)} "
                f"image tokens, but tokenizer sequence contains "
                f"{image_count}."
            )

        image_positions = torch.nonzero(
            image_mask[0],
            as_tuple=False,
        ).flatten()

        image_mask_t1 = torch.zeros_like(image_mask)
        image_mask_t2 = torch.zeros_like(image_mask)

        image_count_t1 = 0
        image_count_t2 = 0
        image_grid_index_t1 = None
        image_grid_index_t2 = None

        cursor = 0

        for grid_index, (label, count) in enumerate(
            zip(image_labels, image_counts_in_order)
        ):
            temporal_mask = self._mask_from_positions(
                shape=image_mask.shape,
                positions=image_positions,
                start=cursor,
                count=count,
            )

            if label == "t1":
                image_mask_t1 |= temporal_mask
                image_count_t1 += count
                image_grid_index_t1 = grid_index

            elif label == "t2":
                image_mask_t2 |= temporal_mask
                image_count_t2 += count
                image_grid_index_t2 = grid_index

            cursor += count

        return {
            "prompt_text": prompt_text,
            "inputs": inputs,

            "image_mask": image_mask,
            "image_mask_t1": image_mask_t1,
            "image_mask_t2": image_mask_t2,

            "point_mask": point_mask,
            "point_mask_t1": point_mask_t1,
            "point_mask_t2": point_mask_t2,

            "task_mask": task_mask,

            "image_count": image_count,
            "image_count_t1": image_count_t1,
            "image_count_t2": image_count_t2,

            "point_count": point_count,
            "point_count_t1": n_point_t1,
            "point_count_t2": n_point_t2,

            "task_count": task_count,

            "image_grid_index_t1": image_grid_index_t1,
            "image_grid_index_t2": image_grid_index_t2,
        }

    # ==================================================================
    # Trainability helpers
    # ==================================================================

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
    print("qwen3vl_backbone.py temporal import OK")
    print("Default checkpoint:", DEFAULT_MODEL_DIR)
