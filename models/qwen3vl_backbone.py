"""
Batch-aware Qwen3-VL backbone wrapper for temporal PAIR.

Supports true vectorized 2D batching:
    B prompts + B image T1 + B image T2
        -> one Qwen processor batch
        -> one Qwen forward

The wrapper also keeps the single-sample API backward compatible.

For point tokens the input-normalization/mask API is batch-aware, but true
batched 3D still depends on PointAdapter producing per-sample token sets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


class Qwen3VLBackbone(nn.Module):
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR,
                 dtype: torch.dtype = torch.bfloat16,
                 device: Union[str, torch.device] = "cuda",
                 device_map: Optional[Union[str, Dict[str, Any]]] = "cuda",
                 local_files_only: bool = True,
                 point_token: str = "<POINT>", task_token: str = "<TASK>"):
        super().__init__()
        self.model_dir = model_dir
        self.dtype = dtype
        self.device_name = str(device)
        self.device_map = device_map
        self.local_files_only = local_files_only
        self.point_token = point_token
        self.task_token = task_token

        self.processor = AutoProcessor.from_pretrained(
            model_dir, local_files_only=local_files_only
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [self.point_token, self.task_token]
        })
        self.point_token_id = self.tokenizer.convert_tokens_to_ids(self.point_token)
        self.task_token_id = self.tokenizer.convert_tokens_to_ids(self.task_token)

        load_kwargs = {"dtype": dtype, "local_files_only": local_files_only}
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, **load_kwargs
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        if device_map is None:
            self.model.to(device)

        self.hidden_size = self.model.config.text_config.hidden_size
        self.image_token_id = self.model.config.image_token_id
        self.vision_spatial_merge_size = self.model.config.vision_config.spatial_merge_size

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def freeze(self) -> None:
        self.model.requires_grad_(False)

    def unfreeze(self) -> None:
        self.model.requires_grad_(True)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Batch normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_batch(prompt) -> Tuple[List[str], bool]:
        if isinstance(prompt, str):
            return [prompt], True
        if isinstance(prompt, (list, tuple)) and prompt and all(isinstance(x, str) for x in prompt):
            return list(prompt), False
        raise TypeError("prompt must be str or a non-empty sequence[str]")

    @staticmethod
    def _value_batch(value, batch_size: int, name: str):
        if value is None:
            return [None] * batch_size
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(f"{name} has {len(value)} items, expected {batch_size}")
            return list(value)
        if batch_size != 1:
            raise ValueError(f"{name} must be a sequence of length {batch_size}")
        return [value]

    def _point_token_batch(self, value, batch_size: int, name: str):
        if value is None:
            return [None] * batch_size

        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(f"{name} has {len(value)} items, expected {batch_size}")
            values = list(value)
        elif torch.is_tensor(value):
            if value.ndim == 2:
                if batch_size != 1:
                    raise ValueError(f"{name} [N,D] is valid only for batch_size=1")
                values = [value]
            elif value.ndim == 3:
                if value.shape[0] != batch_size:
                    raise ValueError(
                        f"{name} batch dimension {value.shape[0]} != {batch_size}"
                    )
                values = [value[i] for i in range(batch_size)]
            else:
                raise ValueError(f"{name} must be [N,D], [B,N,D], list, or None")
        else:
            raise TypeError(f"{name} must be tensor/list/None")

        for i, tensor in enumerate(values):
            if tensor is None:
                continue
            if not torch.is_tensor(tensor) or tensor.ndim != 2:
                raise ValueError(f"{name}[{i}] must be [N,D]")
            if tensor.shape[1] != self.hidden_size:
                raise ValueError(
                    f"{name}[{i}] hidden dim {tensor.shape[1]} != Qwen {self.hidden_size}"
                )
            if tensor.shape[0] <= 0:
                raise ValueError(f"{name}[{i}] contains zero tokens")
        return values

    def _validate_prompt(self, prompt: str):
        if self.point_token in prompt:
            raise ValueError(
                f"Do not manually include {self.point_token}; PAIR inserts it automatically"
            )
        if prompt.count(self.task_token) > 1:
            raise ValueError(f"Prompt contains more than one {self.task_token}")

    def _build_messages(self, prompt, image_t1, image_t2, n_point_t1, n_point_t2):
        self._validate_prompt(prompt)
        content = []
        if image_t1 is not None:
            content += [{"type": "text", "text": "Time 1 image:"},
                        {"type": "image", "image": image_t1}]
        if image_t2 is not None:
            content += [{"type": "text", "text": "Time 2 image:"},
                        {"type": "image", "image": image_t2}]
        if n_point_t1:
            content.append({
                "type": "text",
                "text": "Time 1 point cloud:\n" + self.point_token * n_point_t1,
            })
        if n_point_t2:
            content.append({
                "type": "text",
                "text": "Time 2 point cloud:\n" + self.point_token * n_point_t2,
            })

        task_text = prompt.rstrip()
        if self.task_token not in prompt:
            task_text += "\n" + self.task_token
        content.append({"type": "text", "text": task_text})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _image_token_counts(image_grid_thw, merge):
        if image_grid_thw is None:
            return []
        counts = []
        for row in image_grid_thw:
            t, h_patch, w_patch = [int(x.item()) for x in row]
            if h_patch % merge or w_patch % merge:
                raise RuntimeError("Qwen image grid is not divisible by spatial_merge_size")
            counts.append(t * (h_patch // merge) * (w_patch // merge))
        return counts

    @staticmethod
    def _assign_positions(mask, batch_idx, positions, start, count):
        if count == 0:
            return
        selected = positions[start:start + count]
        if selected.numel() != count:
            raise RuntimeError(
                f"Could not assign {count} token positions for batch item {batch_idx}"
            )
        mask[batch_idx, selected] = True

    # ------------------------------------------------------------------
    # Native batched processor input
    # ------------------------------------------------------------------

    def prepare_inputs(self, *, prompt, images_t1=None, images_t2=None,
                       point_tokens_t1=None, point_tokens_t2=None):
        prompts, single_input = self._prompt_batch(prompt)
        bsz = len(prompts)
        images1 = self._value_batch(images_t1, bsz, "images_t1")
        images2 = self._value_batch(images_t2, bsz, "images_t2")
        points1 = self._point_token_batch(point_tokens_t1, bsz, "point_tokens_t1")
        points2 = self._point_token_batch(point_tokens_t2, bsz, "point_tokens_t2")

        messages_batch = []
        image_list = []
        image_records = []
        point_counts_t1, point_counts_t2 = [], []

        for b in range(bsz):
            n1 = 0 if points1[b] is None else int(points1[b].shape[0])
            n2 = 0 if points2[b] is None else int(points2[b].shape[0])
            point_counts_t1.append(n1)
            point_counts_t2.append(n2)

            messages = self._build_messages(
                prompts[b], images1[b], images2[b], n1, n2
            )
            messages_batch.append(messages)

            if images1[b] is not None:
                image_list.append(images1[b])
                image_records.append((b, "t1"))
            if images2[b] is not None:
                image_list.append(images2[b])
                image_records.append((b, "t2"))

        prompt_texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]

        kwargs = dict(text=prompt_texts, padding=True, return_tensors="pt")
        if image_list:
            kwargs["images"] = image_list
        inputs = self.processor(**kwargs)

        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != bsz:
            raise RuntimeError(
                f"Processor returned batch {input_ids.shape[0]}, expected {bsz}"
            )

        image_mask = input_ids == self.image_token_id
        point_mask = input_ids == self.point_token_id
        task_mask = input_ids == self.task_token_id
        point_mask_t1 = torch.zeros_like(point_mask)
        point_mask_t2 = torch.zeros_like(point_mask)
        image_mask_t1 = torch.zeros_like(image_mask)
        image_mask_t2 = torch.zeros_like(image_mask)

        # Every sample owns exactly one task token.
        task_counts = task_mask.sum(1)
        if not torch.all(task_counts == 1):
            raise RuntimeError(
                f"Expected one {self.task_token} per sample, got {task_counts.tolist()}"
            )

        # Per-sample point temporal masks.
        for b in range(bsz):
            positions = torch.nonzero(point_mask[b], as_tuple=False).flatten()
            expected = point_counts_t1[b] + point_counts_t2[b]
            if positions.numel() != expected:
                raise RuntimeError(
                    f"Batch {b}: expected {expected} point placeholders, "
                    f"tokenizer produced {positions.numel()}"
                )
            self._assign_positions(
                point_mask_t1, b, positions, 0, point_counts_t1[b]
            )
            self._assign_positions(
                point_mask_t2, b, positions, point_counts_t1[b], point_counts_t2[b]
            )

        # Qwen stores one image_grid_thw row per flattened supplied image.
        grid = inputs.get("image_grid_thw")
        image_counts_in_order = self._image_token_counts(
            grid, int(self.vision_spatial_merge_size)
        )
        if len(image_counts_in_order) != len(image_records):
            raise RuntimeError(
                f"Qwen produced {len(image_counts_in_order)} image grids for "
                f"{len(image_records)} supplied images"
            )

        image_counts_t1 = [0] * bsz
        image_counts_t2 = [0] * bsz
        image_grid_indices_t1 = [None] * bsz
        image_grid_indices_t2 = [None] * bsz
        image_position_cursor = [0] * bsz
        image_positions = [
            torch.nonzero(image_mask[b], as_tuple=False).flatten()
            for b in range(bsz)
        ]

        for grid_idx, ((b, label), count) in enumerate(
            zip(image_records, image_counts_in_order)
        ):
            start = image_position_cursor[b]
            target = image_mask_t1 if label == "t1" else image_mask_t2
            self._assign_positions(target, b, image_positions[b], start, count)
            image_position_cursor[b] += count

            if label == "t1":
                image_counts_t1[b] += count
                image_grid_indices_t1[b] = grid_idx
            else:
                image_counts_t2[b] += count
                image_grid_indices_t2[b] = grid_idx

        for b in range(bsz):
            if image_position_cursor[b] != int(image_mask[b].sum().item()):
                raise RuntimeError(
                    f"Batch {b}: image token accounting mismatch "
                    f"{image_position_cursor[b]} != {int(image_mask[b].sum().item())}"
                )

        return {
            "batch_size": bsz,
            "single_input": single_input,
            "prompt_text": prompt_texts[0] if single_input else prompt_texts,
            "prompt_texts": prompt_texts,
            "inputs": inputs,

            "image_mask": image_mask,
            "image_mask_t1": image_mask_t1,
            "image_mask_t2": image_mask_t2,
            "point_mask": point_mask,
            "point_mask_t1": point_mask_t1,
            "point_mask_t2": point_mask_t2,
            "task_mask": task_mask,

            "image_records": image_records,
            "image_counts_in_order": image_counts_in_order,
            "image_counts_t1": image_counts_t1,
            "image_counts_t2": image_counts_t2,
            "point_counts_t1": point_counts_t1,
            "point_counts_t2": point_counts_t2,
            "image_grid_indices_t1": image_grid_indices_t1,
            "image_grid_indices_t2": image_grid_indices_t2,

            # Backward-compatible single-sample aliases.
            "image_count": int(image_mask.sum().item()),
            "image_count_t1": image_counts_t1[0] if single_input else sum(image_counts_t1),
            "image_count_t2": image_counts_t2[0] if single_input else sum(image_counts_t2),
            "point_count": int(point_mask.sum().item()),
            "point_count_t1": point_counts_t1[0] if single_input else sum(point_counts_t1),
            "point_count_t2": point_counts_t2[0] if single_input else sum(point_counts_t2),
            "task_count": int(task_mask.sum().item()),
            "image_grid_index_t1": image_grid_indices_t1[0] if single_input else None,
            "image_grid_index_t2": image_grid_indices_t2[0] if single_input else None,
        }


if __name__ == "__main__":
    print("qwen3vl_backbone.py batch-aware temporal import OK")
