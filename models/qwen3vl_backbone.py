"""Qwen3-VL backbone for PAIR native multimodal temporal reasoning.

Two-stage 3D flow: PointAdapter.encode_single -> llm_point_tokens -> this
backbone -> llm_point_t1/t2 (T_mm) -> PointAdapter.fuse_temporal.
For 2D, native Qwen vision DeepStack D5/D11/D17 and visual-position
LLM hidden T_mm are returned without any custom <PYRAMID> tokens.

Outputs from forward():
  task_hidden       [B,qwen_dim]  -> task Query builder (change_decoder.py)
  llm_visual_t1/t2  list[B] of [1,Hm,Wm,qwen_dim] -> ImageAdapter
  llm_point_t1/t2   list[B] of [Ki,qwen_dim] -> PointAdapter
  premerge_t1/t2    maps from selected native vision levels [B,C,Hv,Wv]

Box proposals require a SEPARATE generated sequence, and require grounding
supervision/fine-tuning for reliable PAIR 2D/3D detection. Generation is not a
latent prediction head and is not automatically trained by mask loss.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import warnings
import torch.nn.functional as F

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
                 point_token: str = "<POINT>", task_token: str = "<TASK>",
                 vision_intermediate_layers: Optional[Sequence[int]] = None):
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
            "additional_special_tokens": [
                self.point_token,
                self.task_token,
            ]
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

        # <TASK> is a PAIR-specific readout token.  The Qwen base embedding
        # table is frozen under LoRA/frozen tuning, so keeping the newly added
        # row inside that table would leave <TASK> random and untrainable.
        # Maintain one tiny standalone trainable vector and replace the frozen
        # embedding output at <TASK> positions on every embedding forward.
        # This adds only hidden_size parameters (2560 for 4B / 4096 for 8B).
        with torch.no_grad():
            task_init = (
                self.model.get_input_embeddings()
                .weight[self.task_token_id]
                .detach()
                .float()
                .clone()
            )
        self.task_token_embedding = nn.Parameter(task_init)
        self._task_embedding_hook_handle = (
            self.model.get_input_embeddings().register_forward_hook(
                self._replace_task_token_embedding
            )
        )
        self.image_token_id = self.model.config.image_token_id

        vision_config = self.model.config.vision_config
        self.vision_hidden_size = int(vision_config.hidden_size)
        self.vision_patch_size = int(vision_config.patch_size)
        self.vision_spatial_merge_size = int(vision_config.spatial_merge_size)

        # PAIR uses the checkpoint's three native Qwen DeepStack depths
        # before the spatial merger.  The compact PAIR config may provide the
        # expected preset values; we validate them against the actual checkpoint
        # so a mismatched 4B/8B path cannot silently run with the wrong hooks.
        configured_deepstack = tuple(
            int(x) for x in getattr(vision_config, "deepstack_visual_indexes", ())
        )
        if len(configured_deepstack) != 3:
            raise RuntimeError(
                "PAIR requires exactly three Qwen deepstack_visual_indexes, "
                f"but this checkpoint reports {configured_deepstack}"
            )

        if vision_intermediate_layers is None:
            requested_deepstack = configured_deepstack
        else:
            requested_deepstack = tuple(int(x) for x in vision_intermediate_layers)
            if len(requested_deepstack) != 3:
                raise ValueError(
                    "vision_intermediate_layers must contain exactly three block "
                    f"indices, got {requested_deepstack}"
                )
            if requested_deepstack != configured_deepstack:
                raise RuntimeError(
                    "PAIR Qwen preset/checkpoint mismatch: config requests "
                    f"DeepStack layers {requested_deepstack}, but checkpoint "
                    f"reports {configured_deepstack}."
                )

        self.vision_intermediate_layers = requested_deepstack

        self._vision_intermediate_cache: Dict[int, torch.Tensor] = {}
        self._vision_hook_handles = []
        self._register_vision_intermediate_hooks()

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
    # PAIR: pre-merge Qwen vision features
    # ------------------------------------------------------------------

    def _vision_module(self) -> nn.Module:
        base_model = getattr(self.model, "model", None)
        vision = getattr(base_model, "visual", None)
        if vision is None:
            raise RuntimeError(
                "Could not locate Qwen vision module at self.model.model.visual"
            )
        return vision

    def _register_vision_intermediate_hooks(self) -> None:
        vision = self._vision_module()
        blocks = getattr(vision, "blocks", None)
        if blocks is None:
            raise RuntimeError("Qwen vision module does not expose .blocks")

        depth = len(blocks)
        for layer_idx in self.vision_intermediate_layers:
            if layer_idx < 0 or layer_idx >= depth:
                raise RuntimeError(
                    f"Requested vision layer {layer_idx}, but Qwen vision depth is {depth}"
                )

            def _capture(_module, _inputs, output, *, _layer_idx=layer_idx):
                hidden = output
                if isinstance(hidden, (tuple, list)):
                    if not hidden:
                        raise RuntimeError(
                            f"Qwen vision layer {_layer_idx} returned an empty output"
                        )
                    hidden = hidden[0]
                if not torch.is_tensor(hidden):
                    hidden = getattr(hidden, "last_hidden_state", None)
                if not torch.is_tensor(hidden) or hidden.ndim != 2:
                    shape = getattr(hidden, "shape", None)
                    raise RuntimeError(
                        f"Unexpected output from Qwen vision layer {_layer_idx}: "
                        f"type={type(hidden).__name__}, shape={shape}"
                    )
                if hidden.shape[-1] != self.vision_hidden_size:
                    raise RuntimeError(
                        f"Qwen vision layer {_layer_idx} hidden dim {hidden.shape[-1]} "
                        f"!= expected {self.vision_hidden_size}"
                    )
                # Do not detach: this remains valid if the vision tower is later
                # unfrozen.  Current PAIR keeps the vision tower frozen.
                self._vision_intermediate_cache[_layer_idx] = hidden

            self._vision_hook_handles.append(
                blocks[layer_idx].register_forward_hook(_capture)
            )

    def clear_vision_intermediate_cache(self) -> None:
        self._vision_intermediate_cache.clear()

    def get_vision_intermediate_flat(
        self,
        layers: Optional[Sequence[int]] = None,
        *,
        require_all: bool = True,
    ) -> Dict[int, torch.Tensor]:
        requested = (
            self.vision_intermediate_layers
            if layers is None
            else tuple(int(x) for x in layers)
        )
        result = {
            layer_idx: self._vision_intermediate_cache[layer_idx]
            for layer_idx in requested
            if layer_idx in self._vision_intermediate_cache
        }
        if require_all and len(result) != len(requested):
            missing = [x for x in requested if x not in result]
            raise RuntimeError(
                "Qwen pre-merge vision features are incomplete. "
                f"Missing layers {missing}. Run a Qwen forward with images first."
            )
        return result

    def _restore_premerge_spatial_layout(
        self,
        flat_feature: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Restore Qwen's merge-grouped patch order to [T, C, H, W].

        Qwen's image processor groups patches as
            [T, H/merge, W/merge, merge_h, merge_w]
        before flattening so that every consecutive merge^2 patches can be fed
        to the patch merger.  A direct ``view(T, H, W, C)`` would therefore
        scramble the spatial layout.
        """
        if flat_feature.ndim != 2:
            raise ValueError(
                f"flat_feature must be [N,C], got {tuple(flat_feature.shape)}"
            )

        t, h_patch, w_patch = [int(x.item()) for x in grid_thw]
        merge = int(self.vision_spatial_merge_size)
        if h_patch % merge or w_patch % merge:
            raise RuntimeError(
                f"Qwen patch grid {(t, h_patch, w_patch)} is not divisible by "
                f"spatial_merge_size={merge}"
            )

        expected = t * h_patch * w_patch
        if flat_feature.shape[0] != expected:
            raise RuntimeError(
                f"Pre-merge token count {flat_feature.shape[0]} != grid product {expected}"
            )

        channels = flat_feature.shape[-1]
        feature = flat_feature.reshape(
            t,
            h_patch // merge,
            w_patch // merge,
            merge,
            merge,
            channels,
        )
        feature = feature.permute(0, 1, 3, 2, 4, 5).contiguous()
        feature = feature.reshape(t, h_patch, w_patch, channels)
        return feature.permute(0, 3, 1, 2).contiguous()

    def get_temporal_vision_feature_maps(
        self,
        prepared: Dict[str, Any],
        layers: Optional[Sequence[int]] = None,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """Return native pre-merge T1/T2 maps as [B,C,H,W].

        This method is intentionally 2D-only.  It expects every sample in the
        prepared batch to contain exactly one T1 image and one T2 image.  It
        does not resample the maps into the 1/4, 1/8, 1/16 pseudo-pyramid; that
        belongs to the downstream 2D decoder path.
        """
        requested = (
            self.vision_intermediate_layers
            if layers is None
            else tuple(int(x) for x in layers)
        )
        cached = self.get_vision_intermediate_flat(requested, require_all=True)

        inputs = prepared.get("inputs")
        if inputs is None:
            raise ValueError("prepared does not contain 'inputs'")
        grid = inputs.get("image_grid_thw")
        if grid is None:
            raise RuntimeError("No image_grid_thw is available for 2D vision features")

        image_records = list(prepared.get("image_records", ()))
        if len(image_records) != int(grid.shape[0]):
            raise RuntimeError(
                f"image_records has {len(image_records)} entries but image_grid_thw "
                f"has {int(grid.shape[0])}"
            )

        batch_size = int(prepared.get("batch_size", 0))
        if batch_size <= 0:
            raise RuntimeError(f"Invalid prepared batch_size={batch_size}")

        raw_counts = [
            int(row[0].item()) * int(row[1].item()) * int(row[2].item())
            for row in grid
        ]
        expected_total = sum(raw_counts)

        temporal_lists: Dict[str, Dict[int, List[Optional[torch.Tensor]]]] = {
            "t1": {layer_idx: [None] * batch_size for layer_idx in requested},
            "t2": {layer_idx: [None] * batch_size for layer_idx in requested},
        }

        for layer_idx in requested:
            flat = cached[layer_idx]
            if flat.shape[0] != expected_total:
                raise RuntimeError(
                    f"Layer {layer_idx} has {flat.shape[0]} pre-merge tokens, "
                    f"expected {expected_total} from image_grid_thw"
                )
            pieces = torch.split(flat, raw_counts, dim=0)

            for record_idx, ((batch_idx, label), piece) in enumerate(
                zip(image_records, pieces)
            ):
                if label not in ("t1", "t2"):
                    raise RuntimeError(f"Unexpected image record label {label!r}")
                feature_tchw = self._restore_premerge_spatial_layout(
                    piece, grid[record_idx]
                )
                if feature_tchw.shape[0] != 1:
                    raise RuntimeError(
                        "PAIR 2D V2 expects still-image features with grid_t=1, "
                        f"got grid_t={feature_tchw.shape[0]}"
                    )
                temporal_lists[label][layer_idx][int(batch_idx)] = feature_tchw[0]

        stacked: Dict[str, Dict[int, torch.Tensor]] = {"t1": {}, "t2": {}}
        for label in ("t1", "t2"):
            for layer_idx in requested:
                features = temporal_lists[label][layer_idx]
                missing = [i for i, feature in enumerate(features) if feature is None]
                if missing:
                    raise RuntimeError(
                        f"Missing {label.upper()} image features at layer {layer_idx} "
                        f"for batch items {missing}"
                    )
                shapes = [tuple(feature.shape) for feature in features if feature is not None]
                if len(set(shapes)) != 1:
                    raise RuntimeError(
                        f"Cannot stack {label.upper()} layer {layer_idx} feature maps "
                        f"with different shapes: {shapes}"
                    )
                stacked[label][layer_idx] = torch.stack(
                    [feature for feature in features if feature is not None], dim=0
                )

        return stacked

    def _replace_task_token_embedding(self, module, args, output):
        """Make the added <TASK> embedding trainable with frozen Qwen weights."""
        if not args or not torch.is_tensor(args[0]):
            return output
        ids = args[0]
        if not torch.is_tensor(output) or output.ndim != 3 or ids.shape != output.shape[:2]:
            return output
        mask = ids.eq(self.task_token_id)
        if not bool(mask.any()):
            return output
        updated = output.clone()
        updated[mask] = self.task_token_embedding.to(device=output.device, dtype=output.dtype)
        return updated

    # ------------------------------------------------------------------
    # Shared batch and modality preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_batch(prompt: Union[str, Sequence[str]]) -> Tuple[List[str], bool]:
        if isinstance(prompt, str):
            return [prompt], True
        if isinstance(prompt, (tuple, list)) and prompt and all(isinstance(x, str) for x in prompt):
            return list(prompt), False
        raise TypeError("prompt must be a string or a non-empty sequence of strings")

    @staticmethod
    def _value_batch(value: Any, batch_size: int, name: str) -> List[Any]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(f"{name} has {len(value)} items, expected {batch_size}")
            return list(value)
        if batch_size != 1:
            raise ValueError(f"{name} must be a list/tuple of length {batch_size}")
        return [value]

    def _point_token_batch(self, value: Any, batch_size: int, name: str) -> List[Optional[torch.Tensor]]:
        if value is None:
            return [None] * batch_size
        if torch.is_tensor(value):
            if value.ndim == 2 and batch_size == 1:
                value = [value]
            elif value.ndim == 3 and value.shape[0] == batch_size:
                value = list(value.unbind(0))
            else:
                raise ValueError(f"{name} must be [K,D] for B=1 or [B,K,D]")
        if not isinstance(value, (list, tuple)) or len(value) != batch_size:
            raise ValueError(f"{name} must contain {batch_size} point-token tensors")
        result = list(value)
        for b, token in enumerate(result):
            if token is None:
                continue
            if not torch.is_tensor(token) or token.ndim != 2 or token.shape[1] != self.hidden_size:
                raise ValueError(f"{name}[{b}] must be [K,{self.hidden_size}]")
            if token.shape[0] == 0 or not torch.is_floating_point(token):
                raise ValueError(f"{name}[{b}] must have nonempty floating tokens")
        return result

    def _validate_prompt(self, prompt: str) -> None:
        for token in (self.point_token, self.task_token, "<PYRAMID>"):
            if token in prompt:
                raise ValueError(f"Do not manually add internal token {token!r} to prompt")

    def _build_messages(self, prompt: str, image_t1: Any, image_t2: Any,
                        point_count_t1: int, point_count_t2: int) -> List[Dict[str, Any]]:
        self._validate_prompt(prompt)
        content = [{"type": "text", "text": prompt.rstrip()}]
        # The standard Qwen image content retains native vision DeepStack logic.
        for time, image, count in (
            ("Time 1", image_t1, point_count_t1),
            ("Time 2", image_t2, point_count_t2),
        ):
            if image is not None:
                content.extend([
                    {"type": "text", "text": f"{time} image:"},
                    {"type": "image", "image": image},
                ])
            if count:
                content.append({"type": "text", "text": f"{time} point cloud:\n" + self.point_token * count})
        # One TASK readout after all T1/T2 modalities for causal conditioning.
        content.append({"type": "text", "text": self.task_token})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _image_token_counts(image_grid_thw: Optional[torch.Tensor], merge: int) -> List[int]:
        if image_grid_thw is None:
            return []
        counts = []
        for t, h, w in image_grid_thw.tolist():
            if h % merge or w % merge:
                raise ValueError(f"Image patch grid {(t, h, w)} is not divisible by {merge}")
            counts.append(int(t * (h // merge) * (w // merge)))
        return counts

    def prepare_inputs(
        self, *, prompt: Union[str, Sequence[str]], images_t1: Any = None,
        images_t2: Any = None, point_tokens_t1: Any = None,
        point_tokens_t2: Any = None,
    ) -> Dict[str, Any]:
        """Build one native Qwen processor batch with exact temporal masks.

        Point tokens are embedded by an embedding hook at forward time, NOT via
        tokenizer embedding weights. This preserves gradients to PointAdapter.
        """
        self.clear_vision_intermediate_cache()
        prompts, single_input = self._prompt_batch(prompt)
        batch_size = len(prompts)
        images1 = self._value_batch(images_t1, batch_size, "images_t1")
        images2 = self._value_batch(images_t2, batch_size, "images_t2")
        points1 = self._point_token_batch(point_tokens_t1, batch_size, "point_tokens_t1")
        points2 = self._point_token_batch(point_tokens_t2, batch_size, "point_tokens_t2")

        messages, image_list, image_records = [], [], []
        point_counts_t1, point_counts_t2 = [], []
        for b in range(batch_size):
            n1 = 0 if points1[b] is None else int(points1[b].shape[0])
            n2 = 0 if points2[b] is None else int(points2[b].shape[0])
            point_counts_t1.append(n1)
            point_counts_t2.append(n2)
            messages.append(self._build_messages(prompts[b], images1[b], images2[b], n1, n2))
            for label, img in (("t1", images1[b]), ("t2", images2[b])):
                if img is not None:
                    image_list.append(img)
                    image_records.append((b, label))

        prompt_texts = [self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ) for msg in messages]
        kwargs = {"text": prompt_texts, "padding": True, "return_tensors": "pt"}
        if image_list:
            kwargs["images"] = image_list
        inputs = self.processor(**kwargs)
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != batch_size:
            raise RuntimeError("Processor batch size mismatch")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise RuntimeError("Processor attention_mask shape mismatch")

        image_mask = input_ids.eq(self.image_token_id)
        point_mask = input_ids.eq(self.point_token_id)
        task_mask = input_ids.eq(self.task_token_id)
        if not bool((task_mask.sum(dim=1) == 1).all()):
            raise RuntimeError("Each sample must contain exactly one <TASK> position")

        point_mask_t1, point_mask_t2 = torch.zeros_like(point_mask), torch.zeros_like(point_mask)
        image_mask_t1, image_mask_t2 = torch.zeros_like(image_mask), torch.zeros_like(image_mask)
        for b in range(batch_size):
            positions = torch.nonzero(point_mask[b], as_tuple=False).flatten()
            expected = point_counts_t1[b] + point_counts_t2[b]
            if positions.numel() != expected:
                raise RuntimeError(f"Batch {b}: point tokens {positions.numel()} != {expected}")
            point_mask_t1[b, positions[:point_counts_t1[b]]] = True
            point_mask_t2[b, positions[point_counts_t1[b]:]] = True
            # Processor must not move point placeholders into temporal disorder.
            if point_counts_t1[b] and point_counts_t2[b]:
                if not bool((positions[:point_counts_t1[b]].max() < positions[point_counts_t1[b]:].min()).item()):
                    raise RuntimeError("Point placeholder ordering invalid")

        grid = inputs.get("image_grid_thw")
        image_counts = self._image_token_counts(grid, self.vision_spatial_merge_size)
        if len(image_counts) != len(image_records):
            raise RuntimeError("Image records / Qwen image_grid_thw mismatch")
        grid_t1, grid_t2 = [None]*batch_size, [None]*batch_size
        counts_t1, counts_t2 = [0]*batch_size, [0]*batch_size
        cursor = [0]*batch_size
        for grid_idx, ((b, label), count) in enumerate(zip(image_records, image_counts)):
            positions = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            selected = positions[cursor[b]:cursor[b]+count]
            if selected.numel() != count:
                raise RuntimeError(f"Batch {b}: native image token count mismatch")
            target = image_mask_t1 if label == "t1" else image_mask_t2
            target[b, selected] = True
            cursor[b] += count
            if label == "t1":
                grid_t1[b], counts_t1[b] = grid_idx, count
            else:
                grid_t2[b], counts_t2[b] = grid_idx, count
        for b in range(batch_size):
            if cursor[b] != int(image_mask[b].sum()):
                raise RuntimeError(f"Batch {b}: unassigned Qwen image tokens")

        return {
            "batch_size": batch_size,
            "single_input": single_input,
            "prompt_texts": prompt_texts,
            "inputs": inputs,
            "point_tokens_t1": points1,
            "point_tokens_t2": points2,
            "point_counts_t1": point_counts_t1,
            "point_counts_t2": point_counts_t2,
            "point_mask": point_mask,
            "point_mask_t1": point_mask_t1,
            "point_mask_t2": point_mask_t2,
            "image_mask": image_mask,
            "image_mask_t1": image_mask_t1,
            "image_mask_t2": image_mask_t2,
            "task_mask": task_mask,
            "image_records": image_records,
            "image_counts_in_order": image_counts,
            "image_grid_indices_t1": grid_t1,
            "image_grid_indices_t2": grid_t2,
            "image_counts_t1": counts_t1,
            "image_counts_t2": counts_t2,
        }

    # ------------------------------------------------------------------
    # Native Qwen hooks, temporal hidden readout
    # ------------------------------------------------------------------

    def _language_module(self) -> nn.Module:
        # Supports the base Qwen model as well as common PEFT LoRA wrappers.
        queue, visited = [self.model], set()
        while queue:
            module = queue.pop(0)
            if module is None or id(module) in visited:
                continue
            visited.add(id(module))
            child = getattr(module, "language_model", None)
            if isinstance(child, nn.Module):
                return child
            for field in ("base_model", "model"):
                nxt = getattr(module, field, None)
                if isinstance(nxt, nn.Module) and nxt is not module:
                    queue.append(nxt)
        raise RuntimeError("Qwen language_model module not found")

    def _make_point_injection_hook(self, prepared: Dict[str, Any], stats: Dict[str, int]):
        mask = prepared["point_mask"]
        bsz, length = mask.shape
        point_t1, point_t2 = prepared["point_tokens_t1"], prepared["point_tokens_t2"]
        for b in range(bsz):
            expected = sum(0 if x is None else x.shape[0] for x in (point_t1[b], point_t2[b]))
            if int(mask[b].sum()) != expected:
                raise RuntimeError(f"Batch {b}: point token injection mismatch")

        def hook(module, args, output):
            if not args or not torch.is_tensor(args[0]):
                return output
            ids = args[0]
            # During generate(), the cached incremental steps contain only the
            # newly generated token. Inject only the full prompt prefill.
            if ids.shape != (bsz, length) or not torch.is_tensor(output):
                return output
            if output.shape != (bsz, length, self.hidden_size):
                raise RuntimeError("Qwen token embedding dimensions changed")
            if not bool(mask.any()):
                return output
            out = output.clone()
            for b in range(bsz):
                parts = [part for part in (point_t1[b], point_t2[b]) if part is not None]
                if not parts:
                    continue
                values = torch.cat(parts, dim=0).to(device=out.device, dtype=out.dtype)
                out[b, mask[b].to(out.device)] = values
                stats["replaced"] += int(values.shape[0])
            stats["prefill_calls"] += 1
            return out
        return hook

    @staticmethod
    def _capture_language_hidden(store: Dict[str, torch.Tensor]):
        def hook(module, args, output):
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, (tuple, list)) and output:
                hidden = output[0]
            if torch.is_tensor(hidden):
                store["last_hidden"] = hidden
        return hook

    @staticmethod
    def _image_hidden_shape(inputs: Dict[str, Any], grid_idx: Optional[int], merge: int):
        if grid_idx is None:
            return None
        t, hp, wp = [int(x) for x in inputs["image_grid_thw"][grid_idx].tolist()]
        if hp % merge or wp % merge:
            raise RuntimeError("Invalid image patch grid shape")
        return t, hp//merge, wp//merge

    def extract_temporal_hidden(self, *, last_hidden: torch.Tensor,
                                prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Read Qwen T_mm from native image/point input positions, never LM head.

        Returns per-phase lists to preserve ragged point token counts. All
        tensors retain gradients when Qwen LoRA is trainable.
        """
        bsz = prepared["batch_size"]
        if last_hidden.ndim != 3 or last_hidden.shape[:2] != prepared["inputs"]["input_ids"].shape:
            raise ValueError("Qwen last_hidden must match prepared input_ids [B,L,D]")
        if last_hidden.shape[-1] != self.hidden_size:
            raise ValueError("Qwen last hidden dimension mismatch")

        def gather(mask_key):
            mask = prepared[mask_key].to(device=last_hidden.device)
            return [last_hidden[b, mask[b]] for b in range(bsz)]

        task_list = gather("task_mask")
        if any(x.shape != (1, self.hidden_size) for x in task_list):
            raise RuntimeError("Each sample must have exactly one task_hidden")
        task_hidden = torch.cat(task_list, dim=0)

        point1, point2 = gather("point_mask_t1"), gather("point_mask_t2")
        for b in range(bsz):
            if point1[b].shape[0] != prepared["point_counts_t1"][b]:
                raise RuntimeError(f"T1 point readout count mismatch for batch {b}")
            if point2[b].shape[0] != prepared["point_counts_t2"][b]:
                raise RuntimeError(f"T2 point readout count mismatch for batch {b}")

        def gather_visual(mask_key, grid_indices):
            parts = gather(mask_key)
            outputs = []
            for b, (tokens, idx) in enumerate(zip(parts, grid_indices)):
                shape = self._image_hidden_shape(prepared["inputs"], idx,
                                                 self.vision_spatial_merge_size)
                if shape is None:
                    if tokens.numel():
                        raise RuntimeError(f"Image {mask_key} grid missing for sample {b}")
                    outputs.append(None)
                else:
                    if tokens.shape[0] != shape[0]*shape[1]*shape[2]:
                        raise RuntimeError(f"Image {mask_key} token shape/count mismatch")
                    outputs.append(tokens.reshape(*shape, self.hidden_size))
            return outputs

        return {
            "task_hidden": task_hidden,
            "llm_visual_t1": gather_visual("image_mask_t1", prepared["image_grid_indices_t1"]),
            "llm_visual_t2": gather_visual("image_mask_t2", prepared["image_grid_indices_t2"]),
            "llm_point_t1": point1,
            "llm_point_t2": point2,
        }

    def _append_grounding_answers(self, prepared, targets, *, max_target_tokens=768):
        """Append supervised assistant answers to the SAME multimodal pass.

        All old masks are remapped to the new left-padded positions. The task
        hidden and T_mm readouts are strictly BEFORE the answer text.
        `grounding_label_positions` identifies the hidden vectors predicting
        each answer token (causal shift of one).
        """
        if len(targets) != prepared["batch_size"]:
            raise ValueError("grounding_targets must contain one answer per sample")
        inputs = prepared["inputs"]
        ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", torch.ones_like(ids))
        bsz = ids.shape[0]
        mask_names = ("point_mask", "point_mask_t1", "point_mask_t2",
                      "image_mask", "image_mask_t1", "image_mask_t2", "task_mask")
        answers = []
        for target in targets:
            if not isinstance(target, str):
                raise TypeError("Grounding targets must be preformatted JSON strings")
            extra = self.tokenizer.encode(target, add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                extra.append(int(self.tokenizer.eos_token_id))
            if not extra or len(extra) > max_target_tokens:
                raise ValueError(f"Box target length {len(extra)} exceeds max_target_tokens={max_target_tokens}")
            answers.append(extra)
        rows = []
        max_len = max(int(attn[b].sum()) + len(answers[b]) for b in range(bsz))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("Qwen tokenizer requires pad_token_id for supervised grounding")
        new_ids = ids.new_full((bsz,max_len),int(pad_id))
        new_attn = attn.new_zeros((bsz,max_len))
        masks = {name: prepared[name].new_zeros((bsz,max_len)) for name in mask_names}
        label_positions = []
        label_targets = []
        for b, answer in enumerate(answers):
            original_positions = torch.nonzero(attn[b].bool(),as_tuple=False).flatten()
            pre = int(original_positions.numel())
            if pre < 1:
                raise RuntimeError("Qwen grounding context is empty")
            total = pre + len(answer)
            offset = max_len-total
            new_ids[b,offset:offset+pre] = ids[b,original_positions]
            new_ids[b,offset+pre:offset+total] = torch.tensor(answer,device=ids.device,dtype=ids.dtype)
            new_attn[b,offset:offset+total] = 1
            for name in mask_names:
                masks[name][b,offset:offset+pre] = prepared[name][b,original_positions]
            for k,token in enumerate(answer):
                label_positions.append((b,offset+pre-1+k))
                label_targets.append(token)
        inputs["input_ids"] = new_ids
        inputs["attention_mask"] = new_attn
        prepared.update(masks)
        prepared["grounding_label_positions"] = torch.tensor(label_positions,dtype=torch.long)
        prepared["grounding_label_targets"] = torch.tensor(label_targets,dtype=torch.long)
        prepared["grounding_num_target_tokens"] = sum(map(len,answers))
        return prepared

    def _grounding_cross_entropy(self, hidden, prepared, *, chunk_size=64):
        positions = prepared["grounding_label_positions"].to(hidden.device)
        targets = prepared["grounding_label_targets"].to(hidden.device)
        if positions.shape[0] == 0:
            raise ValueError("No grounding tokens were supervised")
        # Compute LM logits ONLY on supervised positions, in chunks. Avoid
        # allocating [B,sequence_length,vocab] full vocabulary logits.
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0,positions.shape[0],chunk_size):
            sub = positions[start:start+chunk_size]
            selected = hidden[sub[:,0],sub[:,1]]
            logits = self.model.lm_head(selected)
            total = total + F.cross_entropy(
                logits.float(), targets[start:start+chunk_size], reduction="sum"
            )
        return total / targets.numel()

    def forward(self, *, prompt: Union[str, Sequence[str]], images_t1: Any = None,
                images_t2: Any = None, point_tokens_t1: Any = None,
                point_tokens_t2: Any = None, return_logits: bool = False,
                use_cache: bool = False, grounding_targets: Optional[Sequence[str]] = None,
                grounding_kind: Optional[str] = None,
                grounding_scene_bounds=None, grounding_max_boxes: int = 16,
                grounding_max_target_tokens: int = 768, **qwen_kwargs) -> Dict[str, Any]:
        """One Qwen forward; capture Q and T_mm without duplicating LM layers.

        Boxes are produced separately by generate_box_proposals(), not inferred
        from hidden values. point_tokens should come from PointAdapter.encode_single.
        """
        prompts, _ = self._prompt_batch(prompt)
        if grounding_targets is not None:
            if grounding_kind not in ("2d", "3d"):
                raise ValueError("grounding_kind must be '2d' or '3d' when targets are supplied")
            prompts = self._grounding_prompts(
                prompts, kind=grounding_kind,
                scene_bounds=grounding_scene_bounds,
                max_boxes=grounding_max_boxes,
            )
        prepared = self.prepare_inputs(
            prompt=prompts, images_t1=images_t1, images_t2=images_t2,
            point_tokens_t1=point_tokens_t1, point_tokens_t2=point_tokens_t2,
        )
        if grounding_targets is not None:
            prepared = self._append_grounding_answers(
                prepared, grounding_targets, max_target_tokens=grounding_max_target_tokens
            )
        inputs = {
            k: value.to(self.model_device) if torch.is_tensor(value) else value
            for k, value in prepared["inputs"].items()
        }
        prepared["inputs"] = inputs
        capture: Dict[str, torch.Tensor] = {}
        stats = {"replaced": 0, "prefill_calls": 0}
        point_handle = None
        if bool(prepared["point_mask"].any()):
            point_handle = self.model.get_input_embeddings().register_forward_hook(
                self._make_point_injection_hook(prepared, stats)
            )
        language_handle = self._language_module().register_forward_hook(
            self._capture_language_hidden(capture)
        )
        if not return_logits and "logits_to_keep" not in qwen_kwargs:
            qwen_kwargs["logits_to_keep"] = 1
        try:
            outputs = self.model(**inputs, return_dict=True, use_cache=use_cache, **qwen_kwargs)
        finally:
            language_handle.remove()
            if point_handle is not None:
                point_handle.remove()

        if point_handle is not None:
            expected = sum(prepared["point_counts_t1"]) + sum(prepared["point_counts_t2"])
            if stats["prefill_calls"] != 1 or stats["replaced"] != expected:
                raise RuntimeError(f"Point injection failed: {stats} != {expected} tokens")
        if "last_hidden" not in capture:
            raise RuntimeError("Qwen language_model forward did not expose last_hidden_state")
        temporal = self.extract_temporal_hidden(last_hidden=capture["last_hidden"], prepared=prepared)
        images_present = len(prepared["image_records"]) > 0
        premerge = (self.get_temporal_vision_feature_maps(prepared)
                    if images_present else {"t1": {}, "t2": {}})
        grounding_loss = (
            self._grounding_cross_entropy(capture["last_hidden"], prepared)
            if grounding_targets is not None else None
        )
        return {
            **temporal,
            "grounding_loss": grounding_loss,
            "premerge_t1": premerge["t1"],
            "premerge_t2": premerge["t2"],
            "prepared": prepared,
            "qwen_outputs": outputs if return_logits else None,
        }

    # ------------------------------------------------------------------
    # Optional autoregressive box proposal interface
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_box_json(text: str, *, kind: str, max_boxes: int,
                        scene_bounds=None) -> List[Dict[str, Any]]:
        """Parse JSON proposal sequence. No implicit guessing of coordinate units."""
        # A Qwen response may contain a brief preamble or markdown JSON fence.
        start, stop = text.find("{"), text.rfind("}")
        if start < 0 or stop <= start:
            raise ValueError("Box response does not contain a JSON object")
        try:
            root = json.loads(text[start:stop+1])
        except json.JSONDecodeError as exc:
            raise ValueError("Box response is not valid JSON") from exc
        if not isinstance(root, dict) or not isinstance(root.get("boxes"), list):
            raise ValueError("Box response must be an object with 'boxes' array")
        boxes = root["boxes"]
        if len(boxes) > max_boxes:
            raise ValueError(f"Box count {len(boxes)} exceeds max_boxes={max_boxes}")
        size = 4 if kind == "2d" else 6
        answer = []
        for i, proposal in enumerate(boxes):
            if not isinstance(proposal, dict):
                raise ValueError(f"Box {i} must be a JSON object")
            coords = proposal.get("box")
            score = proposal.get("score", 1.0)
            if not isinstance(coords, (list, tuple)) or len(coords) != size:
                raise ValueError(f"Box {i} needs {size} xyz/xyxy coordinates")
            try:
                vals = [float(v) for v in coords]
                confidence = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Box {i} coordinates or score are not numeric") from exc
            if not bool(torch.isfinite(torch.tensor(vals+[confidence])).all()):
                raise ValueError(f"Box {i} contains non-finite value")
            if not (0 <= confidence <= 1):
                raise ValueError(f"Box {i} score must be in [0,1]")
            half = size//2
            if any(vals[j] >= vals[j+half] for j in range(half)):
                raise ValueError(f"Box {i} has nonpositive width/height/depth")
            if kind == "2d":
                if any(v < 0 or v > 1000 for v in vals):
                    raise ValueError("Qwen 2D Box must use [0,1000] coordinates")
                # Qwen grounding normally uses a 0..1000 box grid. The PAIR
                # decoder consistently accepts normalized 0..1 xyxy.
                vals = [v / 1000.0 for v in vals]
            else:
                if scene_bounds is None:
                    raise ValueError("3D normalized Box decoding requires world scene_bounds")
                if any(v < 0 or v > 1000 for v in vals):
                    raise ValueError("Qwen 3D Box must use [0,1000] normalized coordinates")
                bnd = torch.as_tensor(scene_bounds,device="cpu",dtype=torch.float64)
                if bnd.shape != (2,3):
                    raise ValueError("scene_bounds must be [2,3]")
                # Convert Qwen 0..1000 box coordinates to original XYZ used by
                # PointAdapter and the 3D Soft Spatial Gate.
                lo,span = bnd[0],(bnd[1]-bnd[0]).clamp_min(1e-3)
                norm = torch.tensor(vals,dtype=torch.float64).view(2,3)/1000.0
                world = lo.unsqueeze(0)+norm*span.unsqueeze(0)
                vals = world.reshape(-1).tolist()
            answer.append({"box": vals, "score": confidence})
        return answer

    @staticmethod
    def _grounding_prompts(prompts, *, kind, scene_bounds, max_boxes):
        """One EXACT instruction template for teacher forcing and inference."""
        if kind == "3d":
            if scene_bounds is None or len(scene_bounds) != len(prompts):
                raise ValueError("3D box generation requires scene_bounds [B,2,3]")
            context = []
            for b, bounds in enumerate(scene_bounds):
                if torch.is_tensor(bounds):
                    bounds = bounds.detach().cpu().tolist()
                if len(bounds) != 2 or any(len(row) != 3 for row in bounds):
                    raise ValueError("scene_bounds must contain [min_xyz,max_xyz] pairs")
                context.append(
                    "The point clouds share the world XYZ coordinate frame. "
                    f"Bounds are {json.dumps(bounds)}. Propose axis-aligned regions "
                    "as 6 integers [xmin,ymin,zmin,xmax,ymax,zmax] normalized to "
                    "[0,1000] relative to those shared XYZ bounds (NOT raw world coordinates)."
                )
        else:
            context = ["Give Qwen-style image coordinates [x1,y1,x2,y2] on a [0,1000] grid. For example, half-width is 500."]*len(prompts)
        grounding_prompts = [
            p.rstrip() + "\nFind up to " + str(max_boxes) + " candidate CHANGE REGIONS. "
            + context[b]
            + " Use regions, not one box per object. There may be zero changes. "
              "Reply ONLY with strict JSON: {\"boxes\":[{\"box\":[...],\"score\":0.9},...]}. "
              "Do not include markdown or other text."
            for b,p in enumerate(prompts)
        ]
        return grounding_prompts

    def generate_box_proposals(
        self, *, prompt: Union[str, Sequence[str]], kind: str,
        images_t1: Any = None, images_t2: Any = None,
        point_tokens_t1: Any = None, point_tokens_t2: Any = None,
        scene_bounds: Optional[Sequence[Sequence[Sequence[float]]]] = None,
        max_boxes: int = 64, max_new_tokens: int = 512,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        """Generate optional multi-region proposals via Qwen LM head.

        Separate from forward() because discrete generated boxes are NOT a
        differentiable prediction head. Grounding requires additional data/loss.
        Returns a padded tensor contract consumed by PAIRChangeDecoder.
        """
        if kind not in ("2d", "3d"):
            raise ValueError("kind must be '2d' or '3d'")
        if max_boxes < 0 or max_new_tokens <= 0:
            raise ValueError("max_boxes >= 0 and max_new_tokens > 0 required")
        prompts, _ = self._prompt_batch(prompt)
        grounding_prompts = self._grounding_prompts(
            prompts, kind=kind, scene_bounds=scene_bounds, max_boxes=max_boxes
        )
        prepared = self.prepare_inputs(
            prompt=grounding_prompts, images_t1=images_t1, images_t2=images_t2,
            point_tokens_t1=point_tokens_t1, point_tokens_t2=point_tokens_t2,
        )
        inputs = {k: v.to(self.model_device) if torch.is_tensor(v) else v
                  for k,v in prepared["inputs"].items()}
        prepared["inputs"] = inputs
        stats = {"replaced": 0, "prefill_calls": 0}
        handle = None
        if bool(prepared["point_mask"].any()):
            handle = self.model.get_input_embeddings().register_forward_hook(
                self._make_point_injection_hook(prepared, stats)
            )
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, **generate_kwargs
                )
        finally:
            if handle is not None:
                handle.remove()
        if handle is not None:
            expected = sum(prepared["point_counts_t1"]) + sum(prepared["point_counts_t2"])
            if stats["prefill_calls"] != 1 or stats["replaced"] != expected:
                raise RuntimeError("Point tokens not injected during box generation prefill")
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        if sequences.ndim != 2 or sequences.shape[0] != prepared["batch_size"]:
            raise RuntimeError("Qwen generation must return one sequence per batch item")
        generated_only = sequences[:, inputs["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        # Box is SOFT guidance. An invalid generated JSON/coordinate response
        # must not prevent a valid full-scene mask prediction or validation.
        proposals = []
        parse_failures = 0
        for b, answer_text in enumerate(texts):
            try:
                items = self._parse_box_json(
                    answer_text, kind=kind, max_boxes=max_boxes,
                    scene_bounds=scene_bounds[b] if kind == "3d" else None
                )
            except ValueError as exc:
                parse_failures += 1
                warnings.warn(f"Qwen box response for sample {b} rejected ({exc}); "
                              "falling back to full-scene prediction without Boxes")
                items = []
            proposals.append(items)
        dim = 4 if kind == "2d" else 6
        boxes = torch.zeros(len(proposals), max_boxes, dim, device=self.model_device)
        scores = torch.zeros(len(proposals), max_boxes, device=self.model_device)
        valid = torch.zeros(len(proposals), max_boxes, dtype=torch.bool, device=self.model_device)
        for b, items in enumerate(proposals):
            for i, item in enumerate(items):
                boxes[b,i] = torch.tensor(item["box"], device=boxes.device)
                scores[b,i] = item["score"]
                valid[b,i] = True
        return {
            ("boxes_2d" if kind == "2d" else "boxes_3d"): boxes,
            "box_scores": scores,
            "box_valid": valid,
            "box_texts": texts,
            "box_parse_failures": parse_failures,
        }


if __name__ == "__main__":
    print("PAIR Qwen native DeepStack backbone; invoke after loading checkpoint")
