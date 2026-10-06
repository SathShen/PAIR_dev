"""
Batch-aware Qwen3-VL backbone wrapper for temporal PAIR.

Supports true vectorized 2D batching:
    B prompts + B image T1 + B image T2
        -> one Qwen processor batch
        -> one Qwen forward

The wrapper also keeps the single-sample API backward compatible.

PAIR V2 additionally exposes the three Qwen DeepStack vision features before
spatial merging.  Their exact block indices are checkpoint-specific (for
example 4B and 8B use different depths) and are validated against the loaded
checkpoint config.  The normal Qwen forward path is left untouched: features
are captured with forward hooks while the native vision encoder runs.

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
                 point_token: str = "<POINT>", task_token: str = "<TASK>",
                 pyramid_token: str = "<PYRAMID>",
                 vision_intermediate_layers: Optional[Sequence[int]] = None):
        super().__init__()
        self.model_dir = model_dir
        self.dtype = dtype
        self.device_name = str(device)
        self.device_map = device_map
        self.local_files_only = local_files_only
        self.point_token = point_token
        self.task_token = task_token
        self.pyramid_token = pyramid_token

        self.processor = AutoProcessor.from_pretrained(
            model_dir, local_files_only=local_files_only
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                self.point_token,
                self.task_token,
                self.pyramid_token,
            ]
        })
        self.point_token_id = self.tokenizer.convert_tokens_to_ids(self.point_token)
        self.task_token_id = self.tokenizer.convert_tokens_to_ids(self.task_token)
        self.pyramid_token_id = self.tokenizer.convert_tokens_to_ids(
            self.pyramid_token
        )

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

        # PAIR V2 uses the checkpoint's three native Qwen DeepStack depths
        # before the spatial merger.  The compact PAIR config may provide the
        # expected preset values; we validate them against the actual checkpoint
        # so a mismatched 4B/8B path cannot silently run with the wrong hooks.
        configured_deepstack = tuple(
            int(x) for x in getattr(vision_config, "deepstack_visual_indexes", ())
        )
        if len(configured_deepstack) != 3:
            raise RuntimeError(
                "PAIR V2 requires exactly three Qwen deepstack_visual_indexes, "
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
    # PAIR V2: pre-merge Qwen vision features
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

    def _replace_task_token_embedding(self, module, args, output):
        """Replace only <TASK> embedding positions with the trainable PAIR vector."""
        if not args or not torch.is_tensor(args[0]):
            return output
        input_ids = args[0]
        if not (torch.is_tensor(output) and output.ndim == 3):
            return output
        if input_ids.shape != output.shape[:2]:
            return output

        task_mask = input_ids.eq(self.task_token_id)
        if not bool(task_mask.any()):
            return output

        out = output.clone()
        task_vector = self.task_token_embedding.to(
            device=out.device, dtype=out.dtype
        )
        out[task_mask] = task_vector
        return out

    def _validate_prompt(self, prompt: str):
        if self.point_token in prompt:
            raise ValueError(
                f"Do not manually include {self.point_token}; PAIR inserts it automatically"
            )
        if self.task_token in prompt:
            raise ValueError(
                f"Do not manually include {self.task_token}; PAIR appends it as the final readout token"
            )
        if self.pyramid_token in prompt:
            raise ValueError(
                f"Do not manually include {self.pyramid_token}; "
                "PAIR inserts image-pyramid reasoning tokens automatically"
            )

    def _build_messages(
        self,
        prompt,
        image_t1,
        image_t2,
        n_point_t1,
        n_point_t2,
        pyramid_plan=None,
    ):
        self._validate_prompt(prompt)

        # Causal Qwen order:
        #   task instruction -> Time 1 modalities -> Time 2 modalities -> <TASK>.
        #
        # For 2D PAIR, each temporal image is explicitly described to Qwen as
        # a four-level visual pyramid.  P4/P8/P16 are custom compressed
        # DeepStack token blocks and P32 is the native merged Qwen image-token
        # block.  The textual labels are intentionally part of the LLM prompt:
        # the model is told which pyramid level, decoder scale, source Vision
        # layer and compressed token grid each custom block represents.
        content = [{"type": "text", "text": prompt.rstrip()}]

        def append_image_pyramid(time_label, image):
            if image is None:
                return

            if pyramid_plan:
                content.append({
                    "type": "text",
                    "text": (
                        f"{time_label} visual pyramid. "
                        "The next token blocks are multi-scale Qwen Vision features "
                        "for the same image. Use their level/scale labels when "
                        "reasoning about semantic content and temporal change."
                    ),
                })
                # Shallow -> deep so the later/deeper causal states can see the
                # earlier high-resolution semantic context.
                for item in pyramid_plan:
                    count = int(item["count"])
                    content.append({
                        "type": "text",
                        "text": (
                            f"{time_label} pyramid {item['level']} "
                            f"(decoder scale {item['scale']}; derived from Qwen "
                            f"Vision DeepStack layer {item['layer_idx']}; "
                            f"compressed to {item['grid']}x{item['grid']} "
                            "reasoning tokens):\n"
                            + self.pyramid_token * count
                        ),
                    })

            content.extend([
                {
                    "type": "text",
                    "text": (
                        f"{time_label} pyramid P32 "
                        "(decoder scale 1/32; native merged Qwen image tokens):"
                    ),
                },
                {"type": "image", "image": image},
            ])

        append_image_pyramid("Time 1", image_t1)
        if n_point_t1:
            content.append({
                "type": "text",
                "text": "Time 1 point cloud:\n" + self.point_token * n_point_t1,
            })

        append_image_pyramid("Time 2", image_t2)
        if n_point_t2:
            content.append({
                "type": "text",
                "text": "Time 2 point cloud:\n" + self.point_token * n_point_t2,
            })

        # Keep exactly one PAIR readout marker after the complete bi-temporal
        # context.  Its hidden state can therefore summarize every pyramid
        # level from both times.
        content.append({"type": "text", "text": self.task_token})
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

    def prepare_inputs(
        self,
        *,
        prompt,
        images_t1=None,
        images_t2=None,
        point_tokens_t1=None,
        point_tokens_t2=None,
        pyramid_tokens_per_level: int = 0,
        pyramid_grid_size: int = 8,
    ):
        # Prevent stale 2D features from a previous Qwen forward from being
        # consumed accidentally (especially when alternating 2D and 3D batches).
        self.clear_vision_intermediate_cache()

        prompts, single_input = self._prompt_batch(prompt)
        bsz = len(prompts)
        images1 = self._value_batch(images_t1, bsz, "images_t1")
        images2 = self._value_batch(images_t2, bsz, "images_t2")
        points1 = self._point_token_batch(point_tokens_t1, bsz, "point_tokens_t1")
        points2 = self._point_token_batch(point_tokens_t2, bsz, "point_tokens_t2")

        pyramid_tokens_per_level = int(pyramid_tokens_per_level)
        pyramid_grid_size = int(pyramid_grid_size)
        if pyramid_tokens_per_level < 0:
            raise ValueError("pyramid_tokens_per_level must be >= 0")
        if pyramid_grid_size <= 0:
            raise ValueError("pyramid_grid_size must be > 0")
        if pyramid_tokens_per_level not in (0, pyramid_grid_size * pyramid_grid_size):
            raise ValueError(
                "PAIR currently expects one square compressed token grid per "
                "pyramid level: pyramid_tokens_per_level must equal "
                "pyramid_grid_size**2"
            )

        shallow_idx, middle_idx, deep_idx = self.vision_intermediate_layers
        pyramid_plan = []
        if pyramid_tokens_per_level > 0:
            pyramid_plan = [
                {
                    "level": "P4",
                    "key": "p4",
                    "scale": "1/4",
                    "layer_idx": int(shallow_idx),
                    "count": pyramid_tokens_per_level,
                    "grid": pyramid_grid_size,
                },
                {
                    "level": "P8",
                    "key": "p8",
                    "scale": "1/8",
                    "layer_idx": int(middle_idx),
                    "count": pyramid_tokens_per_level,
                    "grid": pyramid_grid_size,
                },
                {
                    "level": "P16",
                    "key": "p16",
                    "scale": "1/16",
                    "layer_idx": int(deep_idx),
                    "count": pyramid_tokens_per_level,
                    "grid": pyramid_grid_size,
                },
            ]

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
                prompts[b],
                images1[b],
                images2[b],
                n1,
                n2,
                pyramid_plan=pyramid_plan,
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
        pyramid_mask = input_ids == self.pyramid_token_id
        point_mask_t1 = torch.zeros_like(point_mask)
        point_mask_t2 = torch.zeros_like(point_mask)
        image_mask_t1 = torch.zeros_like(image_mask)
        image_mask_t2 = torch.zeros_like(image_mask)

        pyramid_masks_t1 = {
            item["key"]: torch.zeros_like(pyramid_mask)
            for item in pyramid_plan
        }
        pyramid_masks_t2 = {
            item["key"]: torch.zeros_like(pyramid_mask)
            for item in pyramid_plan
        }

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

        # Assign the custom pyramid placeholders in exactly the same causal
        # order used by _build_messages: T1 P4/P8/P16, then T2 P4/P8/P16.
        if pyramid_plan:
            for b in range(bsz):
                positions = torch.nonzero(
                    pyramid_mask[b], as_tuple=False
                ).flatten()
                has_t1 = images1[b] is not None
                has_t2 = images2[b] is not None
                expected = (int(has_t1) + int(has_t2)) * sum(
                    int(item["count"]) for item in pyramid_plan
                )
                if positions.numel() != expected:
                    raise RuntimeError(
                        f"Batch {b}: expected {expected} pyramid placeholders, "
                        f"tokenizer produced {positions.numel()}"
                    )
                cursor = 0
                for label, enabled, target_masks in (
                    ("t1", has_t1, pyramid_masks_t1),
                    ("t2", has_t2, pyramid_masks_t2),
                ):
                    if not enabled:
                        continue
                    for item in pyramid_plan:
                        count = int(item["count"])
                        self._assign_positions(
                            target_masks[item["key"]],
                            b,
                            positions,
                            cursor,
                            count,
                        )
                        cursor += count
                if cursor != positions.numel():
                    raise RuntimeError(
                        f"Batch {b}: pyramid placeholder accounting mismatch "
                        f"{cursor} != {positions.numel()}"
                    )
        elif bool(pyramid_mask.any()):
            raise RuntimeError(
                "Prompt unexpectedly contains pyramid placeholders while "
                "pyramid_tokens_per_level=0"
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
            "pyramid_mask": pyramid_mask,
            "pyramid_masks_t1": pyramid_masks_t1,
            "pyramid_masks_t2": pyramid_masks_t2,
            "pyramid_plan": tuple(dict(item) for item in pyramid_plan),
            "pyramid_tokens_per_level": pyramid_tokens_per_level,
            "pyramid_grid_size": pyramid_grid_size,

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