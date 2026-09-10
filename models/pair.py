"""
PAIR: Prompt-Aware Image-Point Reasoning.

Unified temporal backbone/model for:
    image pair                  -> 2d
    point-cloud pair            -> 3d
    image + point-cloud pair    -> 2d3d

Routing is inferred from supplied modalities. An explicit task_mode is accepted
only as a consistency check.

3D path:
    point_dict T1/T2
        -> frozen Utonia
        -> PointAdapter
             dense [N, decoder_dim]      -> UnifiedChangeDecoder
             reasoning [K, qwen_dim]     -> Qwen <POINT> tokens
        -> Qwen reasoning hidden
        -> semantic + event prediction

Point protocol at the PAIR boundary:
    coord      [N,3] mandatory
    rgb        [N,3] optional
    intensity  [N,1] optional

Missing optional fields are never fabricated at dataset level. During ragged
batching, zero tensors are created only when another sample in the same batch
contains that field, and a matching validity mask is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.change_decoder import (
    TemporalLinks,
    UnifiedChangeDecoder,
    UnifiedTokenSet,
    build_identity_temporal_links,
)
from models.lora import apply_qwen_lora
from models.point_adapter import PointAdapter, PointAdapterConfig
from models.point_encoder import UtoniaPointEncoder, UtoniaPointEncoderConfig
from models.qwen3vl_backbone import Qwen3VLBackbone


SUPPORTED_TASK_MODES = ("2d", "3d", "2d3d")
POINT_TEMPORAL_K = 3
POINT_KNN_CHUNK_SIZE = 2048


@dataclass
class PAIROutput:
    # Dense pre-Qwen modality features.
    image_dense_t1: Optional[torch.Tensor] = None
    image_dense_t2: Optional[torch.Tensor] = None
    point_dense_t1: Optional[torch.Tensor] = None
    point_dense_t2: Optional[torch.Tensor] = None

    # Qwen reasoning features.
    image_hidden_t1: Optional[torch.Tensor] = None
    image_hidden_t2: Optional[torch.Tensor] = None
    point_hidden_t1: Optional[torch.Tensor] = None
    point_hidden_t2: Optional[torch.Tensor] = None
    task_hidden: Optional[torch.Tensor] = None

    logits: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    generated_text: Optional[Any] = None
    aux: Optional[Dict[str, Any]] = None


# =============================================================================
# Backbone orchestration
# =============================================================================

class PAIRBackbone(nn.Module):
    def __init__(self, qwen_backbone: nn.Module, point_encoder: Optional[nn.Module] = None, point_adapter: Optional[nn.Module] = None):
        super().__init__()
        self.qwen_backbone = qwen_backbone
        self.point_encoder = point_encoder
        self.point_adapter = point_adapter

    # -------------------------------------------------------------------------
    # Route inference
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_task_mode(task_mode: str) -> str:
        aliases = {
            "2d": "2d", "image": "2d", "image_only": "2d",
            "3d": "3d", "point": "3d", "point_only": "3d",
            "2d3d": "2d3d", "2d+3d": "2d3d", "image_point": "2d3d", "multimodal": "2d3d",
        }
        task_mode = str(task_mode).lower().strip()
        if task_mode not in aliases:
            raise ValueError(f"Unsupported task_mode={task_mode!r}. Supported: {SUPPORTED_TASK_MODES}")
        return aliases[task_mode]

    @classmethod
    def _resolve_task_mode(cls, *, task_mode, images_t1, images_t2, point_dict_t1, point_dict_t2) -> str:
        has_image_t1, has_image_t2 = images_t1 is not None, images_t2 is not None
        has_point_t1, has_point_t2 = point_dict_t1 is not None, point_dict_t2 is not None

        if has_image_t1 != has_image_t2:
            raise ValueError("Temporal image input must provide both images_t1 and images_t2")
        if has_point_t1 != has_point_t2:
            raise ValueError("Temporal point input must provide both point_dict_t1 and point_dict_t2")

        has_images = has_image_t1 and has_image_t2
        has_points = has_point_t1 and has_point_t2
        if has_images and has_points:
            inferred = "2d3d"
        elif has_images:
            inferred = "2d"
        elif has_points:
            inferred = "3d"
        else:
            raise ValueError("PAIR needs an image pair, a point-cloud pair, or both")

        if task_mode is not None:
            explicit = cls._normalize_task_mode(task_mode)
            if explicit != inferred:
                raise ValueError(f"task_mode={explicit!r} conflicts with supplied modalities, which imply {inferred!r}")
        return inferred

    def _validate_modules(self, task_mode: str):
        if task_mode in ("3d", "2d3d"):
            if self.point_encoder is None:
                raise RuntimeError("3D input requested but point_encoder is not configured")
            if self.point_adapter is None:
                raise RuntimeError("3D input requested but point_adapter is not configured")

    # -------------------------------------------------------------------------
    # Qwen module resolution before/after PEFT wrapping
    # -------------------------------------------------------------------------

    def _find_submodule(self, name: str):
        queue = [self.qwen_backbone.model]
        seen = set()
        while queue:
            module = queue.pop(0)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, name):
                return getattr(module, name)
            for attr in ("base_model", "model"):
                child = getattr(module, attr, None)
                if child is not None and child is not module:
                    queue.append(child)
        raise AttributeError(f"Could not resolve Qwen submodule {name!r}")

    def visual_module(self):
        return self._find_submodule("visual")

    def language_model_module(self):
        return self._find_submodule("language_model")

    # -------------------------------------------------------------------------
    # Shared 3D branch
    # -------------------------------------------------------------------------

    def encode_points(self, point_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        raw point topology
            -> frozen Utonia [N,1386]
            -> PointAdapter
                 dense [N,decoder_dim]
                 reasoning [K,qwen_dim]

        intensity is carried through PointEncoder and fused inside PointAdapter.
        rgb is passed through the point protocol; whether Utonia consumes it is
        owned by the PointEncoder wrapper, not by PAIR.
        """
        point_encoded = self.point_encoder(point_dict)
        adapter_out = self.point_adapter.forward_with_metadata(point_encoded)

        if adapter_out.dense_features.shape[0] != point_encoded.coord.shape[0]:
            raise RuntimeError("PointAdapter dense topology does not match PointEncoder topology")
        if adapter_out.batch.shape[0] != point_encoded.coord.shape[0]:
            raise RuntimeError("PointAdapter batch topology does not match point topology")

        return {
            "point_encoded": point_encoded,
            "point_adapter_output": adapter_out,
            "point_tokens": adapter_out.tokens,
            "point_token_coord": adapter_out.sampled_coord,
            "point_token_indices": adapter_out.sampled_indices,
            "point_dense": adapter_out.dense_features,
            "point_dense_coord": point_encoded.coord,
            "point_dense_batch": adapter_out.batch,
            "point_dense_offset": adapter_out.offset,
            "intensity_used": adapter_out.intensity_used,
            "pooled_voxel_count": adapter_out.pooled_voxel_count,
            "effective_voxel_size": adapter_out.effective_voxel_size,
        }

    @staticmethod
    def _dense_point_fields(point_out) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        if point_out is None:
            return None, None, None, None
        return (
            point_out["point_dense"],
            point_out["point_dense_coord"],
            point_out["point_dense_batch"],
            point_out["point_dense_offset"],
        )

    @staticmethod
    def _flatten_point_metadata(value, *, batch_size: int, name: str):
        """
        PointAdapter may return Tensor for B=1 or list[Tensor] for ragged B>1.
        Flatten to one token tensor plus matching batch IDs.
        """
        if value is None:
            return None, None

        if torch.is_tensor(value):
            if batch_size != 1:
                raise ValueError(f"{name} Tensor form is valid only for batch_size=1")
            if value.ndim < 1:
                raise ValueError(f"{name} must have a token dimension")
            batch_ids = torch.zeros(value.shape[0], dtype=torch.long, device=value.device)
            return value, batch_ids

        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{name} must be Tensor/list/tuple/None")
        if len(value) != batch_size:
            raise ValueError(f"{name} has {len(value)} entries, expected batch_size={batch_size}")

        parts, ids = [], []
        for batch_id, item in enumerate(value):
            if item is None:
                continue
            if not torch.is_tensor(item):
                raise TypeError(f"{name}[{batch_id}] must be a Tensor")
            if item.ndim < 1:
                raise ValueError(f"{name}[{batch_id}] must have a token dimension")
            if item.shape[0] == 0:
                continue
            parts.append(item)
            ids.append(torch.full((item.shape[0],), batch_id, dtype=torch.long, device=item.device))

        if not parts:
            return None, None
        return torch.cat(parts, dim=0), torch.cat(ids, dim=0)

    # -------------------------------------------------------------------------
    # Point-token injection into Qwen
    # -------------------------------------------------------------------------

    def _point_batch(self, point_tokens, batch_size):
        return self.qwen_backbone._point_token_batch(point_tokens, batch_size, "point_tokens")

    def _concat_point_tokens(self, t1, t2, batch_size):
        a = self._point_batch(t1, batch_size)
        b = self._point_batch(t2, batch_size)
        out = []
        for x, y in zip(a, b):
            parts = [z for z in (x, y) if z is not None]
            out.append(None if not parts else torch.cat(parts, dim=0))
        return out

    @staticmethod
    def _validate_point_layout(point_tokens, point_mask):
        if len(point_tokens) != point_mask.shape[0]:
            raise RuntimeError("Point token batch size does not match point mask")
        for batch_id, tokens in enumerate(point_tokens):
            expected = 0 if tokens is None else int(tokens.shape[0])
            actual = int(point_mask[batch_id].sum().item())
            if expected != actual:
                raise RuntimeError(f"Batch {batch_id}: prepared {expected} point tokens but prompt has {actual}")

    def _make_point_injection_hook(self, *, point_tokens, point_mask, full_seq_len, stats):
        self._validate_point_layout(point_tokens, point_mask)
        hidden_size = self.qwen_backbone.hidden_size
        batch_size = len(point_tokens)

        def hook(module, args, output):
            stats["calls"] += 1
            if not (torch.is_tensor(output) and output.ndim == 3):
                return output
            if output.shape != (batch_size, full_seq_len, hidden_size):
                return output

            out = output.clone()
            replaced = False
            for batch_id, tokens in enumerate(point_tokens):
                if tokens is None:
                    continue
                mask = point_mask[batch_id].to(out.device)
                out[batch_id, mask] = tokens.to(device=out.device, dtype=out.dtype)
                replaced = True
            stats["replaced"] = replaced
            return out

        return hook

    # -------------------------------------------------------------------------
    # Capture hooks
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_visual_capture_hook(store):
        def hook(module, args, output):
            if isinstance(output, (tuple, list)) and output:
                store["dense"] = output[0]
            elif torch.is_tensor(output):
                store["dense"] = output
        return hook

    @staticmethod
    def _make_language_capture_hook(store):
        def hook(module, args, output):
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, (tuple, list)) and output:
                hidden = output[0]
            if hidden is not None:
                store["last_hidden"] = hidden
        return hook

    # -------------------------------------------------------------------------
    # Image helpers
    # -------------------------------------------------------------------------

    def _image_token_shape(self, inputs, grid_index):
        if grid_index is None or "image_grid_thw" not in inputs:
            return None
        grid = inputs["image_grid_thw"][grid_index]
        t, hp, wp = [int(x.item()) for x in grid]
        merge = int(self.qwen_backbone.vision_spatial_merge_size)
        if hp % merge or wp % merge:
            raise RuntimeError("Qwen image grid is not divisible by spatial_merge_size")
        return t, hp // merge, wp // merge

    def _image_token_shapes(self, inputs, indices):
        return [self._image_token_shape(inputs, idx) for idx in indices]

    @staticmethod
    def _split_visual_dense(vision_dense, prepared):
        if vision_dense is None:
            return None, None, None, None

        records = prepared["image_records"]
        counts = prepared["image_counts_in_order"]
        batch_size = prepared["batch_size"]
        t1_parts = [[] for _ in range(batch_size)]
        t2_parts = [[] for _ in range(batch_size)]
        cursor = 0

        for (batch_id, label), count in zip(records, counts):
            part = vision_dense[cursor:cursor + count]
            if part.shape[0] != count:
                raise RuntimeError("Vision dense token accounting mismatch")
            if label == "t1":
                t1_parts[batch_id].append(part)
            else:
                t2_parts[batch_id].append(part)
            cursor += count

        if cursor != vision_dense.shape[0]:
            raise RuntimeError(f"Vision dense count {vision_dense.shape[0]} != consumed {cursor}")

        def flatten(parts):
            features, batch_ids = [], []
            for batch_id, chunks in enumerate(parts):
                if not chunks:
                    continue
                x = torch.cat(chunks, dim=0)
                features.append(x)
                batch_ids.append(torch.full((x.shape[0],), batch_id, dtype=torch.long, device=x.device))
            if not features:
                return None, None
            return torch.cat(features, dim=0), torch.cat(batch_ids, dim=0)

        f1, b1 = flatten(t1_parts)
        f2, b2 = flatten(t2_parts)
        return f1, f2, b1, b2

    @staticmethod
    def _reshape_single(tokens, shape):
        if tokens is None or shape is None:
            return None
        t, h, w = shape
        if tokens.shape[0] != t * h * w:
            raise RuntimeError(f"Image token count {tokens.shape[0]} does not match grid {shape}")
        return tokens.reshape(t, h, w, tokens.shape[-1])

    def _split_by_batch(self, tokens, batch_ids, shapes):
        if tokens is None:
            return [None for _ in shapes]
        return [self._reshape_single(tokens[batch_ids == batch_id], shape) for batch_id, shape in enumerate(shapes)]

    # -------------------------------------------------------------------------
    # Qwen hidden extraction
    # -------------------------------------------------------------------------

    def _extract_multimodal_hidden(self, *, last_hidden, prepared, inputs):
        device = last_hidden.device
        batch_size = prepared["batch_size"]
        image_mask_t1 = prepared["image_mask_t1"].to(device)
        image_mask_t2 = prepared["image_mask_t2"].to(device)
        point_mask_t1 = prepared["point_mask_t1"].to(device)
        point_mask_t2 = prepared["point_mask_t2"].to(device)
        task_mask = prepared["task_mask"].to(device)

        def flatten(mask):
            parts, batch_ids = [], []
            for batch_id in range(batch_size):
                selected = last_hidden[batch_id][mask[batch_id]]
                if selected.numel() == 0:
                    continue
                parts.append(selected)
                batch_ids.append(torch.full((selected.shape[0],), batch_id, dtype=torch.long, device=device))
            if not parts:
                return None, None
            return torch.cat(parts, dim=0), torch.cat(batch_ids, dim=0)

        image_hidden_t1, image_batch_t1 = flatten(image_mask_t1)
        image_hidden_t2, image_batch_t2 = flatten(image_mask_t2)
        point_hidden_t1, point_batch_t1 = flatten(point_mask_t1)
        point_hidden_t2, point_batch_t2 = flatten(point_mask_t2)

        task_parts = []
        for batch_id in range(batch_size):
            x = last_hidden[batch_id][task_mask[batch_id]]
            if x.shape[0] != 1:
                raise RuntimeError(f"Batch {batch_id}: expected one task hidden, got {x.shape[0]}")
            task_parts.append(x[0])
        task_hidden = torch.stack(task_parts, dim=0)

        shapes_t1 = self._image_token_shapes(inputs, prepared["image_grid_indices_t1"])
        shapes_t2 = self._image_token_shapes(inputs, prepared["image_grid_indices_t2"])

        return {
            "image_hidden_t1": image_hidden_t1,
            "image_hidden_t2": image_hidden_t2,
            "point_hidden_t1": point_hidden_t1,
            "point_hidden_t2": point_hidden_t2,
            "task_hidden": task_hidden,
            "image_batch_ids_t1": image_batch_t1,
            "image_batch_ids_t2": image_batch_t2,
            "point_batch_ids_t1": point_batch_t1,
            "point_batch_ids_t2": point_batch_t2,
            "image_hidden_2d_t1_list": self._split_by_batch(image_hidden_t1, image_batch_t1, shapes_t1),
            "image_hidden_2d_t2_list": self._split_by_batch(image_hidden_t2, image_batch_t2, shapes_t2),
        }

    # -------------------------------------------------------------------------
    # Qwen preparation
    # -------------------------------------------------------------------------

    def _prepare_qwen(self, *, prompt, images_t1, images_t2, point_tokens_t1, point_tokens_t2):
        prepared = self.qwen_backbone.prepare_inputs(
            prompt=prompt,
            images_t1=images_t1,
            images_t2=images_t2,
            point_tokens_t1=point_tokens_t1,
            point_tokens_t2=point_tokens_t2,
        )
        device = self.qwen_backbone.model_device
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in prepared["inputs"].items()}
        point_tokens = self._concat_point_tokens(point_tokens_t1, point_tokens_t2, prepared["batch_size"])
        return {**prepared, "inputs": inputs, "point_tokens": point_tokens}

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        *,
        prompt,
        images_t1=None,
        images_t2=None,
        point_dict_t1=None,
        point_dict_t2=None,
        task_mode: Optional[str] = None,
        return_logits: bool = True,
        return_hidden_states: bool = True,
        return_dense_features: bool = True,
        use_cache: bool = False,
        **qwen_kwargs,
    ):
        task_mode = self._resolve_task_mode(
            task_mode=task_mode,
            images_t1=images_t1,
            images_t2=images_t2,
            point_dict_t1=point_dict_t1,
            point_dict_t2=point_dict_t2,
        )
        self._validate_modules(task_mode)

        point_out_t1 = point_out_t2 = None
        point_tokens_t1 = point_tokens_t2 = None
        if task_mode in ("3d", "2d3d"):
            point_out_t1 = self.encode_points(point_dict_t1)
            point_out_t2 = self.encode_points(point_dict_t2)
            point_tokens_t1 = point_out_t1["point_tokens"]
            point_tokens_t2 = point_out_t2["point_tokens"]

        prepared = self._prepare_qwen(
            prompt=prompt,
            images_t1=images_t1 if task_mode in ("2d", "2d3d") else None,
            images_t2=images_t2 if task_mode in ("2d", "2d3d") else None,
            point_tokens_t1=point_tokens_t1 if task_mode in ("3d", "2d3d") else None,
            point_tokens_t2=point_tokens_t2 if task_mode in ("3d", "2d3d") else None,
        )
        inputs = prepared["inputs"]
        point_tokens = prepared["point_tokens"]
        stats = {"calls": 0, "replaced": False}
        visual_capture, language_capture, handles = {}, {}, []

        if any(item is not None for item in point_tokens):
            handles.append(
                self.qwen_backbone.model.get_input_embeddings().register_forward_hook(
                    self._make_point_injection_hook(
                        point_tokens=point_tokens,
                        point_mask=prepared["point_mask"],
                        full_seq_len=inputs["input_ids"].shape[1],
                        stats=stats,
                    )
                )
            )

        if return_dense_features and task_mode in ("2d", "2d3d"):
            handles.append(self.visual_module().register_forward_hook(self._make_visual_capture_hook(visual_capture)))
        if return_hidden_states:
            handles.append(self.language_model_module().register_forward_hook(self._make_language_capture_hook(language_capture)))
        if not return_logits and "logits_to_keep" not in qwen_kwargs:
            qwen_kwargs["logits_to_keep"] = 1

        try:
            qwen_outputs = self.qwen_backbone.model(
                **inputs, return_dict=True, use_cache=use_cache, **qwen_kwargs
            )
        finally:
            for handle in handles:
                handle.remove()

        if any(item is not None for item in point_tokens) and not stats["replaced"]:
            raise RuntimeError("Temporal point tokens were prepared but not injected into Qwen")

        # Image dense features.
        image_dense_t1 = image_dense_t2 = None
        image_dense_batch_t1 = image_dense_batch_t2 = None
        if return_dense_features and task_mode in ("2d", "2d3d"):
            if "dense" not in visual_capture:
                raise RuntimeError("Qwen vision forward ran but image dense features were not captured")
            image_dense_t1, image_dense_t2, image_dense_batch_t1, image_dense_batch_t2 = self._split_visual_dense(
                visual_capture["dense"], prepared
            )

        # Qwen reasoning features.
        hidden = {}
        if return_hidden_states:
            if "last_hidden" not in language_capture:
                raise RuntimeError("Qwen language hidden state was not captured")
            hidden = self._extract_multimodal_hidden(
                last_hidden=language_capture["last_hidden"], prepared=prepared, inputs=inputs
            )

        # Point dense topology from PointAdapter.
        point_dense_t1, point_dense_coord_t1, point_dense_batch_t1, point_dense_offset_t1 = self._dense_point_fields(point_out_t1)
        point_dense_t2, point_dense_coord_t2, point_dense_batch_t2, point_dense_offset_t2 = self._dense_point_fields(point_out_t2)
        if not return_dense_features:
            point_dense_t1 = point_dense_t2 = None

        # Flatten reasoning coordinates to match Qwen point hidden topology.
        point_reasoning_coord_t1, point_coord_batch_t1 = self._flatten_point_metadata(
            None if point_out_t1 is None else point_out_t1["point_token_coord"],
            batch_size=prepared["batch_size"],
            name="point_token_coord_t1",
        )
        point_reasoning_coord_t2, point_coord_batch_t2 = self._flatten_point_metadata(
            None if point_out_t2 is None else point_out_t2["point_token_coord"],
            batch_size=prepared["batch_size"],
            name="point_token_coord_t2",
        )

        if return_hidden_states:
            checks = (
                ("t1", hidden.get("point_hidden_t1"), point_reasoning_coord_t1, hidden.get("point_batch_ids_t1"), point_coord_batch_t1),
                ("t2", hidden.get("point_hidden_t2"), point_reasoning_coord_t2, hidden.get("point_batch_ids_t2"), point_coord_batch_t2),
            )
            for name, reasoning_hidden, reasoning_coord, hidden_batch, coord_batch in checks:
                if reasoning_hidden is None:
                    if reasoning_coord is not None:
                        raise RuntimeError(f"Point reasoning coordinates exist for {name} but Qwen reasoning hidden is missing")
                    continue
                if reasoning_coord is None:
                    raise RuntimeError(f"Qwen point hidden exists for {name} but reasoning coordinates are missing")
                if reasoning_hidden.shape[0] != reasoning_coord.shape[0]:
                    raise RuntimeError(
                        f"Point reasoning topology mismatch at {name}: "
                        f"hidden={reasoning_hidden.shape[0]}, coord={reasoning_coord.shape[0]}"
                    )
                if hidden_batch is None or coord_batch is None:
                    raise RuntimeError(f"Point reasoning batch metadata missing at {name}")
                if not torch.equal(hidden_batch, coord_batch.to(hidden_batch.device)):
                    raise RuntimeError(f"Point reasoning batch IDs mismatch at {name}")

        # 2D topology metadata.
        shapes_t1 = self._image_token_shapes(inputs, prepared["image_grid_indices_t1"])
        shapes_t2 = self._image_token_shapes(inputs, prepared["image_grid_indices_t2"])
        single = prepared["single_input"]
        dense_2d_t1 = self._split_by_batch(image_dense_t1, image_dense_batch_t1, shapes_t1)
        dense_2d_t2 = self._split_by_batch(image_dense_t2, image_dense_batch_t2, shapes_t2)

        aux = {
            "task_mode": task_mode,
            "batch_size": prepared["batch_size"],
            "image_dense_batch_ids_t1": image_dense_batch_t1,
            "image_dense_batch_ids_t2": image_dense_batch_t2,
            "image_reasoning_batch_ids_t1": hidden.get("image_batch_ids_t1"),
            "image_reasoning_batch_ids_t2": hidden.get("image_batch_ids_t2"),
            "image_token_shapes_t1": shapes_t1,
            "image_token_shapes_t2": shapes_t2,
            "image_dense_2d_t1_list": dense_2d_t1,
            "image_dense_2d_t2_list": dense_2d_t2,
            "image_hidden_2d_t1_list": hidden.get("image_hidden_2d_t1_list"),
            "image_hidden_2d_t2_list": hidden.get("image_hidden_2d_t2_list"),
            "point_encoded_t1": None if point_out_t1 is None else point_out_t1["point_encoded"],
            "point_encoded_t2": None if point_out_t2 is None else point_out_t2["point_encoded"],
            "point_adapter_output_t1": None if point_out_t1 is None else point_out_t1["point_adapter_output"],
            "point_adapter_output_t2": None if point_out_t2 is None else point_out_t2["point_adapter_output"],
            "point_dense_coord_t1": point_dense_coord_t1,
            "point_dense_coord_t2": point_dense_coord_t2,
            "point_dense_batch_t1": point_dense_batch_t1,
            "point_dense_batch_t2": point_dense_batch_t2,
            "point_dense_offset_t1": point_dense_offset_t1,
            "point_dense_offset_t2": point_dense_offset_t2,
            "point_reasoning_coord_t1": point_reasoning_coord_t1,
            "point_reasoning_coord_t2": point_reasoning_coord_t2,
            "point_reasoning_batch_ids_t1": hidden.get("point_batch_ids_t1"),
            "point_reasoning_batch_ids_t2": hidden.get("point_batch_ids_t2"),
            "point_tokens_t1": point_tokens_t1,
            "point_tokens_t2": point_tokens_t2,
            "point_token_indices_t1": None if point_out_t1 is None else point_out_t1["point_token_indices"],
            "point_token_indices_t2": None if point_out_t2 is None else point_out_t2["point_token_indices"],
            "point_intensity_used_t1": False if point_out_t1 is None else bool(point_out_t1["intensity_used"]),
            "point_intensity_used_t2": False if point_out_t2 is None else bool(point_out_t2["intensity_used"]),
            "point_pooled_voxel_count_t1": None if point_out_t1 is None else point_out_t1["pooled_voxel_count"],
            "point_pooled_voxel_count_t2": None if point_out_t2 is None else point_out_t2["pooled_voxel_count"],
            "point_effective_token_voxel_size_t1": None if point_out_t1 is None else point_out_t1["effective_voxel_size"],
            "point_effective_token_voxel_size_t2": None if point_out_t2 is None else point_out_t2["effective_voxel_size"],
            "prompt_text": prepared["prompt_text"],
            "prompt_texts": prepared["prompt_texts"],
            "image_counts_t1": prepared["image_counts_t1"],
            "image_counts_t2": prepared["image_counts_t2"],
            "point_counts_t1": prepared["point_counts_t1"],
            "point_counts_t2": prepared["point_counts_t2"],
            "task_count": prepared["task_count"],
            "point_injection_calls": stats["calls"],
            "point_injection_replaced": stats["replaced"],
        }

        if single:
            aux.update({
                "image_token_shape_t1": shapes_t1[0],
                "image_token_shape_t2": shapes_t2[0],
                "image_dense_2d_t1": dense_2d_t1[0],
                "image_dense_2d_t2": dense_2d_t2[0],
                "image_hidden_2d_t1": (hidden.get("image_hidden_2d_t1_list") or [None])[0],
                "image_hidden_2d_t2": (hidden.get("image_hidden_2d_t2_list") or [None])[0],
            })

        return PAIROutput(
            image_dense_t1=image_dense_t1,
            image_dense_t2=image_dense_t2,
            point_dense_t1=point_dense_t1,
            point_dense_t2=point_dense_t2,
            image_hidden_t1=hidden.get("image_hidden_t1"),
            image_hidden_t2=hidden.get("image_hidden_t2"),
            point_hidden_t1=hidden.get("point_hidden_t1"),
            point_hidden_t2=hidden.get("point_hidden_t2"),
            task_hidden=hidden.get("task_hidden"),
            logits=qwen_outputs.logits if return_logits else None,
            aux=aux,
        )

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt,
        images_t1=None,
        images_t2=None,
        point_dict_t1=None,
        point_dict_t2=None,
        task_mode: Optional[str] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        **generate_kwargs,
    ):
        task_mode = self._resolve_task_mode(
            task_mode=task_mode,
            images_t1=images_t1,
            images_t2=images_t2,
            point_dict_t1=point_dict_t1,
            point_dict_t2=point_dict_t2,
        )
        self._validate_modules(task_mode)

        point_tokens_t1 = point_tokens_t2 = None
        if task_mode in ("3d", "2d3d"):
            point_out_t1 = self.encode_points(point_dict_t1)
            point_out_t2 = self.encode_points(point_dict_t2)
            point_tokens_t1 = point_out_t1["point_tokens"]
            point_tokens_t2 = point_out_t2["point_tokens"]

        prepared = self._prepare_qwen(
            prompt=prompt,
            images_t1=images_t1 if task_mode in ("2d", "2d3d") else None,
            images_t2=images_t2 if task_mode in ("2d", "2d3d") else None,
            point_tokens_t1=point_tokens_t1 if task_mode in ("3d", "2d3d") else None,
            point_tokens_t2=point_tokens_t2 if task_mode in ("3d", "2d3d") else None,
        )
        inputs = prepared["inputs"]
        point_tokens = prepared["point_tokens"]
        stats = {"calls": 0, "replaced": False}
        handle = None

        if any(item is not None for item in point_tokens):
            handle = self.qwen_backbone.model.get_input_embeddings().register_forward_hook(
                self._make_point_injection_hook(
                    point_tokens=point_tokens,
                    point_mask=prepared["point_mask"],
                    full_seq_len=inputs["input_ids"].shape[1],
                    stats=stats,
                )
            )

        try:
            ids = self.qwen_backbone.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **generate_kwargs
            )
        finally:
            if handle is not None:
                handle.remove()

        if any(item is not None for item in point_tokens) and not stats["replaced"]:
            raise RuntimeError("Point tokens were prepared but not injected during generation")

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompt_lens = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
        else:
            prompt_lens = attention_mask.sum(dim=1).tolist()

        texts = []
        for batch_id, prompt_len in enumerate(prompt_lens):
            new_ids = ids[batch_id, int(prompt_len):]
            texts.append(self.qwen_backbone.processor.decode(new_ids, skip_special_tokens=True))

        return PAIROutput(
            generated_ids=ids,
            generated_text=texts[0] if prepared["single_input"] else texts,
            aux={"task_mode": task_mode, "batch_size": prepared["batch_size"]},
        )


# =============================================================================
# Dense model helpers
# =============================================================================

class ImageDenseAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        return self.proj(x)


def tensor_to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if not torch.is_tensor(image):
        raise TypeError(f"Image must be PIL.Image or Tensor, got {type(image).__name__}")

    x = image.detach().cpu().float()
    if x.ndim != 3 or x.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected image [C,H,W], got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("Image contains NaN/Inf")
    if float(x.min()) < -1e-4 or float(x.max()) > 1.0001:
        raise ValueError(f"Expected image in [0,1], got [{float(x.min())}, {float(x.max())}]")

    x = x.clamp(0, 1).mul(255).round().to(torch.uint8)
    arr = x.permute(1, 2, 0).contiguous().numpy()
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def make_grid_positions(shape, device):
    if shape is None:
        raise RuntimeError("Missing image token shape")
    t, h, w = shape
    if t != 1:
        raise NotImplementedError(f"Current 2D path expects T=1, got {shape}")

    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    zz = torch.zeros(h * w, device=device)
    return torch.stack((xx.reshape(-1), yy.reshape(-1), zz), dim=1)


def make_batched_image_token_set(features, shapes, batch_ids):
    if features is None or batch_ids is None:
        raise RuntimeError("Missing batched image features/batch IDs")

    positions, expected_ids = [], []
    for batch_id, shape in enumerate(shapes):
        pos = make_grid_positions(shape, features.device)
        positions.append(pos)
        expected_ids.append(torch.full((pos.shape[0],), batch_id, dtype=torch.long, device=features.device))

    positions = torch.cat(positions, dim=0)
    expected_ids = torch.cat(expected_ids, dim=0)
    batch_ids = batch_ids.to(features.device).long()

    if features.shape[0] != positions.shape[0]:
        raise RuntimeError(f"Feature/position count mismatch: {features.shape[0]} vs {positions.shape[0]}")
    if not torch.equal(batch_ids, expected_ids):
        raise RuntimeError("PAIR image token order/batch IDs do not match sample-major layout")

    n = features.shape[0]
    return UnifiedTokenSet(
        features=features,
        positions=positions,
        modality_ids=torch.zeros(n, dtype=torch.long, device=features.device),
        batch_ids=batch_ids,
    )


def make_point_token_set(features, positions, batch_ids, feature_dim):
    if features is None or positions is None or batch_ids is None:
        raise RuntimeError("Missing point feature/position/batch metadata")
    if features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError(f"Point features must be [N,{feature_dim}], got {tuple(features.shape)}")

    n = features.shape[0]
    positions = positions.to(features.device, dtype=torch.float32)
    batch_ids = batch_ids.to(features.device, dtype=torch.long)
    if positions.shape != (n, 3):
        raise ValueError(f"Point positions must be [N,3], got {tuple(positions.shape)}")
    if batch_ids.shape != (n,):
        raise ValueError(f"Point batch IDs must be [N], got {tuple(batch_ids.shape)}")

    return UnifiedTokenSet(
        features=features,
        positions=positions,
        modality_ids=torch.ones(n, dtype=torch.long, device=features.device),
        batch_ids=batch_ids,
    )


def restore_2d_prediction_batch(prediction, shapes_t1, shapes_t2, output_sizes):
    def restore_semantic(logits, shapes):
        chunks, cursor = [], 0
        k = logits.shape[1]
        for shape, output_size in zip(shapes, output_sizes):
            t, h, w = shape
            n = t * h * w
            if t != 1:
                raise RuntimeError(f"Expected T=1, got {shape}")
            part = logits[cursor:cursor + n]
            if part.shape[0] != n:
                raise RuntimeError("Semantic token split mismatch")
            x = part.T.reshape(1, k, h, w)
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
            chunks.append(x[0].permute(1, 2, 0).reshape(-1, k))
            cursor += n
        if cursor != logits.shape[0]:
            raise RuntimeError("Unconsumed semantic logits after 2D restore")
        return torch.cat(chunks, dim=0)

    def restore_change(logits, shapes):
        if logits is None:
            raise RuntimeError("2D decoder returned no binary change logits")
        chunks, cursor = [], 0
        for shape, output_size in zip(shapes, output_sizes):
            t, h, w = shape
            n = t * h * w
            if t != 1:
                raise RuntimeError(f"Expected T=1, got {shape}")
            part = logits[cursor:cursor + n]
            if part.numel() != n:
                raise RuntimeError("Change token split mismatch")
            x = part.reshape(1, 1, h, w)
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
            chunks.append(x[0, 0].reshape(-1))
            cursor += n
        if cursor != logits.numel():
            raise RuntimeError("Unconsumed change logits after 2D restore")
        return torch.cat(chunks, dim=0)

    prediction.semantic_logits_t1 = restore_semantic(prediction.semantic_logits_t1, shapes_t1)
    prediction.semantic_logits_t2 = restore_semantic(prediction.semantic_logits_t2, shapes_t2)
    prediction.change_logits_t1 = restore_change(prediction.change_logits_t1, shapes_t1)
    prediction.change_logits_t2 = restore_change(prediction.change_logits_t2, shapes_t2)
    return prediction


# =============================================================================
# Ragged point protocol batching
# =============================================================================

def _as_point_list(value, name):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, dict) for x in value):
        return list(value)
    raise TypeError(f"{name} must be a point dictionary or non-empty list of point dictionaries")


def _validate_optional_mask(mask, n, name, device):
    if mask is None:
        return torch.ones((n, 1), dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(1)
    if mask.shape != (n, 1):
        raise ValueError(f"{name} must be [N] or [N,1], got {tuple(mask.shape)}")
    return mask.bool()


def batch_point_dicts(point_dicts, device):
    """
    Build one ragged runtime point batch.

    Input sample protocol:
        coord      [N,3] mandatory
        rgb        [N,3] optional
        intensity  [N,1] optional

    No fake rgb/intensity is created when the whole batch lacks that field.
    For mixed availability only, missing entries receive runtime zeros plus
    rgb_mask/intensity_mask=False so downstream code can distinguish absence.
    """
    point_dicts = _as_point_list(point_dicts, "point_dicts")
    coords, batch_ids, counts = [], [], []

    any_rgb = any(item.get("rgb") is not None for item in point_dicts)
    any_intensity = any(item.get("intensity") is not None for item in point_dicts)
    rgbs, rgb_masks = [], []
    intensities, intensity_masks = [], []

    for batch_id, item in enumerate(point_dicts):
        coord = item.get("coord")
        if coord is None or not torch.is_tensor(coord):
            raise TypeError(f"point_dicts[{batch_id}]['coord'] must be a Tensor")
        if coord.ndim != 2 or coord.shape[1] != 3 or coord.shape[0] == 0:
            raise ValueError(f"point_dicts[{batch_id}]['coord'] must be non-empty [N,3]")

        coord = coord.to(device=device, dtype=torch.float32, non_blocking=True)
        n = coord.shape[0]
        coords.append(coord)
        counts.append(n)
        batch_ids.append(torch.full((n,), batch_id, dtype=torch.long, device=device))

        if any_rgb:
            rgb = item.get("rgb")
            if rgb is None:
                rgb = torch.zeros((n, 3), dtype=torch.float32, device=device)
                rgb_mask = torch.zeros((n, 1), dtype=torch.bool, device=device)
            else:
                rgb = torch.as_tensor(rgb, dtype=torch.float32, device=device)
                if rgb.shape != (n, 3):
                    raise ValueError(f"point_dicts[{batch_id}]['rgb'] must be [N,3], got {tuple(rgb.shape)}")
                rgb_mask = _validate_optional_mask(item.get("rgb_mask"), n, f"point_dicts[{batch_id}]['rgb_mask']", device)
            rgbs.append(rgb)
            rgb_masks.append(rgb_mask)

        if any_intensity:
            intensity = item.get("intensity")
            if intensity is None:
                intensity = torch.zeros((n, 1), dtype=torch.float32, device=device)
                intensity_mask = torch.zeros((n, 1), dtype=torch.bool, device=device)
            else:
                intensity = torch.as_tensor(intensity, dtype=torch.float32, device=device)
                if intensity.ndim == 1:
                    intensity = intensity.unsqueeze(1)
                if intensity.shape != (n, 1):
                    raise ValueError(
                        f"point_dicts[{batch_id}]['intensity'] must be [N] or [N,1], got {tuple(intensity.shape)}"
                    )
                intensity_mask = _validate_optional_mask(
                    item.get("intensity_mask"), n, f"point_dicts[{batch_id}]['intensity_mask']", device
                )
            intensities.append(intensity)
            intensity_masks.append(intensity_mask)

    result = {
        "coord": torch.cat(coords, dim=0),
        "batch": torch.cat(batch_ids, dim=0),
        "offset": torch.tensor(counts, dtype=torch.long, device=device).cumsum(0),
    }
    if any_rgb:
        result["rgb"] = torch.cat(rgbs, dim=0)
        result["rgb_mask"] = torch.cat(rgb_masks, dim=0)
    if any_intensity:
        result["intensity"] = torch.cat(intensities, dim=0)
        result["intensity_mask"] = torch.cat(intensity_masks, dim=0)
    return result


# =============================================================================
# 3D temporal links
# =============================================================================

@torch.no_grad()
def build_batched_knn_temporal_links(
    target_positions,
    target_batch_ids,
    source_positions,
    source_batch_ids,
    k=POINT_TEMPORAL_K,
    chunk_size=POINT_KNN_CHUNK_SIZE,
):
    """
    Geometry-only temporal correspondence for ragged 3D point sets.
    Each target point links to K nearest opposite-epoch points in the same item.
    """
    if target_positions.ndim != 2 or target_positions.shape[1] != 3:
        raise ValueError("target_positions must be [N,3]")
    if source_positions.ndim != 2 or source_positions.shape[1] != 3:
        raise ValueError("source_positions must be [M,3]")
    if k <= 0 or chunk_size <= 0:
        raise ValueError("k and chunk_size must be > 0")

    device = target_positions.device
    n_target = target_positions.shape[0]
    indices = torch.full((n_target, k), -1, dtype=torch.long, device=device)
    weights = torch.zeros((n_target, k), dtype=torch.float32, device=device)

    target_batch_ids = target_batch_ids.to(device=device, dtype=torch.long)
    source_batch_ids = source_batch_ids.to(device=device, dtype=torch.long)
    target_xyz = target_positions.float()
    source_xyz = source_positions.to(device).float()

    for batch_id in torch.unique(target_batch_ids).tolist():
        target_global = torch.nonzero(target_batch_ids == batch_id, as_tuple=False).flatten()
        source_global = torch.nonzero(source_batch_ids == batch_id, as_tuple=False).flatten()
        if source_global.numel() == 0:
            raise RuntimeError(f"3D temporal link source is empty for batch item {batch_id}")

        kk = min(int(k), int(source_global.numel()))
        source_local_xyz = source_xyz[source_global]
        for start in range(0, target_global.numel(), chunk_size):
            target_chunk_global = target_global[start:start + chunk_size]
            distances = torch.cdist(target_xyz[target_chunk_global], source_local_xyz)
            dist, local_index = torch.topk(distances, k=kk, dim=1, largest=False, sorted=True)
            indices[target_chunk_global, :kk] = source_global[local_index]
            weights[target_chunk_global, :kk] = 1.0 / dist.clamp_min(1e-3)

    return TemporalLinks(source_indices=indices, weights=weights)


# =============================================================================
# Full PAIR model
# =============================================================================

class PAIRModel(nn.Module):
    """
    Complete PAIR model.

    train.py should construct only:
        model = PAIRModel.from_config(experiment.model, device)

    Modules:
        Qwen3-VL Vision      frozen in frozen/LoRA modes
        Qwen language       frozen / LoRA / full
        Utonia              always frozen
        PointAdapter         trainable
        ImageDenseAdapter    trainable
        Unified decoder      trainable
    """

    def __init__(self, backbone, decoder, image_adapter, qwen_tuning):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.image_adapter = image_adapter
        self.qwen_tuning = str(qwen_tuning).lower()

    @property
    def qwen_backbone(self):
        return self.backbone.qwen_backbone

    @property
    def point_encoder(self):
        return self.backbone.point_encoder

    @property
    def point_adapter(self):
        return self.backbone.point_adapter

    @classmethod
    def from_config(cls, model_config, device):
        cfg = dict(model_config)
        device = torch.device(device)
        device_str = str(device)

        qwen = Qwen3VLBackbone(
            model_dir=str(cfg["qwen_model"]),
            dtype=torch.bfloat16,
            device=device_str,
            device_map=device_str,
            local_files_only=True,
        )

        qwen_tuning = str(cfg.get("qwen_tuning", "lora")).lower()
        if qwen_tuning == "frozen":
            qwen.freeze()
        elif qwen_tuning == "full":
            qwen.unfreeze()
        elif qwen_tuning == "lora":
            lora = dict(cfg.get("lora", {}))
            targets = lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            if isinstance(targets, str):
                targets = [x.strip() for x in targets.split(",") if x.strip()]
            apply_qwen_lora(
                qwen,
                r=int(lora.get("r", 16)),
                alpha=int(lora.get("alpha", 32)),
                dropout=float(lora.get("dropout", 0.05)),
                target_modules=tuple(targets),
            )
        else:
            raise ValueError("model.qwen_tuning must be one of: frozen, lora, full")

        decoder_dim = int(cfg.get("decoder_dim", 256))
        point_cfg = dict(cfg.get("point_encoder", {}))
        checkpoint = point_cfg.get("checkpoint")
        if not checkpoint:
            raise KeyError("model.point_encoder.checkpoint is required")

        point_encoder = UtoniaPointEncoder(
            UtoniaPointEncoderConfig(
                checkpoint=str(checkpoint),
                voxel_size=float(point_cfg.get("voxel_size", 0.5)),
            )
        ).to(device)
        point_adapter = PointAdapter(
            PointAdapterConfig(
                in_dim=point_encoder.output_dim,
                dense_dim=decoder_dim,
                out_dim=qwen.hidden_size,
                num_tokens=int(cfg.get("max_point_reasoning_tokens", 512)),
            )
        ).to(device)

        backbone = PAIRBackbone(qwen_backbone=qwen, point_encoder=point_encoder, point_adapter=point_adapter)
        image_adapter = ImageDenseAdapter(qwen.hidden_size, decoder_dim).to(device)
        decoder = UnifiedChangeDecoder(qwen_dim=qwen.hidden_size, decoder_dim=decoder_dim).to(device)
        return cls(backbone, decoder, image_adapter, qwen_tuning).to(device)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.qwen_tuning in ("frozen", "lora"):
            self.backbone.visual_module().eval()
        if mode and self.qwen_tuning == "frozen":
            self.qwen_backbone.model.eval()
        return self

    # -------------------------------------------------------------------------
    # 2D: existing binary-change path
    # -------------------------------------------------------------------------

    def forward_2d(self, images_t1, images_t2, prompts, class_names, output_sizes):
        if not (len(images_t1) == len(images_t2) == len(prompts) == len(output_sizes)):
            raise ValueError("Batched 2D inputs have inconsistent lengths")

        images_t1 = [tensor_to_pil(x) for x in images_t1]
        images_t2 = [tensor_to_pil(x) for x in images_t2]
        kwargs = dict(
            prompt=prompts,
            images_t1=images_t1,
            images_t2=images_t2,
            return_logits=False,
            return_hidden_states=True,
            return_dense_features=True,
            use_cache=False,
        )

        # Frozen Qwen is safe under no_grad for 2D because all trainable 2D
        # modules are downstream from Qwen.
        if self.qwen_tuning == "frozen":
            with torch.no_grad():
                out = self.backbone(**kwargs)
        else:
            out = self.backbone(**kwargs)

        if out.image_dense_t1 is None or out.image_dense_t2 is None:
            raise RuntimeError("PAIR did not expose Qwen Vision dense features")
        if out.image_hidden_t1 is None or out.image_hidden_t2 is None or out.task_hidden is None:
            raise RuntimeError("PAIR did not expose Qwen reasoning features")

        shapes1 = out.aux["image_token_shapes_t1"]
        shapes2 = out.aux["image_token_shapes_t2"]
        for batch_id, (shape1, shape2) in enumerate(zip(shapes1, shapes2)):
            if shape1 != shape2:
                raise RuntimeError(
                    f"Aligned 2D temporal links require identical T1/T2 grids; "
                    f"batch {batch_id}: {shape1} vs {shape2}"
                )

        dense1 = self.image_adapter(out.image_dense_t1)
        dense2 = self.image_adapter(out.image_dense_t2)
        dense_t1 = make_batched_image_token_set(dense1, shapes1, out.aux["image_dense_batch_ids_t1"])
        dense_t2 = make_batched_image_token_set(dense2, shapes2, out.aux["image_dense_batch_ids_t2"])
        reasoning_t1 = make_batched_image_token_set(
            out.image_hidden_t1, shapes1, out.aux["image_reasoning_batch_ids_t1"]
        )
        reasoning_t2 = make_batched_image_token_set(
            out.image_hidden_t2, shapes2, out.aux["image_reasoning_batch_ids_t2"]
        )

        if dense1.shape[0] != dense2.shape[0]:
            raise RuntimeError("Aligned 2D batch has different total T1/T2 token counts")

        prediction = self.decoder(
            dense_t1=dense_t1,
            dense_t2=dense_t2,
            reasoning_t1=reasoning_t1,
            reasoning_t2=reasoning_t2,
            task_hidden=out.task_hidden,
            links_t1_to_t2=build_identity_temporal_links(dense1.shape[0], device=dense1.device),
            links_t2_to_t1=build_identity_temporal_links(dense2.shape[0], device=dense2.device),
            class_names=class_names,
            qwen_backbone=self.qwen_backbone,
            detach_qwen_class_encoder=True,
            prediction_type="binary",
        )
        return restore_2d_prediction_batch(prediction, shapes1, shapes2, output_sizes)

    # -------------------------------------------------------------------------
    # 3D: semantic + 6-class event path
    # -------------------------------------------------------------------------

    def forward_3d(self, point_dicts_t1, point_dicts_t2, prompts, class_names):
        point_dicts_t1 = _as_point_list(point_dicts_t1, "point_dicts_t1")
        point_dicts_t2 = _as_point_list(point_dicts_t2, "point_dicts_t2")
        if not (len(point_dicts_t1) == len(point_dicts_t2) == len(prompts)):
            raise ValueError("Batched 3D inputs have inconsistent lengths")

        device = self.qwen_backbone.model_device
        point_t1 = batch_point_dicts(point_dicts_t1, device)
        point_t2 = batch_point_dicts(point_dicts_t2, device)

        # Important: even if Qwen weights are frozen, do NOT wrap this forward
        # in no_grad. PointAdapter reasoning tokens are trainable inputs and
        # gradients must pass through the fixed Qwen transformer.
        out = self.backbone(
            prompt=prompts,
            point_dict_t1=point_t1,
            point_dict_t2=point_t2,
            return_logits=False,
            return_hidden_states=True,
            return_dense_features=True,
            use_cache=False,
        )

        if out.point_dense_t1 is None or out.point_dense_t2 is None:
            raise RuntimeError("PAIR did not expose PointAdapter dense features")
        if out.point_hidden_t1 is None or out.point_hidden_t2 is None or out.task_hidden is None:
            raise RuntimeError("PAIR did not expose Qwen point reasoning features")

        dense_t1 = make_point_token_set(
            out.point_dense_t1,
            out.aux["point_dense_coord_t1"],
            out.aux["point_dense_batch_t1"],
            self.decoder.decoder_dim,
        )
        dense_t2 = make_point_token_set(
            out.point_dense_t2,
            out.aux["point_dense_coord_t2"],
            out.aux["point_dense_batch_t2"],
            self.decoder.decoder_dim,
        )
        reasoning_t1 = make_point_token_set(
            out.point_hidden_t1,
            out.aux["point_reasoning_coord_t1"],
            out.aux["point_reasoning_batch_ids_t1"],
            self.decoder.qwen_dim,
        )
        reasoning_t2 = make_point_token_set(
            out.point_hidden_t2,
            out.aux["point_reasoning_coord_t2"],
            out.aux["point_reasoning_batch_ids_t2"],
            self.decoder.qwen_dim,
        )

        links_t1_to_t2 = build_batched_knn_temporal_links(
            dense_t1.positions, dense_t1.batch_ids, dense_t2.positions, dense_t2.batch_ids
        )
        links_t2_to_t1 = build_batched_knn_temporal_links(
            dense_t2.positions, dense_t2.batch_ids, dense_t1.positions, dense_t1.batch_ids
        )

        prediction = self.decoder(
            dense_t1=dense_t1,
            dense_t2=dense_t2,
            reasoning_t1=reasoning_t1,
            reasoning_t2=reasoning_t2,
            task_hidden=out.task_hidden,
            links_t1_to_t2=links_t1_to_t2,
            links_t2_to_t1=links_t2_to_t1,
            class_names=class_names,
            qwen_backbone=self.qwen_backbone,
            detach_qwen_class_encoder=True,
            prediction_type="event",
        )

        if prediction.change_logits_t1 is not None or prediction.change_logits_t2 is not None:
            raise RuntimeError("3D decoder must not return binary change logits")
        if prediction.event_logits_t1 is None or prediction.event_logits_t2 is None:
            raise RuntimeError("3D decoder did not return event logits")
        if prediction.event_logits_t1.shape != (dense_t1.features.shape[0], 6):
            raise RuntimeError(f"Unexpected T1 event shape: {tuple(prediction.event_logits_t1.shape)}")
        if prediction.event_logits_t2.shape != (dense_t2.features.shape[0], 6):
            raise RuntimeError(f"Unexpected T2 event shape: {tuple(prediction.event_logits_t2.shape)}")

        return prediction

    def forward_2d3d(self, **kwargs):
        raise NotImplementedError(
            "PAIR 2D+3D decoder wiring is intentionally deferred until world-coordinate "
            "image-token positions and cross-modal temporal links are finalized. "
            "Do not substitute normalized image-grid coordinates for real world coordinates."
        )

    def forward(self, task_mode=None, **kwargs):
        if task_mode is None:
            has_images = kwargs.get("images_t1") is not None or kwargs.get("images_t2") is not None
            has_points = kwargs.get("point_dicts_t1") is not None or kwargs.get("point_dicts_t2") is not None
            if has_images and has_points:
                task_mode = "2d3d"
            elif has_images:
                task_mode = "2d"
            elif has_points:
                task_mode = "3d"
            else:
                raise ValueError("Cannot infer PAIR route from empty inputs")

        task_mode = str(task_mode).lower()
        if task_mode == "2d":
            return self.forward_2d(**kwargs)
        if task_mode == "3d":
            return self.forward_3d(**kwargs)
        if task_mode in ("2d3d", "2d+3d"):
            return self.forward_2d3d(**kwargs)
        raise ValueError(f"Unsupported task_mode={task_mode!r}")

    @torch.no_grad()
    def generate(self, **kwargs):
        return self.backbone.generate(**kwargs)


PAIR = PAIRModel
