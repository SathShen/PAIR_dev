"""PAIR model orchestration: Qwen3-VL, 2D/3D spatial adapters and shared decoder.

Architecture (same notation for 2D and 3D):
    Backbone multilevel features + Qwen token-position T_mm -> F_fuse
    F_fuse -> Decoder Memory (K/V)
    F_fuse + original-resolution detail -> F_pixel / F_point
    Qwen <TASK> hidden + task descriptions -> Query (Q)
    PAIRChangeDecoder(Q, Memory, dense features, optional multi-box regions)
        -> Semantic + Binary Change (2D) or Semantic + Event (3D) logits.

This file ONLY orchestrates modules.  No custom <PYRAMID> injection,
no temporal feature matching/MLPs, no prediction head or box soft gate here.
Box proposals can be provided externally for diagnostics or predicted directly
as a fixed-size set from Qwen-conditioned spatial features.  Formal validation
does not use autoregressive JSON generation.  2D+3D spatial alignment remains
intentionally unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image

from models.change_decoder import PAIRChangeDecoder
from models.image_adapter import ImageAdapter
from models.point_adapter import PointAdapter, PointAdapterConfig
from models.point_encoder import UtoniaPointEncoder, UtoniaPointEncoderConfig
from models.qwen3vl_backbone import Qwen3VLBackbone
from models.lora import (
    apply_qwen_lora,
    apply_qwen_vision_lora,
    set_custom_lora_training,
)

SUPPORTED_TASK_MODES = ("2d", "3d", "2d3d")


def _normalize_route(task_mode: str) -> str:
    aliases = {
        "2d": "2d", "image": "2d", "image_only": "2d",
        "3d": "3d", "point": "3d", "point_only": "3d",
        "2d3d": "2d3d", "2d+3d": "2d3d", "image_point": "2d3d",
        "multimodal": "2d3d",
    }
    key = str(task_mode).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported task_mode={task_mode!r}; use {SUPPORTED_TASK_MODES}")
    return aliases[key]


def _as_prompts(prompts: Any) -> list[str]:
    if isinstance(prompts, str):
        result = [prompts]
    elif isinstance(prompts, (list, tuple)) and prompts and all(isinstance(p, str) for p in prompts):
        result = list(prompts)
    else:
        raise TypeError("prompts must be a string or nonempty sequence of strings")
    if any(not p.strip() for p in result):
        raise ValueError("prompts cannot contain empty text")
    return result


def _as_images(images: Any, count: int, name: str) -> list[Any]:
    if torch.is_tensor(images) and images.ndim == 4:
        result = list(images.unbind(0))
    elif isinstance(images, (tuple, list)):
        result = list(images)
    elif count == 1 and (torch.is_tensor(images) or isinstance(images, Image.Image)):
        result = [images]
    else:
        raise TypeError(f"{name} must be a [B,3,H,W] tensor or a list of images")
    if len(result) != count:
        raise ValueError(f"{name}: {len(result)} images, expected {count}")
    return result


def _image_to_pil(image: Any) -> Image.Image:
    """Qwen processor input; original (unquantized) RGB still feeds ImageAdapter."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] != 3:
        raise TypeError("Qwen image input must be PIL RGB or Tensor [3,H,W]")
    data = image.detach().to(device="cpu")
    if data.dtype == torch.uint8:
        arr = data.permute(1, 2, 0).contiguous().numpy()
    elif torch.is_floating_point(data):
        if not bool(torch.isfinite(data).all()) or bool((data < -1e-5).any()) or bool((data > 1 + 1e-5).any()):
            raise ValueError("Floating-point RGB images must be finite and in [0,1]")
        arr = data.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    else:
        raise TypeError("Image must be uint8 or floating-point RGB")
    return Image.fromarray(arr, "RGB")


def _as_point_list(value: Any, name: str) -> list[dict]:
    if isinstance(value, Mapping):
        values = [dict(value)]
    elif isinstance(value, (tuple, list)) and value and all(isinstance(x, Mapping) for x in value):
        values = [dict(x) for x in value]
    else:
        raise TypeError(f"{name} must be a point dictionary or nonempty list of dictionaries")
    return values


def _batch_point_dicts(items: Sequence[Mapping[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    """Pack original points without altering XYZ or silently synthesizing validity.

    RGB, normal and intensity are optional. If at least one sample has a field,
    other samples receive zeros with presence=False. Utonia may internally use
    its own zero defaults, but PointAdapter receives the same explicit masks.
    """
    points = _as_point_list(items, "point_dicts")
    out: Dict[str, torch.Tensor] = {}
    coordinates, batch_ids, counts = [], [], []
    for b, item in enumerate(points):
        xyz = item.get("coord")
        if xyz is None:
            raise KeyError(f"point_dicts[{b}] requires 'coord'")
        xyz = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
            raise ValueError(f"point_dicts[{b}]['coord'] must be nonempty [N,3]")
        if not bool(torch.isfinite(xyz).all()):
            raise ValueError(f"point_dicts[{b}]['coord'] has NaN/Inf")
        count = int(xyz.shape[0])
        coordinates.append(xyz)
        batch_ids.append(torch.full((count,), b, device=device, dtype=torch.long))
        counts.append(count)
    out["coord"] = torch.cat(coordinates, dim=0)
    out["batch"] = torch.cat(batch_ids, dim=0)
    out["offset"] = torch.tensor(counts, device=device, dtype=torch.long).cumsum(0)

    for field, width in (("rgb", 3), ("normal", 3), ("intensity", 1)):
        if not any(item.get(field) is not None for item in points):
            continue
        values, masks = [], []
        for b, (item, count) in enumerate(zip(points, counts)):
            present = item.get(field) is not None
            if present:
                x = torch.as_tensor(item[field], dtype=torch.float32, device=device)
                if width == 1 and x.ndim == 1:
                    x = x.unsqueeze(-1)
                if x.shape != (count, width) or not bool(torch.isfinite(x).all()):
                    raise ValueError(f"point_dicts[{b}][{field!r}] must be finite [{count},{width}]")
            else:
                x = torch.zeros(count, width, device=device, dtype=torch.float32)
            mask_value = item.get(field + "_mask") if present else None
            if mask_value is None:
                valid = torch.full((count, 1), present, device=device, dtype=torch.bool)
            else:
                valid = torch.as_tensor(mask_value, device=device)
                if valid.ndim == 1:
                    valid = valid.unsqueeze(-1)
                if valid.shape != (count, 1) or not bool(((valid == 0) | (valid == 1)).all()):
                    raise ValueError(f"point_dicts[{b}][{field+'_mask'!r}] needs [{count},1] bool/0/1")
                valid = valid.bool()
            values.append(x)
            masks.append(valid)
        out[field] = torch.cat(values, dim=0)
        out[field + "_mask"] = torch.cat(masks, dim=0)
    return out


class PAIRBackbone(nn.Module):
    """Own Qwen/Utonia/PointAdapter and keep the historical train.py paths."""

    def __init__(self, qwen_backbone: nn.Module,
                 point_encoder: Optional[nn.Module] = None,
                 point_adapter: Optional[nn.Module] = None):
        super().__init__()
        self.qwen_backbone = qwen_backbone
        self.point_encoder = point_encoder
        self.point_adapter = point_adapter

    def visual_module(self) -> nn.Module:
        # Handles PEFT's optional base_model/model nesting.
        queue = [self.qwen_backbone.model]
        seen: set[int] = set()
        while queue:
            model = queue.pop(0)
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            visual = getattr(model, "visual", None)
            if isinstance(visual, nn.Module):
                return visual
            for name in ("base_model", "model"):
                child = getattr(model, name, None)
                if isinstance(child, nn.Module) and child is not model:
                    queue.append(child)
        raise RuntimeError("Qwen visual module not found inside current PEFT wrapper")

    def language_model_module(self) -> nn.Module:
        return self.qwen_backbone._language_module()

    def encode_points(self, point_dict: Dict[str, torch.Tensor]):
        if self.point_encoder is None or self.point_adapter is None:
            raise RuntimeError("PAIR 3D route is not configured")
        encoded = self.point_encoder(point_dict)
        single = self.point_adapter.encode_single(encoded, raw_point_dict=point_dict)
        if single.original_point_count != point_dict["coord"].shape[0]:
            raise RuntimeError("Utonia inverse mapping lost original point topology")
        if single.voxel_inverse is not None:
            if single.voxel_inverse.shape[0] != single.original_point_count:
                raise RuntimeError("Sparse Utonia voxel_inverse does not cover original points")
        return single


class PAIRModel(nn.Module):
    """PAIR entry point with direct parallel Box set prediction.

    Box proposals are predicted from Qwen TASK hidden + decoder spatial memory in
    one forward pass. Autoregressive JSON grounding is no longer part of the
    training or formal validation path.
    """

    def __init__(
        self,
        backbone: PAIRBackbone,
        decoder: PAIRChangeDecoder,
        image_adapter: Optional[ImageAdapter],
        *,
        qwen_tuning: str = "lora",
        vision_lora_enabled: bool = False,
        point_lora_enabled: bool = False,
        enable_2d: bool = True,
        enable_3d: bool = True,
        enable_semantic: bool = True,
        box_max_proposals: int = 64,
        box_num_queries: int = 16,
    ) -> None:
        super().__init__()
        if not (enable_2d or enable_3d):
            raise ValueError("At least one PAIR route must be enabled")
        if box_num_queries < 1 or box_num_queries > box_max_proposals:
            raise ValueError("box_num_queries must be in [1, box_max_proposals]")
        self.backbone = backbone
        self.decoder = decoder
        self.image_adapter = image_adapter
        self.qwen_tuning = str(qwen_tuning).lower()
        self.vision_lora_enabled = bool(vision_lora_enabled)
        self.point_lora_enabled = bool(point_lora_enabled)
        self.enable_2d = bool(enable_2d)
        self.enable_3d = bool(enable_3d)
        self.enable_semantic = bool(enable_semantic)
        self.box_max_proposals = int(box_max_proposals)
        self.box_num_queries = int(box_num_queries)

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
    def from_config(
        cls, model_config: Mapping[str, Any], device: Any, *,
        enable_2d: bool = True, enable_3d: bool = True,
        enable_semantic: bool = True,
    ) -> "PAIRModel":
        cfg = dict(model_config)
        device = torch.device(device)
        if not (enable_2d or enable_3d):
            raise ValueError("PAIRModel needs at least one active route")
        tuning = str(cfg.get("qwen_tuning", "lora")).lower().strip()
        if tuning not in ("lora", "frozen", "full"):
            raise ValueError("qwen_tuning must be 'lora', 'frozen' or 'full'")

        qwen_kwargs: Dict[str, Any] = dict(
            model_dir=str(cfg["qwen_model"]),
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device=device,
            device_map=str(device) if device.type == "cuda" else None,
            local_files_only=bool(cfg.get("local_files_only", True)),
            vision_intermediate_layers=cfg.get("vision_intermediate_layers"),
        )
        qwen = Qwen3VLBackbone(**qwen_kwargs)
        if tuning == "frozen":
            qwen.freeze()
        elif tuning == "full":
            qwen.unfreeze()
        else:
            lora_cfg = dict(cfg.get("lora", {}))
            target = lora_cfg.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
            if isinstance(target, str):
                target = tuple(x.strip() for x in target.split(",") if x.strip())
            apply_qwen_lora(
                qwen,
                r=int(lora_cfg.get("r", 16)),
                alpha=int(lora_cfg.get("alpha", 32)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=tuple(target),
            )

        vision_cfg = dict(cfg.get("vision_lora", {}))
        vision_enabled = bool(vision_cfg.get("enabled", False) and enable_2d)
        if vision_enabled:
            if tuning == "full":
                raise ValueError("Vision LoRA is incompatible with qwen_tuning='full'")
            apply_qwen_vision_lora(
                qwen,
                r=int(vision_cfg.get("r", 16)),
                alpha=float(vision_cfg.get("alpha", 32)),
                dropout=float(vision_cfg.get("dropout", 0.05)),
            )

        decoder_dim = int(cfg.get("decoder_dim", 256))
        point_encoder = point_adapter = None
        point_lora_cfg = dict(cfg.get("point_lora", {}))
        point_lora_enabled = bool(enable_3d and point_lora_cfg.get("enabled", False))
        if enable_3d:
            encoder_cfg = dict(cfg.get("point_encoder", {}))
            if not encoder_cfg.get("checkpoint"):
                raise KeyError("model.point_encoder.checkpoint is required for 3D")
            point_encoder = UtoniaPointEncoder(UtoniaPointEncoderConfig(
                checkpoint=str(encoder_cfg["checkpoint"]),
                voxel_size=float(encoder_cfg.get("voxel_size", 0.5)),
                lora_enabled=point_lora_enabled,
                lora_r=int(point_lora_cfg.get("r", 8)),
                lora_alpha=float(point_lora_cfg.get("alpha", 16)),
                lora_dropout=float(point_lora_cfg.get("dropout", 0.05)),
            )).to(device)
            raw_point_adapter_cfg = dict(cfg.get("point_adapter", {}))
            raw_point_adapter_cfg.setdefault("utonia_dim", point_encoder.output_dim)
            raw_point_adapter_cfg.setdefault("decoder_dim", decoder_dim)
            raw_point_adapter_cfg.setdefault("qwen_dim", qwen.hidden_size)
            raw_point_adapter_cfg.setdefault(
                "llm_tokens_per_cloud", int(cfg.get("max_point_reasoning_tokens", 512))
            )
            allowed = {f.name for f in fields(PointAdapterConfig)}
            unknown = set(raw_point_adapter_cfg) - allowed
            if unknown:
                raise ValueError(f"Unknown point_adapter settings: {sorted(unknown)}")
            point_adapter = PointAdapter(PointAdapterConfig(**raw_point_adapter_cfg)).to(device)

        image_adapter = None
        if enable_2d:
            image_cfg = dict(cfg.get("image_adapter", {}))
            image_cfg.setdefault("vision_dim", qwen.vision_hidden_size)
            image_cfg.setdefault("qwen_dim", qwen.hidden_size)
            image_cfg.setdefault("decoder_dim", decoder_dim)
            image_cfg.setdefault("layer_indices", qwen.vision_intermediate_layers)
            image_adapter = ImageAdapter(**image_cfg).to(device)

        old_dec_cfg = dict(cfg.get("unified_decoder", {}))
        dec_cfg = dict(cfg.get("change_decoder", {}))
        dec_cfg.setdefault("qwen_dim", qwen.hidden_size)
        dec_cfg.setdefault("decoder_dim", decoder_dim)
        dec_cfg.setdefault("num_layers", int(old_dec_cfg.get("num_layers", 2)))
        dec_cfg.setdefault("num_heads", int(old_dec_cfg.get("num_heads", 8)))
        dec_cfg.setdefault("mlp_ratio", float(old_dec_cfg.get("mlp_ratio", 4.0)))
        dec_cfg.setdefault("dropout", float(old_dec_cfg.get("dropout", 0.0)))

        box_guidance_cfg = dict(cfg.get("box_guidance", {}))
        dec_cfg.setdefault("max_box_proposals", int(box_guidance_cfg.get("max_proposals", 64)))
        proposal_cfg = dict(cfg.get("box_proposal", {}))
        # Backward-compatible fallback: the old generation max_boxes becomes K.
        legacy_generation_cfg = dict(cfg.get("box_generation", {}))
        dec_cfg.setdefault(
            "box_num_queries",
            int(proposal_cfg.get("num_queries", legacy_generation_cfg.get("max_boxes", 16))),
        )
        dec_cfg.setdefault("box_score_threshold", float(proposal_cfg.get("score_threshold", 0.5)))

        decoder = PAIRChangeDecoder(**dec_cfg).to(device)
        backbone = PAIRBackbone(qwen, point_encoder, point_adapter)
        return cls(
            backbone,
            decoder,
            image_adapter,
            qwen_tuning=tuning,
            vision_lora_enabled=vision_enabled,
            point_lora_enabled=point_lora_enabled,
            enable_2d=enable_2d,
            enable_3d=enable_3d,
            enable_semantic=enable_semantic,
            box_max_proposals=int(dec_cfg["max_box_proposals"]),
            box_num_queries=int(dec_cfg["box_num_queries"]),
        ).to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.qwen_tuning in ("frozen", "lora"):
            visual = self.backbone.visual_module()
            visual.eval()
            set_custom_lora_training(visual, bool(mode) and self.vision_lora_enabled)
        if mode and self.qwen_tuning == "frozen":
            self.qwen_backbone.model.eval()
            if self.vision_lora_enabled:
                set_custom_lora_training(self.backbone.visual_module(), True)
        return self

    @staticmethod
    def _validate_explicit_box_args(boxes, valid, scores, name: str) -> None:
        if boxes is None and (valid is not None or scores is not None):
            raise ValueError(f"{name}: box_valid/box_scores require explicit boxes")

    def forward_2d(
        self,
        images_t1,
        images_t2,
        prompts,
        class_names,
        output_sizes=None,
        prediction_mode="scd",
        *,
        boxes_2d: Optional[torch.Tensor] = None,
        box_valid: Optional[torch.Tensor] = None,
        box_scores: Optional[torch.Tensor] = None,
        generate_boxes: Optional[bool] = None,
        return_maps: bool = False,
        detach_qwen_class_encoder: bool = True,
        grounding_targets: Optional[Sequence[str]] = None,
    ):
        if not self.enable_2d or self.image_adapter is None:
            raise RuntimeError("2D route not configured")
        if grounding_targets is not None:
            raise RuntimeError(
                "Text teacher-forced Box grounding was removed. Use direct Box set prediction."
            )
        prompts = _as_prompts(prompts)
        first = _as_images(images_t1, len(prompts), "images_t1")
        second = _as_images(images_t2, len(prompts), "images_t2")
        mode = str(prediction_mode).strip().lower()
        if mode not in ("scd", "bcd"):
            raise ValueError("prediction_mode must be 'scd' or 'bcd'")
        semantic_classes = None if mode == "bcd" else class_names
        if mode == "scd" and not semantic_classes:
            raise ValueError("SCD requires semantic class_names")
        self._validate_explicit_box_args(boxes_2d, box_valid, box_scores, "2D")
        predict_boxes = bool(generate_boxes)
        if predict_boxes and boxes_2d is not None:
            raise ValueError("Direct Box prediction cannot be combined with explicit boxes_2d")

        images1 = [_image_to_pil(x) for x in first]
        images2 = [_image_to_pil(x) for x in second]
        qwen_out = self.qwen_backbone(
            prompt=prompts,
            images_t1=images1,
            images_t2=images2,
        )
        features = self.image_adapter(
            premerge_t1=qwen_out["premerge_t1"],
            premerge_t2=qwen_out["premerge_t2"],
            llm_visual_t1=qwen_out["llm_visual_t1"],
            llm_visual_t2=qwen_out["llm_visual_t2"],
            images_t1=first,
            images_t2=second,
            output_sizes=output_sizes,
        )
        if not self.training:
            self.qwen_backbone.clear_vision_intermediate_cache()
            qwen_out.pop("premerge_t1", None)
            qwen_out.pop("premerge_t2", None)
            qwen_out.pop("llm_visual_t1", None)
            qwen_out.pop("llm_visual_t2", None)
            qwen_out.pop("prepared", None)

        return self.decoder.forward_2d(
            qwen_backbone=self.qwen_backbone,
            task_hidden=qwen_out["task_hidden"],
            prediction_mode=mode,
            class_names=semantic_classes,
            return_maps=return_maps,
            detach_qwen_class_encoder=detach_qwen_class_encoder,
            boxes_2d=boxes_2d,
            box_valid=box_valid,
            box_scores=box_scores,
            predict_boxes=predict_boxes,
            **features.decoder_inputs(),
        )

    def forward_3d(
        self,
        point_dicts_t1,
        point_dicts_t2,
        prompts,
        class_names,
        *,
        boxes_3d: Optional[torch.Tensor] = None,
        box_valid: Optional[torch.Tensor] = None,
        box_scores: Optional[torch.Tensor] = None,
        generate_boxes: Optional[bool] = None,
        detach_qwen_class_encoder: bool = True,
        grounding_targets: Optional[Sequence[str]] = None,
    ):
        if not self.enable_3d or self.point_adapter is None:
            raise RuntimeError("3D route not configured")
        if grounding_targets is not None:
            raise RuntimeError(
                "Text teacher-forced Box grounding was removed. Use direct Box set prediction."
            )
        prompts = _as_prompts(prompts)
        first = _as_point_list(point_dicts_t1, "point_dicts_t1")
        second = _as_point_list(point_dicts_t2, "point_dicts_t2")
        if len(first) != len(prompts) or len(second) != len(prompts):
            raise ValueError("Point-cloud batch and prompt batch sizes differ")
        if not class_names:
            raise ValueError("3D semantic class_names are required")
        self._validate_explicit_box_args(boxes_3d, box_valid, box_scores, "3D")
        predict_boxes = bool(generate_boxes)
        if predict_boxes and boxes_3d is not None:
            raise ValueError("Direct Box prediction cannot be combined with explicit boxes_3d")

        device = self.qwen_backbone.model_device
        batch1 = _batch_point_dicts(first, device)
        batch2 = _batch_point_dicts(second, device)
        t1 = self.backbone.encode_points(batch1)
        t2 = self.backbone.encode_points(batch2)

        qwen_out = self.qwen_backbone(
            prompt=prompts,
            point_tokens_t1=t1.llm_point_tokens,
            point_tokens_t2=t2.llm_point_tokens,
        )
        features = self.point_adapter.fuse_temporal(
            t1,
            t2,
            llm_point_t1=qwen_out["llm_point_t1"],
            llm_point_t2=qwen_out["llm_point_t2"],
        )
        if not self.training:
            qwen_out.pop("llm_point_t1", None)
            qwen_out.pop("llm_point_t2", None)
            qwen_out.pop("prepared", None)

        return self.decoder.forward_3d(
            qwen_backbone=self.qwen_backbone,
            task_hidden=qwen_out["task_hidden"],
            class_names=class_names,
            detach_qwen_class_encoder=detach_qwen_class_encoder,
            boxes_3d=boxes_3d,
            box_valid=box_valid,
            box_scores=box_scores,
            predict_boxes=predict_boxes,
            **features.decoder_inputs(),
        )

    def forward_2d3d(self, **kwargs):
        raise NotImplementedError(
            "2D+3D unified prediction requires calibrated, world-coordinate "
            "image/point alignment; neither image-grid nor XYZ token indices "
            "can be treated as cross-modal correspondence."
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
                raise ValueError("No 2D images or 3D point clouds supplied")
        mode = _normalize_route(task_mode)
        if mode == "2d":
            return self.forward_2d(**kwargs)
        if mode == "3d":
            return self.forward_3d(**kwargs)
        return self.forward_2d3d(**kwargs)

    def generate_box_proposals(self, **kwargs):
        raise RuntimeError(
            "PAIR formal Box prediction is now direct set prediction inside the decoder; "
            "autoregressive Qwen JSON generation is no longer used."
        )

    def generate(self, **kwargs):
        return self.generate_box_proposals(**kwargs)


PAIR = PAIRModel
