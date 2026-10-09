"""PAIR two-temporal image adapter for Qwen3-VL Native DeepStack.

This file owns ONLY image features, from Qwen's spatial ViT intermediate maps
and native LLM visual-position hidden states to decoder-ready spatial memory
and full-resolution dense pixel features. It NEVER defines prediction heads,
box proposal modules, losses, or Qwen prompt/token injection.

Input contracts (Qwen3VLBackbone/PAIRBackbone):
  premerge_t1/t2: Mapping[int, Tensor[B,vision_dim,Hv,Wv]]
                  Spatial order MUST already have been restored by
                  Qwen3VLBackbone.get_temporal_vision_feature_maps().
  llm_visual_t1/t2: sequence of B Tensor[1,Hm,Wm,qwen_dim] (per-image readout),
                    or a stacked Tensor[B,qwen_dim,Hm,Wm].
  images_t1/t2: Tensor[B,3,H,W], sequence of Tensor[3,H,W], or RGB PIL images.
                    Values of float tensors must be [0,1]; uint8 is [0,255].

Output contracts (PAIRChangeDecoder.forward_2d):
  memory:             [B,2*Hv*Wv,D] = concatenate(T1 memory,T2 memory)
  memory_time_ids:    [B,2*Hv*Wv], 1 for T1 and 2 for T2
  pixel_features_t1: [B,D,Hp,Wp]
  pixel_features_t2: [B,D,Hp,Wp]
  output_sizes:       per-sample (H,W), for final logit interpolation/flattening

Memory deliberately has NO new positional embedding: the ViT intermediates and
LLM visual hidden states already contain Qwen's position-dependent features.
The effect of an independent Decoder positional encoding remains an ablation.

The decoder, not this adapter, owns MultiBoxGuidance, region queries, all
semantic/change logits, mask flattening and logit resizing for loss.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


ImageInput = Union[torch.Tensor, Sequence[Any]]


def _group_count(channels: int, desired: int = 16) -> int:
    return max(g for g in range(min(desired, channels), 0, -1) if channels % g == 0)


def _norm(channels: int, desired: int = 16) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels, desired), channels)


class _ViTLevelFusion(nn.Module):
    """1x1/GN + DW3x3/GELU, with a small learned residual correction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _norm(out_channels),
        )
        self.depthwise = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1,
            groups=out_channels, bias=False,
        )
        self.activate = nn.GELU()
        self.correction_strength = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        correction = self.activate(self.depthwise(x))
        return x + self.correction_strength.to(dtype=x.dtype) * correction


class _PixelShuffleStage(nn.Module):
    """Learnable x2 upsampling; does not pretend to recover lost RGB detail."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.ops = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            _norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


@dataclass
class ImageSingleAdapterOutput:
    """Single-temporal F_fuse and F_pixel.

    One phase of the same encode_single/fuse_temporal pipeline as PointAdapter.
    memory_features: [B,D,Hv,Wv] at the Qwen visual grid.
    pixel_features: [B,D,Hp,Wp] at the common RGB resolution.
    """

    memory_features: torch.Tensor
    pixel_features: torch.Tensor

@dataclass
class ImageAdapterOutput:
    """F_fuse sampled/flattened as memory and full-resolution F_pixel for both phases."""

    memory: torch.Tensor
    memory_time_ids: torch.Tensor
    pixel_features_t1: torch.Tensor
    pixel_features_t2: torch.Tensor
    output_sizes: Tuple[Tuple[int, int], ...]

    def decoder_inputs(self) -> Dict[str, Any]:
        return {
            "memory": self.memory,
            "memory_time_ids": self.memory_time_ids,
            "pixel_features_t1": self.pixel_features_t1,
            "pixel_features_t2": self.pixel_features_t2,
            "output_sizes": self.output_sizes,
        }

class ImageAdapter(nn.Module):

    # Modules shared by T1 and T2.
    def __init__(
        self,
        *,
        vision_dim: int = 1024,
        qwen_dim: int = 2560,
        decoder_dim: int = 256,
        layer_indices: Sequence[int] = (5, 11, 17),
        up_channels: Tuple[int, int] = (128, 64),
        rgb_channels: int = 32,
    ) -> None:
        super().__init__()
        layers = tuple(int(x) for x in layer_indices)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("layer_indices must be nonempty, with no duplicates")
        if min(vision_dim, qwen_dim, decoder_dim, rgb_channels, *up_channels) <= 0:
            raise ValueError("All channel dimensions must be positive")
        if len(up_channels) != 2:
            raise ValueError("up_channels must specify exactly two PixelShuffle stages")

        self.vision_dim = int(vision_dim)
        self.qwen_dim = int(qwen_dim)
        self.decoder_dim = int(decoder_dim)
        self.layer_indices = layers

        self.level_fusion = nn.ModuleDict({
            str(idx): _ViTLevelFusion(vision_dim, decoder_dim)
            for idx in layers
        })
        self.vit_fuse = nn.Sequential(
            nn.Conv2d(len(layers) * decoder_dim, decoder_dim, kernel_size=1, bias=False),
            _norm(decoder_dim), nn.GELU(),
        )
        # T_mm: native Qwen LLM hidden states from *visual token positions*.
        # Do not inject additional <PYRAMID> or create independent LLM tokens.
        self.llm_visual_project = nn.Conv2d(qwen_dim, decoder_dim, kernel_size=1)
        self.spatial_memory_norm = _norm(decoder_dim)

        self.upsample1 = _PixelShuffleStage(decoder_dim, up_channels[0])
        self.upsample2 = _PixelShuffleStage(up_channels[0], up_channels[1])
        self.rgb_detail = nn.Sequential(
            nn.Conv2d(3, rgb_channels, kernel_size=3, padding=1, bias=False),
            _norm(rgb_channels, 8), nn.GELU(),
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
            _norm(rgb_channels, 8), nn.GELU(),
        )
        self.pixel_fuse = nn.Sequential(
            nn.Conv2d(up_channels[1] + rgb_channels, decoder_dim, kernel_size=1, bias=False),
            _norm(decoder_dim), nn.GELU(),
        )
    # Input preparation and validation.
    @staticmethod
    def _cast_to_module(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Match parameter dtype/device with or without autocast enabled."""
        parameter = next(module.parameters())
        return x.to(device=parameter.device, dtype=parameter.dtype)

    def _validate_vision_maps(
        self,
        maps: Mapping[int, torch.Tensor],
        phase: str,
    ) -> Tuple[int, int, int, torch.device]:
        if not isinstance(maps, Mapping):
            raise TypeError(f"{phase} must be a mapping of ViT layer index to BCHW")
        expected = set(self.layer_indices)
        if not expected.issubset(set(maps.keys())):
            raise ValueError(f"{phase} missing ViT layers {sorted(expected - set(maps.keys()))}")
        reference = maps[self.layer_indices[0]]
        if not torch.is_tensor(reference) or reference.ndim != 4:
            raise ValueError(f"{phase}: expected ViT feature [B,{self.vision_dim},H,W]")
        b, c, h, w = reference.shape
        if b < 1 or min(h, w) < 1 or c != self.vision_dim:
            raise ValueError(f"{phase}: invalid ViT feature {tuple(reference.shape)}")
        for idx in self.layer_indices:
            x = maps[idx]
            if not torch.is_tensor(x) or x.shape != reference.shape:
                raise ValueError(f"{phase}: layer {idx} has different shape")
            if x.device != reference.device or not torch.is_floating_point(x):
                raise ValueError(f"{phase}: layer {idx} device/dtype is invalid")
        return b, h, w, reference.device

    def _visual_to_bchw(self, value: Any, *, batch_size: int, name: str) -> torch.Tensor:
        """Accept native Qwen [1,H,W,C] list, NOT a blindly flattened array."""
        if torch.is_tensor(value):
            if value.ndim != 4 or value.shape[0] != batch_size or value.shape[1] != self.qwen_dim:
                raise ValueError(f"{name} tensor must be [B,{self.qwen_dim},Hm,Wm]")
            return value
        if not isinstance(value, (list, tuple)) or len(value) != batch_size:
            raise ValueError(f"{name} must be a list of {batch_size} native [1,H,W,C] maps")
        maps = []
        for i, item in enumerate(value):
            if not torch.is_tensor(item) or item.ndim != 4 or item.shape[0] != 1 or item.shape[-1] != self.qwen_dim:
                raise ValueError(f"{name}[{i}] must be [1,Hm,Wm,{self.qwen_dim}]")
            maps.append(item[0].permute(2, 0, 1).contiguous())
        shapes = {tuple(item.shape) for item in maps}
        if len(shapes) != 1:
            raise ValueError(f"{name} native visual grids differ in this batch: {sorted(shapes)}")
        return torch.stack(maps, dim=0)

    @staticmethod
    def _image_tensor(image: Any, *, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(image):
            from PIL import Image
            import numpy as np
            if not isinstance(image, Image.Image):
                raise TypeError("RGB image must be a tensor or PIL.Image")
            array = np.array(image.convert("RGB"), copy=True)
            image = torch.from_numpy(array).permute(2, 0, 1)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected RGB image [3,H,W], got {tuple(image.shape)}")
        if image.dtype == torch.uint8:
            image = image.to(device=device, dtype=torch.float32) / 255.0
        else:
            if not torch.is_floating_point(image):
                raise TypeError("Image must be uint8 or a floating-point tensor")
            image = image.to(device=device, dtype=torch.float32)
        if not torch.isfinite(image).all() or (image < -1e-5).any() or (image > 1.00001).any():
            raise ValueError("RGB inputs must be finite and normalized to [0,1]")
        return image.clamp(0.0, 1.0)

    def _prepare_rgb_pair(
        self,
        images_t1: ImageInput,
        images_t2: ImageInput,
        *,
        batch_size: int,
        device: torch.device,
        pixel_size: Optional[Tuple[int, int]],
        output_sizes: Optional[Sequence[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[int, int], ...]]:
        def unpack(value: ImageInput, name: str):
            if torch.is_tensor(value):
                if value.ndim != 4 or value.shape[0] != batch_size:
                    raise ValueError(f"{name} must be a tensor [B,3,H,W]")
                return list(value.unbind(dim=0))
            if not isinstance(value, (list, tuple)) or len(value) != batch_size:
                raise ValueError(f"{name} must have {batch_size} images")
            return list(value)

        imgs1 = [self._image_tensor(x, device=device) for x in unpack(images_t1, 'images_t1')]
        imgs2 = [self._image_tensor(x, device=device) for x in unpack(images_t2, 'images_t2')]
        original_sizes = tuple(tuple(x.shape[-2:]) for x in imgs1)
        if any(tuple(imgs2[i].shape[-2:]) != original_sizes[i] for i in range(batch_size)):
            raise ValueError("T1 and T2 original RGB sizes must match for every sample")
        if output_sizes is None:
            target_sizes = original_sizes
        else:
            if len(output_sizes) != batch_size:
                raise ValueError("output_sizes must have B (H,W) entries")
            target_sizes = tuple((int(h), int(w)) for h,w in output_sizes)
            if any(h < 1 or w < 1 for h,w in target_sizes):
                raise ValueError("output_sizes must be positive")

        if pixel_size is None:
            if len(set(original_sizes)) != 1:
                raise ValueError("Mixed RGB sizes require explicit common pixel_size")
            pixel_size = original_sizes[0]
        hp, wp = (int(pixel_size[0]), int(pixel_size[1]))
        if hp < 1 or wp < 1:
            raise ValueError("pixel_size must be positive")

        def resize_and_stack(items):
            resized = [
                F.interpolate(x.unsqueeze(0), size=(hp,wp), mode='bilinear',
                              align_corners=False)[0]
                if tuple(x.shape[-2:]) != (hp,wp) else x
                for x in items
            ]
            return torch.stack(resized,dim=0)

        return resize_and_stack(imgs1), resize_and_stack(imgs2), target_sizes

    # Single-temporal spatial memory and full-resolution features.
    def _encode_spatial_memory(
        self,
        premerge: Mapping[int, torch.Tensor],
        llm_visual: Any,
        *,
        batch_size: int,
        spatial_hw: Tuple[int,int],
        name: str,
    ) -> torch.Tensor:
        features = [
            self.level_fusion[str(idx)](
                self._cast_to_module(premerge[idx], self.level_fusion[str(idx)])
            )
            for idx in self.layer_indices
        ]
        fused = self.vit_fuse(torch.cat(features, dim=1))
        hidden = self._visual_to_bchw(llm_visual, batch_size=batch_size, name=name)
        projected = self.llm_visual_project(
            self._cast_to_module(hidden, self.llm_visual_project)
        )
        if tuple(projected.shape[-2:]) != spatial_hw:
            projected = F.interpolate(
                projected, size=spatial_hw, mode='bilinear', align_corners=False
            )
        if fused.shape != projected.shape:
            raise RuntimeError(f"{name}: projected LLM hidden does not match ViT grid")
        return self.spatial_memory_norm(fused + projected)

    def _encode_spatial_features(
        self,
        memory_map: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        high = self.upsample2(self.upsample1(memory_map))
        target_hw = tuple(rgb.shape[-2:])
        if tuple(high.shape[-2:]) != target_hw:
            high = F.interpolate(high, size=target_hw, mode='bilinear', align_corners=False)
        detail = self.rgb_detail(self._cast_to_module(rgb, self.rgb_detail))
        return self.pixel_fuse(torch.cat((high,detail),dim=1))

    def encode_single(
        self,
        *,
        premerge: Mapping[int, torch.Tensor],
        llm_visual: Any,
        rgb: torch.Tensor,
        name: str = 'llm_visual',
    ) -> ImageSingleAdapterOutput:
        """Encode one temporal image after shared RGB preparation.

        premerge is the restored ViT map dictionary, llm_visual contains
        native Qwen LLM visual-position hidden (not a positional embedding).
        rgb must be an already prepared [B,3,Hp,Wp] floating-point tensor.
        Both temporal calls use exactly the same learnable weights.
        """
        b, hv, wv, device = self._validate_vision_maps(premerge, name + '_premerge')
        if next(self.parameters()).device != device:
            raise ValueError('ImageAdapter and ViT features must be on the same device')
        if not torch.is_tensor(rgb) or rgb.ndim != 4 or rgb.shape[:2] != (b, 3):
            raise ValueError('rgb must be a prepared [B,3,Hp,Wp] tensor')
        if rgb.device != device:
            raise ValueError('rgb and ViT maps must be on the same device')
        spatial_memory_map = self._encode_spatial_memory(
            premerge, llm_visual, batch_size=b, spatial_hw=(hv, wv), name=name,
        )
        pixel_features = self._encode_spatial_features(spatial_memory_map, rgb)
        return ImageSingleAdapterOutput(
            memory_features=spatial_memory_map,
            pixel_features=pixel_features,
        )

    # Two-temporal features, memory, and prediction bridge.
    def fuse_temporal(
        self,
        t1: ImageSingleAdapterOutput,
        t2: ImageSingleAdapterOutput,
        *,
        output_sizes: Sequence[Tuple[int, int]],
    ) -> ImageAdapterOutput:
        """Combine two-temporal K/V Memory and full-resolution pixel features.

        Exactly the same public staged pipeline as PointAdapter.fuse_temporal.
        The Memory interface is shared, while the Dense feature names remain
        modality-specific (pixel_features vs point_features).
        """
        mem1, mem2 = t1.memory_features, t2.memory_features
        if mem1.ndim != 4 or mem2.shape != mem1.shape:
            raise ValueError('T1/T2 image memory maps must have the same [B,D,Hv,Wv] shape')
        b, channels, hv, wv = mem1.shape
        if channels != self.decoder_dim:
            raise ValueError('Image memory channel count differs from decoder_dim')
        if t1.pixel_features.shape != t2.pixel_features.shape:
            raise ValueError('T1/T2 full-resolution image feature shapes must match')
        if len(output_sizes) != b:
            raise ValueError('output_sizes must contain B entries')
        sizes = tuple((int(h), int(w)) for h, w in output_sizes)
        if any(h <= 0 or w <= 0 for h, w in sizes):
            raise ValueError('output_sizes must be positive')

        memory = torch.cat((mem1.flatten(2).transpose(1, 2),
                            mem2.flatten(2).transpose(1, 2)), dim=1)
        count = hv * wv
        memory_time_ids = torch.cat((
            torch.ones((b, count), device=mem1.device, dtype=torch.long),
            torch.full((b, count), 2, device=mem1.device, dtype=torch.long),
        ), dim=1)
        return ImageAdapterOutput(
            memory=memory,
            memory_time_ids=memory_time_ids,
            pixel_features_t1=t1.pixel_features,
            pixel_features_t2=t2.pixel_features,
            output_sizes=sizes,
        )

    def forward(
        self,
        *,
        premerge_t1: Mapping[int, torch.Tensor],
        premerge_t2: Mapping[int, torch.Tensor],
        llm_visual_t1: Any,
        llm_visual_t2: Any,
        images_t1: ImageInput,
        images_t2: ImageInput,
        pixel_size: Optional[Tuple[int, int]] = None,
        output_sizes: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> ImageAdapterOutput:
        """Encode T1/T2 once, then combine their spatial features and memory."""
        # Only extract dimensions for RGB preparation. Do not validate premerge
        # twice: encode_single() validates the complete ViT map for each phase.
        if not isinstance(premerge_t1, Mapping):
            raise TypeError("premerge_t1 must map ViT layer indices to tensors")
        reference = premerge_t1.get(self.layer_indices[0])
        if not torch.is_tensor(reference) or reference.ndim != 4:
            raise ValueError("premerge_t1 is missing a BCHW reference layer")
        batch_size = reference.shape[0]
        device = reference.device

        rgb_t1, rgb_t2, sizes = self._prepare_rgb_pair(
            images_t1, images_t2, batch_size=batch_size, device=device,
            pixel_size=pixel_size, output_sizes=output_sizes,
        )
        t1 = self.encode_single(
            premerge=premerge_t1, llm_visual=llm_visual_t1,
            rgb=rgb_t1, name="llm_visual_t1",
        )
        t2 = self.encode_single(
            premerge=premerge_t2, llm_visual=llm_visual_t2,
            rgb=rgb_t2, name="llm_visual_t2",
        )
        return self.fuse_temporal(t1, t2, output_sizes=sizes)

    def predict(
        self,
        *,
        decoder: nn.Module,
        qwen_backbone: Any,
        task_hidden: torch.Tensor,
        prediction_mode: str = 'scd',
        class_names: Optional[Dict[int,str]] = None,
        boxes_2d: Optional[torch.Tensor] = None,
        box_valid: Optional[torch.Tensor] = None,
        box_scores: Optional[torch.Tensor] = None,
        return_maps: bool = False,
        detach_qwen_class_encoder: bool = True,
        **image_kwargs,
    ):
        """Convenience bridge to existing PAIRChangeDecoder.forward_2d.

        The image adapter does not produce logits itself. All prediction heads
        and optional MultiBoxGuidance remain inside change_decoder.py.
        """
        prepared = self.forward(**image_kwargs)
        return decoder.forward_2d(
            **prepared.decoder_inputs(),
            qwen_backbone=qwen_backbone,
            task_hidden=task_hidden,
            prediction_mode=prediction_mode,
            class_names=class_names,
            boxes_2d=boxes_2d,
            box_valid=box_valid,
            box_scores=box_scores,
            return_maps=return_maps,
            detach_qwen_class_encoder=detach_qwen_class_encoder,
        )
