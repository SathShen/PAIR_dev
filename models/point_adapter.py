"""PAIR 3D adapter: Utonia and native Qwen point-position hidden -> spatial features.

The public terminology mirrors image_adapter.py:
  * LLM path: pretrained Utonia [N,utonia_dim] -> spatial sampling ->
    llm_point_tokens [K,qwen_dim], consumed by Qwen's <POINT> positions.
  * Spatial path: recovered Utonia stages [54,108,216,432,576] -> separate
    learned level projections -> fused spatial features [M,D]. Native Qwen
    point-position hidden is then projected/aligned and fused into K/V memory.
  * Dense path: fused spatial features -> voxel-representative detail MLP ->
    point_features [M,D]. K/V is sampled BEFORE detail,
    matching ImageAdapter's memory-then-high-resolution feature sequence.
  * Decoder handles all temporal matching and Event feature creation.
    This adapter only returns memory, sparse voxel-point features, and XYZ.

PointSingleAdapterOutput is a SINGLE-cloud output. PointAdapterOutput matches
ImageAdapterOutput's PAIR-level role: a two-epoch spatial adapter result with
memory, memory_time_ids, features, and decoder_inputs(). All prediction heads
and 3D MultiBoxGuidance remain in change_decoder.py.

This module does NOT create the task Query: Qwen returns task_hidden after it
processes llm_point_tokens, and change_decoder.py builds initial task Queries.
LLM point-position hidden must be supplied after Qwen forward in the SAME
order as llm_point_tokens; sampled XYZ provides the geometry for alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


@dataclass
class PointAdapterConfig:
    # Utonia output width; freezing is controlled by the outer model.
    utonia_dim: int = 1386
    # Utonia encoder stage channel order matches point_encoder._recover_multiscale.
    # For nonstandard small models, None falls back to a single stage; set this
    # explicitly for custom multi-stage Utonia encoders.
    utonia_stage_channels: Optional[Tuple[int, ...]] = None
    utonia_chunk_size: int = 4096
    utonia_checkpoint: bool = True

    # Unified decoder working width.
    decoder_dim: int = 256

    # Qwen hidden width.
    qwen_dim: int = 2560

    # Maximum number of LLM point tokens per cloud.
    llm_tokens_per_cloud: int = 512

    # Optional intensity side branch.
    intensity_hidden_dim: int = 64

    # LLM-point-token spatial sampling.
    llm_sampling: str = "voxel"  # "voxel" or "uniform"
    llm_voxel_size: Optional[float] = None
    llm_voxel_oversample_factor: float = 4.0
    llm_max_fps_candidates: int = 8192

    # Spatial position added to Qwen <POINT> input embeddings.
    use_xyz_pos: bool = True
    xyz_hidden_dim: int = 128
    use_layer_norm: bool = True

    # High-frequency branch. The MLP works on 16 input channels, NOT 1386.
    detail_hidden_dim: int = 48
    detail_chunk_size: int = 8192
    detail_checkpoint: bool = True
    detail_residual_init: float = 0.10
    subvoxel_size: float = 0.5

    # Decoder Memory uses a subset of Utonia's M voxel representatives.
    # Final low-channel logits are expanded to every original point.
    memory_tokens_per_cloud: int = 1024
    memory_cell_size: float = 0.5

    # Bounded XYZ alignment from Qwen point-position hidden to original points.
    llm_align_chunk_size: int = 2048
    llm_align_neighbors: int = 3


@dataclass
class PointSingleAdapterOutput:
    """Single epoch before Qwen: Utonia memory base, point detail and LLM tokens."""
    # Tensor for B=1, list[Tensor] for B>1.
    llm_point_tokens: TensorOrList

    # Samples of RAW Utonia features [K,utonia_dim], before point projection.
    llm_sampled_features: TensorOrList
    llm_sampled_xyz: TensorOrList
    llm_sampled_indices: TensorOrList

    # Deliberately NOT final point features: Qwen hidden is fused AFTER this
    # stage, and original-resolution detail is attached only after K/V creation.
    memory_features: torch.Tensor      # [N,D] 5 Utonia levels fused, pre-Qwen
    point_detail: torch.Tensor         # [N,16] raw XYZ/RGB/normal/intensity
    point_intensity: Optional[torch.Tensor]
    point_intensity_mask: Optional[torch.Tensor]
    point_xyz: torch.Tensor
    point_batch: torch.Tensor
    point_offset: torch.Tensor

    original_point_count: int
    llm_pooled_voxel_count: Union[int, List[int]]
    llm_effective_voxel_size: Union[Optional[float], List[Optional[float]]]
    intensity_used: bool
    voxel_inverse: Optional[torch.Tensor] = None  # [N_original] -> [M_sparse]
    original_xyz: Optional[torch.Tensor] = None


@dataclass
class PointAdapterOutput:
    """Two-temporal spatial memory and per-point features.

    Equivalent in role to ImageAdapterOutput; output field names retain the
    exact contract expected by PAIRChangeDecoder.forward_3d().
    """
    memory: torch.Tensor                 # [B,L,256], zero-padded
    memory_time_ids: torch.Tensor        # [B,L], 0=pad, 1=T1, 2=T2
    point_features_t1: torch.Tensor      # [M1,256] voxel representatives
    point_features_t2: torch.Tensor      # [M2,256]
    point_batch_t1: torch.Tensor         # [M1]
    point_batch_t2: torch.Tensor         # [M2]
    point_xyz_t1: torch.Tensor           # [M1,3], original world reference frame
    point_xyz_t2: torch.Tensor           # [M2,3]
    box_scene_bounds: torch.Tensor       # [B,2,3], common T1/T2 bounds
    point_inverse_t1: Optional[torch.Tensor] = None  # [N_original] -> [M_t1]
    point_inverse_t2: Optional[torch.Tensor] = None  # [N_original] -> [M_t2]

    def decoder_inputs(self) -> Dict[str, torch.Tensor]:
        return {
            "memory": self.memory,
            "memory_time_ids": self.memory_time_ids,
            "point_features_t1": self.point_features_t1,
            "point_features_t2": self.point_features_t2,
            "point_batch_t1": self.point_batch_t1,
            "point_batch_t2": self.point_batch_t2,
            "point_xyz_t1": self.point_xyz_t1,
            "point_xyz_t2": self.point_xyz_t2,
            "box_scene_bounds": self.box_scene_bounds,
            "point_inverse_t1": self.point_inverse_t1,
            "point_inverse_t2": self.point_inverse_t2,
        }


def _offset_from_batch(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 1:
        raise ValueError(f"batch must be [N], got {tuple(batch.shape)}")
    if batch.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=batch.device)

    batch = batch.long()
    if int(batch.min().item()) != 0:
        raise ValueError("batch IDs must start at 0")
    if (batch[1:] < batch[:-1]).any():
        raise ValueError("batch IDs must be grouped contiguously")

    num_batches = int(batch[-1].item()) + 1
    counts = torch.bincount(batch, minlength=num_batches)
    if (counts == 0).any():
        raise ValueError("batch IDs must be contiguous with no missing IDs")

    return torch.cumsum(counts, dim=0)


class PointAdapter(nn.Module):

    # Modules shared by T1 and T2.
    def __init__(self, config: Optional[PointAdapterConfig] = None):
        super().__init__()
        self.config = config or PointAdapterConfig()
        cfg = self.config

        if min(cfg.utonia_dim, cfg.decoder_dim, cfg.qwen_dim, cfg.llm_tokens_per_cloud) <= 0:
            raise ValueError("feature dimensions and llm_tokens_per_cloud must be > 0")
        if cfg.utonia_chunk_size <= 0:
            raise ValueError("utonia_chunk_size must be positive")
        if cfg.utonia_stage_channels is None:
            # Official Utonia has five recovered scales, not one 1386D level.
            self.utonia_stage_channels = (
                (54, 108, 216, 432, 576) if cfg.utonia_dim == 1386
                else (cfg.utonia_dim,)
            )
        else:
            self.utonia_stage_channels = tuple(int(v) for v in cfg.utonia_stage_channels)
        if (not self.utonia_stage_channels
                or min(self.utonia_stage_channels) <= 0
                or sum(self.utonia_stage_channels) != cfg.utonia_dim):
            raise ValueError("utonia_stage_channels must be positive and sum to utonia_dim")
        if cfg.intensity_hidden_dim <= 0:
            raise ValueError("intensity_hidden_dim must be > 0")
        if cfg.llm_sampling not in ("voxel", "uniform"):
            raise ValueError("llm_sampling must be 'voxel' or 'uniform'")
        if cfg.llm_voxel_size is not None and cfg.llm_voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if cfg.llm_voxel_oversample_factor <= 0:
            raise ValueError("voxel_oversample_factor must be > 0")
        if cfg.llm_max_fps_candidates < cfg.llm_tokens_per_cloud:
            raise ValueError("llm_max_fps_candidates must be >= llm_tokens_per_cloud")
        if min(cfg.detail_hidden_dim, cfg.detail_chunk_size, cfg.memory_tokens_per_cloud) <= 0:
            raise ValueError("point-detail and memory dimensions must be positive")
        if min(cfg.llm_align_chunk_size, cfg.llm_align_neighbors) <= 0:
            raise ValueError("llm_align_chunk_size and llm_align_neighbors must be positive")
        if min(cfg.subvoxel_size, cfg.memory_cell_size) <= 0:
            raise ValueError("point spatial cell sizes must be positive")
        if not 0 < cfg.detail_residual_init < 1:
            raise ValueError("detail_residual_init must be inside (0,1)")

        # ------------------------------------------------------------------
        # Utonia multi-scale feature fusion, corresponding to 2D level_fusion
        # and vit_fuse. Spatial scales were already mapped to original points
        # by point_encoder; split according to its documented concatenation.
        # Chunk projection avoids retaining N x (5*decoder_dim) activations.
        # ------------------------------------------------------------------
        self.level_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, cfg.decoder_dim),
                nn.LayerNorm(cfg.decoder_dim),
                nn.GELU(),
            )
            for channels in self.utonia_stage_channels
        ])
        self.utonia_fuse = nn.Sequential(
            nn.Linear(len(self.utonia_stage_channels) * cfg.decoder_dim, cfg.decoder_dim),
            nn.LayerNorm(cfg.decoder_dim),
            nn.GELU(),
        )

        # Optional LiDAR radiometry for the original-resolution detail branch.
        self.point_intensity_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, cfg.intensity_hidden_dim),
            nn.LayerNorm(cfg.intensity_hidden_dim),
            nn.GELU(),
        )

        # Residual intensity adapter. The last layer starts at zero so even on
        # an intensity-equipped dataset training begins exactly geometry-only.
        self.point_intensity_fuse = nn.Sequential(
            nn.Linear(
                cfg.decoder_dim + cfg.intensity_hidden_dim,
                cfg.decoder_dim,
            ),
            nn.GELU(),
            nn.Linear(cfg.decoder_dim, cfg.decoder_dim),
        )
        nn.init.zeros_(self.point_intensity_fuse[-1].weight)
        nn.init.zeros_(self.point_intensity_fuse[-1].bias)

        # ------------------------------------------------------------------
        # LLM point-token branch: RAW Utonia feature -> Qwen hidden space.
        #
        # REVISION:
        #   LLM: [M,1386] -> voxel/FPS -> [K,1386] -> [K,qwen_dim].
        # ------------------------------------------------------------------
        self.llm_token_project = nn.Linear(cfg.utonia_dim, cfg.qwen_dim)

        if cfg.use_xyz_pos:
            self.llm_xyz_position = nn.Sequential(
                nn.Linear(3, cfg.xyz_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.xyz_hidden_dim, cfg.qwen_dim),
            )
        else:
            self.llm_xyz_position = None

        self.llm_token_norm = (
            nn.LayerNorm(cfg.qwen_dim)
            if cfg.use_layer_norm
            else nn.Identity()
        )

        # Fixed 16 channels: normalized local XYZ(3), sub-voxel XYZ(3),
        # RGB(3)+presence(1), normal(3)+presence(1), intensity(1)+presence(1).
        self.point_detail_mlp = nn.Sequential(
            nn.Linear(16, cfg.detail_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.detail_hidden_dim, cfg.decoder_dim),
        )
        self.point_detail_norm = nn.LayerNorm(cfg.decoder_dim)
        self.point_detail_strength = nn.Parameter(torch.tensor(
            math.log(cfg.detail_residual_init / (1 - cfg.detail_residual_init))
        ))

        # Mirrors image_adapter.llm_visual_project: this projects native
        # post-Qwen point-position hidden, NOT the pre-Qwen point tokens.
        self.llm_point_project = nn.Linear(cfg.qwen_dim, cfg.decoder_dim)
        self.spatial_memory_norm = nn.LayerNorm(cfg.decoder_dim)

        self.utonia_dim = cfg.utonia_dim
        self.decoder_dim = cfg.decoder_dim
        self.qwen_dim = cfg.qwen_dim
        self.llm_tokens_per_cloud = cfg.llm_tokens_per_cloud

    # Input preparation and validation.
    def _extract(self, point_encoded: Any):
        if torch.is_tensor(point_encoded):
            features = point_encoded
            coord = batch = offset = None
            intensity = intensity_mask = None
        else:
            if isinstance(point_encoded, dict):
                getter = point_encoded.get
            else:
                getter = lambda name, default=None: getattr(
                    point_encoded, name, default
                )

            features = getter("features", None)
            if features is None:
                features = getter("feat", None)

            coord = getter("coord", None)
            batch = getter("batch", None)
            offset = getter("offset", None)
            intensity = getter("intensity", None)
            intensity_mask = getter("intensity_mask", None)

        if features is None or not torch.is_tensor(features):
            raise TypeError("PointAdapter needs Utonia features [N,C]")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(
                f"features must be non-empty [N,C], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.utonia_dim:
            raise ValueError(
                f"expected Utonia dim {self.utonia_dim}, got {features.shape[1]}"
            )

        n = int(features.shape[0])
        device = features.device

        if coord is None or not torch.is_tensor(coord):
            raise TypeError("PointAdapter requires coord [N,3]")
        if coord.shape != (n, 3):
            raise ValueError(f"coord must be [N,3], got {tuple(coord.shape)}")
        coord = coord.to(device=device)

        if batch is None:
            if offset is None:
                batch = torch.zeros(n, dtype=torch.long, device=device)
                offset = torch.tensor([n], dtype=torch.long, device=device)
            else:
                offset = torch.as_tensor(
                    offset, dtype=torch.long, device=device
                )
                if (
                    offset.ndim != 1
                    or offset.numel() == 0
                    or int(offset[-1].item()) != n
                ):
                    raise ValueError("invalid offset")

                starts = torch.cat([offset.new_zeros(1), offset[:-1]])
                counts = offset - starts
                if (counts <= 0).any():
                    raise ValueError("every batch item must contain points")

                batch = torch.repeat_interleave(
                    torch.arange(
                        offset.numel(),
                        dtype=torch.long,
                        device=device,
                    ),
                    counts,
                )
        else:
            batch = torch.as_tensor(batch, dtype=torch.long, device=device)
            if batch.shape != (n,):
                raise ValueError(f"batch must be [N], got {tuple(batch.shape)}")

            derived_offset = _offset_from_batch(batch)
            if offset is not None:
                offset = torch.as_tensor(
                    offset, dtype=torch.long, device=device
                )
                if not torch.equal(offset, derived_offset):
                    raise ValueError("batch and offset describe different layouts")
            offset = derived_offset

        if intensity is None:
            if intensity_mask is not None:
                raise ValueError(
                    "intensity_mask exists but intensity is absent"
                )
        else:
            intensity = torch.as_tensor(
                intensity, dtype=torch.float32, device=device
            )
            if intensity.ndim == 1:
                intensity = intensity.unsqueeze(1)
            if intensity.shape != (n, 1):
                raise ValueError(
                    f"intensity must be [N] or [N,1], got {tuple(intensity.shape)}"
                )
            if not torch.isfinite(intensity).all():
                raise ValueError("intensity contains NaN/Inf")

            if intensity_mask is None:
                intensity_mask = torch.ones(
                    (n, 1), dtype=torch.bool, device=device
                )
            else:
                intensity_mask = torch.as_tensor(
                    intensity_mask, device=device
                )
                if intensity_mask.ndim == 1:
                    intensity_mask = intensity_mask.unsqueeze(1)
                if intensity_mask.shape != (n, 1):
                    raise ValueError(
                        "intensity_mask must be [N] or [N,1]"
                    )
                if intensity_mask.dtype != torch.bool:
                    if not torch.all(
                        (intensity_mask == 0) | (intensity_mask == 1)
                    ):
                        raise ValueError(
                            "intensity_mask must contain bool or 0/1"
                        )
                    intensity_mask = intensity_mask.bool()

        return (
            features,
            coord,
            batch,
            offset,
            intensity,
            intensity_mask,
        )

    # Multi-scale Utonia features, before Qwen hidden and raw-point detail.
    def _fuse_utonia_chunk(self, features: torch.Tensor) -> torch.Tensor:
        levels = torch.split(features, self.utonia_stage_channels, dim=1)
        projected = [module(level) for module, level in zip(self.level_fusion, levels)]
        return self.utonia_fuse(torch.cat(projected, dim=1))

    def _project_utonia_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.utonia_dim:
            raise ValueError(f"Utonia features must be [N,{self.utonia_dim}]")
        outputs = []
        dtype = self.level_fusion[0][0].weight.dtype
        for start in range(0, features.shape[0], self.config.utonia_chunk_size):
            part = features[start:start + self.config.utonia_chunk_size].to(dtype=dtype)
            if self.training and self.config.utonia_checkpoint:
                fused = checkpoint(self._fuse_utonia_chunk, part, use_reentrant=False)
            else:
                fused = self._fuse_utonia_chunk(part)
            outputs.append(fused)
        return torch.cat(outputs, dim=0)

    # The intensity side path is original-resolution detail, NOT coarse memory.
    def _fuse_intensity(
        self,
        point_features: torch.Tensor,
        intensity: Optional[torch.Tensor],
        intensity_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if intensity is None:
            return point_features
        if intensity_mask is None:
            raise RuntimeError("intensity exists without intensity_mask")
        mask = intensity_mask.to(dtype=point_features.dtype)
        encoded = self.point_intensity_encoder(
            (intensity * mask).to(dtype=self.point_intensity_encoder[0].weight.dtype)
        ).to(dtype=point_features.dtype) * mask
        correction = self.point_intensity_fuse(
            torch.cat((point_features, encoded), dim=1).to(
                dtype=self.point_intensity_fuse[0].weight.dtype
            )
        ).to(dtype=point_features.dtype)
        return point_features + correction * mask

    @staticmethod
    def _raw_attribute(raw: Optional[Dict[str, torch.Tensor]], name: str,
                       n: int, width: int, device: torch.device,
                       dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if raw is None or raw.get(name) is None:
            return (torch.zeros(n, width, device=device, dtype=dtype),
                    torch.zeros(n, 1, device=device, dtype=dtype))
        x = torch.as_tensor(raw[name], device=device, dtype=dtype)
        if width == 1 and x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.shape != (n, width):
            raise ValueError(f"raw_point_dict[{name!r}] expected [{n},{width}], got {tuple(x.shape)}")
        mask = raw.get(name + '_mask')
        if mask is None:
            mask = torch.ones((n, 1), device=device, dtype=dtype)
        else:
            mask = torch.as_tensor(mask, device=device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1)
            if mask.shape != (n, 1):
                raise ValueError(f"{name}_mask must be [{n},1]")
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(f"{name}_mask must contain only 0/1 values")
            mask = mask.to(dtype=dtype)
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN/Inf")
        return x * mask, mask

    def _prepare_point_detail(self, coord: torch.Tensor, batch: torch.Tensor,
                      raw: Optional[Dict[str, torch.Tensor]],
                      fallback_intensity: Optional[torch.Tensor],
                      fallback_mask: Optional[torch.Tensor]) -> torch.Tensor:
        n = coord.shape[0]
        xyz = coord.float()
        local = torch.empty_like(xyz)
        subvoxel = torch.empty_like(xyz)
        # Per-cloud coordinates avoid giant UTM/world-coordinate magnitudes.
        for b in range(int(batch[-1].item()) + 1):
            idx = torch.nonzero(batch == b, as_tuple=False).flatten()
            cloud = xyz[idx]
            origin = cloud.amin(0)
            span = (cloud.amax(0) - origin).clamp_min(1e-3)
            local[idx] = 2.0 * ((cloud - origin) / span) - 1.0
            frac = torch.remainder((cloud - origin) / self.config.subvoxel_size, 1.0)
            subvoxel[idx] = 2.0 * frac - 1.0
        rgb, rgb_valid = self._raw_attribute(raw, 'rgb', n, 3, xyz.device, xyz.dtype)
        # Some sources use RGB 0..255, others use 0..1.
        if rgb_valid.any() and rgb.detach().abs().amax() > 1.5:
            rgb = rgb / 255.0
        normal, normal_valid = self._raw_attribute(raw, 'normal', n, 3, xyz.device, xyz.dtype)
        if raw is None or raw.get('intensity') is None:
            if fallback_intensity is None:
                intensity = torch.zeros(n, 1, device=xyz.device, dtype=xyz.dtype)
                int_valid = torch.zeros_like(intensity)
            else:
                intensity = fallback_intensity.to(device=xyz.device, dtype=xyz.dtype)
                int_valid = (fallback_mask.to(device=xyz.device, dtype=xyz.dtype)
                             if fallback_mask is not None else torch.ones_like(intensity))
                intensity = intensity * int_valid
        else:
            intensity, int_valid = self._raw_attribute(raw, 'intensity', n, 1, xyz.device, xyz.dtype)
        detail = torch.cat((local, subvoxel, rgb, rgb_valid, normal,
                            normal_valid, intensity, int_valid), dim=1)
        if detail.shape != (n, 16):
            raise AssertionError('raw detail input must have 16 channels')
        return detail

    def _fuse_point_detail(self, dense: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        proj = self.point_detail_mlp[0]
        outputs = []
        step = self.config.detail_chunk_size
        for i in range(0, detail.shape[0], step):
            inp = detail[i:i + step].to(device=dense.device, dtype=proj.weight.dtype)
            if self.training and self.config.detail_checkpoint:
                delta = checkpoint(self.point_detail_mlp, inp, use_reentrant=False)
            else:
                delta = self.point_detail_mlp(inp)
            outputs.append(delta.to(dtype=dense.dtype))
        correction = torch.cat(outputs, dim=0)
        return dense + torch.sigmoid(self.point_detail_strength) * self.point_detail_norm(correction)

    def _encode_spatial_features(
        self, memory_features: torch.Tensor, single: PointSingleAdapterOutput,
    ) -> torch.Tensor:
        """Fused K/V memory -> all original points + raw detail, as in 2D."""
        detailed = self._fuse_point_detail(memory_features, single.point_detail)
        return self._fuse_intensity(
            detailed, single.point_intensity, single.point_intensity_mask,
        )

    # Native Qwen point-position hidden: project and align back to every point.
    def _llm_hidden_list(
        self, value: TensorOrList, single: PointSingleAdapterOutput, name: str,
    ) -> List[torch.Tensor]:
        """Normalize post-Qwen readouts to one [Ki,qwen_dim] tensor per cloud.

        The Ki rows MUST match llm_point_tokens order, including any voxel/FPS
        sampling. This is not the same as the pre-Qwen llm_point_tokens tensor.
        """
        counts = single.llm_sampled_xyz
        xyz_list = [counts] if torch.is_tensor(counts) else counts
        batch_size = len(xyz_list)
        if torch.is_tensor(value):
            if batch_size == 1 and value.ndim == 2:
                values = [value]
            elif value.ndim == 3 and value.shape[0] == batch_size:
                values = [value[i, :xyz_list[i].shape[0]] for i in range(batch_size)]
                # Padded [B,K,C] is permitted, but never truncate real tokens.
                if any(value.shape[1] < x.shape[0] for x in xyz_list):
                    raise ValueError(f"{name}: padded tensor has fewer tokens than sampled points")
            else:
                raise ValueError(f"{name}: expected [K,C] for B=1 or [B,K,C]")
        elif isinstance(value, (tuple, list)) and len(value) == batch_size:
            values = list(value)
        else:
            raise ValueError(f"{name}: expected {batch_size} per-cloud Qwen hidden tensors")
        for bid, (hidden, xyz) in enumerate(zip(values, xyz_list)):
            expected = (xyz.shape[0], self.qwen_dim)
            if not torch.is_tensor(hidden) or hidden.shape != expected:
                raise ValueError(
                    f"{name}[{bid}]: post-Qwen hidden must be {expected}; "
                    f"got {getattr(hidden, 'shape', None)}"
                )
            if hidden.device != single.memory_features.device:
                raise ValueError(f"{name}[{bid}]: LLM hidden and Utonia features must share device")
            if not hidden.is_floating_point():
                raise TypeError(f"{name}[{bid}]: LLM hidden must be floating-point")
        return values

    def _encode_spatial_memory(
        self, single: PointSingleAdapterOutput, llm_point: TensorOrList, name: str,
    ) -> torch.Tensor:
        """Utonia + native LLM hidden, analogous to ImageAdapter memory fusion.

        Interpolation: nearest 3 (or fewer) sampled Qwen token positions in XYZ,
        inverse-distance weighted. Compute small [chunk,Ki] distance matrices,
        NEVER [N,Ki] globally or [N,Ki,D]. Gradients flow to the Qwen hidden
        states and llm_point_project; only XYZ nearest-neighbor selection is
        non-differentiable, as in conventional point-cloud interpolation.
        """
        hidden_list = self._llm_hidden_list(llm_point, single, name)
        xyz_list = (single.llm_sampled_xyz if isinstance(single.llm_sampled_xyz, list)
                    else [single.llm_sampled_xyz])
        b = single.point_batch
        original = single.memory_features
        output = []
        for bid, (hidden, sampled_xyz) in enumerate(zip(hidden_list, xyz_list)):
            idx = torch.nonzero(b == bid, as_tuple=False).flatten()
            if idx.numel() == 0 or sampled_xyz.shape[0] == 0:
                raise ValueError(f"{name}: cloud {bid} has no points or Qwen tokens")
            projected = self.llm_point_project(
                hidden.to(dtype=self.llm_point_project.weight.dtype)
            )
            # Work with local coordinates, not huge UTM/ECEF coordinates.
            # Preserve the input-point order within each batch item.
            origin = single.point_xyz[idx[0]].detach().float()
            src = sampled_xyz.detach().float() - origin
            k = min(self.config.llm_align_neighbors, src.shape[0])
            for start in range(0, idx.numel(), self.config.llm_align_chunk_size):
                selection = idx[start:start + self.config.llm_align_chunk_size]
                target = single.point_xyz[selection].detach().float() - origin
                with torch.no_grad():
                    distances = torch.cdist(target, src)
                    nearest_distance, nearest = distances.topk(k, largest=False, dim=1)
                    weights = 1.0 / nearest_distance.clamp_min(1e-4)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                # Only [chunk,k,D], not [chunk,Ki,D].
                aligned = (projected[nearest] * weights.to(projected.dtype).unsqueeze(-1)).sum(dim=1)
                output.append(original[selection] + aligned.to(dtype=original.dtype))
        return self.spatial_memory_norm(torch.cat(output, dim=0))

    # Single-temporal Qwen LLM point tokens.
    def _uniform_indices(self, n: int, device: torch.device):
        k = min(n, self.llm_tokens_per_cloud)
        if k == n:
            return torch.arange(n, device=device)
        return torch.linspace(0, n - 1, k, device=device).long()

    def _estimate_voxel_size(self, coord: torch.Tensor) -> float:
        if self.config.llm_voxel_size is not None:
            return float(self.config.llm_voxel_size)

        extent = (
            coord.float().amax(0) - coord.float().amin(0)
        ).clamp_min(1e-6)

        xy_extent = float(torch.max(extent[:2]).item())
        if xy_extent <= 1e-6:
            xy_extent = float(torch.max(extent).item())

        target_regions = max(
            float(self.llm_tokens_per_cloud)
            * self.config.llm_voxel_oversample_factor,
            1.0,
        )
        cells_per_axis = math.sqrt(target_regions)
        return max(xy_extent / max(cells_per_axis, 1.0), 1e-6)

    def _voxel_pool(
        self,
        features: torch.Tensor,
        coord: torch.Tensor,
    ):
        """
        Mean-pool RAW Utonia features and XYZ inside spatial voxels.

        features is intentionally [N,utonia_dim] here: LLM sampling happens
        before the point projection to decoder_dim.
        """
        llm_voxel_size = self._estimate_voxel_size(coord)

        # Pool in FP32 for numerical stability, then restore source dtype.
        feat_f = features.float()
        coord_f = coord.float()

        origin = coord_f.amin(0, keepdim=True)
        grid = torch.floor((coord_f - origin) / llm_voxel_size).long()

        _, inverse = torch.unique(
            grid,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        m = int(inverse.max().item()) + 1

        pooled_feat = torch.zeros(
            m,
            feat_f.shape[1],
            device=features.device,
            dtype=feat_f.dtype,
        )
        pooled_coord = torch.zeros(
            m,
            3,
            device=coord.device,
            dtype=coord_f.dtype,
        )
        counts = torch.zeros(
            m,
            1,
            device=features.device,
            dtype=feat_f.dtype,
        )

        pooled_feat.index_add_(0, inverse, feat_f)
        pooled_coord.index_add_(0, inverse, coord_f)
        counts.index_add_(
            0,
            inverse,
            torch.ones(
                features.shape[0],
                1,
                device=features.device,
                dtype=feat_f.dtype,
            ),
        )
        counts.clamp_min_(1.0)

        pooled_feat = pooled_feat / counts
        pooled_coord = pooled_coord / counts

        return (
            pooled_feat.to(features.dtype),
            pooled_coord.to(coord.dtype),
            llm_voxel_size,
        )

    def _preselect_candidates(self, coord: torch.Tensor):
        n = int(coord.shape[0])
        limit = self.config.llm_max_fps_candidates
        if n <= limit:
            return torch.arange(n, device=coord.device)

        # Deterministic thinning before O(KN) FPS if the voxel set is huge.
        return torch.linspace(
            0,
            n - 1,
            limit,
            device=coord.device,
        ).long()

    @staticmethod
    def _fps_indices(coord: torch.Tensor, k: int):
        n = int(coord.shape[0])
        if k >= n:
            return torch.arange(n, device=coord.device)

        xyz = coord.float()
        selected = torch.empty(
            k,
            dtype=torch.long,
            device=coord.device,
        )

        centroid = xyz.mean(0, keepdim=True)
        current = torch.argmax(((xyz - centroid) ** 2).sum(1))
        min_dist = torch.full(
            (n,),
            float("inf"),
            device=coord.device,
            dtype=torch.float32,
        )

        for i in range(k):
            selected[i] = current
            distance = ((xyz - xyz[current].unsqueeze(0)) ** 2).sum(1)
            min_dist = torch.minimum(min_dist, distance)
            current = torch.argmax(min_dist)

        return selected

    def _voxel_resample(
        self,
        features: torch.Tensor,
        coord: torch.Tensor,
    ):
        pooled_feat, pooled_coord, llm_voxel_size = self._voxel_pool(
            features,
            coord,
        )
        pooled_count = int(pooled_feat.shape[0])

        candidates = self._preselect_candidates(pooled_coord)
        candidate_coord = pooled_coord[candidates]

        k = min(self.llm_tokens_per_cloud, int(candidate_coord.shape[0]))
        local_idx = self._fps_indices(candidate_coord, k)
        selected_idx = candidates[local_idx]

        return (
            pooled_feat[selected_idx],
            pooled_coord[selected_idx],
            selected_idx,
            pooled_count,
            llm_voxel_size,
        )

    @staticmethod
    def _normalize_llm_xyz(
        llm_sampled_xyz: torch.Tensor,
        dense_coord: torch.Tensor,
    ):
        dense = dense_coord.float()
        xyz_min = dense.amin(0)
        xyz_max = dense.amax(0)
        center = (xyz_min + xyz_max) * 0.5
        scale = ((xyz_max - xyz_min).max() * 0.5).clamp_min(1e-6)
        return (llm_sampled_xyz.float() - center) / scale

    def _project_llm_point_tokens(
        self,
        llm_sampled_features: torch.Tensor,
        llm_sampled_xyz: torch.Tensor,
        dense_coord: torch.Tensor,
    ):
        if llm_sampled_features.ndim != 2:
            raise ValueError(
                "llm_sampled_features must be [K,C], got "
                f"{tuple(llm_sampled_features.shape)}"
            )
        if llm_sampled_features.shape[1] != self.utonia_dim:
            raise ValueError(
                "LLM point-token branch must receive raw Utonia features with "
                f"dim={self.utonia_dim}, got {llm_sampled_features.shape[1]}"
            )

        # Raw Utonia feature [K,1386] -> Qwen hidden [K,2560].
        llm_point_tokens = self.llm_token_project(llm_sampled_features.to(dtype=self.llm_token_project.weight.dtype))

        if self.llm_xyz_position is not None:
            xyz = self._normalize_llm_xyz(llm_sampled_xyz, dense_coord)
            xyz_embed = self.llm_xyz_position(
                xyz.to(dtype=self.llm_xyz_position[0].weight.dtype)
            ).to(dtype=llm_point_tokens.dtype)
            llm_point_tokens = llm_point_tokens + xyz_embed

        return self.llm_token_norm(llm_point_tokens)

    def encode_single(self, point_encoded: Any,
                              *, raw_point_dict: Optional[Dict[str,torch.Tensor]] = None
                              ) -> PointSingleAdapterOutput:
        (
            features,
            coord,
            batch,
            offset,
            intensity,
            intensity_mask,
        ) = self._extract(point_encoded)

        # --------------------------------------------------------------
        # Branch A: Utonia multi-scale fusion, before Qwen hidden / detail.
        # --------------------------------------------------------------
        memory_features = self._project_utonia_features(features)
        # Raw point attributes are provided in original N-point order.
        # Select the exact Utonia voxel representatives before detail fusion;
        # never allocate a [N,256] or [N,1386] intermediate here.
        getter = (point_encoded.get if isinstance(point_encoded, dict)
                  else lambda k, default=None: getattr(point_encoded, k, default))
        representative = getter('representative_indices')
        inverse = getter('voxel_inverse')
        original_xyz = getter('original_coord')
        if representative is not None:
            representative = representative.to(device=coord.device, dtype=torch.long)
            if representative.shape != (features.shape[0],):
                raise ValueError('representative_indices must be [M]')
            if inverse is None or original_xyz is None:
                raise ValueError('sparse Utonia output requires voxel_inverse and original_coord')
            if inverse.ndim != 1 or int(inverse.numel()) != int(original_xyz.shape[0]):
                raise ValueError('voxel_inverse must map N original points to M voxels')
            if inverse.device != coord.device or inverse.min() < 0 or inverse.max() >= coord.shape[0]:
                raise ValueError('voxel_inverse out of bounds or on wrong device')
            if raw_point_dict is not None:
                raw_point_dict = {
                    name: (value.index_select(0, representative) if
                           torch.is_tensor(value) and value.ndim >= 1 and
                           value.shape[0] == original_xyz.shape[0] else value)
                    for name, value in raw_point_dict.items()
                }
        point_detail = self._prepare_point_detail(
            coord, batch, raw_point_dict, intensity, intensity_mask,
        )
        intensity_used = bool(intensity_mask.any().item()) if intensity_mask is not None else False

        # --------------------------------------------------------------
        # Branch B: Qwen LLM point tokens.
        #
        # Sample raw Utonia features BEFORE point projection to decoder_dim.
        # --------------------------------------------------------------
        num_batches = int(offset.numel())

        llm_tokens_all = []
        llm_features_all = []
        llm_xyz_all = []
        llm_indices_all = []
        pooled_counts = []
        voxel_sizes = []

        for batch_id in range(num_batches):
            mask = batch == batch_id
            if not mask.any():
                raise RuntimeError(
                    f"empty cloud at batch index {batch_id}"
                )

            # LLM input tokens are sampled from raw Utonia, not from point_features.
            local_llm_features = features[mask]  # [Nb,1386]
            local_coord = coord[mask]

            if self.config.llm_sampling == "uniform":
                idx = self._uniform_indices(
                    local_llm_features.shape[0],
                    local_llm_features.device,
                )
                llm_sampled_features = local_llm_features[idx]
                llm_sampled_xyz = local_coord[idx]
                pooled_count = int(local_llm_features.shape[0])
                llm_voxel_size = None
            else:
                (
                    llm_sampled_features,
                    llm_sampled_xyz,
                    idx,
                    pooled_count,
                    llm_voxel_size,
                ) = self._voxel_resample(
                    local_llm_features,
                    local_coord,
                )

            llm_point_tokens = self._project_llm_point_tokens(
                llm_sampled_features,
                llm_sampled_xyz,
                local_coord,
            )

            if llm_point_tokens.ndim != 2 or llm_point_tokens.shape[1] != self.qwen_dim:
                raise RuntimeError(
                    f"unexpected token shape {tuple(llm_point_tokens.shape)}"
                )
            if llm_point_tokens.shape[0] > self.llm_tokens_per_cloud:
                raise RuntimeError(
                    "Qwen <POINT> token budget exceeded"
                )
            if not torch.isfinite(llm_point_tokens).all():
                raise RuntimeError(
                    "PointAdapter produced NaN/Inf LLM point tokens"
                )

            llm_tokens_all.append(llm_point_tokens)
            llm_features_all.append(llm_sampled_features)
            llm_xyz_all.append(llm_sampled_xyz)
            llm_indices_all.append(idx)
            pooled_counts.append(pooled_count)
            voxel_sizes.append(
                None if llm_voxel_size is None else float(llm_voxel_size)
            )

        if num_batches == 1:
            llm_tokens_out = llm_tokens_all[0]
            llm_features_out = llm_features_all[0]
            llm_xyz_out = llm_xyz_all[0]
            llm_indices_out = llm_indices_all[0]
            pooled_counts_out = pooled_counts[0]
            voxel_sizes_out = voxel_sizes[0]
        else:
            llm_tokens_out = llm_tokens_all
            llm_features_out = llm_features_all
            llm_xyz_out = llm_xyz_all
            llm_indices_out = llm_indices_all
            pooled_counts_out = pooled_counts
            voxel_sizes_out = voxel_sizes

        return PointSingleAdapterOutput(
            llm_point_tokens=llm_tokens_out,
            llm_sampled_features=llm_features_out,
            llm_sampled_xyz=llm_xyz_out,
            llm_sampled_indices=llm_indices_out,
            memory_features=memory_features,
            point_detail=point_detail,
            point_intensity=intensity,
            point_intensity_mask=intensity_mask,
            point_xyz=coord,
            point_batch=batch,
            point_offset=offset,
            original_point_count=int(original_xyz.shape[0]) if original_xyz is not None
                                 else int(features.shape[0]),
            llm_pooled_voxel_count=pooled_counts_out,
            llm_effective_voxel_size=voxel_sizes_out,
            intensity_used=intensity_used,
            voxel_inverse=inverse,
            original_xyz=original_xyz,
        )

    # Spatial memory representative sampling (not temporal matching).
    def _sample_spatial_memory(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        # Extract one point per occupied cell, then cap the memory budget.
        # No dense spatial [num_voxels,256] pooling tensor is instantiated.
        with torch.no_grad():
            origin = xyz.float().amin(dim=0)
            grid = torch.floor((xyz.float() - origin) / self.config.memory_cell_size).long()
            extent = grid.amax(dim=0) + 1
            if math.prod(int(x) for x in extent.tolist()) >= 1 << 62:
                raise OverflowError('memory grid cannot be represented by int64')
            key = grid[:,0] * (extent[1] * extent[2]) + grid[:,1] * extent[2] + grid[:,2]
            ordered, perm = torch.sort(key)
            first = torch.ones_like(ordered, dtype=torch.bool)
            first[1:] = ordered[1:] != ordered[:-1]
            selected = perm[first]
            # Spatially dispersed deterministic representatives from occupied cells.
            # Hash sorting avoids taking only a single corner of the XY extent.
            hashed = ((key[selected] * 6364136223846793005 +
                       1442695040888963407) & ((1 << 62) - 1))
            selected = selected[torch.argsort(hashed)]
            cap = self.config.memory_tokens_per_cloud
            if selected.numel() > cap:
                selected = selected[:cap]
        return features[selected]  # already normalized by _encode_spatial_memory

    def fuse_temporal(
        self, t1: PointSingleAdapterOutput, t2: PointSingleAdapterOutput,
        *, llm_point_t1: TensorOrList, llm_point_t2: TensorOrList,
    ) -> PointAdapterOutput:
        """Fuse each phase's Utonia and T_mm; return Memory, F_point, and XYZ.

        This adapter does NOT match T1/T2 points or create F_event.
        """
        if t1.memory_features.shape[1] != self.decoder_dim or t2.memory_features.shape[1] != self.decoder_dim:
            raise ValueError('memory feature width differs from decoder_dim')
        if t1.point_batch.device != t2.point_batch.device or int(t1.point_batch[-1]) != int(t2.point_batch[-1]):
            raise ValueError('T1/T2 must have the same number of cloud batch items')
        batch_size = int(t1.point_offset.numel())
        if batch_size != int(t2.point_offset.numel()):
            raise ValueError('T1/T2 offset length mismatch')
        fused_t1 = self._encode_spatial_memory(t1, llm_point_t1, 'llm_point_t1')
        fused_t2 = self._encode_spatial_memory(t2, llm_point_t2, 'llm_point_t2')
        point_t1 = self._encode_spatial_features(fused_t1, t1)
        point_t2 = self._encode_spatial_features(fused_t2, t2)
        xyz_t1, xyz_t2 = t1.point_xyz, t2.point_xyz
        batch_t1, batch_t2 = t1.point_batch, t2.point_batch
        spatial_memories = []
        bounds = []
        for bid in range(batch_size):
            ix1 = torch.nonzero(batch_t1 == bid, as_tuple=False).flatten()
            ix2 = torch.nonzero(batch_t2 == bid, as_tuple=False).flatten()
            if ix1.numel() == 0 or ix2.numel() == 0:
                raise ValueError('T1 and T2 must have points in every batch item')
            coords1, coords2 = xyz_t1[ix1], xyz_t2[ix2]
            # Grounding uses the complete ORIGINAL world-coordinate extent,
            # not the voxel representative extent (which can omit extrema).
            if t1.original_xyz is not None and t2.original_xyz is not None:
                if t1.voxel_inverse is None or t2.voxel_inverse is None:
                    raise ValueError('original XYZ requires complete voxel inverse mapping')
                # Original batch IDs follow original point -> voxel index ->
                # sparse batch; avoids assuming samples have equal sizes.
                orig1 = t1.original_xyz[batch_t1[t1.voxel_inverse] == bid]
                orig2 = t2.original_xyz[batch_t2[t2.voxel_inverse] == bid]
                lower = torch.minimum(orig1.amin(0), orig2.amin(0))
                upper = torch.maximum(orig1.amax(0), orig2.amax(0))
            else:
                lower = torch.minimum(coords1.amin(0), coords2.amin(0))
                upper = torch.maximum(coords1.amax(0), coords2.amax(0))
            bounds.append(torch.stack((lower, upper), dim=0))
            spatial_memories.append((
                self._sample_spatial_memory(fused_t1[ix1], coords1),
                self._sample_spatial_memory(fused_t2[ix2], coords2),
            ))
        max_length = max(a.shape[0] + b.shape[0] for a, b in spatial_memories)
        memory = point_t1.new_zeros((batch_size, max_length, self.decoder_dim))
        memory_time_ids = torch.zeros((batch_size, max_length), device=memory.device, dtype=torch.long)
        for bid, (a, b) in enumerate(spatial_memories):
            n1, n2 = a.shape[0], b.shape[0]
            memory[bid, :n1] = a
            memory[bid, n1:n1+n2] = b
            memory_time_ids[bid, :n1] = 1
            memory_time_ids[bid, n1:n1+n2] = 2
        return PointAdapterOutput(
            memory=memory, memory_time_ids=memory_time_ids,
            point_features_t1=point_t1, point_features_t2=point_t2,
            point_batch_t1=batch_t1, point_batch_t2=batch_t2,
            point_xyz_t1=xyz_t1, point_xyz_t2=xyz_t2,
            box_scene_bounds=torch.stack(bounds, 0),
            point_inverse_t1=t1.voxel_inverse,
            point_inverse_t2=t2.voxel_inverse,
        )

    def forward(
        self,
        *,
        point_encoded_t1: Any,
        point_encoded_t2: Any,
        llm_point_t1: TensorOrList,
        llm_point_t2: TensorOrList,
        raw_point_dict_t1: Optional[Dict[str, torch.Tensor]] = None,
        raw_point_dict_t2: Optional[Dict[str, torch.Tensor]] = None,
    ) -> PointAdapterOutput:
        """Build two-temporal spatial features, analogous to ImageAdapter.forward.

        If Qwen <POINT> tokens are needed first, call encode_single() on each
        epoch, pass .llm_point_tokens to the Qwen backbone, and reuse those
        outputs with fuse_temporal() to avoid encoding Utonia twice.
        """
        t1 = self.encode_single(point_encoded_t1, raw_point_dict=raw_point_dict_t1)
        t2 = self.encode_single(point_encoded_t2, raw_point_dict=raw_point_dict_t2)
        return self.fuse_temporal(
            t1, t2, llm_point_t1=llm_point_t1, llm_point_t2=llm_point_t2,
        )

    def predict(
        self, *, decoder: nn.Module, qwen_backbone: Any,
        task_hidden: torch.Tensor, class_names: Dict[int, str],
        point_encoded_t1: Any, point_encoded_t2: Any,
        llm_point_t1: TensorOrList, llm_point_t2: TensorOrList,
        raw_point_dict_t1: Optional[Dict[str,torch.Tensor]] = None,
        raw_point_dict_t2: Optional[Dict[str,torch.Tensor]] = None,
        boxes_3d: Optional[torch.Tensor] = None,
        box_valid: Optional[torch.Tensor] = None,
        box_scores: Optional[torch.Tensor] = None,
        **decoder_kwargs,
    ):
        """Generate 3D logits through the shared Change Decoder."""
        prepared = self.forward(
            point_encoded_t1=point_encoded_t1,
            point_encoded_t2=point_encoded_t2,
            llm_point_t1=llm_point_t1, llm_point_t2=llm_point_t2,
            raw_point_dict_t1=raw_point_dict_t1,
            raw_point_dict_t2=raw_point_dict_t2,
        )
        return decoder.forward_3d(
            qwen_backbone=qwen_backbone, task_hidden=task_hidden,
            class_names=class_names, **prepared.decoder_inputs(),
            boxes_3d=boxes_3d, box_valid=box_valid, box_scores=box_scores,
            **decoder_kwargs,
        )

    # Parameter statistics.
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
