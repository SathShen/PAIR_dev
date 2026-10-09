"""PAIR 3D point adapter: Utonia + LLM point tokens + spatial prediction features.

The public terminology mirrors image_adapter.py:
  * LLM path: pretrained Utonia [N,utonia_dim] -> spatial sampling ->
    llm_point_tokens [K,qwen_dim], consumed by Qwen's <POINT> positions.
  * Spatial path: original point Utonia -> point detail MLP -> full-resolution
    point_features [N,decoder_dim]. This builds dense spatial memory (K/V).
  * Temporal path: T1/T2 coordinate-based matching -> event_features,
    and capped per-epoch spatial memory tokens for shared Query Decoder.

PointSingleAdapterOutput is a SINGLE-cloud output. PointAdapterOutput matches
ImageAdapterOutput's PAIR-level role: a two-epoch spatial adapter result with
memory, memory_time_ids, features, and decoder_inputs(). All prediction heads
and 3D MultiBoxGuidance remain in change_decoder.py.

This module does NOT create the task Query: Qwen returns task_hidden after it
processes llm_point_tokens, and change_decoder.py builds initial task Queries.
Nor does it currently re-inject LLM point-position hidden into 3D memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


@dataclass
class PointAdapterConfig:
    # Frozen Utonia output width.
    utonia_dim: int = 1386

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

    # Coarse coordinate-based temporal comparison; no N1 x N2 cdist.
    temporal_cell_size: float = 0.5
    event_hidden_dim: int = 64
    event_chunk_size: int = 8192
    event_checkpoint: bool = True
    event_residual_init: float = 0.10

    # Spatial representatives are used only for Decoder Memory. Final
    # logits still cover EVERY original input point.
    memory_tokens_per_cloud: int = 1024
    memory_cell_size: float = 0.5


@dataclass
class PointSingleAdapterOutput:
    """Single-epoch Utonia: LLM point tokens and full-resolution point features."""
    # Tensor for B=1, list[Tensor] for B>1.
    llm_point_tokens: TensorOrList

    # Samples of RAW Utonia features [K,utonia_dim], before point projection.
    llm_sampled_features: TensorOrList
    llm_sampled_xyz: TensorOrList
    llm_sampled_indices: TensorOrList

    # Original point topology, ready for the new shared change decoder.
    point_features: torch.Tensor
    point_xyz: torch.Tensor
    point_batch: torch.Tensor
    point_offset: torch.Tensor

    original_point_count: int
    llm_pooled_voxel_count: Union[int, List[int]]
    llm_effective_voxel_size: Union[Optional[float], List[Optional[float]]]
    intensity_used: bool


@dataclass
class PointAdapterOutput:
    """Two-temporal spatial memory and per-point features.

    Equivalent in role to ImageAdapterOutput; output field names retain the
    exact contract expected by PAIRChangeDecoder.forward_3d().
    """
    memory: torch.Tensor                 # [B,L,256], zero-padded
    memory_time_ids: torch.Tensor        # [B,L], 0=pad, 1=T1, 2=T2
    point_features_t1: torch.Tensor      # [N1,256]
    point_features_t2: torch.Tensor      # [N2,256]
    event_features_t1: torch.Tensor      # [N1,256]
    event_features_t2: torch.Tensor      # [N2,256]
    point_batch_t1: torch.Tensor         # [N1]
    point_batch_t2: torch.Tensor         # [N2]
    point_xyz_t1: torch.Tensor           # [N1,3], original reference frame
    point_xyz_t2: torch.Tensor           # [N2,3]
    box_scene_bounds: torch.Tensor       # [B,2,3], common T1/T2 bounds
    temporal_match_ratio_t1: float
    temporal_match_ratio_t2: float

    def decoder_inputs(self) -> Dict[str, torch.Tensor]:
        return {
            "memory": self.memory,
            "memory_time_ids": self.memory_time_ids,
            "point_features_t1": self.point_features_t1,
            "point_features_t2": self.point_features_t2,
            "event_features_t1": self.event_features_t1,
            "event_features_t2": self.event_features_t2,
            "point_batch_t1": self.point_batch_t1,
            "point_batch_t2": self.point_batch_t2,
            "point_xyz_t1": self.point_xyz_t1,
            "point_xyz_t2": self.point_xyz_t2,
            "box_scene_bounds": self.box_scene_bounds,
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
    """
    Utonia 3D adapter with two paths, named consistently with ImageAdapter.

    LLM input path:
        raw Utonia [N,utonia_dim] -> spatial sampling [K,utonia_dim]
        -> llm_token_project -> [K,qwen_dim] -> XYZ embedding -> <POINT>.
        These are LLM input tokens, NOT Transformer Decoder Queries.

    Spatial prediction path:
        recovered Utonia [N,utonia_dim] -> point_project [N,decoder_dim]
        -> optional intensity and raw-point detail residuals.
        Temporal geometry fusion produces full-point Event features and
        time-labeled spatial Memory, used as Decoder K/V.

    The point head and Multi-box guidance are in change_decoder.py.
    """

    def __init__(self, config: Optional[PointAdapterConfig] = None):
        super().__init__()
        self.config = config or PointAdapterConfig()
        cfg = self.config

        if min(cfg.utonia_dim, cfg.decoder_dim, cfg.qwen_dim, cfg.llm_tokens_per_cloud) <= 0:
            raise ValueError("feature dimensions and llm_tokens_per_cloud must be > 0")
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
        if min(cfg.detail_hidden_dim, cfg.detail_chunk_size, cfg.event_hidden_dim,
               cfg.event_chunk_size, cfg.memory_tokens_per_cloud) <= 0:
            raise ValueError("point-detail, event, and memory dimensions must be positive")
        if min(cfg.subvoxel_size, cfg.temporal_cell_size, cfg.memory_cell_size) <= 0:
            raise ValueError("point spatial cell sizes must be positive")
        if not (0 < cfg.detail_residual_init < 1 and 0 < cfg.event_residual_init < 1):
            raise ValueError("detail/event residual strengths must be inside (0,1)")

        # ------------------------------------------------------------------
        # Spatial feature branch: frozen Utonia representation -> PAIR decoder width.
        # ------------------------------------------------------------------
        self.point_project = nn.Sequential(
            nn.Linear(cfg.utonia_dim, cfg.decoder_dim),
            nn.LayerNorm(cfg.decoder_dim),
            nn.GELU(),
        )

        # Optional LiDAR radiometry for the spatial feature branch.
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
        #   LLM: [N,1386] -> voxel/FPS -> [K,1386] -> [K,qwen_dim].
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

        # Pairwise geometry affects Event features, not semantic features.
        # Event match is deliberately approximate (same spatial cell), with
        # exact zero memory if no match; later kNN refinement can replace it.
        self.temporal_event_mlp = nn.Sequential(
            nn.Linear(2 * cfg.decoder_dim + 2, cfg.event_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.event_hidden_dim, cfg.decoder_dim),
        )
        self.temporal_event_norm = nn.LayerNorm(cfg.decoder_dim)
        self.temporal_event_strength = nn.Parameter(torch.tensor(
            math.log(cfg.event_residual_init / (1 - cfg.event_residual_init))
        ))
        self.spatial_memory_norm = nn.LayerNorm(cfg.decoder_dim)

        self.utonia_dim = cfg.utonia_dim
        self.decoder_dim = cfg.decoder_dim
        self.qwen_dim = cfg.qwen_dim
        self.llm_tokens_per_cloud = cfg.llm_tokens_per_cloud

    # ------------------------------------------------------------------
    # Input normalization / validation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Spatial feature branch
    # ------------------------------------------------------------------
    def _project_point_features(
        self,
        features: torch.Tensor,
        intensity: Optional[torch.Tensor],
        intensity_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        # Utonia [N,1386] -> decoder feature [N,256].
        geometry = self.point_project(features.to(dtype=self.point_project[0].weight.dtype))

        # NYC-SCD and other XYZ-only datasets take this exact path.
        if intensity is None:
            return geometry, False

        if intensity_mask is None:
            raise RuntimeError(
                "internal error: intensity exists but intensity_mask is None"
            )

        mask = intensity_mask.to(dtype=geometry.dtype)

        intensity_feature = self.point_intensity_encoder(
            intensity.to(dtype=self.point_intensity_encoder[0].weight.dtype)
            * mask.to(dtype=self.point_intensity_encoder[0].weight.dtype)
        ).to(dtype=geometry.dtype)

        # Keep invalid/missing intensity points exactly zero in the side branch.
        intensity_feature = intensity_feature * mask

        delta = self.point_intensity_fuse(
            torch.cat([geometry, intensity_feature], dim=1).to(
                dtype=self.point_intensity_fuse[0].weight.dtype)
        ).to(dtype=geometry.dtype)

        # Invalid/missing intensity points are exactly geometry-only.
        dense = geometry + delta * mask
        return dense, bool(intensity_mask.any().item())

    # ------------------------------------------------------------------
    # Original-resolution point detail. Inputs are NOT fed into Utonia.
    # ------------------------------------------------------------------
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
        self,
        features: torch.Tensor,
        coord: torch.Tensor,
        batch: torch.Tensor,
        intensity: Optional[torch.Tensor],
        intensity_mask: Optional[torch.Tensor],
        raw_point_dict: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, bool]:
        """Utonia + original-resolution point detail, analogous to ImageAdapter.

        This does not alter sampling or voxel inverse mapping: features here
        are already aligned with the original input point order.
        """
        point_features, intensity_used = self._project_point_features(
            features, intensity, intensity_mask,
        )
        detail = self._prepare_point_detail(
            coord, batch, raw_point_dict, intensity, intensity_mask,
        )
        return self._fuse_point_detail(point_features, detail), intensity_used

    @staticmethod
    def _packed_grid(xyz: torch.Tensor, origin: torch.Tensor,
                     cell_size: float, limits: torch.Tensor) -> torch.Tensor:
        grid = torch.floor((xyz.float() - origin) / cell_size).long()
        if bool((grid < 0).any()):
            raise ValueError('coordinate origin is inconsistent across epochs')
        nx, ny, nz = (int(v) for v in limits.tolist())
        if nx * ny * nz >= (1 << 62):
            raise OverflowError('spatial grid cannot be packed safely in int64')
        return (grid[:, 0] * (ny * nz) + grid[:, 1] * nz + grid[:, 2])

    @staticmethod
    def _shared_grid_keys(coord1: torch.Tensor, coord2: torch.Tensor,
                          cell_size: float):
        xyz1, xyz2 = coord1.float(), coord2.float()
        origin = torch.minimum(xyz1.amin(0), xyz2.amin(0))
        maximum = torch.maximum(xyz1.amax(0), xyz2.amax(0))
        limits = torch.floor((maximum - origin) / cell_size).long() + 2
        return (PointAdapter._packed_grid(xyz1, origin, cell_size, limits),
                PointAdapter._packed_grid(xyz2, origin, cell_size, limits))

    @staticmethod
    def _same_voxel_lookup(target_key: torch.Tensor, source_key: torch.Tensor):
        # No [Nt,Ns] distance matrix, only sorting and searchsorted.
        ordered, perm = torch.sort(source_key)
        ix = torch.searchsorted(ordered.contiguous(), target_key.contiguous())
        ix = ix.clamp(max=ordered.numel() - 1)
        valid = ordered[ix] == target_key
        return perm[ix], valid

    def _fuse_temporal_features(self, local: torch.Tensor, other: torch.Tensor,
                        local_xyz: torch.Tensor, other_xyz: torch.Tensor,
                        source_indices: torch.Tensor, matched: torch.Tensor):
        parts = []
        step = self.config.event_chunk_size
        dense_dtype = self.temporal_event_mlp[0].weight.dtype
        for start in range(0, local.shape[0], step):
            end = min(start + step, local.shape[0])
            a = local[start:end]
            pick = source_indices[start:end]
            mask = matched[start:end, None]
            b = other[pick] * mask.to(dtype=other.dtype)
            distance = ((local_xyz[start:end].float() - other_xyz[pick].float())
                        .square().sum(-1, keepdim=True).sqrt())
            distance = (distance / self.config.temporal_cell_size).clamp(max=4.0)
            distance = distance * mask.float()
            fields = torch.cat((a, b-a, distance.to(dtype=a.dtype),
                                mask.to(dtype=a.dtype)), dim=-1).to(dense_dtype)
            if self.training and self.config.event_checkpoint:
                delta = checkpoint(self.temporal_event_mlp, fields, use_reentrant=False)
            else:
                delta = self.temporal_event_mlp(fields)
            parts.append(a + torch.sigmoid(self.temporal_event_strength) *
                         self.temporal_event_norm(delta.to(a.dtype)))
        return torch.cat(parts, dim=0)

    def _encode_spatial_memory(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
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
        return self.spatial_memory_norm(features[selected])

    def fuse_temporal(self, t1: PointSingleAdapterOutput,
                     t2: PointSingleAdapterOutput) -> PointAdapterOutput:
        """Construct shared 2D/3D QueryDecoder-compatible memory and 3D Event features.

        Batches are ragged, phases must share their metric coordinate reference.
        Temporal matching is SAME-CELL, NOT exact nearest-neighbor correspondence.
        """
        if t1.point_features.shape[1] != self.decoder_dim or t2.point_features.shape[1] != self.decoder_dim:
            raise ValueError('point feature width differs from decoder_dim')
        if t1.point_batch.device != t2.point_batch.device or t1.point_batch[-1] != t2.point_batch[-1]:
            raise ValueError('T1/T2 must have the same number of cloud batch items')
        B = int(t1.point_offset.numel())
        x1, x2 = t1.point_features, t2.point_features
        coords1, coords2 = t1.point_xyz, t2.point_xyz
        batches1, batches2 = t1.point_batch, t2.point_batch
        event1, event2 = [], []
        memories = []
        bounds = []
        matches1 = matches2 = 0
        for bid in range(B):
            ix1 = torch.nonzero(batches1 == bid, as_tuple=False).flatten()
            ix2 = torch.nonzero(batches2 == bid, as_tuple=False).flatten()
            if ix1.numel() == 0 or ix2.numel() == 0:
                raise ValueError('T1 and T2 must have points in every batch item')
            a, b = x1[ix1], x2[ix2]
            xyz1, xyz2 = coords1[ix1], coords2[ix2]
            lower = torch.minimum(xyz1.amin(0), xyz2.amin(0))
            upper = torch.maximum(xyz1.amax(0), xyz2.amax(0))
            bounds.append(torch.stack((lower, upper), dim=0))
            with torch.no_grad():
                key1,key2 = self._shared_grid_keys(xyz1, xyz2, self.config.temporal_cell_size)
                to2, valid1 = self._same_voxel_lookup(key1, key2)
                to1, valid2 = self._same_voxel_lookup(key2, key1)
            matches1 += int(valid1.sum().item())
            matches2 += int(valid2.sum().item())
            event1.append(self._fuse_temporal_features(a, b, xyz1, xyz2, to2, valid1))
            event2.append(self._fuse_temporal_features(b, a, xyz2, xyz1, to1, valid2))
            m1 = self._encode_spatial_memory(a, xyz1)
            m2 = self._encode_spatial_memory(b, xyz2)
            memories.append((m1,m2))
        # max lengths vary per batch. Padding is ignored through time_ids=0.
        max_len = max(m1.shape[0] + m2.shape[0] for m1,m2 in memories)
        memory = x1.new_zeros((B,max_len,self.decoder_dim))
        time_ids = torch.zeros((B,max_len),device=x1.device,dtype=torch.long)
        for bid,(m1,m2) in enumerate(memories):
            n1,n2 = m1.shape[0],m2.shape[0]
            memory[bid,:n1] = m1
            memory[bid,n1:n1+n2] = m2
            time_ids[bid,:n1] = 1
            time_ids[bid,n1:n1+n2] = 2
        return PointAdapterOutput(
            memory=memory, memory_time_ids=time_ids,
            point_features_t1=x1, point_features_t2=x2,
            event_features_t1=torch.cat(event1,0),
            event_features_t2=torch.cat(event2,0),
            point_batch_t1=batches1, point_batch_t2=batches2,
            point_xyz_t1=coords1, point_xyz_t2=coords2,
            box_scene_bounds=torch.stack(bounds,0),
            temporal_match_ratio_t1=matches1/max(1,x1.shape[0]),
            temporal_match_ratio_t2=matches2/max(1,x2.shape[0]),
        )

    def forward(
        self,
        *,
        point_encoded_t1: Any,
        point_encoded_t2: Any,
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
        return self.fuse_temporal(t1, t2)

    def predict(
        self, *, decoder: nn.Module, qwen_backbone: Any,
        task_hidden: torch.Tensor, class_names: Dict[int, str],
        point_encoded_t1: Any, point_encoded_t2: Any,
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
            raw_point_dict_t1=raw_point_dict_t1,
            raw_point_dict_t2=raw_point_dict_t2,
        )
        return decoder.forward_3d(
            qwen_backbone=qwen_backbone, task_hidden=task_hidden,
            class_names=class_names, **prepared.decoder_inputs(),
            boxes_3d=boxes_3d, box_valid=box_valid, box_scores=box_scores,
            **decoder_kwargs,
        )

    # ------------------------------------------------------------------
    # Per-cloud spatial sampling for the LLM point-token branch
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Qwen LLM-point-token projection
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # One-epoch feature encoding (called before and/or after Qwen forward)
    # ------------------------------------------------------------------
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
        # Branch A: full-resolution spatial point features.
        # --------------------------------------------------------------
        point_features, intensity_used = self._encode_spatial_features(
            features, coord, batch, intensity, intensity_mask, raw_point_dict,
        )

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
            point_features=point_features,
            point_xyz=coord,
            point_batch=batch,
            point_offset=offset,
            original_point_count=int(features.shape[0]),
            llm_pooled_voxel_count=pooled_counts_out,
            llm_effective_voxel_size=voxel_sizes_out,
            intensity_used=intensity_used,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


if __name__ == "__main__":
    cfg = PointAdapterConfig()
    adapter = PointAdapter(cfg)

    n1, n2 = 2500, 4200
    n = n1 + n2

    coord = torch.randn(n, 3)
    feat = torch.randn(n, cfg.utonia_dim)
    batch = torch.cat([
        torch.zeros(n1, dtype=torch.long),
        torch.ones(n2, dtype=torch.long),
    ])

    intensity = torch.rand(n, 1)
    intensity_mask = torch.cat([
        torch.zeros(n1, 1, dtype=torch.bool),
        torch.ones(n2, 1, dtype=torch.bool),
    ])

    out = adapter.encode_single({
        "features": feat,
        "coord": coord,
        "batch": batch,
        "intensity": intensity,
        "intensity_mask": intensity_mask,
    })

    assert out.point_features.shape == (n, cfg.decoder_dim)
    assert isinstance(out.llm_point_tokens, list) and len(out.llm_point_tokens) == 2
    assert all(
        x.ndim == 2
        and x.shape[1] == cfg.qwen_dim
        and x.shape[0] <= cfg.llm_tokens_per_cloud
        for x in out.llm_point_tokens
    )

    # LLM samples retain raw Utonia channel width.
    assert isinstance(out.llm_sampled_features, list)
    assert all(
        x.ndim == 2 and x.shape[1] == cfg.utonia_dim
        for x in out.llm_sampled_features
    )

    # Geometry-only path must not require intensity.
    geo = adapter.encode_single({
        "features": feat[:n1],
        "coord": coord[:n1],
        "batch": torch.zeros(n1, dtype=torch.long),
    })
    assert geo.point_features.shape == (n1, cfg.decoder_dim)
    assert geo.intensity_used is False
    assert geo.llm_sampled_features.shape[1] == cfg.utonia_dim

    print("PointAdapter standalone tests: PASS")
    print("dense:", tuple(out.point_features.shape))
    print("tokens:", [tuple(x.shape) for x in out.llm_point_tokens])
    print(
        "sampled raw Utonia features:",
        [tuple(x.shape) for x in out.llm_sampled_features],
    )
    print("trainable params:", adapter.trainable_parameter_count())
