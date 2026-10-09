"""
PAIR point adapter: Utonia dense, original-point detail, temporal memory and Qwen tokens.

Revised 3D path
---------------
Frozen Utonia dense feature [N,1386]
    |
    |-- dense branch
    |     -> geometry projector 1386 -> 256
    |     -> optional intensity residual adapter
    |     -> PAIR dense feature [N,256]
    |
    `-- reasoning branch (BEFORE dense adaptation)
          -> per-cloud voxel pooling on raw Utonia [N,1386]
          -> FPS to <=512 representatives
          -> token projector 1386 -> 2560
          -> + XYZ positional embedding
          -> <=512 Qwen reasoning tokens per cloud

This deliberately decouples the dense-decoder bottleneck from the Qwen
reasoning branch. The reasoning branch no longer has to pass through the
1386 -> 256 geometry projector before spatial compression.

Intensity never enters Utonia's pretrained 9-D input stem. In this version,
intensity is fused only into the dense decoder branch. Missing intensity is
represented by absence/mask, not by pretending that intensity == 0.
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
    in_dim: int = 1386

    # Unified decoder working width.
    dense_dim: int = 256

    # Qwen hidden width.
    out_dim: int = 2560

    # Maximum number of reasoning tokens per cloud.
    num_tokens: int = 512

    # Optional intensity side branch.
    intensity_hidden_dim: int = 64

    # Reasoning-token spatial sampling.
    sampling: str = "voxel"  # "voxel" or "uniform"
    voxel_size: Optional[float] = None
    voxel_oversample_factor: float = 4.0
    max_fps_candidates: int = 8192

    # Qwen-token XYZ positional encoding.
    use_xyz_pos: bool = True
    xyz_hidden_dim: int = 128
    use_layer_norm: bool = True

    # High-frequency branch. The MLP works on 16 input channels, NOT 1386.
    detail_hidden_dim: int = 48
    detail_chunk_size: int = 8192
    detail_checkpoint: bool = True
    detail_residual_init: float = 0.10
    subvoxel_size: float = 0.5

    # Coarse, coordinate-based two-epoch matching; no N1 x N2 cdist.
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
class PointAdapterOutput:
    # Tensor for B=1, list[Tensor] for B>1.
    tokens: TensorOrList

    # IMPORTANT: these are now sampled RAW Utonia features [K, in_dim],
    # not sampled dense-decoder features [K, dense_dim].
    sampled_features: TensorOrList
    sampled_coord: TensorOrList
    sampled_indices: TensorOrList

    # Original point topology, ready for the new shared change decoder.
    dense_features: torch.Tensor
    coord: torch.Tensor
    batch: torch.Tensor
    offset: torch.Tensor

    source_point_count: int
    pooled_voxel_count: Union[int, List[int]]
    effective_voxel_size: Union[Optional[float], List[Optional[float]]]
    intensity_used: bool


@dataclass
class PointPairAdapterOutput:
    """Two-epoch, ragged 3D dense features ready for PAIRChangeDecoder.forward_3d.

    memory is sampled for affordable query cross-attention; semantic/event
    features retain the original point counts and ordering from PointEncoder.
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
    Two-branch adapter after frozen Utonia.

    Dense branch:
        original-point recovered Utonia [N,in_dim]
          -> geometry_proj -> [N,dense_dim]
          -> optional intensity residual fusion
          -> original XYZ/RGB/normal/intensity point-wise MLP residual
          -> [N,dense_dim] for PAIRChangeDecoder

    Temporal pair branch:
        shared coordinate grid -> matched opposite-epoch features
          -> event features [N1/N2,dense_dim]
        spatial memory samples per epoch -> padded [B,L,dense_dim]
        no all-pairs distance matrix, no extra point-classification head

    Reasoning branch:
        raw Utonia [N,in_dim]
          -> voxel pooling / FPS
          -> [K,in_dim]
          -> token_proj -> [K,out_dim]
          -> + XYZ embedding
          -> Qwen <POINT> tokens

    The two branches split directly from the raw Utonia representation.
    """

    def __init__(self, config: Optional[PointAdapterConfig] = None):
        super().__init__()
        self.config = config or PointAdapterConfig()
        cfg = self.config

        if min(cfg.in_dim, cfg.dense_dim, cfg.out_dim, cfg.num_tokens) <= 0:
            raise ValueError("feature dimensions and num_tokens must be > 0")
        if cfg.intensity_hidden_dim <= 0:
            raise ValueError("intensity_hidden_dim must be > 0")
        if cfg.sampling not in ("voxel", "uniform"):
            raise ValueError("sampling must be 'voxel' or 'uniform'")
        if cfg.voxel_size is not None and cfg.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if cfg.voxel_oversample_factor <= 0:
            raise ValueError("voxel_oversample_factor must be > 0")
        if cfg.max_fps_candidates < cfg.num_tokens:
            raise ValueError("max_fps_candidates must be >= num_tokens")
        if min(cfg.detail_hidden_dim, cfg.detail_chunk_size, cfg.event_hidden_dim,
               cfg.event_chunk_size, cfg.memory_tokens_per_cloud) <= 0:
            raise ValueError("point-detail, event, and memory dimensions must be positive")
        if min(cfg.subvoxel_size, cfg.temporal_cell_size, cfg.memory_cell_size) <= 0:
            raise ValueError("point spatial cell sizes must be positive")
        if not (0 < cfg.detail_residual_init < 1 and 0 < cfg.event_residual_init < 1):
            raise ValueError("detail/event residual strengths must be inside (0,1)")

        # ------------------------------------------------------------------
        # Dense branch: frozen Utonia representation -> PAIR decoder width.
        # ------------------------------------------------------------------
        self.geometry_proj = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.dense_dim),
            nn.LayerNorm(cfg.dense_dim),
            nn.GELU(),
        )

        # Optional LiDAR radiometry for the dense branch.
        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, cfg.intensity_hidden_dim),
            nn.LayerNorm(cfg.intensity_hidden_dim),
            nn.GELU(),
        )

        # Residual intensity adapter. The last layer starts at zero so even on
        # an intensity-equipped dataset training begins exactly geometry-only.
        self.intensity_fusion = nn.Sequential(
            nn.Linear(
                cfg.dense_dim + cfg.intensity_hidden_dim,
                cfg.dense_dim,
            ),
            nn.GELU(),
            nn.Linear(cfg.dense_dim, cfg.dense_dim),
        )
        nn.init.zeros_(self.intensity_fusion[-1].weight)
        nn.init.zeros_(self.intensity_fusion[-1].bias)

        # ------------------------------------------------------------------
        # Reasoning branch: RAW Utonia feature -> Qwen hidden space.
        #
        # REVISION:
        #   old:  [N,1386] -> geometry_proj -> [N,256] -> voxel/FPS -> 2560
        #   new:  [N,1386] -> voxel/FPS -> [K,1386] -> 2560
        # ------------------------------------------------------------------
        self.token_proj = nn.Linear(cfg.in_dim, cfg.out_dim)

        if cfg.use_xyz_pos:
            self.xyz_mlp = nn.Sequential(
                nn.Linear(3, cfg.xyz_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.xyz_hidden_dim, cfg.out_dim),
            )
        else:
            self.xyz_mlp = None

        self.token_norm = (
            nn.LayerNorm(cfg.out_dim)
            if cfg.use_layer_norm
            else nn.Identity()
        )

        # Fixed 16 channels: normalized local XYZ(3), sub-voxel XYZ(3),
        # RGB(3)+presence(1), normal(3)+presence(1), intensity(1)+presence(1).
        self.detail_mlp = nn.Sequential(
            nn.Linear(16, cfg.detail_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.detail_hidden_dim, cfg.dense_dim),
        )
        self.detail_norm = nn.LayerNorm(cfg.dense_dim)
        self.detail_strength = nn.Parameter(torch.tensor(
            math.log(cfg.detail_residual_init / (1 - cfg.detail_residual_init))
        ))

        # Pairwise geometry affects Event features, not semantic features.
        # Event match is deliberately approximate (same spatial cell), with
        # exact zero memory if no match; later kNN refinement can replace it.
        self.event_mlp = nn.Sequential(
            nn.Linear(2 * cfg.dense_dim + 2, cfg.event_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.event_hidden_dim, cfg.dense_dim),
        )
        self.event_norm = nn.LayerNorm(cfg.dense_dim)
        self.event_strength = nn.Parameter(torch.tensor(
            math.log(cfg.event_residual_init / (1 - cfg.event_residual_init))
        ))
        self.memory_norm = nn.LayerNorm(cfg.dense_dim)

        self.in_dim = cfg.in_dim
        self.dense_dim = cfg.dense_dim
        self.out_dim = cfg.out_dim
        self.num_tokens = cfg.num_tokens

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
        if features.shape[1] != self.in_dim:
            raise ValueError(
                f"expected Utonia dim {self.in_dim}, got {features.shape[1]}"
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
    # Dense decoder branch
    # ------------------------------------------------------------------
    def make_dense_features(
        self,
        features: torch.Tensor,
        intensity: Optional[torch.Tensor],
        intensity_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        # Utonia [N,1386] -> decoder feature [N,256].
        geometry = self.geometry_proj(features.to(dtype=self.geometry_proj[0].weight.dtype))

        # NYC-SCD and other XYZ-only datasets take this exact path.
        if intensity is None:
            return geometry, False

        if intensity_mask is None:
            raise RuntimeError(
                "internal error: intensity exists but intensity_mask is None"
            )

        mask = intensity_mask.to(dtype=geometry.dtype)

        intensity_feature = self.intensity_encoder(
            intensity.to(dtype=self.intensity_encoder[0].weight.dtype)
            * mask.to(dtype=self.intensity_encoder[0].weight.dtype)
        ).to(dtype=geometry.dtype)

        # Keep invalid/missing intensity points exactly zero in the side branch.
        intensity_feature = intensity_feature * mask

        delta = self.intensity_fusion(
            torch.cat([geometry, intensity_feature], dim=1).to(
                dtype=self.intensity_fusion[0].weight.dtype)
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

    def _detail_input(self, coord: torch.Tensor, batch: torch.Tensor,
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

    def _add_detail(self, dense: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        proj = self.detail_mlp[0]
        outputs = []
        step = self.config.detail_chunk_size
        for i in range(0, detail.shape[0], step):
            inp = detail[i:i + step].to(device=dense.device, dtype=proj.weight.dtype)
            if self.training and self.config.detail_checkpoint:
                delta = checkpoint(self.detail_mlp, inp, use_reentrant=False)
            else:
                delta = self.detail_mlp(inp)
            outputs.append(delta.to(dtype=dense.dtype))
        correction = torch.cat(outputs, dim=0)
        return dense + torch.sigmoid(self.detail_strength) * self.detail_norm(correction)

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

    def _event_features(self, local: torch.Tensor, other: torch.Tensor,
                        local_xyz: torch.Tensor, other_xyz: torch.Tensor,
                        source_indices: torch.Tensor, matched: torch.Tensor):
        parts = []
        step = self.config.event_chunk_size
        dense_dtype = self.event_mlp[0].weight.dtype
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
                delta = checkpoint(self.event_mlp, fields, use_reentrant=False)
            else:
                delta = self.event_mlp(fields)
            parts.append(a + torch.sigmoid(self.event_strength) *
                         self.event_norm(delta.to(a.dtype)))
        return torch.cat(parts, dim=0)

    def _memory_tokens(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
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
        return self.memory_norm(features[selected])

    def forward_pair(self, t1: PointAdapterOutput,
                     t2: PointAdapterOutput) -> PointPairAdapterOutput:
        """Construct shared 2D/3D QueryDecoder-compatible memory and 3D Event features.

        Batches are ragged, phases must share their metric coordinate reference.
        Temporal matching is SAME-CELL, NOT exact nearest-neighbor correspondence.
        """
        if t1.dense_features.shape[1] != self.dense_dim or t2.dense_features.shape[1] != self.dense_dim:
            raise ValueError('point feature width differs from decoder_dim')
        if t1.batch.device != t2.batch.device or t1.batch[-1] != t2.batch[-1]:
            raise ValueError('T1/T2 must have the same number of cloud batch items')
        B = int(t1.offset.numel())
        x1, x2 = t1.dense_features, t2.dense_features
        coords1, coords2 = t1.coord, t2.coord
        batches1, batches2 = t1.batch, t2.batch
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
            event1.append(self._event_features(a, b, xyz1, xyz2, to2, valid1))
            event2.append(self._event_features(b, a, xyz2, xyz1, to1, valid2))
            m1 = self._memory_tokens(a, xyz1)
            m2 = self._memory_tokens(b, xyz2)
            memories.append((m1,m2))
        # max lengths vary per batch. Padding is ignored through time_ids=0.
        max_len = max(m1.shape[0] + m2.shape[0] for m1,m2 in memories)
        memory = x1.new_zeros((B,max_len,self.dense_dim))
        time_ids = torch.zeros((B,max_len),device=x1.device,dtype=torch.long)
        for bid,(m1,m2) in enumerate(memories):
            n1,n2 = m1.shape[0],m2.shape[0]
            memory[bid,:n1] = m1
            memory[bid,n1:n1+n2] = m2
            time_ids[bid,:n1] = 1
            time_ids[bid,n1:n1+n2] = 2
        return PointPairAdapterOutput(
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

    def predict(self, *, decoder: nn.Module, qwen_backbone,
                task_hidden: torch.Tensor, class_names: Dict[int,str],
                point_encoded_t1: Any, point_encoded_t2: Any,
                raw_point_dict_t1: Optional[Dict[str,torch.Tensor]] = None,
                raw_point_dict_t2: Optional[Dict[str,torch.Tensor]] = None,
                boxes_3d: Optional[torch.Tensor] = None,
                box_valid: Optional[torch.Tensor] = None,
                box_scores: Optional[torch.Tensor] = None,
                **decoder_kwargs):
        """Convenience integration; does not duplicate Decoder heads."""
        a = self.forward_with_metadata(point_encoded_t1, raw_point_dict=raw_point_dict_t1)
        b = self.forward_with_metadata(point_encoded_t2, raw_point_dict=raw_point_dict_t2)
        pair = self.forward_pair(a,b)
        return decoder.forward_3d(
            qwen_backbone=qwen_backbone, task_hidden=task_hidden,
            class_names=class_names, **pair.decoder_inputs(),
            boxes_3d=boxes_3d, box_valid=box_valid, box_scores=box_scores,
            **decoder_kwargs,
        )

    # ------------------------------------------------------------------
    # Per-cloud spatial sampling for the reasoning branch
    # ------------------------------------------------------------------
    def _uniform_indices(self, n: int, device: torch.device):
        k = min(n, self.num_tokens)
        if k == n:
            return torch.arange(n, device=device)
        return torch.linspace(0, n - 1, k, device=device).long()

    def _estimate_voxel_size(self, coord: torch.Tensor) -> float:
        if self.config.voxel_size is not None:
            return float(self.config.voxel_size)

        extent = (
            coord.float().amax(0) - coord.float().amin(0)
        ).clamp_min(1e-6)

        xy_extent = float(torch.max(extent[:2]).item())
        if xy_extent <= 1e-6:
            xy_extent = float(torch.max(extent).item())

        target_regions = max(
            float(self.num_tokens)
            * self.config.voxel_oversample_factor,
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

        features is intentionally [N,in_dim] here. This is the architectural
        change: voxel compression happens before geometry_proj / dense_dim.
        """
        voxel_size = self._estimate_voxel_size(coord)

        # Pool in FP32 for numerical stability, then restore source dtype.
        feat_f = features.float()
        coord_f = coord.float()

        origin = coord_f.amin(0, keepdim=True)
        grid = torch.floor((coord_f - origin) / voxel_size).long()

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
            voxel_size,
        )

    def _preselect_candidates(self, coord: torch.Tensor):
        n = int(coord.shape[0])
        limit = self.config.max_fps_candidates
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
        pooled_feat, pooled_coord, voxel_size = self._voxel_pool(
            features,
            coord,
        )
        pooled_count = int(pooled_feat.shape[0])

        candidates = self._preselect_candidates(pooled_coord)
        candidate_coord = pooled_coord[candidates]

        k = min(self.num_tokens, int(candidate_coord.shape[0]))
        local_idx = self._fps_indices(candidate_coord, k)
        selected_idx = candidates[local_idx]

        return (
            pooled_feat[selected_idx],
            pooled_coord[selected_idx],
            selected_idx,
            pooled_count,
            voxel_size,
        )

    # ------------------------------------------------------------------
    # Qwen reasoning-token projection
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_xyz(
        sampled_coord: torch.Tensor,
        dense_coord: torch.Tensor,
    ):
        dense = dense_coord.float()
        xyz_min = dense.amin(0)
        xyz_max = dense.amax(0)
        center = (xyz_min + xyz_max) * 0.5
        scale = ((xyz_max - xyz_min).max() * 0.5).clamp_min(1e-6)
        return (sampled_coord.float() - center) / scale

    def _project_tokens(
        self,
        sampled_features: torch.Tensor,
        sampled_coord: torch.Tensor,
        dense_coord: torch.Tensor,
    ):
        if sampled_features.ndim != 2:
            raise ValueError(
                "sampled_features must be [K,C], got "
                f"{tuple(sampled_features.shape)}"
            )
        if sampled_features.shape[1] != self.in_dim:
            raise ValueError(
                "reasoning branch must receive raw Utonia features with "
                f"dim={self.in_dim}, got {sampled_features.shape[1]}"
            )

        # Raw Utonia feature [K,1386] -> Qwen hidden [K,2560].
        tokens = self.token_proj(sampled_features.to(dtype=self.token_proj.weight.dtype))

        if self.xyz_mlp is not None:
            xyz = self._normalize_xyz(sampled_coord, dense_coord)
            xyz_embed = self.xyz_mlp(
                xyz.to(dtype=self.xyz_mlp[0].weight.dtype)
            ).to(dtype=tokens.dtype)
            tokens = tokens + xyz_embed

        return self.token_norm(tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward_with_metadata(self, point_encoded: Any,
                              *, raw_point_dict: Optional[Dict[str,torch.Tensor]] = None
                              ) -> PointAdapterOutput:
        (
            features,
            coord,
            batch,
            offset,
            intensity,
            intensity_mask,
        ) = self._extract(point_encoded)

        # --------------------------------------------------------------
        # Branch A: dense decoder feature.
        # --------------------------------------------------------------
        dense_features, intensity_used = self.make_dense_features(
            features,
            intensity,
            intensity_mask,
        )

        # Recover original-point high-frequency information; no extra NxN.
        detail = self._detail_input(coord, batch, raw_point_dict,
                                    intensity, intensity_mask)
        dense_features = self._add_detail(dense_features, detail)

        # --------------------------------------------------------------
        # Branch B: Qwen reasoning tokens.
        #
        # IMPORTANT: sample RAW Utonia features, NOT dense_features.
        # Voxel/FPS therefore happens before the 1386 -> 256 dense adapter.
        # --------------------------------------------------------------
        num_batches = int(offset.numel())

        tokens_all = []
        sampled_features_all = []
        sampled_coord_all = []
        sampled_indices_all = []
        pooled_counts = []
        voxel_sizes = []

        for batch_id in range(num_batches):
            mask = batch == batch_id
            if not mask.any():
                raise RuntimeError(
                    f"empty cloud at batch index {batch_id}"
                )

            # THIS is the key architectural change.
            local_reasoning_features = features[mask]  # [Nb,1386]
            local_coord = coord[mask]

            if self.config.sampling == "uniform":
                idx = self._uniform_indices(
                    local_reasoning_features.shape[0],
                    local_reasoning_features.device,
                )
                sampled_features = local_reasoning_features[idx]
                sampled_coord = local_coord[idx]
                pooled_count = int(local_reasoning_features.shape[0])
                voxel_size = None
            else:
                (
                    sampled_features,
                    sampled_coord,
                    idx,
                    pooled_count,
                    voxel_size,
                ) = self._voxel_resample(
                    local_reasoning_features,
                    local_coord,
                )

            tokens = self._project_tokens(
                sampled_features,
                sampled_coord,
                local_coord,
            )

            if tokens.ndim != 2 or tokens.shape[1] != self.out_dim:
                raise RuntimeError(
                    f"unexpected token shape {tuple(tokens.shape)}"
                )
            if tokens.shape[0] > self.num_tokens:
                raise RuntimeError(
                    "point reasoning-token budget exceeded"
                )
            if not torch.isfinite(tokens).all():
                raise RuntimeError(
                    "PointAdapter produced NaN/Inf reasoning tokens"
                )

            tokens_all.append(tokens)
            sampled_features_all.append(sampled_features)
            sampled_coord_all.append(sampled_coord)
            sampled_indices_all.append(idx)
            pooled_counts.append(pooled_count)
            voxel_sizes.append(
                None if voxel_size is None else float(voxel_size)
            )

        if num_batches == 1:
            tokens_out = tokens_all[0]
            sampled_features_out = sampled_features_all[0]
            sampled_coord_out = sampled_coord_all[0]
            sampled_indices_out = sampled_indices_all[0]
            pooled_counts_out = pooled_counts[0]
            voxel_sizes_out = voxel_sizes[0]
        else:
            tokens_out = tokens_all
            sampled_features_out = sampled_features_all
            sampled_coord_out = sampled_coord_all
            sampled_indices_out = sampled_indices_all
            pooled_counts_out = pooled_counts
            voxel_sizes_out = voxel_sizes

        return PointAdapterOutput(
            tokens=tokens_out,
            sampled_features=sampled_features_out,
            sampled_coord=sampled_coord_out,
            sampled_indices=sampled_indices_out,
            dense_features=dense_features,
            coord=coord,
            batch=batch,
            offset=offset,
            source_point_count=int(features.shape[0]),
            pooled_voxel_count=pooled_counts_out,
            effective_voxel_size=voxel_sizes_out,
            intensity_used=intensity_used,
        )

    def forward(self, point_encoded: Any):
        return self.forward_with_metadata(point_encoded).tokens

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
    feat = torch.randn(n, cfg.in_dim)
    batch = torch.cat([
        torch.zeros(n1, dtype=torch.long),
        torch.ones(n2, dtype=torch.long),
    ])

    intensity = torch.rand(n, 1)
    intensity_mask = torch.cat([
        torch.zeros(n1, 1, dtype=torch.bool),
        torch.ones(n2, 1, dtype=torch.bool),
    ])

    out = adapter.forward_with_metadata({
        "features": feat,
        "coord": coord,
        "batch": batch,
        "intensity": intensity,
        "intensity_mask": intensity_mask,
    })

    assert out.dense_features.shape == (n, cfg.dense_dim)
    assert isinstance(out.tokens, list) and len(out.tokens) == 2
    assert all(
        x.ndim == 2
        and x.shape[1] == cfg.out_dim
        and x.shape[0] <= cfg.num_tokens
        for x in out.tokens
    )

    # New architecture invariant: sampled reasoning features retain Utonia dim.
    assert isinstance(out.sampled_features, list)
    assert all(
        x.ndim == 2 and x.shape[1] == cfg.in_dim
        for x in out.sampled_features
    )

    # Geometry-only path must not require intensity.
    geo = adapter.forward_with_metadata({
        "features": feat[:n1],
        "coord": coord[:n1],
        "batch": torch.zeros(n1, dtype=torch.long),
    })
    assert geo.dense_features.shape == (n1, cfg.dense_dim)
    assert geo.intensity_used is False
    assert geo.sampled_features.shape[1] == cfg.in_dim

    print("PointAdapter standalone tests: PASS")
    print("dense:", tuple(out.dense_features.shape))
    print("tokens:", [tuple(x.shape) for x in out.tokens])
    print(
        "sampled raw Utonia features:",
        [tuple(x.shape) for x in out.sampled_features],
    )
    print("trainable params:", adapter.trainable_parameter_count())
