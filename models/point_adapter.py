"""
PAIR point adapter for frozen Utonia + optional LiDAR intensity.

3D path
-------
Utonia dense feature [N,1386]
    -> geometry projector 1386 -> 256
    -> optional intensity residual adapter
    -> PAIR dense feature [N,256]
    -> per-cloud voxel pooling + FPS
    -> token projector 256 -> 2560
    -> <=512 Qwen reasoning tokens per cloud

Intensity never enters Utonia's pretrained 9-D input stem. Missing intensity
is represented by absence/mask, not by pretending that intensity == 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


@dataclass
class PointAdapterConfig:
    in_dim: int = 1386
    dense_dim: int = 256
    out_dim: int = 2560
    num_tokens: int = 512
    intensity_hidden_dim: int = 64

    sampling: str = "voxel"
    voxel_size: Optional[float] = None
    voxel_oversample_factor: float = 4.0
    max_fps_candidates: int = 8192

    use_xyz_pos: bool = True
    xyz_hidden_dim: int = 128
    use_layer_norm: bool = True


@dataclass
class PointAdapterOutput:
    # Tensor for B=1, list[Tensor] for B>1.
    tokens: TensorOrList
    sampled_features: TensorOrList
    sampled_coord: TensorOrList
    sampled_indices: TensorOrList

    # Original point topology, ready for Unified Decoder.
    dense_features: torch.Tensor
    batch: torch.Tensor
    offset: torch.Tensor

    source_point_count: int
    pooled_voxel_count: Union[int, List[int]]
    effective_voxel_size: Union[Optional[float], List[Optional[float]]]
    intensity_used: bool


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

        # Frozen Utonia representation -> PAIR working dimension.
        self.geometry_proj = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.dense_dim),
            nn.LayerNorm(cfg.dense_dim),
            nn.GELU(),
        )

        # Optional LiDAR radiometry.
        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, cfg.intensity_hidden_dim),
            nn.LayerNorm(cfg.intensity_hidden_dim),
            nn.GELU(),
        )

        # Residual intensity adapter. Its last layer starts at zero so even on
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

        # Dense -> Qwen reasoning-token space.
        self.token_proj = nn.Linear(cfg.dense_dim, cfg.out_dim)

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

        self.in_dim = cfg.in_dim
        self.dense_dim = cfg.dense_dim
        self.out_dim = cfg.out_dim
        self.num_tokens = cfg.num_tokens

    # ------------------------------------------------------------------
    # Input
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
                        offset.numel(), dtype=torch.long, device=device
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
            features, coord, batch, offset, intensity, intensity_mask
        )

    # ------------------------------------------------------------------
    # Dense feature fusion
    # ------------------------------------------------------------------

    def make_dense_features(
        self,
        features: torch.Tensor,
        intensity: Optional[torch.Tensor],
        intensity_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        geometry = self.geometry_proj(features)

        # NYC-SCD and other XYZ-only datasets take this exact path.
        if intensity is None:
            return geometry, False

        mask = intensity_mask.to(dtype=geometry.dtype)

        intensity_feature = self.intensity_encoder(
            intensity.to(dtype=self.intensity_encoder[0].weight.dtype)
            * mask.to(dtype=self.intensity_encoder[0].weight.dtype)
        ).to(dtype=geometry.dtype)
        intensity_feature = intensity_feature * mask

        delta = self.intensity_fusion(
            torch.cat([geometry, intensity_feature], dim=1)
        )

        # Invalid/missing intensity points are exactly geometry-only.
        dense = geometry + delta * mask
        return dense, bool(intensity_mask.any().item())

    # ------------------------------------------------------------------
    # Per-cloud spatial sampling
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
        voxel_size = self._estimate_voxel_size(coord)

        feat_f = features.float()
        coord_f = coord.float()
        origin = coord_f.amin(0, keepdim=True)
        grid = torch.floor((coord_f - origin) / voxel_size).long()

        _, inverse = torch.unique(
            grid, dim=0, sorted=True, return_inverse=True
        )
        m = int(inverse.max().item()) + 1

        pooled_feat = torch.zeros(
            m, feat_f.shape[1], device=features.device, dtype=feat_f.dtype
        )
        pooled_coord = torch.zeros(
            m, 3, device=coord.device, dtype=coord_f.dtype
        )
        counts = torch.zeros(
            m, 1, device=features.device, dtype=feat_f.dtype
        )

        pooled_feat.index_add_(0, inverse, feat_f)
        pooled_coord.index_add_(0, inverse, coord_f)
        counts.index_add_(
            0,
            inverse,
            torch.ones(
                features.shape[0], 1,
                device=features.device, dtype=feat_f.dtype,
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
        return torch.linspace(
            0, n - 1, limit, device=coord.device
        ).long()

    @staticmethod
    def _fps_indices(coord: torch.Tensor, k: int):
        n = int(coord.shape[0])
        if k >= n:
            return torch.arange(n, device=coord.device)

        xyz = coord.float()
        selected = torch.empty(k, dtype=torch.long, device=coord.device)

        centroid = xyz.mean(0, keepdim=True)
        current = torch.argmax(((xyz - centroid) ** 2).sum(1))
        min_dist = torch.full(
            (n,), float("inf"),
            device=coord.device, dtype=torch.float32,
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
            features, coord
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
    # Qwen token projection
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
        tokens = self.token_proj(sampled_features)

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

    def forward_with_metadata(self, point_encoded: Any) -> PointAdapterOutput:
        (
            features,
            coord,
            batch,
            offset,
            intensity,
            intensity_mask,
        ) = self._extract(point_encoded)

        dense_features, intensity_used = self.make_dense_features(
            features, intensity, intensity_mask
        )

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
                raise RuntimeError(f"empty cloud at batch index {batch_id}")

            local_features = dense_features[mask]
            local_coord = coord[mask]

            if self.config.sampling == "uniform":
                idx = self._uniform_indices(
                    local_features.shape[0], local_features.device
                )
                sampled_features = local_features[idx]
                sampled_coord = local_coord[idx]
                pooled_count = int(local_features.shape[0])
                voxel_size = None
            else:
                (
                    sampled_features,
                    sampled_coord,
                    idx,
                    pooled_count,
                    voxel_size,
                ) = self._voxel_resample(local_features, local_coord)

            tokens = self._project_tokens(
                sampled_features, sampled_coord, local_coord
            )

            if tokens.ndim != 2 or tokens.shape[1] != self.out_dim:
                raise RuntimeError(
                    f"unexpected token shape {tuple(tokens.shape)}"
                )
            if tokens.shape[0] > self.num_tokens:
                raise RuntimeError("point reasoning-token budget exceeded")
            if not torch.isfinite(tokens).all():
                raise RuntimeError("PointAdapter produced NaN/Inf")

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
            p.numel() for p in self.parameters() if p.requires_grad
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

    # Geometry-only path must not require intensity.
    geo = adapter.forward_with_metadata({
        "features": feat[:n1],
        "coord": coord[:n1],
        "batch": torch.zeros(n1, dtype=torch.long),
    })
    assert geo.dense_features.shape == (n1, cfg.dense_dim)
    assert geo.intensity_used is False

    print("PointAdapter standalone tests: PASS")
    print("dense:", tuple(out.dense_features.shape))
    print("tokens:", [tuple(x.shape) for x in out.tokens])
    print("trainable params:", adapter.trainable_parameter_count())
