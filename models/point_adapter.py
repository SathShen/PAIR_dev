"""
Spatial PointAdapter for PAIR.

Default path:
    PTv3 dense features [N,64] + XYZ
      -> adaptive voxel mean pooling
      -> spatial FPS token selection (max 512)
      -> feature projection 64->2560
      +  XYZ positional MLP
      -> LayerNorm
      -> Qwen point tokens [K,2560], K<=512

The old uniform sampler is kept as an ablation/compatibility option.
PTv3 dense features remain untouched for the future dense 3D decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn


@dataclass
class PointAdapterConfig:
    in_dim: int = 64
    out_dim: int = 2560

    # Maximum Qwen point-token budget PER cloud / PER time step.
    num_tokens: int = 512

    # New default. Keep "uniform" for ablation.
    sampling: str = "voxel"

    # None = estimate from scene XY extent.
    voxel_size: Optional[float] = None
    voxel_oversample_factor: float = 4.0

    # FPS is only run on at most this many pooled candidates.
    max_fps_candidates: int = 8192

    # Explicit geometry in Qwen token space.
    use_xyz_pos: bool = True
    xyz_hidden_dim: int = 128
    xyz_init_scale: float = 0.1

    use_layer_norm: bool = True


@dataclass
class PointAdapterOutput:
    tokens: torch.Tensor
    sampled_features: torch.Tensor
    sampled_coord: Optional[torch.Tensor]
    sampled_indices: torch.Tensor

    source_point_count: int
    pooled_voxel_count: int
    effective_voxel_size: Optional[float]


class PointAdapter(nn.Module):
    """PTv3 dense feature -> spatial Qwen point tokens."""

    def __init__(self, config: Optional[PointAdapterConfig] = None):
        super().__init__()
        self.config = config or PointAdapterConfig()
        cfg = self.config

        if cfg.num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if cfg.in_dim <= 0 or cfg.out_dim <= 0:
            raise ValueError("in_dim/out_dim must be > 0")
        if cfg.sampling not in ("voxel", "uniform"):
            raise ValueError("sampling must be 'voxel' or 'uniform'")
        if cfg.voxel_size is not None and cfg.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if cfg.voxel_oversample_factor <= 0:
            raise ValueError("voxel_oversample_factor must be > 0")
        if cfg.max_fps_candidates < cfg.num_tokens:
            raise ValueError("max_fps_candidates must be >= num_tokens")

        self.feature_proj = nn.Linear(cfg.in_dim, cfg.out_dim)

        if cfg.use_xyz_pos:
            self.xyz_mlp = nn.Sequential(
                nn.Linear(3, cfg.xyz_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.xyz_hidden_dim, cfg.out_dim),
            )
            self.xyz_scale = nn.Parameter(
                torch.tensor(float(cfg.xyz_init_scale), dtype=torch.float32)
            )
        else:
            self.xyz_mlp = None
            self.register_parameter("xyz_scale", None)

        self.norm = (
            nn.LayerNorm(cfg.out_dim)
            if cfg.use_layer_norm
            else nn.Identity()
        )

        self.in_dim = cfg.in_dim
        self.out_dim = cfg.out_dim
        self.num_tokens = cfg.num_tokens

    # ------------------------------------------------------------------
    # Input extraction
    # ------------------------------------------------------------------
    def _extract_dense_features(
        self, point_encoded: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if torch.is_tensor(point_encoded):
            features, coord, batch = point_encoded, None, None
        else:
            features = getattr(point_encoded, "features", None)
            coord = getattr(point_encoded, "coord", None)
            batch = getattr(point_encoded, "batch", None)

            if features is None and isinstance(point_encoded, dict):
                features = point_encoded.get(
                    "features", point_encoded.get("feat", None)
                )
                coord = point_encoded.get("coord", coord)
                batch = point_encoded.get("batch", batch)

        if features is None or not torch.is_tensor(features):
            raise TypeError("PointAdapter needs PTv3 features [N,C]")
        if features.ndim != 2:
            raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
        if features.shape[1] != self.in_dim:
            raise ValueError(
                f"expected feature dim {self.in_dim}, got {features.shape[1]}"
            )
        if features.shape[0] == 0:
            raise ValueError("empty point cloud")

        if coord is not None:
            if not torch.is_tensor(coord):
                raise TypeError("coord must be a tensor")
            if coord.shape != (features.shape[0], 3):
                raise ValueError(
                    f"coord must be [N,3], got {tuple(coord.shape)}"
                )

        if self.config.sampling == "voxel" and coord is None:
            raise ValueError("sampling='voxel' requires coord")

        if batch is not None:
            if not torch.is_tensor(batch) or batch.shape != (features.shape[0],):
                raise ValueError("batch must be [N]")
            if torch.unique(batch).numel() != 1:
                raise NotImplementedError(
                    "PointAdapter currently handles one cloud per call"
                )

        return features, coord, batch

    # ------------------------------------------------------------------
    # Uniform baseline
    # ------------------------------------------------------------------
    def _uniform_indices(self, n: int, device: torch.device) -> torch.Tensor:
        k = min(n, self.num_tokens)
        if k == n:
            return torch.arange(n, device=device)
        return torch.linspace(0, n - 1, k, device=device).long()

    # ------------------------------------------------------------------
    # Voxel pooling
    # ------------------------------------------------------------------
    def _estimate_voxel_size(self, coord: torch.Tensor) -> float:
        if self.config.voxel_size is not None:
            return float(self.config.voxel_size)

        c = coord.float()
        extent = (c.amax(0) - c.amin(0)).clamp_min(1e-6)

        # Remote sensing is predominantly XY-extended / 2.5D.
        xy_extent = float(torch.max(extent[:2]).item())
        if xy_extent <= 1e-6:
            xy_extent = float(torch.max(extent).item())

        target_regions = max(
            float(self.num_tokens) * self.config.voxel_oversample_factor,
            1.0,
        )
        cells_per_axis = math.sqrt(target_regions)
        return max(xy_extent / max(cells_per_axis, 1.0), 1e-6)

    def _voxel_pool(
        self, features: torch.Tensor, coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
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
            torch.ones(features.shape[0], 1, device=features.device),
        )

        counts.clamp_min_(1.0)
        pooled_feat = pooled_feat / counts
        pooled_coord = pooled_coord / counts

        return (
            pooled_feat.to(features.dtype),
            pooled_coord.to(coord.dtype),
            voxel_size,
        )

    # ------------------------------------------------------------------
    # Spatial selection
    # ------------------------------------------------------------------
    def _preselect_candidates(self, coord: torch.Tensor) -> torch.Tensor:
        m = coord.shape[0]
        limit = self.config.max_fps_candidates
        if m <= limit:
            return torch.arange(m, device=coord.device)
        return torch.linspace(0, m - 1, limit, device=coord.device).long()

    @staticmethod
    def _fps_indices(coord: torch.Tensor, k: int) -> torch.Tensor:
        """Deterministic FPS over pooled voxel centroids."""
        n = coord.shape[0]
        if k >= n:
            return torch.arange(n, device=coord.device)

        xyz = coord.float()
        selected = torch.empty(k, dtype=torch.long, device=coord.device)

        centroid = xyz.mean(0, keepdim=True)
        current = torch.argmax(((xyz - centroid) ** 2).sum(1))

        min_dist = torch.full(
            (n,), float("inf"), device=coord.device, dtype=torch.float32
        )

        for i in range(k):
            selected[i] = current
            d = ((xyz - xyz[current].unsqueeze(0)) ** 2).sum(1)
            min_dist = torch.minimum(min_dist, d)
            current = torch.argmax(min_dist)

        return selected

    def _voxel_resample(
        self, features: torch.Tensor, coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
        pooled_feat, pooled_coord, voxel_size = self._voxel_pool(
            features, coord
        )
        pooled_count = pooled_feat.shape[0]

        candidates = self._preselect_candidates(pooled_coord)
        candidate_coord = pooled_coord[candidates]

        k = min(self.num_tokens, candidate_coord.shape[0])
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
    # XYZ position encoding
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_xyz(
        sampled_coord: torch.Tensor, dense_coord: torch.Tensor
    ) -> torch.Tensor:
        dense = dense_coord.float()
        xyz_min = dense.amin(0)
        xyz_max = dense.amax(0)
        center = (xyz_min + xyz_max) * 0.5

        # One isotropic scale preserves relative X/Y/Z geometry.
        scale = ((xyz_max - xyz_min).max() * 0.5).clamp_min(1e-6)
        return (sampled_coord.float() - center) / scale

    def project(
        self,
        sampled_features: torch.Tensor,
        sampled_coord: Optional[torch.Tensor],
        dense_coord: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tokens = self.feature_proj(sampled_features)

        if self.config.use_xyz_pos:
            if sampled_coord is None or dense_coord is None:
                raise ValueError("XYZ position encoding requires coordinates")

            xyz = self._normalize_xyz(sampled_coord, dense_coord)
            xyz_embed = self.xyz_mlp(
                xyz.to(dtype=self.feature_proj.weight.dtype)
            ).to(tokens.dtype)

            tokens = tokens + self.xyz_scale.to(tokens.dtype) * xyz_embed

        return self.norm(tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward_with_metadata(self, point_encoded: Any) -> PointAdapterOutput:
        features, coord, _ = self._extract_dense_features(point_encoded)
        source_count = features.shape[0]

        if self.config.sampling == "uniform":
            idx = self._uniform_indices(source_count, features.device)
            sampled_feat = features[idx]
            sampled_coord = coord[idx] if coord is not None else None
            pooled_count = source_count
            voxel_size = None
        else:
            (
                sampled_feat,
                sampled_coord,
                idx,
                pooled_count,
                voxel_size,
            ) = self._voxel_resample(features, coord)

        tokens = self.project(sampled_feat, sampled_coord, coord)

        if tokens.ndim != 2 or tokens.shape[1] != self.out_dim:
            raise RuntimeError(f"unexpected token shape: {tuple(tokens.shape)}")
        if tokens.shape[0] > self.num_tokens:
            raise RuntimeError("token budget exceeded")
        if not torch.isfinite(tokens).all():
            raise RuntimeError("PointAdapter produced NaN/Inf")

        return PointAdapterOutput(
            tokens=tokens,
            sampled_features=sampled_feat,
            sampled_coord=sampled_coord,
            sampled_indices=idx,
            source_point_count=int(source_count),
            pooled_voxel_count=int(pooled_count),
            effective_voxel_size=(
                None if voxel_size is None else float(voxel_size)
            ),
        )

    def forward(self, point_encoded: Any) -> torch.Tensor:
        return self.forward_with_metadata(point_encoded).tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = PointAdapterConfig()
    adapter = PointAdapter(cfg)

    axis = torch.arange(16, dtype=torch.float32)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    coord = torch.stack(
        [x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=1
    ) * 0.05
    feat = torch.randn(coord.shape[0], cfg.in_dim)

    out = adapter.forward_with_metadata(
        {
            "features": feat,
            "coord": coord,
            "batch": torch.zeros(coord.shape[0], dtype=torch.long),
        }
    )

    print("point_adapter.py spatial standalone test OK")
    print("Dense input:", tuple(feat.shape))
    print("Occupied voxels:", out.pooled_voxel_count)
    print("Effective voxel size:", out.effective_voxel_size)
    print("Spatial features:", tuple(out.sampled_features.shape))
    print("Spatial coords:", tuple(out.sampled_coord.shape))
    print("Point tokens:", tuple(out.tokens.shape))
    print("Parameters:", adapter.parameter_count())
