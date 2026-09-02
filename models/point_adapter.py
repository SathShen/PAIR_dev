"""
Point adapter for PAIR.

Validated V1 path
-----------------
PTv3 dense point features [N, 64]
    -> simple uniform resampling
    -> [N_token, 64]
    -> Linear(64, 2560)
    -> LayerNorm
    -> Qwen-compatible point tokens [N_token, 2560]

This intentionally matches the successful PTv3-Qwen smoke test.

Important:
    - V1 uses a deliberately simple deterministic resampler.
    - No FPS, learned queries, cross-attention, XYZ encoding, or modality
      embedding is introduced yet.
    - The adapter only converts dense PTv3 features into a small set of
      Qwen-width tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PointAdapterConfig:
    """Configuration for the V1 PAIR point adapter."""

    in_dim: int = 64
    out_dim: int = 2560
    num_tokens: int = 32

    # Keep the first implementation identical in spirit to the smoke test.
    sampling: str = "uniform"

    use_layer_norm: bool = True


@dataclass
class PointAdapterOutput:
    """
    Optional detailed output for debugging and future decoder work.

    tokens:
        Qwen-compatible point tokens, [N_token, out_dim].

    sampled_features:
        PTv3 features before projection, [N_token, in_dim].

    sampled_coord:
        Coordinates corresponding to sampled PTv3 features, if available.

    sampled_indices:
        Indices into the dense PTv3 output.
    """

    tokens: torch.Tensor
    sampled_features: torch.Tensor
    sampled_coord: Optional[torch.Tensor]
    sampled_indices: torch.Tensor


class PointAdapter(nn.Module):
    """
    Convert dense PTv3 features into a compact set of Qwen point tokens.

    The normal forward() method returns only the token tensor so it can plug
    directly into PAIRModel:

        point_encoded = point_encoder(point_dict)
        point_tokens = point_adapter(point_encoded)
        qwen_backbone(..., point_tokens=point_tokens)

    Use forward_with_metadata() when sampled coordinates / indices are needed.
    """

    def __init__(
        self,
        config: Optional[PointAdapterConfig] = None,
    ):
        super().__init__()

        self.config = config or PointAdapterConfig()

        if self.config.num_tokens <= 0:
            raise ValueError("num_tokens must be > 0.")

        if self.config.in_dim <= 0:
            raise ValueError("in_dim must be > 0.")

        if self.config.out_dim <= 0:
            raise ValueError("out_dim must be > 0.")

        if self.config.sampling != "uniform":
            raise ValueError(
                "PAIR PointAdapter V1 currently supports only "
                "sampling='uniform'."
            )

        self.proj = nn.Linear(
            self.config.in_dim,
            self.config.out_dim,
        )

        self.norm = (
            nn.LayerNorm(self.config.out_dim)
            if self.config.use_layer_norm
            else nn.Identity()
        )

        self.in_dim = self.config.in_dim
        self.out_dim = self.config.out_dim
        self.num_tokens = self.config.num_tokens

    # ------------------------------------------------------------------
    # Input extraction / validation
    # ------------------------------------------------------------------

    def _extract_dense_features(
        self,
        point_encoded: Any,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Accept the PTv3Output wrapper used by PAIR.

        For convenience during debugging, a raw [N, C] tensor is also
        accepted.
        """

        if torch.is_tensor(point_encoded):
            features = point_encoded
            coord = None
            batch = None

        else:
            # Preferred PAIR PTv3Output interface.
            features = getattr(point_encoded, "features", None)
            coord = getattr(point_encoded, "coord", None)
            batch = getattr(point_encoded, "batch", None)

            # Small compatibility fallback for dict-style representations.
            if features is None and isinstance(point_encoded, dict):
                features = point_encoded.get(
                    "features",
                    point_encoded.get("feat", None),
                )
                coord = point_encoded.get("coord", coord)
                batch = point_encoded.get("batch", batch)

        if features is None:
            raise TypeError(
                "PointAdapter expected a PTv3Output-like object with "
                "'.features', a dict containing 'features'/'feat', or a "
                "raw feature tensor."
            )

        if not torch.is_tensor(features):
            raise TypeError("Dense point features must be a torch.Tensor.")

        if features.ndim != 2:
            raise ValueError(
                "Dense point features must have shape [N, C], "
                f"got {tuple(features.shape)}."
            )

        if features.shape[1] != self.in_dim:
            raise ValueError(
                f"PointAdapter expects feature dim {self.in_dim}, "
                f"got {features.shape[1]}."
            )

        if features.shape[0] < self.num_tokens:
            raise ValueError(
                f"PointAdapter needs at least {self.num_tokens} points "
                f"for V1 uniform sampling, got {features.shape[0]}."
            )

        if coord is not None:
            if not torch.is_tensor(coord):
                raise TypeError("coord must be a torch.Tensor when provided.")

            if coord.ndim != 2 or coord.shape != (features.shape[0], 3):
                raise ValueError(
                    "coord must have shape [N, 3] matching point features, "
                    f"got {tuple(coord.shape)}."
                )

        # The current Qwen injection wrapper is intentionally batch-size 1.
        # Catch multi-cloud input here rather than silently mixing clouds.
        if batch is not None:
            if not torch.is_tensor(batch):
                raise TypeError("batch must be a torch.Tensor when provided.")

            if batch.ndim != 1 or batch.shape[0] != features.shape[0]:
                raise ValueError(
                    "batch must have shape [N] matching point features."
                )

            unique_batches = torch.unique(batch)

            if unique_batches.numel() != 1:
                raise NotImplementedError(
                    "PointAdapter V1 currently supports one point cloud per "
                    "forward call because Qwen3VLBackbone V1 point injection "
                    "is batch-size 1."
                )

        return features, coord, batch

    # ------------------------------------------------------------------
    # V1 resampling
    # ------------------------------------------------------------------

    def _uniform_indices(
        self,
        num_points: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Deterministically select num_tokens approximately evenly spaced
        indices over the current dense PTv3 ordering.

        This is intentionally the same simple strategy used in the successful
        integration smoke test. It is a compatibility baseline, not the final
        geometric resampler.
        """

        return torch.linspace(
            0,
            num_points - 1,
            self.num_tokens,
            device=device,
        ).long()

    def resample(
        self,
        features: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        if self.config.sampling != "uniform":
            raise RuntimeError(
                f"Unsupported sampling mode: {self.config.sampling}"
            )

        indices = self._uniform_indices(
            num_points=features.shape[0],
            device=features.device,
        )

        sampled_features = features[indices]

        sampled_coord = (
            coord[indices]
            if coord is not None
            else None
        )

        return sampled_features, sampled_coord, indices

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        sampled_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project PTv3 features into Qwen's hidden dimension.

        PTv3 currently outputs FP32 and this module stays FP32 by default.
        Qwen3VLBackbone performs the final cast to the Qwen embedding dtype
        (BF16 in the validated setup) at injection time.
        """

        tokens = self.proj(sampled_features)
        tokens = self.norm(tokens)

        return tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_with_metadata(
        self,
        point_encoded: Any,
    ) -> PointAdapterOutput:
        features, coord, _batch = self._extract_dense_features(
            point_encoded
        )

        sampled_features, sampled_coord, sampled_indices = self.resample(
            features=features,
            coord=coord,
        )

        tokens = self.project(sampled_features)

        if tokens.shape != (
            self.num_tokens,
            self.out_dim,
        ):
            raise RuntimeError(
                "Unexpected PointAdapter output shape: "
                f"{tuple(tokens.shape)}; expected "
                f"({self.num_tokens}, {self.out_dim})."
            )

        if not torch.isfinite(tokens).all():
            raise RuntimeError(
                "PointAdapter produced NaN or Inf values."
            )

        return PointAdapterOutput(
            tokens=tokens,
            sampled_features=sampled_features,
            sampled_coord=sampled_coord,
            sampled_indices=sampled_indices,
        )

    def forward(
        self,
        point_encoded: Any,
    ) -> torch.Tensor:
        """
        Return only Qwen-compatible point tokens [N_token, out_dim].
        """

        return self.forward_with_metadata(point_encoded).tokens

    # ------------------------------------------------------------------
    # Trainability / parameter helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)

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

    # Minimal CPU-only shape check.
    feat = torch.randn(4096, cfg.in_dim)
    coord = torch.randn(4096, 3)

    out = adapter.forward_with_metadata(
        {
            "features": feat,
            "coord": coord,
            "batch": torch.zeros(4096, dtype=torch.long),
        }
    )

    print("point_adapter.py standalone shape test OK")
    print("Dense input:", tuple(feat.shape))
    print("Sampled features:", tuple(out.sampled_features.shape))
    print("Sampled coords:", tuple(out.sampled_coord.shape))
    print("Point tokens:", tuple(out.tokens.shape))
    print("Parameters:", adapter.parameter_count())
