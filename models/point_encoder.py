"""
PTv3 point-cloud backbone for PAIR.

This module wraps the standalone official PointTransformerV3 implementation
that has already passed our forward smoke test.

Validated V1 path
-----------------
raw point cloud
    -> PointTransformerV3
    -> encoder + decoder
    -> dense point features [N, 64]

The resampling/tokenization step does NOT belong here. It will be handled by
point_adapter.py so this module remains a clean dense 3D feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.ptv3.model import PointTransformerV3


@dataclass
class PTv3Config:
    """
    PTv3 configuration used by PAIR.

    Defaults match the standalone configuration that was already validated
    successfully on the current RTX 4090 D / PyTorch 2.12.1 environment.
    """

    in_channels: int = 6

    order: Tuple[str, ...] = (
        "z",
        "z-trans",
        "hilbert",
        "hilbert-trans",
    )

    stride: Tuple[int, ...] = (2, 2, 2, 2)

    enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2)
    enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32)
    enc_patch_size: Tuple[int, ...] = (128, 128, 128, 128, 128)

    dec_depths: Tuple[int, ...] = (2, 2, 2, 2)
    dec_channels: Tuple[int, ...] = (64, 64, 128, 256)
    dec_num_head: Tuple[int, ...] = (4, 4, 8, 16)
    dec_patch_size: Tuple[int, ...] = (128, 128, 128, 128)

    drop_path: float = 0.0
    shuffle_orders: bool = False

    enable_rpe: bool = False
    enable_flash: bool = False

    upcast_attention: bool = True
    upcast_softmax: bool = True

    cls_mode: bool = False

    pdnorm_bn: bool = False
    pdnorm_ln: bool = False

    # Extra arguments can be added later without changing PAIR.py.
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PTv3Output:
    """
    Standardized output passed from PTv3 to point_adapter.py.

    features:
        Dense per-point PTv3 features, normally [N, 64].

    coord:
        Point coordinates, [N, 3].

    batch:
        Batch index of every point, [N].

    offset:
        Optional Pointcept-style cumulative point counts.

    raw:
        Original Point object returned by the official PTv3 implementation.
        Kept for debugging / future access to serialization information.
    """

    features: torch.Tensor
    coord: torch.Tensor
    batch: torch.Tensor
    offset: Optional[torch.Tensor] = None
    raw: Optional[Any] = None


class PTv3Backbone(nn.Module):
    """
    Dense PTv3 backbone used by PAIR.

    V1 deliberately keeps the exact full encoder-decoder PTv3 topology that
    passed smoke testing. The output therefore returns to original point
    resolution instead of exposing only the coarsest encoder representation.
    """

    def __init__(
        self,
        config: Optional[PTv3Config] = None,
    ):
        super().__init__()

        self.config = config or PTv3Config()

        cfg = self.config

        kwargs = dict(
            in_channels=cfg.in_channels,
            order=cfg.order,
            stride=cfg.stride,

            enc_depths=cfg.enc_depths,
            enc_channels=cfg.enc_channels,
            enc_num_head=cfg.enc_num_head,
            enc_patch_size=cfg.enc_patch_size,

            dec_depths=cfg.dec_depths,
            dec_channels=cfg.dec_channels,
            dec_num_head=cfg.dec_num_head,
            dec_patch_size=cfg.dec_patch_size,

            drop_path=cfg.drop_path,
            shuffle_orders=cfg.shuffle_orders,

            enable_rpe=cfg.enable_rpe,
            enable_flash=cfg.enable_flash,

            upcast_attention=cfg.upcast_attention,
            upcast_softmax=cfg.upcast_softmax,

            cls_mode=cfg.cls_mode,

            pdnorm_bn=cfg.pdnorm_bn,
            pdnorm_ln=cfg.pdnorm_ln,
        )

        kwargs.update(cfg.extra_kwargs)

        self.model = PointTransformerV3(**kwargs)

        # With cls_mode=False, the full decoder returns dec_channels[0]
        # channels at original point resolution. This is 64 in our V1 config.
        self.output_dim = (
            cfg.enc_channels[-1]
            if cfg.cls_mode
            else cfg.dec_channels[0]
        )

        self.in_channels = cfg.in_channels

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_point_dict(
        self,
        point_dict: Dict[str, torch.Tensor],
    ) -> None:
        if not isinstance(point_dict, dict):
            raise TypeError(
                "point_dict must be a dictionary compatible with PTv3."
            )

        if "feat" not in point_dict:
            raise KeyError("point_dict must contain 'feat'.")

        if "coord" not in point_dict:
            raise KeyError("point_dict must contain 'coord'.")

        feat = point_dict["feat"]
        coord = point_dict["coord"]

        if feat.ndim != 2:
            raise ValueError(
                f"'feat' must have shape [N, C], got {tuple(feat.shape)}."
            )

        if feat.shape[1] != self.in_channels:
            raise ValueError(
                f"PTv3 expects {self.in_channels} input channels, "
                f"got {feat.shape[1]}."
            )

        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(
                f"'coord' must have shape [N, 3], got {tuple(coord.shape)}."
            )

        if feat.shape[0] != coord.shape[0]:
            raise ValueError(
                "'feat' and 'coord' must contain the same number of points."
            )

        if "grid_coord" not in point_dict and "grid_size" not in point_dict:
            raise KeyError(
                "point_dict must contain either 'grid_coord' or 'grid_size'."
            )

        if "batch" not in point_dict and "offset" not in point_dict:
            raise KeyError(
                "point_dict must contain either 'batch' or 'offset'."
            )

        if "batch" in point_dict:
            batch = point_dict["batch"]

            if batch.ndim != 1:
                raise ValueError(
                    f"'batch' must have shape [N], got {tuple(batch.shape)}."
                )

            if batch.shape[0] != feat.shape[0]:
                raise ValueError(
                    "'batch' must have one entry for every point."
                )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        point_dict: Dict[str, torch.Tensor],
    ) -> PTv3Output:
        """
        Parameters
        ----------
        point_dict:
            PTv3 / Pointcept-style point dictionary.

            Minimum supported forms:

            {
                "coord":      [N, 3],
                "grid_coord": [N, 3],
                "feat":       [N, C],
                "batch":      [N],
            }

            or

            {
                "coord":     [N, 3],
                "grid_size": float,
                "feat":      [N, C],
                "batch":     [N],
            }

        Returns
        -------
        PTv3Output
            Dense per-point representation for point_adapter.py.
        """

        self._validate_point_dict(point_dict)

        point = self.model(point_dict)

        features = point.feat
        coord = point.coord
        batch = point.batch

        offset = getattr(point, "offset", None)

        if features.ndim != 2:
            raise RuntimeError(
                "Unexpected PTv3 output feature shape: "
                f"{tuple(features.shape)}"
            )

        if not self.config.cls_mode:
            if features.shape[0] != coord.shape[0]:
                raise RuntimeError(
                    "PTv3 decoder did not return one feature per output point."
                )

            if features.shape[1] != self.output_dim:
                raise RuntimeError(
                    f"Expected PTv3 output dim {self.output_dim}, "
                    f"got {features.shape[1]}."
                )

        return PTv3Output(
            features=features,
            coord=coord,
            batch=batch,
            offset=offset,
            raw=point,
        )

    # ------------------------------------------------------------------
    # Trainability helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all PTv3 parameters."""
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze all PTv3 parameters."""
        self.requires_grad_(True)

    def trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def parameter_count(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
        )


# Compatibility alias for places where PAIR refers to a generic point encoder.
PTv3Encoder = PTv3Backbone


if __name__ == "__main__":
    cfg = PTv3Config()

    print("ptv3.py import scaffold OK")
    print("Input channels:", cfg.in_channels)
    print("Expected V1 dense output channels:", cfg.dec_channels[0])
    print("FlashAttention enabled:", cfg.enable_flash)
