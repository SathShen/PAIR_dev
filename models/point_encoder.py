"""
Frozen Utonia point-cloud encoder for PAIR.

PAIR 3D path
------------
cropped point cloud [N, 3]
    -> voxel sampling
    -> Utonia pretrained encoder (frozen)
    -> concatenate all five encoder-scale features
    -> map sampled features back to the original cropped point topology
    -> dense point features [N, D]

The same Utonia instance is shared by T1 and T2 through PAIRModel.

Utonia input
------------
The official Utonia model is pretrained with:

    feat = [coord, color, normal]  # 9 channels

For point clouds without color or normals, Utonia explicitly supports zeros
for those missing modalities. NYC-SCD therefore remains an XYZ-only dataset;
we do NOT invent RGB or normals in the prepared files.

Spatial scale
-------------
Utonia's official transform first scales coordinates and then applies a
0.01-unit GridSample. PAIR exposes the physically meaningful `voxel_size`
instead of exposing both of those implementation details.

    scale = 0.01 / voxel_size

Thus voxel_size=0.5 m corresponds to scale=0.02 for meter-based NYC-SCD.

The initial 0.5 m choice matches the NYC-SCD / ME-CPT point resolution used
in our current 3D setup. It is a model hyperparameter and can be ablated later.

Important topology guarantee
----------------------------
Utonia sees one representative per occupied voxel, but PAIR needs dense
per-point features for change decoding. This wrapper preserves the inverse
voxel map and returns one feature for every original cropped input point:

    output.features.shape[0] == input.coord.shape[0]

No labels are consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import importlib.util

import torch
import torch.nn as nn


UTONIA_INTERNAL_GRID_SIZE = 0.01


@dataclass(frozen=True)
class UtoniaPointEncoderConfig:
    checkpoint: str = "/data2/sht/checkpoints/Utonia/utonia.pth"
    voxel_size: float = 0.5


@dataclass
class PointEncoderOutput:
    """
    Dense point representation consumed by PointAdapter / Unified Decoder.

    features:
        Utonia multi-scale feature for every original cropped input point,
        [N, output_dim].

    coord:
        Original PAIR point coordinates, [N, 3]. These are NOT the scaled
        coordinates used internally by Utonia.

    batch:
        Batch ID for every original point, [N].

    offset:
        Pointcept-style cumulative point counts, [B].
    """

    features: torch.Tensor
    coord: torch.Tensor
    batch: torch.Tensor
    offset: torch.Tensor

    # Optional LiDAR radiometry. Utonia itself does not consume intensity;
    # it is carried unchanged to PointAdapter for PAIR-side fusion.
    intensity: Optional[torch.Tensor] = None
    intensity_mask: Optional[torch.Tensor] = None


def _load_checkpoint(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def _batch_from_offset(
    offset: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    if offset.ndim != 1:
        raise ValueError(
            f"'offset' must be [B], got {tuple(offset.shape)}"
        )

    if offset.numel() == 0:
        if num_points != 0:
            raise ValueError(
                "Empty offset cannot describe a non-empty point cloud"
            )
        return torch.empty(
            0,
            dtype=torch.long,
            device=offset.device,
        )

    offset = offset.long()

    if int(offset[-1].item()) != num_points:
        raise ValueError(
            f"offset[-1]={int(offset[-1].item())} "
            f"does not match N={num_points}"
        )

    start = torch.cat(
        [
            offset.new_zeros(1),
            offset[:-1],
        ]
    )
    counts = offset - start

    if (counts <= 0).any():
        raise ValueError(
            "Every point cloud in a batch must contain at least one point"
        )

    return torch.repeat_interleave(
        torch.arange(
            offset.numel(),
            device=offset.device,
            dtype=torch.long,
        ),
        counts,
    )


def _offset_from_batch(
    batch: torch.Tensor,
) -> torch.Tensor:
    if batch.ndim != 1:
        raise ValueError(
            f"'batch' must be [N], got {tuple(batch.shape)}"
        )

    if batch.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=batch.device,
        )

    batch = batch.long()

    if int(batch.min().item()) != 0:
        raise ValueError(
            "PAIR point batch IDs must start at 0"
        )

    if (batch[1:] < batch[:-1]).any():
        raise ValueError(
            "PAIR point batches must be contiguous and grouped by batch ID"
        )

    num_batches = int(
        batch[-1].item()
    ) + 1

    counts = torch.bincount(
        batch,
        minlength=num_batches,
    )

    if (counts == 0).any():
        raise ValueError(
            "PAIR point batch IDs must be contiguous with no missing IDs"
        )

    return torch.cumsum(
        counts,
        dim=0,
    )


def _normalize_grid_per_batch(
    grid_coord: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Match the important behavior of Utonia GridSample:
    each cloud's integer grid starts from zero.
    """
    out = torch.empty_like(
        grid_coord
    )

    num_batches = (
        int(batch[-1].item()) + 1
        if batch.numel()
        else 0
    )

    for batch_id in range(
        num_batches
    ):
        mask = batch == batch_id

        if not mask.any():
            raise RuntimeError(
                f"Empty batch ID {batch_id}"
            )

        values = grid_coord[
            mask
        ]

        out[
            mask
        ] = (
            values
            - values.amin(
                dim=0,
                keepdim=True,
            )
        )

    return out


def _select_voxel_representatives(
    inverse: torch.Tensor,
    num_voxels: int,
    *,
    randomize: bool,
) -> torch.Tensor:
    """
    Pick exactly one original point for every occupied voxel.

    Training:
        random representative, matching the spirit of Utonia's official
        GridSample(mode="train").

    Evaluation:
        deterministic first representative.
    """
    num_points = int(
        inverse.numel()
    )

    if num_points == 0:
        raise ValueError(
            "Cannot voxel-sample an empty point cloud"
        )

    device = inverse.device

    if randomize:
        permutation = torch.randperm(
            num_points,
            device=device,
        )

        rank = torch.empty(
            num_points,
            dtype=torch.long,
            device=device,
        )

        rank[
            permutation
        ] = torch.arange(
            num_points,
            dtype=torch.long,
            device=device,
        )

        best_rank = torch.full(
            (num_voxels,),
            num_points,
            dtype=torch.long,
            device=device,
        )

        best_rank.scatter_reduce_(
            0,
            inverse,
            rank,
            reduce="amin",
            include_self=True,
        )

        representative = permutation[
            best_rank
        ]

    else:
        point_index = torch.arange(
            num_points,
            dtype=torch.long,
            device=device,
        )

        representative = torch.full(
            (num_voxels,),
            num_points,
            dtype=torch.long,
            device=device,
        )

        representative.scatter_reduce_(
            0,
            inverse,
            point_index,
            reduce="amin",
            include_self=True,
        )

    if (
        representative.numel()
        != num_voxels
    ):
        raise RuntimeError(
            "Voxel representative count mismatch"
        )

    if (
        representative < 0
    ).any() or (
        representative >= num_points
    ).any():
        raise RuntimeError(
            "Invalid voxel representative index"
        )

    return representative


class UtoniaPointEncoder(nn.Module):
    """
    Frozen Utonia foundation encoder used by PAIR.

    The Utonia parameters are always frozen. Calling parent_model.train()
    keeps this wrapper's preprocessing in train mode (random voxel
    representative selection), while the pretrained Utonia network itself
    remains in eval mode.
    """

    def __init__(
        self,
        config: Optional[
            UtoniaPointEncoderConfig
        ] = None,
    ):
        super().__init__()

        self.config = (
            config
            or UtoniaPointEncoderConfig()
        )

        if (
            self.config.voxel_size
            <= 0
        ):
            raise ValueError(
                "Utonia voxel_size must be > 0"
            )

        checkpoint_path = (
            Path(
                self.config.checkpoint
            )
            .expanduser()
            .resolve()
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Utonia checkpoint not found:\n"
                f"  {checkpoint_path}"
            )

        try:
            from models.utonia.model import (
                PointTransformerV3,
            )
        except ImportError as exc:
            raise ImportError(
                "PAIR now uses the official Utonia package for its 3D "
                "foundation encoder.\n"
                "Install/copy the official Pointcept/Utonia package into "
                "the current 'pair' environment before running 3D:\n"
                "  https://github.com/Pointcept/Utonia"
            ) from exc

        checkpoint = _load_checkpoint(
            checkpoint_path
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise RuntimeError(
                "Unexpected Utonia checkpoint format: "
                "expected a dictionary"
            )

        if (
            "config" not in checkpoint
            or "state_dict" not in checkpoint
        ):
            raise RuntimeError(
                "Utonia checkpoint must contain "
                "'config' and 'state_dict'"
            )

        model_config = dict(
            checkpoint[
                "config"
            ]
        )

        # Utonia is used as an encoder-only foundation model. The official
        # inference checkpoint should already declare this; fail loudly if
        # an incompatible checkpoint is supplied.
        if not bool(
            model_config.get(
                "enc_mode",
                False,
            )
        ):
            raise RuntimeError(
                "PAIR requires the encoder-only Utonia checkpoint "
                "(config['enc_mode'] must be True)"
            )

        in_channels = int(
            model_config.get(
                "in_channels",
                -1,
            )
        )

        if in_channels != 9:
            raise RuntimeError(
                "PAIR expects the official Utonia "
                "[coord,color,normal] input with 9 channels; "
                f"checkpoint declares in_channels={in_channels}"
            )

        enc_channels = tuple(
            int(x)
            for x in model_config.get(
                "enc_channels",
                (),
            )
        )

        if len(
            enc_channels
        ) < 2:
            raise RuntimeError(
                "Utonia checkpoint has invalid enc_channels"
            )

        # Utonia makes FlashAttention optional in its own demos. Disabling
        # it changes execution only, not parameter shapes, so the same
        # pretrained state_dict remains valid.
        flash_available = (
            importlib.util.find_spec(
                "flash_attn"
            )
            is not None
        )

        if (
            model_config.get(
                "enable_flash",
                False,
            )
            and not flash_available
        ):
            model_config[
                "enable_flash"
            ] = False

            print(
                "Utonia: flash_attn is not installed; "
                "using the official non-FlashAttention path."
            )

        self.model = (
            PointTransformerV3(
                **model_config
            )
        )

        self.model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True,
        )

        # Foundation encoder is frozen by design.
        self.model.requires_grad_(
            False
        )
        self.model.eval()

        self.checkpoint_path = str(
            checkpoint_path
        )
        self.model_config = (
            model_config
        )
        self.stage_channels = (
            enc_channels
        )

        # Official quantitative use concatenates features from all encoder
        # scales while mapping back through all four pooling levels.
        self.output_dim = sum(
            enc_channels
        )

        self.voxel_size = float(
            self.config.voxel_size
        )

        self.scale = (
            UTONIA_INTERNAL_GRID_SIZE
            / self.voxel_size
        )

        total_params = self.parameter_count()

        print(
            "Utonia point encoder loaded: "
            f"{total_params / 1e6:.2f}M parameters | "
            "frozen | "
            f"voxel_size={self.voxel_size:g} m | "
            f"dense_dim={self.output_dim}"
        )

    def train(
        self,
        mode: bool = True,
    ):
        """
        Keep the frozen Utonia network in eval mode.

        self.training still follows `mode`, so preprocessing may use random
        voxel representatives during training and deterministic selection
        during evaluation.
        """
        super().train(
            mode
        )

        self.model.eval()

        return self

    def parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )

    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_xyz(
        value: torch.Tensor,
        *,
        name: str,
        num_points: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(
            value
        ):
            raise TypeError(
                f"'{name}' must be a torch.Tensor"
            )

        if (
            value.ndim != 2
            or value.shape
            != (
                num_points,
                3,
            )
        ):
            raise ValueError(
                f"'{name}' must be "
                f"[N,3]={num_points, 3}, "
                f"got {tuple(value.shape)}"
            )

        if not torch.isfinite(
            value
        ).all():
            raise ValueError(
                f"'{name}' contains NaN/Inf"
            )

        return value.float()

    def _resolve_batch(
        self,
        point_dict: Dict[
            str,
            torch.Tensor,
        ],
        num_points: int,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = point_dict.get(
            "batch"
        )
        offset = point_dict.get(
            "offset"
        )

        if (
            batch is None
            and offset is None
        ):
            batch = torch.zeros(
                num_points,
                dtype=torch.long,
                device=device,
            )

            offset = torch.tensor(
                [num_points],
                dtype=torch.long,
                device=device,
            )

        elif batch is None:
            if not torch.is_tensor(
                offset
            ):
                offset = torch.as_tensor(
                    offset,
                    dtype=torch.long,
                    device=device,
                )
            else:
                offset = offset.to(
                    device=device,
                    dtype=torch.long,
                )

            batch = _batch_from_offset(
                offset,
                num_points,
            )

        else:
            if not torch.is_tensor(
                batch
            ):
                batch = torch.as_tensor(
                    batch,
                    dtype=torch.long,
                    device=device,
                )
            else:
                batch = batch.to(
                    device=device,
                    dtype=torch.long,
                )

            if batch.shape != (
                num_points,
            ):
                raise ValueError(
                    f"'batch' must be [N]={num_points}, "
                    f"got {tuple(batch.shape)}"
                )

            derived_offset = (
                _offset_from_batch(
                    batch
                )
            )

            if offset is not None:
                if not torch.is_tensor(
                    offset
                ):
                    offset = torch.as_tensor(
                        offset,
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    offset = offset.to(
                        device=device,
                        dtype=torch.long,
                    )

                if not torch.equal(
                    offset,
                    derived_offset,
                ):
                    raise ValueError(
                        "'batch' and 'offset' describe "
                        "different point layouts"
                    )

            offset = (
                derived_offset
            )

        return (
            batch,
            offset,
        )

    def _optional_color(
        self,
        point_dict,
        coord,
    ):
        color = point_dict.get(
            "color"
        )

        if color is None:
            return torch.zeros_like(
                coord
            )

        color = self._validate_xyz(
            color,
            name="color",
            num_points=coord.shape[0],
        ).to(
            device=coord.device
        )

        # Prepared PAIR point files may eventually use either raw RGB
        # [0,255] or normalized RGB [0,1]. NYC-SCD has no color.
        if (
            color.numel()
            and float(
                color.max().item()
            ) > 1.5
        ):
            color = color / 255.0

        return color

    def _optional_normal(
        self,
        point_dict,
        coord,
    ):
        normal = point_dict.get(
            "normal"
        )

        if normal is None:
            return torch.zeros_like(
                coord
            )

        return self._validate_xyz(
            normal,
            name="normal",
            num_points=coord.shape[0],
        ).to(
            device=coord.device
        )

    def _prepare_utonia_input(
        self,
        point_dict: Dict[
            str,
            torch.Tensor,
        ],
    ):
        if not isinstance(
            point_dict,
            dict,
        ):
            raise TypeError(
                "point_dict must be a dictionary"
            )

        if "coord" not in point_dict:
            raise KeyError(
                "point_dict must contain 'coord'"
            )

        coord = point_dict[
            "coord"
        ]

        if not torch.is_tensor(
            coord
        ):
            raise TypeError(
                "'coord' must be a torch.Tensor"
            )

        if (
            coord.ndim != 2
            or coord.shape[1] != 3
        ):
            raise ValueError(
                f"'coord' must be [N,3], "
                f"got {tuple(coord.shape)}"
            )

        if coord.shape[0] == 0:
            raise ValueError(
                "Utonia cannot encode an empty point cloud"
            )

        if not torch.isfinite(
            coord
        ).all():
            raise ValueError(
                "'coord' contains NaN/Inf"
            )

        coord = coord.float()

        num_points = int(
            coord.shape[0]
        )

        batch, offset = (
            self._resolve_batch(
                point_dict,
                num_points,
                coord.device,
            )
        )

        color = self._optional_color(
            point_dict,
            coord,
        )

        normal = self._optional_normal(
            point_dict,
            coord,
        )

        # Utonia's official transform:
        #     coord <- coord * scale
        #     GridSample(grid_size=0.01)
        #
        # Equivalent physical voxelization:
        #     floor(coord / voxel_size)
        scaled_coord = (
            coord
            * self.scale
        )

        grid_coord = torch.floor(
            scaled_coord
            / UTONIA_INTERNAL_GRID_SIZE
        ).long()

        grid_coord = (
            _normalize_grid_per_batch(
                grid_coord,
                batch,
            )
        )

        voxel_key = torch.cat(
            [
                batch.view(
                    -1,
                    1,
                ),
                grid_coord,
            ],
            dim=1,
        )

        (
            _,
            voxel_inverse,
        ) = torch.unique(
            voxel_key,
            sorted=True,
            return_inverse=True,
            dim=0,
        )

        num_voxels = int(
            voxel_inverse.max().item()
        ) + 1

        representative = (
            _select_voxel_representatives(
                voxel_inverse,
                num_voxels,
                randomize=self.training,
            )
        )

        sampled_coord = (
            scaled_coord[
                representative
            ]
        )

        sampled_grid = (
            grid_coord[
                representative
            ]
        )

        sampled_batch = (
            batch[
                representative
            ]
        )

        sampled_color = (
            color[
                representative
            ]
        )

        sampled_normal = (
            normal[
                representative
            ]
        )

        sampled_offset = (
            _offset_from_batch(
                sampled_batch
            )
        )

        feat = torch.cat(
            [
                sampled_coord,
                sampled_color,
                sampled_normal,
            ],
            dim=1,
        )

        if (
            feat.shape[1]
            != 9
        ):
            raise RuntimeError(
                "Utonia input feature construction "
                f"failed: {tuple(feat.shape)}"
            )

        utonia_input = {
            "coord": sampled_coord,
            "grid_coord": sampled_grid,
            "feat": feat,
            "batch": sampled_batch,
            "offset": sampled_offset,
        }

        return {
            "utonia_input": (
                utonia_input
            ),
            "original_coord": coord,
            "original_batch": batch,
            "original_offset": offset,
            "voxel_inverse": (
                voxel_inverse
            ),
            "num_sampled": (
                num_voxels
            ),
        }

    # ------------------------------------------------------------------
    # Utonia feature recovery
    # ------------------------------------------------------------------

    def _recover_multiscale(
        self,
        point,
    ):
        """
        Concatenate all five Utonia encoder-scale features while mapping
        coarse features back through every pooling level.

        For the official checkpoint:
            54 + 108 + 216 + 432 + 576 = 1386 channels.
        """
        expected_upcasts = (
            len(
                self.stage_channels
            )
            - 1
        )

        for _ in range(
            expected_upcasts
        ):
            if (
                "pooling_parent"
                not in point.keys()
                or "pooling_inverse"
                not in point.keys()
            ):
                raise RuntimeError(
                    "Utonia did not expose the traceable "
                    "pooling hierarchy required for dense "
                    "multi-scale feature recovery"
                )

            parent = point.pop(
                "pooling_parent"
            )

            inverse = point.pop(
                "pooling_inverse"
            )

            parent.feat = torch.cat(
                [
                    parent.feat,
                    point.feat[
                        inverse
                    ],
                ],
                dim=-1,
            )

            point = parent

        if (
            "pooling_parent"
            in point.keys()
            or "pooling_inverse"
            in point.keys()
        ):
            raise RuntimeError(
                "Unexpected extra Utonia pooling level"
            )

        if (
            point.feat.ndim != 2
            or point.feat.shape[1]
            != self.output_dim
        ):
            raise RuntimeError(
                "Unexpected recovered Utonia feature shape: "
                f"{tuple(point.feat.shape)}; "
                f"expected second dim {self.output_dim}"
            )

        return point

    # ------------------------------------------------------------------
    # Optional LiDAR intensity
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_optional_intensity(
        point_dict: Dict[str, torch.Tensor],
        *,
        num_points: int,
        device: torch.device,
    ):
        """
        Carry optional per-point LiDAR intensity through the frozen Utonia
        backbone without feeding it into Utonia.

        intensity:
            Optional [N] or [N,1] tensor.

        intensity_mask:
            Optional [N] or [N,1] bool / 0-1 tensor. If intensity exists and
            no mask is supplied, all points are considered valid.

        Intensity normalization deliberately belongs to dataset preparation,
        not to Utonia. Different LiDAR sensors should use one consistent
        train-split normalization policy rather than per-tile min-max scaling.
        """
        intensity = point_dict.get("intensity")

        if intensity is None:
            if point_dict.get("intensity_mask") is not None:
                raise ValueError(
                    "intensity_mask was provided but intensity is absent"
                )
            return None, None

        if not torch.is_tensor(intensity):
            intensity = torch.as_tensor(
                intensity, dtype=torch.float32, device=device
            )
        else:
            intensity = intensity.to(device=device, dtype=torch.float32)

        if intensity.ndim == 1:
            intensity = intensity.unsqueeze(1)

        if intensity.shape != (num_points, 1):
            raise ValueError(
                f"intensity must be [N] or [N,1], got "
                f"{tuple(intensity.shape)} for N={num_points}"
            )

        if not torch.isfinite(intensity).all():
            raise ValueError("intensity contains NaN/Inf")

        mask = point_dict.get("intensity_mask")
        if mask is None:
            mask = torch.ones(
                (num_points, 1), dtype=torch.bool, device=device
            )
        else:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, device=device)
            else:
                mask = mask.to(device=device)

            if mask.ndim == 1:
                mask = mask.unsqueeze(1)

            if mask.shape != (num_points, 1):
                raise ValueError(
                    f"intensity_mask must be [N] or [N,1], got "
                    f"{tuple(mask.shape)} for N={num_points}"
                )

            if mask.dtype != torch.bool:
                if not torch.all((mask == 0) | (mask == 1)):
                    raise ValueError(
                        "intensity_mask must contain only bool or 0/1"
                    )
                mask = mask.bool()

        return intensity, mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        point_dict: Dict[
            str,
            torch.Tensor,
        ],
    ) -> PointEncoderOutput:
        prepared = (
            self._prepare_utonia_input(
                point_dict
            )
        )

        utonia_input = prepared[
            "utonia_input"
        ]

        # Use no_grad rather than inference_mode: downstream trainable PAIR
        # adapters must be allowed to consume and save these frozen features
        # during their own backward pass.
        with torch.no_grad():
            point = self.model(
                utonia_input
            )

            point = (
                self._recover_multiscale(
                    point
                )
            )

            sampled_features = (
                point.feat
            )

            if (
                sampled_features.shape[0]
                != prepared[
                    "num_sampled"
                ]
            ):
                raise RuntimeError(
                    "Utonia feature count no longer matches "
                    "the voxel-sampled topology"
                )

            dense_features = (
                sampled_features[
                    prepared[
                        "voxel_inverse"
                    ]
                ]
            )

        original_coord = prepared[
            "original_coord"
        ]

        original_batch = prepared[
            "original_batch"
        ]

        original_offset = prepared[
            "original_offset"
        ]

        if (
            dense_features.shape[0]
            != original_coord.shape[0]
        ):
            raise RuntimeError(
                "Utonia inverse mapping failed: "
                f"dense N={dense_features.shape[0]} "
                f"but input N={original_coord.shape[0]}"
            )

        if (
            dense_features.shape[1]
            != self.output_dim
        ):
            raise RuntimeError(
                f"Expected dense Utonia dim "
                f"{self.output_dim}, "
                f"got {dense_features.shape[1]}"
            )

        intensity, intensity_mask = self._extract_optional_intensity(
            point_dict,
            num_points=original_coord.shape[0],
            device=original_coord.device,
        )

        return PointEncoderOutput(
            features=dense_features,
            coord=original_coord,
            batch=original_batch,
            offset=original_offset,
            intensity=intensity,
            intensity_mask=intensity_mask,
        )
