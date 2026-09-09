"""
PAIR standard dataset interface.

Directory is schema
-------------------
Every dataset must be prepared into one of the canonical layouts below.

2D semantic pair:
    root/
    ├── images_t1/
    ├── images_t2/
    ├── semantic_t1/
    ├── semantic_t2/
    └── manifests/
        ├── train.jsonl
        ├── val.jsonl      (optional)
        └── test.jsonl     (optional)

3D semantic pair:
    root/
    ├── points_t1/
    ├── points_t2/
    ├── semantic_t1/
    ├── semantic_t2/
    └── manifests/

    points_t*/sample.npz:
        coord    [N, 3]
        feat     [N, C]

    semantic_t*/sample.npz:
        semantic [N]
        change   [N]

    The point and supervision files of one epoch share exactly the same
    per-point topology and ordering. T1 and T2 are independent point sets:
    N1 does NOT need to equal N2.

2D+3D:
    root/
    ├── images_t1/
    ├── images_t2/
    ├── points_t1/
    ├── points_t2/
    ├── semantic_t1/
    ├── semantic_t2/
    └── manifests/

Supervision is inferred from directories:
    semantic_t1 + semantic_t2  -> semantic_pair
    change + semantic_t2       -> post_semantic
    change                     -> binary

Label policy inside PAIR:
    change      : 0 unchanged, 1 changed
    semantic    : raw dataset class IDs declared in DatasetSpec.class_names
    ignore      : optional explicit DatasetSpec.ignore_raw_id; default None
    UNKNOWN=-1  : internal only, meaning semantic class is not supervised

There is NO implicit magic ignore value in prepared data. In particular, -100
is not silently accepted. Every observed raw semantic ID must either be declared
in class_names or equal the explicitly configured ignore_raw_id. Any other raw
ID raises immediately. When ignore_raw_id is None, all prepared labels are
expected to be fully supervised and no per-element valid mask is generated.

For 2D semantic_pair, T1/T2 share one raster topology, so binary change can be
derived by elementwise semantic comparison when no explicit change mask exists.

For 3D semantic_pair, T1/T2 do NOT share point indices. Each semantic_*.npz
therefore carries its own per-point binary change field and no elementwise
cross-temporal comparison is ever performed.

PAIR's current canonical physical point window is 51.2 m × 51.2 m, matching
a 512 × 512 image at 0.1 m GSD. For pure 3D training, a source pair larger
than this window is cropped on-the-fly using one shared XY window for T1/T2.
Smaller source pairs are used in full. Validation/test data are never randomly
cropped; oversized evaluation sources require deterministic tiling, which is
intentionally left to the evaluation adapter rather than silently discarding
part of the scene.

The training JSON does not repeat modality, manifest path, label mode, image
size, change IDs, spatial window size, alignment policy, or prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


VALID_MODALITIES = {"image", "point"}

# PAIR canonical 2D/3D physical footprint:
# 512 pixels × 0.1 m = 51.2 m.
PAIR_POINT_WINDOW_SIZE_M = 51.2


# =============================================================================
# Dataset / label protocol
# =============================================================================

def normalize_class_name(name: str) -> str:
    return " ".join(
        str(name).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def infer_unchanged_raw_id(class_names: Dict[int, str]) -> Optional[int]:
    """
    Infer an explicit semantic 'unchanged' class by NAME only.
    Intentionally does NOT treat 'background' as unchanged.
    """
    aliases = {"unchanged", "no change", "non change"}
    matches = [
        int(raw_id)
        for raw_id, name in class_names.items()
        if normalize_class_name(name) in aliases
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple unchanged-like classes found: {matches}. "
            "Use one explicit unchanged/no-change class name."
        )
    return matches[0] if matches else None


def infer_binary_class_ids(class_names: Dict[int, str]) -> Tuple[int, int]:
    """
    Infer physical unchanged/changed raw IDs from class_names.

    No changed_raw_id field is stored in DatasetSpec.

    Examples:
        {0: "unchanged", 255: "changed"} -> (0, 255)
        {0: "unchanged", 1: "changed"}   -> (0, 1)
    """
    unchanged_aliases = {"unchanged", "no change", "non change"}
    changed_aliases = {"changed", "change"}

    unchanged = [
        int(raw_id)
        for raw_id, name in class_names.items()
        if normalize_class_name(name) in unchanged_aliases
    ]
    changed = [
        int(raw_id)
        for raw_id, name in class_names.items()
        if normalize_class_name(name) in changed_aliases
    ]

    if len(unchanged) != 1:
        raise ValueError(
            "Binary class_names must contain exactly one unchanged/no-change "
            f"class; found raw IDs {unchanged}."
        )
    if len(changed) != 1:
        raise ValueError(
            "Binary class_names must contain exactly one changed/change "
            f"class; found raw IDs {changed}."
        )
    if unchanged[0] == changed[0]:
        raise ValueError("Binary unchanged and changed raw IDs must differ.")

    return unchanged[0], changed[0]


def route_from_modalities(modalities: Sequence[str]) -> str:
    m = set(modalities)
    if m == {"image"}:
        return "2d"
    if m == {"point"}:
        return "3d"
    if m == {"image", "point"}:
        return "2d3d"
    raise ValueError(f"Unsupported modality set: {sorted(m)}")


@dataclass(frozen=True)
class DatasetSpec:
    """
    Internal normalized dataset description.

    Users do not create this in config. config_loader builds it automatically
    from the prepared PAIR directory and the dataset's class_names.
    """
    name: str
    modalities: Tuple[str, ...]
    label_mode: str
    class_names: Dict[int, str]
    point_grid_size: float = 0.10
    unchanged_raw_id: Optional[int] = None
    ignore_raw_id: Optional[int] = None

    @property
    def route(self) -> str:
        return route_from_modalities(self.modalities)

    @property
    def has_image(self) -> bool:
        return "image" in self.modalities

    @property
    def has_point(self) -> bool:
        return "point" in self.modalities


@dataclass
class CanonicalChangeTarget:
    """
    Common-topology target used by the existing 2D path.

    Valid masks are optional. For a fully supervised semantic-pair dataset with
    ignore_raw_id=None, they stay None and are not materialized. Masks are only
    created when they carry real information, e.g. an explicit ignored class or
    intentionally missing semantic supervision in binary/post-semantic modes.
    """
    change: torch.Tensor
    semantic_t1: torch.Tensor
    semantic_t2: torch.Tensor
    change_valid: Optional[torch.Tensor] = None
    semantic_valid_t1: Optional[torch.Tensor] = None
    semantic_valid_t2: Optional[torch.Tensor] = None


def _long(x):
    if torch.is_tensor(x):
        return x.long()
    return torch.as_tensor(np.asarray(x), dtype=torch.long)


def _declared_class_ids(class_names: Dict[int, str]) -> set[int]:
    if not isinstance(class_names, dict) or not class_names:
        raise TypeError("class_names must be a non-empty Dict[int, str]")
    return {int(raw_id) for raw_id in class_names.keys()}


def validate_ignore_config(
    class_names: Dict[int, str],
    ignore_raw_id: Optional[int],
):
    """
    class_names contains trainable semantic classes only. If a source dataset
    has one explicit ignored class, it is declared separately by ignore_raw_id.
    """
    if ignore_raw_id is None:
        return

    ignore_raw_id = int(ignore_raw_id)
    if ignore_raw_id in _declared_class_ids(class_names):
        raise ValueError(
            f"ignore_raw_id={ignore_raw_id} is also present in class_names. "
            "Ignored classes must not occupy a trainable classifier slot."
        )


def validate_semantic_label_values(
    raw: torch.Tensor,
    class_names: Dict[int, str],
    *,
    ignore_raw_id: Optional[int] = None,
    label_name: str = "semantic",
) -> Optional[torch.Tensor]:
    """
    Validate a raw semantic target against the dataset protocol.

    Returns:
        None when ignore_raw_id is None (all elements are valid), otherwise a
        bool mask where True means the semantic element participates in loss.

    Unknown IDs are never converted or silently ignored.
    """
    raw = _long(raw)
    declared = _declared_class_ids(class_names)
    validate_ignore_config(class_names, ignore_raw_id)

    allowed = torch.zeros_like(raw, dtype=torch.bool)
    for raw_id in declared:
        allowed |= raw == raw_id

    if ignore_raw_id is not None:
        ignore_raw_id = int(ignore_raw_id)
        allowed |= raw == ignore_raw_id

    if not allowed.all():
        bad = torch.unique(raw[~allowed]).cpu().tolist()
        allowed_ids = sorted(declared)
        suffix = (
            f" plus ignore_raw_id={int(ignore_raw_id)}"
            if ignore_raw_id is not None
            else ""
        )
        raise ValueError(
            f"{label_name} contains undeclared raw IDs {bad}. "
            f"class_names declares {allowed_ids}{suffix}."
        )

    if ignore_raw_id is None:
        return None

    return raw != int(ignore_raw_id)


def _validate_canonical_change_values(raw: torch.Tensor):
    """
    Explicit PAIR change supervision is strictly binary.

    There is no implicit -100/void value. If a future change-only dataset has a
    physical ignore code, it must be represented explicitly by that dataset's
    protocol rather than being silently accepted here.
    """
    valid_values = (raw == 0) | (raw == 1)
    if not valid_values.all():
        bad = torch.unique(raw[~valid_values]).cpu().tolist()
        raise ValueError(
            "Canonical PAIR change targets must contain only "
            f"0=unchanged and 1=changed; found {bad}"
        )


def _normalize_binary_change_values(
    raw: torch.Tensor,
    class_names: Dict[int, str],
    *,
    ignore_raw_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert physical binary labels to PAIR internal 0/1 using class_names.

    An optional ignore_raw_id is allowed for binary datasets as an explicit
    source value. No other undeclared value is accepted.
    """
    unchanged_raw_id, changed_raw_id = infer_binary_class_ids(class_names)

    if ignore_raw_id is not None:
        ignore_raw_id = int(ignore_raw_id)
        if ignore_raw_id in {unchanged_raw_id, changed_raw_id}:
            raise ValueError(
                f"ignore_raw_id={ignore_raw_id} conflicts with the binary "
                "unchanged/changed class IDs"
            )

    valid_values = (
        (raw == unchanged_raw_id)
        | (raw == changed_raw_id)
    )
    if ignore_raw_id is not None:
        valid_values |= raw == ignore_raw_id

    if not valid_values.all():
        bad = torch.unique(raw[~valid_values]).cpu().tolist()
        raise ValueError(
            f"Binary change target contains undeclared raw IDs {bad}; "
            f"class_names declares unchanged={unchanged_raw_id}, "
            f"changed={changed_raw_id}"
            + (
                f", ignore_raw_id={ignore_raw_id}."
                if ignore_raw_id is not None
                else "."
            )
        )

    out = torch.zeros_like(raw)
    out[raw == changed_raw_id] = 1

    change_valid = None
    if ignore_raw_id is not None:
        change_valid = raw != ignore_raw_id

    return out, change_valid


def build_canonical_target(
    *,
    label_mode: str,
    semantic_t1=None,
    semantic_t2=None,
    change=None,
    class_names: Optional[Dict[int, str]] = None,
    ignore_raw_id: Optional[int] = None,
) -> CanonicalChangeTarget:
    """
    Build a common-topology target for the existing 2D path.

    Strict policy:
      * semantic IDs must be declared in class_names;
      * one explicitly configured ignore_raw_id is additionally allowed;
      * any other semantic ID raises immediately;
      * canonical explicit change supervision must be 0/1 only;
      * when ignore_raw_id is None, normal semantic-pair data do not allocate
        redundant all-True valid masks.

    Ragged 3D point pairs use build_bitemporal_point_target() instead.
    """
    mode = str(label_mode).lower().strip()
    if mode not in {"semantic_pair", "post_semantic", "binary"}:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    if mode == "semantic_pair":
        if semantic_t1 is None or semantic_t2 is None:
            raise ValueError("semantic_pair requires semantic_t1 and semantic_t2")
        if class_names is None:
            raise ValueError("semantic_pair validation requires class_names")

        s1 = _long(semantic_t1)
        s2 = _long(semantic_t2)

        if s1.shape != s2.shape:
            raise ValueError(
                "Common-topology semantic_pair requires matching shapes; "
                f"got semantic_t1={tuple(s1.shape)} and "
                f"semantic_t2={tuple(s2.shape)}. "
                "Ragged 3D point pairs must use per-epoch supervision bundles."
            )

        sem_valid_t1 = validate_semantic_label_values(
            s1,
            class_names,
            ignore_raw_id=ignore_raw_id,
            label_name="semantic_t1",
        )
        sem_valid_t2 = validate_semantic_label_values(
            s2,
            class_names,
            ignore_raw_id=ignore_raw_id,
            label_name="semantic_t2",
        )

        if change is None:
            ch = (s1 != s2).long()
        else:
            ch = _long(change)
            if ch.shape != s1.shape:
                raise ValueError("change mask must match semantic label shape")
            _validate_canonical_change_values(ch)

        change_valid = None
        if ignore_raw_id is not None:
            # In semantic-pair SCD an ignored semantic pixel is excluded from
            # the derived/paired change objective as well.
            change_valid = sem_valid_t1 & sem_valid_t2

        return CanonicalChangeTarget(
            change=ch,
            semantic_t1=s1,
            semantic_t2=s2,
            change_valid=change_valid,
            semantic_valid_t1=sem_valid_t1,
            semantic_valid_t2=sem_valid_t2,
        )

    if change is None:
        raise ValueError(f"{mode} requires change supervision")

    raw = _long(change)

    if mode == "binary":
        if class_names is None:
            raise ValueError("binary target normalization requires class_names")

        ch, change_valid = _normalize_binary_change_values(
            raw,
            class_names,
            ignore_raw_id=ignore_raw_id,
        )

        # Binary CD has no semantic supervision. Keep shape-compatible
        # placeholder tensors for the current common target interface; their
        # values are never used because semantic_valid is all False.
        s1 = torch.zeros_like(raw)
        s2 = torch.zeros_like(raw)

        sem_valid = torch.zeros_like(raw, dtype=torch.bool)

        return CanonicalChangeTarget(
            change=ch,
            semantic_t1=s1,
            semantic_t2=s2,
            change_valid=change_valid,
            semantic_valid_t1=sem_valid,
            semantic_valid_t2=sem_valid.clone(),
        )

    # post_semantic: explicit change is already canonical 0/1; class_names
    # describe the post-change semantic classes.
    _validate_canonical_change_values(raw)
    ch = raw.clone()

    if semantic_t2 is None:
        raise ValueError("post_semantic requires semantic_t2")
    if class_names is None:
        raise ValueError("post_semantic validation requires class_names")

    s2 = _long(semantic_t2)
    if s2.shape != raw.shape:
        raise ValueError("semantic_t2 must match change mask shape")

    sem_valid_t2 = validate_semantic_label_values(
        s2,
        class_names,
        ignore_raw_id=ignore_raw_id,
        label_name="semantic_t2",
    )

    # T1 semantic supervision is absent in post-semantic mode. The tensor is
    # only a shape-compatible placeholder; semantic_valid_t1 is all False.
    s1 = torch.zeros_like(raw)

    return CanonicalChangeTarget(
        change=ch,
        semantic_t1=s1,
        semantic_t2=s2,
        change_valid=None,
        semantic_valid_t1=torch.zeros_like(raw, dtype=torch.bool),
        semantic_valid_t2=sem_valid_t2,
    )

def build_default_prompt(spec: DatasetSpec) -> str:
    text = (
        "Perform semantic change detection between Time 1 and Time 2. "
        "Identify unchanged and changed regions."
    )

    if spec.label_mode == "semantic_pair":
        text += (
            " For changed regions, infer the semantic class before and after "
            "change."
        )
    elif spec.label_mode == "post_semantic":
        text += (
            " The pre-change semantic class may be unknown, while the "
            "post-change class is supervised."
        )
    elif spec.label_mode == "binary":
        text += (
            " The source dataset supervises change only; semantic classes "
            "before and after change may be unknown."
        )

    if spec.class_names:
        classes = ", ".join(
            f"{raw_id}: {name}"
            for raw_id, name in spec.class_names.items()
        )
        text += " Valid semantic classes are: " + classes + "."

    return text


# =============================================================================
# Generic file readers
# =============================================================================

def read_label_array(
    path: Union[str, Path],
) -> torch.Tensor:
    """
    Read one single-topology label array.

    Used by the existing 2D pipeline. A multi-array NPZ is intentionally
    rejected here; 3D semantic/change bundles use read_point_supervision().
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(
            path,
            allow_pickle=False,
        )

    elif suffix == ".npz":
        with np.load(
            path,
            allow_pickle=False,
        ) as z:
            keys = list(z.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} has multiple arrays {keys}; "
                    "use read_point_supervision() for 3D supervision bundles"
                )
            arr = np.asarray(
                z[keys[0]]
            )

    else:
        arr = np.asarray(
            Image.open(path)
        )

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Expected a single-channel label, got {arr.shape}: {path}"
            )

    return torch.as_tensor(
        np.array(
            arr,
            copy=True,
        ),
        dtype=torch.long,
    )


def read_point_supervision(
    path: Union[str, Path],
) -> Dict[str, torch.Tensor]:
    """
    Read PAIR's per-epoch 3D supervision bundle.

    Required NPZ arrays:
        semantic [N]
        change   [N]

    Both arrays belong to the exact point topology of the corresponding
    points_t*.npz file.
    """
    path = Path(path)

    if path.suffix.lower() != ".npz":
        raise ValueError(
            "PAIR 3D semantic_pair supervision must be prepared as NPZ "
            f"with arrays 'semantic' and 'change': {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        keys = set(z.keys())

        required = {
            "semantic",
            "change",
        }
        missing = sorted(
            required - keys
        )
        if missing:
            raise ValueError(
                f"{path} is missing supervision arrays {missing}; "
                f"available={sorted(keys)}"
            )

        semantic = np.asarray(
            z["semantic"]
        )
        change = np.asarray(
            z["change"]
        )

    if semantic.ndim != 1:
        raise ValueError(
            f"{path}: semantic must be [N], got {semantic.shape}"
        )
    if change.ndim != 1:
        raise ValueError(
            f"{path}: change must be [N], got {change.shape}"
        )
    if semantic.shape != change.shape:
        raise ValueError(
            f"{path}: semantic/change shapes differ: "
            f"{semantic.shape} vs {change.shape}"
        )

    semantic = torch.as_tensor(
        np.array(
            semantic,
            copy=True,
        ),
        dtype=torch.long,
    )
    change = torch.as_tensor(
        np.array(
            change,
            copy=True,
        ),
        dtype=torch.long,
    )

    _validate_canonical_change_values(
        change
    )

    return {
        "semantic": semantic,
        "change": change,
    }


def _read_geotiff(path: Path):
    if path.suffix.lower() not in {
        ".tif",
        ".tiff",
    }:
        return None

    try:
        import rasterio
    except ImportError:
        return None

    with rasterio.open(
        path
    ) as src:
        arr = src.read()
        meta = {
            "crs": (
                str(src.crs)
                if src.crs is not None
                else None
            ),
            "transform": tuple(
                src.transform
            ),
            "bounds": tuple(
                src.bounds
            ),
            "width": int(
                src.width
            ),
            "height": int(
                src.height
            ),
            "gsd_x": abs(
                float(src.transform.a)
            ),
            "gsd_y": abs(
                float(src.transform.e)
            ),
        }

    return (
        np.moveaxis(
            arr,
            0,
            -1,
        ),
        meta,
    )


def read_image(
    path: Union[str, Path],
):
    path = Path(path)
    geo = _read_geotiff(
        path
    )

    if geo is not None:
        arr, meta = geo
    else:
        im = Image.open(
            path
        ).convert(
            "RGB"
        )
        arr = np.asarray(
            im
        )
        meta = {
            "crs": None,
            "transform": None,
            "bounds": None,
            "width": im.width,
            "height": im.height,
            "gsd_x": None,
            "gsd_y": None,
        }

    if arr.ndim == 2:
        arr = arr[..., None]

    if np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        denom = float(
            np.iinfo(
                arr.dtype
            ).max
        )
        arr = (
            arr.astype(
                np.float32
            )
            / denom
        )
    else:
        arr = arr.astype(
            np.float32
        )
        vmax = (
            float(
                np.nanmax(arr)
            )
            if arr.size
            else 1.0
        )
        if vmax > 1.5:
            arr /= (
                255.0
                if vmax <= 255.0
                else 65535.0
            )

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            np.moveaxis(
                arr,
                -1,
                0,
            )
        )
    ).float()

    return (
        tensor,
        meta,
    )


def same_geo_grid(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    atol=1e-7,
) -> bool:
    crs1 = meta1.get(
        "crs"
    )
    crs2 = meta2.get(
        "crs"
    )
    tr1 = meta1.get(
        "transform"
    )
    tr2 = meta2.get(
        "transform"
    )

    if (
        crs1 is None
        and crs2 is None
        and tr1 is None
        and tr2 is None
    ):
        return True

    if (
        crs1 != crs2
        or tr1 is None
        or tr2 is None
    ):
        return False

    return (
        np.allclose(
            np.asarray(tr1),
            np.asarray(tr2),
            atol=atol,
            rtol=0.0,
        )
        and meta1["width"] == meta2["width"]
        and meta1["height"] == meta2["height"]
    )


PointFeatureBuilder = Callable[
    [Dict[str, np.ndarray]],
    np.ndarray,
]
TargetBuilder = Callable[
    [Dict[str, Any], DatasetSpec],
    Dict[str, Any],
]


def _read_las(path: Path):
    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "LAS/LAZ reading requires laspy"
        ) from exc

    las = laspy.read(
        path
    )

    attrs: Dict[str, Any] = {
        "coord": np.stack(
            [
                np.asarray(las.x),
                np.asarray(las.y),
                np.asarray(las.z),
            ],
            axis=1,
        ).astype(
            np.float32
        )
    }

    available = set(
        las.point_format.dimension_names
    )

    for name in (
        "intensity",
        "red",
        "green",
        "blue",
        "nir",
        "classification",
        "return_number",
        "number_of_returns",
    ):
        if name in available:
            attrs[name] = np.asarray(
                getattr(
                    las,
                    name,
                )
            )

    try:
        crs = las.header.parse_crs()
        attrs["_crs"] = (
            None
            if crs is None
            else str(crs)
        )
    except Exception:
        attrs["_crs"] = None

    return attrs


def read_point_cloud(
    path: Union[str, Path],
    *,
    feature_builder: Optional[
        PointFeatureBuilder
    ] = None,
) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    crs = None

    if suffix == ".npz":
        with np.load(
            path,
            allow_pickle=False,
        ) as z:
            if (
                "coord" not in z
                or "feat" not in z
            ):
                raise ValueError(
                    f"{path} must contain arrays named "
                    "'coord' [N,3] and 'feat' [N,C]"
                )

            coord = np.asarray(
                z["coord"],
                dtype=np.float32,
            )
            feat = np.asarray(
                z["feat"],
                dtype=np.float32,
            )

            if "crs" in z:
                crs_value = np.asarray(
                    z["crs"]
                )
                crs = str(
                    crs_value.item()
                    if crs_value.ndim == 0
                    else crs_value
                )

    elif suffix == ".npy":
        arr = np.asarray(
            np.load(
                path,
                allow_pickle=False,
            )
        )

        if (
            arr.ndim != 2
            or arr.shape[1] <= 3
        ):
            raise ValueError(
                f"{path} must be [N,3+C]"
            )

        coord = arr[:, :3].astype(
            np.float32
        )
        feat = arr[:, 3:].astype(
            np.float32
        )

    elif suffix in {
        ".las",
        ".laz",
    }:
        attrs = _read_las(
            path
        )

        coord = attrs["coord"]
        crs = attrs.get(
            "_crs"
        )

        if feature_builder is None:
            raise ValueError(
                "LAS/LAZ input requires a feature_builder. "
                "For final PAIR data, prefer prepared NPZ "
                "with coord+feat."
            )

        feat = np.asarray(
            feature_builder(
                attrs
            ),
            dtype=np.float32,
        )

    else:
        raise ValueError(
            f"Unsupported point-cloud format: {path}"
        )

    if (
        coord.ndim != 2
        or coord.shape[1] != 3
    ):
        raise ValueError(
            f"coord must be [N,3], got {coord.shape}"
        )

    if (
        feat.ndim != 2
        or feat.shape[0] != coord.shape[0]
    ):
        raise ValueError(
            "feat must be [N,C] matching coord, "
            f"got coord={coord.shape}, feat={feat.shape}"
        )

    if not np.isfinite(
        coord
    ).all():
        raise ValueError(
            f"{path}: coord contains NaN/Inf"
        )

    if not np.isfinite(
        feat
    ).all():
        raise ValueError(
            f"{path}: feat contains NaN/Inf"
        )

    return {
        "coord": torch.from_numpy(
            np.ascontiguousarray(
                coord
            )
        ).float(),
        "feat": torch.from_numpy(
            np.ascontiguousarray(
                feat
            )
        ).float(),
        "crs": crs,
    }


# =============================================================================
# Point topology / spatial crop
# =============================================================================

def point_xy_bounds(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    """
    Union XY bounds of a temporal point pair.
    """
    if (
        p1["coord"].shape[0] == 0
        or p2["coord"].shape[0] == 0
    ):
        raise ValueError(
            "Cannot compute bounds of an empty temporal point pair"
        )

    xmin = min(
        float(
            p1["coord"][:, 0].min().item()
        ),
        float(
            p2["coord"][:, 0].min().item()
        ),
    )
    ymin = min(
        float(
            p1["coord"][:, 1].min().item()
        ),
        float(
            p2["coord"][:, 1].min().item()
        ),
    )
    xmax = max(
        float(
            p1["coord"][:, 0].max().item()
        ),
        float(
            p2["coord"][:, 0].max().item()
        ),
    )
    ymax = max(
        float(
            p1["coord"][:, 1].max().item()
        ),
        float(
            p2["coord"][:, 1].max().item()
        ),
    )

    return (
        xmin,
        ymin,
        xmax,
        ymax,
    )


def _window_start(
    low: float,
    high: float,
    size: float,
    *,
    randomize: bool,
) -> float:
    """
    Choose the start coordinate of a fixed physical window.

    If source extent is smaller than the window, center the source inside the
    canonical window. This keeps the returned crop physically size-consistent.
    """
    extent = high - low

    if extent <= size:
        return (
            0.5 * (low + high)
            - 0.5 * size
        )

    if not randomize:
        return (
            0.5 * (low + high - size)
        )

    u = float(
        torch.rand(
            (),
            dtype=torch.float64,
        ).item()
    )

    return (
        low
        + u * (extent - size)
    )


def make_random_pair_window(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    *,
    size: float = PAIR_POINT_WINDOW_SIZE_M,
) -> Tuple[float, float, float, float]:
    """
    Sample ONE XY window and apply it to both epochs.

    The returned window is always size × size in world coordinates.
    """
    if size <= 0:
        raise ValueError(
            "Point spatial window size must be > 0"
        )

    xmin, ymin, xmax, ymax = point_xy_bounds(
        p1,
        p2,
    )

    x0 = _window_start(
        xmin,
        xmax,
        size,
        randomize=True,
    )
    y0 = _window_start(
        ymin,
        ymax,
        size,
        randomize=True,
    )

    return (
        x0,
        y0,
        x0 + size,
        y0 + size,
    )


def crop_points_xy(
    point_data: Dict[str, Any],
    bounds,
):
    xmin, ymin, xmax, ymax = [
        float(v)
        for v in bounds
    ]

    if not (
        xmax > xmin
        and ymax > ymin
    ):
        raise ValueError(
            f"Invalid XY bounds: {bounds}"
        )

    coord = point_data["coord"]

    keep = (
        (coord[:, 0] >= xmin)
        & (coord[:, 0] < xmax)
        & (coord[:, 1] >= ymin)
        & (coord[:, 1] < ymax)
    )

    source_indices = torch.nonzero(
        keep,
        as_tuple=False,
    ).flatten()

    return {
        **point_data,
        "coord": coord[keep],
        "feat": point_data["feat"][keep],
        "_crop_mask": keep,
        "_source_indices": source_indices,
    }


def crop_point_supervision(
    target: Dict[str, torch.Tensor],
    keep: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Apply exactly the same per-point crop mask used by the point cloud.
    """
    if keep.dtype != torch.bool or keep.ndim != 1:
        raise ValueError(
            "Point crop mask must be bool [N]"
        )

    out = {}

    for key in (
        "semantic",
        "change",
    ):
        value = target[key]

        if (
            value.ndim != 1
            or value.shape[0] != keep.shape[0]
        ):
            raise ValueError(
                f"Cannot crop {key}: target shape={tuple(value.shape)}, "
                f"mask shape={tuple(keep.shape)}"
            )

        out[key] = value[keep]

    return out


def build_bitemporal_point_target(
    supervision_t1: Dict[str, torch.Tensor],
    supervision_t2: Dict[str, torch.Tensor],
    *,
    class_names: Dict[int, str],
    ignore_raw_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build ragged 3D semantic-change targets.

    T1 and T2 are independent topologies. No N1 == N2 assumption exists.

    With ignore_raw_id=None (the default), the returned target contains only
    the four real labels:
        semantic_t1, semantic_t2, change_t1, change_t2

    If an explicit semantic ignore_raw_id is configured, per-epoch semantic
    and change valid masks are added. Unknown semantic IDs still raise.
    """
    s1 = _long(supervision_t1["semantic"])
    s2 = _long(supervision_t2["semantic"])
    ch1 = _long(supervision_t1["change"])
    ch2 = _long(supervision_t2["change"])

    if s1.ndim != 1 or ch1.ndim != 1:
        raise ValueError("3D T1 semantic/change targets must be 1D [N1]")
    if s2.ndim != 1 or ch2.ndim != 1:
        raise ValueError("3D T2 semantic/change targets must be 1D [N2]")

    if s1.shape != ch1.shape:
        raise ValueError(
            f"T1 semantic/change shapes differ: "
            f"{tuple(s1.shape)} vs {tuple(ch1.shape)}"
        )
    if s2.shape != ch2.shape:
        raise ValueError(
            f"T2 semantic/change shapes differ: "
            f"{tuple(s2.shape)} vs {tuple(ch2.shape)}"
        )

    sem_valid_t1 = validate_semantic_label_values(
        s1,
        class_names,
        ignore_raw_id=ignore_raw_id,
        label_name="3D semantic_t1",
    )
    sem_valid_t2 = validate_semantic_label_values(
        s2,
        class_names,
        ignore_raw_id=ignore_raw_id,
        label_name="3D semantic_t2",
    )

    _validate_canonical_change_values(ch1)
    _validate_canonical_change_values(ch2)

    target = {
        "semantic_t1": s1,
        "semantic_t2": s2,
        "change_t1": ch1,
        "change_t2": ch2,
    }

    if ignore_raw_id is not None:
        target.update({
            "semantic_valid_t1": sem_valid_t1,
            "semantic_valid_t2": sem_valid_t2,
            # Ignored semantic points are excluded from the corresponding
            # point-wise change objective too.
            "change_valid_t1": sem_valid_t1.clone(),
            "change_valid_t2": sem_valid_t2.clone(),
        })

    return target

def make_ptv3_dict(
    point_data: Dict[str, Any],
    *,
    grid_size: float,
    shared_xyz_origin=None,
) -> Dict[str, torch.Tensor]:
    """
    Convert one cropped epoch to the current PTv3 input dictionary.

    Note:
        grid_coord here is a discrete coordinate used by PTv3 serialization.
        This function does NOT remove/merge points. Explicit voxel
        downsampling, if adopted later, must preserve an inverse mapping.
    """
    if grid_size <= 0:
        raise ValueError(
            "point grid size must be > 0"
        )

    coord = point_data[
        "coord"
    ].clone()

    if shared_xyz_origin is None:
        shared_xyz_origin = tuple(
            float(v)
            for v in coord.amin(
                dim=0
            ).tolist()
        )

    origin = torch.tensor(
        shared_xyz_origin,
        dtype=coord.dtype,
        device=coord.device,
    ).view(
        1,
        3,
    )

    coord = coord - origin

    grid_coord = torch.floor(
        coord / float(grid_size)
    ).long()

    return {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": point_data["feat"],
        "batch": torch.zeros(
            coord.shape[0],
            dtype=torch.long,
        ),
    }


# =============================================================================
# Manifest / dataset
# =============================================================================

def load_jsonl(
    path: Union[str, Path],
) -> List[Dict[str, Any]]:
    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for i, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(
                        line
                    )
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{i}"
                ) from exc

    return records


def resolve_dataset_path(
    path: Union[str, Path],
    dataset_root: Optional[
        Union[str, Path]
    ],
) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    if dataset_root is None:
        raise ValueError(
            f"Relative path {path} requires dataset_root"
        )

    return (
        Path(dataset_root)
        .expanduser()
        .resolve()
        / path
    )


def _infer_split_from_records(
    records: Sequence[Dict[str, Any]],
) -> Optional[str]:
    """
    Fallback split inference when records were supplied directly rather than
    through manifests/train.jsonl etc.
    """
    found = set()

    for record in records:
        sample_id = str(
            record.get(
                "id",
                "",
            )
        ).lower()

        for split in (
            "train",
            "val",
            "test",
        ):
            if sample_id.startswith(
                split + "_"
            ):
                found.add(
                    split
                )
                break

    if len(found) == 1:
        return next(
            iter(found)
        )

    return None


class UnifiedPAIRDataset(Dataset):
    """
    Manifest-driven reader for an already prepared PAIR-standard dataset.

    Directory layout decides route and supervision. Manifest records map
    sample IDs to concrete files.

    3D training spatial policy
    --------------------------
    - source union extent <= 51.2 m in both X/Y:
        use the complete temporal pair.
    - larger pure-3D train source:
        select one random shared 51.2 × 51.2 m XY window.
    - val/test:
        never random-crop. Oversized evaluation scenes must be handled by a
        deterministic tiling adapter rather than silently evaluated partially.

    2D+3D:
        point bounds must come from the common image footprint or an explicit
        manifest bounds field. Points are never independently randomly cropped
        away from the image footprint.
    """

    def __init__(
        self,
        records: Union[
            Sequence[Dict[str, Any]],
            str,
            Path,
        ],
        spec: DatasetSpec,
        *,
        point_feature_builder: Optional[
            PointFeatureBuilder
        ] = None,
        target_builder: Optional[
            TargetBuilder
        ] = None,
    ):
        super().__init__()

        self.manifest_path = None
        self.dataset_root = None

        if isinstance(
            records,
            (str, Path),
        ):
            self.manifest_path = (
                Path(records)
                .expanduser()
                .resolve()
            )

            self.dataset_root = (
                self.manifest_path
                .parent
                .parent
            )

            self.records = load_jsonl(
                self.manifest_path
            )

            manifest_split = (
                self.manifest_path
                .stem
                .lower()
            )

            self.split = (
                manifest_split
                if manifest_split
                in {
                    "train",
                    "val",
                    "test",
                }
                else None
            )

        else:
            self.records = list(
                records
            )

            self.split = _infer_split_from_records(
                self.records
            )

        self.spec = spec
        validate_ignore_config(
            self.spec.class_names,
            self.spec.ignore_raw_id,
        )
        self.route = spec.route

        self.point_feature_builder = (
            point_feature_builder
        )
        self.target_builder = (
            target_builder
        )

        self.prompt = build_default_prompt(
            spec
        )

        if not self.records:
            raise ValueError(
                f"Dataset {spec.name} contains no records: "
                f"{self.manifest_path}"
            )

        if (
            self.spec.has_point
            and self.target_builder is not None
        ):
            raise NotImplementedError(
                "Custom target_builder is not supported for point routes "
                "because point supervision must be cropped with exactly the "
                "same per-point mask as coord/feat."
            )

    def __len__(self):
        return len(
            self.records
        )

    def _path(
        self,
        value: Union[
            str,
            Path,
        ],
    ) -> Path:
        return resolve_dataset_path(
            value,
            self.dataset_root,
        )

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------

    def _load_2d(
        self,
        record,
    ):
        im1, geo1 = read_image(
            self._path(
                record["image_t1"]
            )
        )
        im2, geo2 = read_image(
            self._path(
                record["image_t2"]
            )
        )

        # PAIR-standard data must already be temporally aligned.
        if not same_geo_grid(
            geo1,
            geo2,
        ):
            raise ValueError(
                "T1/T2 rasters are not on one geospatial grid. "
                "PAIR data preparation must align CRS/GSD/affine first."
            )

        if (
            im1.shape[-2:]
            != im2.shape[-2:]
        ):
            raise ValueError(
                f"T1/T2 image sizes differ: "
                f"{tuple(im1.shape[-2:])} vs "
                f"{tuple(im2.shape[-2:])}. "
                "Prepare aligned pairs before training."
            )

        return {
            "images_t1": im1,
            "images_t2": im2,
            "geo_t1": geo1,
            "geo_t2": geo2,
        }

    # ------------------------------------------------------------------
    # 3D
    # ------------------------------------------------------------------

    def _load_point_supervision_pair(
        self,
        record,
        p1,
        p2,
    ):
        if self.spec.label_mode != "semantic_pair":
            raise NotImplementedError(
                "The current point-route loader is connected first for "
                "semantic_pair supervision. Binary/post-semantic 3D topology "
                "will be added when such a dataset is connected."
            )

        if (
            "semantic_t1" not in record
            or "semantic_t2" not in record
        ):
            raise KeyError(
                "3D semantic_pair manifest requires "
                "semantic_t1 and semantic_t2 paths"
            )

        t1 = read_point_supervision(
            self._path(
                record["semantic_t1"]
            )
        )
        t2 = read_point_supervision(
            self._path(
                record["semantic_t2"]
            )
        )

        # Validate the COMPLETE source labels before spatial cropping so an
        # undeclared class cannot be hidden merely because that region was not
        # sampled in this epoch.
        validate_semantic_label_values(
            t1["semantic"],
            self.spec.class_names,
            ignore_raw_id=self.spec.ignore_raw_id,
            label_name=f"{self.spec.name}/{record.get('id')} semantic_t1",
        )
        validate_semantic_label_values(
            t2["semantic"],
            self.spec.class_names,
            ignore_raw_id=self.spec.ignore_raw_id,
            label_name=f"{self.spec.name}/{record.get('id')} semantic_t2",
        )

        n1 = p1["coord"].shape[0]
        n2 = p2["coord"].shape[0]

        if (
            t1["semantic"].shape[0]
            != n1
        ):
            raise ValueError(
                f"{self.spec.name}/{record.get('id')}: "
                "T1 point/supervision length mismatch: "
                f"points={n1}, labels={t1['semantic'].shape[0]}"
            )

        if (
            t2["semantic"].shape[0]
            != n2
        ):
            raise ValueError(
                f"{self.spec.name}/{record.get('id')}: "
                "T2 point/supervision length mismatch: "
                f"points={n2}, labels={t2['semantic'].shape[0]}"
            )

        return (
            t1,
            t2,
        )

    def _choose_point_bounds(
        self,
        record,
        p1,
        p2,
        *,
        reference_bounds=None,
    ):
        """
        Decide one common XY spatial window for both epochs.

        Priority:
        1. reference_bounds from a 2D image footprint;
        2. explicit manifest record["bounds"];
        3. automatic pure-3D policy.
        """
        if reference_bounds is not None:
            if len(reference_bounds) != 4:
                raise ValueError(
                    "reference_bounds must be "
                    "[xmin,ymin,xmax,ymax]"
                )

            return tuple(
                float(v)
                for v in reference_bounds
            )

        explicit = record.get(
            "bounds"
        )

        if explicit is not None:
            if len(explicit) != 4:
                raise ValueError(
                    "bounds must be "
                    "[xmin,ymin,xmax,ymax]"
                )

            return tuple(
                float(v)
                for v in explicit
            )

        if self.route == "2d3d":
            raise ValueError(
                "2D+3D point cropping requires a world-coordinate image "
                "footprint (GeoTIFF bounds) or manifest bounds. "
                "PAIR will not independently random-crop point clouds away "
                "from their paired images."
            )

        xmin, ymin, xmax, ymax = point_xy_bounds(
            p1,
            p2,
        )

        extent_x = (
            xmax - xmin
        )
        extent_y = (
            ymax - ymin
        )

        if (
            extent_x <= PAIR_POINT_WINDOW_SIZE_M
            and extent_y <= PAIR_POINT_WINDOW_SIZE_M
        ):
            return None

        if self.split == "train":
            return make_random_pair_window(
                p1,
                p2,
                size=PAIR_POINT_WINDOW_SIZE_M,
            )

        if self.split in {
            "val",
            "test",
        }:
            raise NotImplementedError(
                f"{self.spec.name}/{record.get('id')}: "
                f"evaluation source extent "
                f"{extent_x:.2f}m × {extent_y:.2f}m exceeds "
                f"PAIR's {PAIR_POINT_WINDOW_SIZE_M:.1f}m window. "
                "Validation/test must use deterministic tiling; random "
                "evaluation cropping is intentionally forbidden."
            )

        raise RuntimeError(
            f"{self.spec.name}/{record.get('id')}: "
            "cannot infer train/val/test split for an oversized 3D source. "
            "Use manifests/train.jsonl, val.jsonl, test.jsonl or explicit bounds."
        )

    def _load_3d(
        self,
        record,
        *,
        reference_bounds=None,
    ):
        p1 = read_point_cloud(
            self._path(
                record["point_t1"]
            ),
            feature_builder=self.point_feature_builder,
        )
        p2 = read_point_cloud(
            self._path(
                record["point_t2"]
            ),
            feature_builder=self.point_feature_builder,
        )

        if (
            p1["coord"].shape[0] == 0
            or p2["coord"].shape[0] == 0
        ):
            raise ValueError(
                "Empty source T1/T2 point cloud"
            )

        supervision_t1, supervision_t2 = (
            self._load_point_supervision_pair(
                record,
                p1,
                p2,
            )
        )

        bounds = self._choose_point_bounds(
            record,
            p1,
            p2,
            reference_bounds=reference_bounds,
        )

        source_n1 = int(
            p1["coord"].shape[0]
        )
        source_n2 = int(
            p2["coord"].shape[0]
        )

        if bounds is not None:
            p1 = crop_points_xy(
                p1,
                bounds,
            )
            p2 = crop_points_xy(
                p2,
                bounds,
            )

            supervision_t1 = crop_point_supervision(
                supervision_t1,
                p1["_crop_mask"],
            )
            supervision_t2 = crop_point_supervision(
                supervision_t2,
                p2["_crop_mask"],
            )

        else:
            p1["_source_indices"] = torch.arange(
                source_n1,
                dtype=torch.long,
            )
            p2["_source_indices"] = torch.arange(
                source_n2,
                dtype=torch.long,
            )

        n1 = int(
            p1["coord"].shape[0]
        )
        n2 = int(
            p2["coord"].shape[0]
        )

        if (
            n1 == 0
            or n2 == 0
        ):
            raise ValueError(
                f"{self.spec.name}/{record.get('id')}: "
                "empty T1/T2 point cloud after common spatial crop; "
                f"bounds={bounds}, N1={n1}, N2={n2}"
            )

        if (
            supervision_t1["semantic"].shape[0]
            != n1
            or supervision_t1["change"].shape[0]
            != n1
        ):
            raise RuntimeError(
                "T1 crop broke point/label topology"
            )

        if (
            supervision_t2["semantic"].shape[0]
            != n2
            or supervision_t2["change"].shape[0]
            != n2
        ):
            raise RuntimeError(
                "T2 crop broke point/label topology"
            )

        common_z0 = min(
            float(
                p1["coord"][:, 2].min().item()
            ),
            float(
                p2["coord"][:, 2].min().item()
            ),
        )

        if bounds is not None:
            shared_xyz_origin = (
                float(bounds[0]),
                float(bounds[1]),
                common_z0,
            )
        else:
            common_min = torch.minimum(
                p1["coord"].amin(
                    dim=0
                ),
                p2["coord"].amin(
                    dim=0
                ),
            )

            shared_xyz_origin = tuple(
                float(v)
                for v in common_min.tolist()
            )

        target = build_bitemporal_point_target(
            supervision_t1,
            supervision_t2,
            class_names=self.spec.class_names,
            ignore_raw_id=self.spec.ignore_raw_id,
        )

        return {
            "point_dict_t1": make_ptv3_dict(
                p1,
                grid_size=self.spec.point_grid_size,
                shared_xyz_origin=shared_xyz_origin,
            ),
            "point_dict_t2": make_ptv3_dict(
                p2,
                grid_size=self.spec.point_grid_size,
                shared_xyz_origin=shared_xyz_origin,
            ),
            "point_crs_t1": p1.get(
                "crs"
            ),
            "point_crs_t2": p2.get(
                "crs"
            ),
            "point_source_indices_t1": p1[
                "_source_indices"
            ],
            "point_source_indices_t2": p2[
                "_source_indices"
            ],
            "point_source_count_t1": source_n1,
            "point_source_count_t2": source_n2,
            "bounds": bounds,
            "shared_xyz_origin": shared_xyz_origin,
            "_point_target": target,
        }

    # ------------------------------------------------------------------
    # Common-topology targets (existing 2D path)
    # ------------------------------------------------------------------

    def _load_targets(
        self,
        record,
    ):
        if self.target_builder is not None:
            return self.target_builder(
                record,
                self.spec,
            )

        s1 = (
            read_label_array(
                self._path(
                    record["semantic_t1"]
                )
            )
            if "semantic_t1" in record
            else None
        )

        s2 = (
            read_label_array(
                self._path(
                    record["semantic_t2"]
                )
            )
            if "semantic_t2" in record
            else None
        )

        ch = (
            read_label_array(
                self._path(
                    record["change"]
                )
            )
            if "change" in record
            else None
        )

        target = build_canonical_target(
            label_mode=self.spec.label_mode,
            semantic_t1=s1,
            semantic_t2=s2,
            change=ch,
            class_names=self.spec.class_names,
            ignore_raw_id=self.spec.ignore_raw_id,
        )

        out = {
            "change": target.change,
            "semantic_t1": target.semantic_t1,
            "semantic_t2": target.semantic_t2,
        }

        if target.change_valid is not None:
            out["change_valid"] = target.change_valid
        if target.semantic_valid_t1 is not None:
            out["semantic_valid_t1"] = target.semantic_valid_t1
        if target.semantic_valid_t2 is not None:
            out["semantic_valid_t2"] = target.semantic_valid_t2

        return out

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def __getitem__(
        self,
        index,
    ):
        record = self.records[
            index
        ]

        sample: Dict[str, Any] = {
            "sample_id": record.get(
                "id",
                str(index),
            ),
            "dataset_name": self.spec.name,
            "route": self.route,
            # Internal backward-compatible alias.
            "task_mode": self.route,
            "prompt": self.prompt,
            "class_names": dict(
                self.spec.class_names
            ),
        }

        if self.spec.has_image:
            sample.update(
                self._load_2d(
                    record
                )
            )

        point_target = None

        if self.spec.has_point:
            reference_bounds = None

            if self.route == "2d3d":
                geo_t1 = sample.get(
                    "geo_t1"
                )
                geo_t2 = sample.get(
                    "geo_t2"
                )

                bounds_t1 = (
                    None
                    if geo_t1 is None
                    else geo_t1.get(
                        "bounds"
                    )
                )
                bounds_t2 = (
                    None
                    if geo_t2 is None
                    else geo_t2.get(
                        "bounds"
                    )
                )

                if (
                    bounds_t1 is not None
                    and bounds_t2 is not None
                ):
                    if not np.allclose(
                        np.asarray(
                            bounds_t1,
                            dtype=np.float64,
                        ),
                        np.asarray(
                            bounds_t2,
                            dtype=np.float64,
                        ),
                        atol=1e-6,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            "T1/T2 image world bounds differ in a 2D+3D sample"
                        )

                    reference_bounds = tuple(
                        float(v)
                        for v in bounds_t1
                    )

                elif record.get(
                    "bounds"
                ) is not None:
                    reference_bounds = tuple(
                        float(v)
                        for v in record[
                            "bounds"
                        ]
                    )

            point_part = self._load_3d(
                record,
                reference_bounds=reference_bounds,
            )

            point_target = point_part.pop(
                "_point_target"
            )

            sample.update(
                point_part
            )

        if self.spec.has_point:
            sample["target"] = point_target
        else:
            sample["target"] = self._load_targets(
                record
            )

        # Existing 2D topology check.
        if (
            self.spec.has_image
            and not self.spec.has_point
        ):
            h, w = sample[
                "images_t1"
            ].shape[-2:]

            for key in (
                "change",
                "semantic_t1",
                "semantic_t2",
            ):
                target = sample[
                    "target"
                ][key]

                if (
                    target.ndim >= 2
                    and tuple(
                        target.shape[-2:]
                    ) != (h, w)
                ):
                    raise ValueError(
                        f"{self.spec.name}/"
                        f"{sample['sample_id']}: "
                        f"{key} shape "
                        f"{tuple(target.shape)} "
                        f"does not match image {(h, w)}"
                    )

        # 3D topology check.
        if self.spec.has_point:
            n1 = sample["point_dict_t1"]["coord"].shape[0]
            n2 = sample["point_dict_t2"]["coord"].shape[0]

            for key in ("semantic_t1", "change_t1"):
                if sample["target"][key].shape[0] != n1:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: "
                        f"{key} length {sample['target'][key].shape[0]} "
                        f"does not match T1 points {n1}"
                    )

            for key in ("semantic_t2", "change_t2"):
                if sample["target"][key].shape[0] != n2:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: "
                        f"{key} length {sample['target'][key].shape[0]} "
                        f"does not match T2 points {n2}"
                    )

            for key in ("semantic_valid_t1", "change_valid_t1"):
                if key in sample["target"] and sample["target"][key].shape[0] != n1:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: "
                        f"{key} length {sample['target'][key].shape[0]} "
                        f"does not match T1 points {n1}"
                    )

            for key in ("semantic_valid_t2", "change_valid_t2"):
                if key in sample["target"] and sample["target"][key].shape[0] != n2:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: "
                        f"{key} length {sample['target'][key].shape[0]} "
                        f"does not match T2 points {n2}"
                    )

        if self.route == "2d3d":
            raster_crs = (
                sample[
                    "geo_t1"
                ].get(
                    "crs"
                )
                if sample.get(
                    "geo_t1"
                ) is not None
                else None
            )

            known_crs = [
                c
                for c in (
                    raster_crs,
                    sample.get(
                        "point_crs_t1"
                    ),
                    sample.get(
                        "point_crs_t2"
                    ),
                )
                if c
            ]

            if (
                known_crs
                and any(
                    c != known_crs[0]
                    for c in known_crs[1:]
                )
            ):
                raise ValueError(
                    "2D+3D sample contains mismatched CRS metadata. "
                    "Prepare all modalities in one CRS before training."
                )

        sample["spatial_meta"] = {
            "bounds": sample.get(
                "bounds",
                record.get(
                    "bounds"
                ),
            ),
            "raster_geo_t1": sample.get(
                "geo_t1"
            ),
            "raster_geo_t2": sample.get(
                "geo_t2"
            ),
            "point_crs_t1": sample.get(
                "point_crs_t1"
            ),
            "point_crs_t2": sample.get(
                "point_crs_t2"
            ),
            "point_grid_size": self.spec.point_grid_size,
            "point_window_size_m": (
                PAIR_POINT_WINDOW_SIZE_M
            ),
            "shared_xyz_origin": sample.get(
                "shared_xyz_origin"
            ),
            "point_source_count_t1": sample.get(
                "point_source_count_t1"
            ),
            "point_source_count_t2": sample.get(
                "point_source_count_t2"
            ),
        }

        return sample


# =============================================================================
# Self-test
# =============================================================================

def _self_test():
    classes = {
        0: "ground",
        1: "building",
        2: "vegetation",
        3: "clutter",
        4: "water",
        5: "other",
    }

    # Fully supervised 2D semantic pair: no redundant valid masks.
    s1 = torch.tensor([[0, 1], [2, 3]])
    s2 = torch.tensor([[0, 4], [2, 5]])
    out = build_canonical_target(
        label_mode="semantic_pair",
        semantic_t1=s1,
        semantic_t2=s2,
        class_names=classes,
    )
    assert torch.equal(out.change, torch.tensor([[0, 1], [0, 1]]))
    assert out.change_valid is None
    assert out.semantic_valid_t1 is None
    assert out.semantic_valid_t2 is None

    # Undeclared semantic labels are errors, not silently mapped to ignore.
    try:
        build_canonical_target(
            label_mode="semantic_pair",
            semantic_t1=torch.tensor([0, 99]),
            semantic_t2=torch.tensor([0, 1]),
            class_names={0: "ground", 1: "building"},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("undeclared semantic ID did not raise")

    # Explicit ignored semantic class is allowed only when configured.
    out = build_canonical_target(
        label_mode="semantic_pair",
        semantic_t1=torch.tensor([0, 5, 1]),
        semantic_t2=torch.tensor([0, 1, 1]),
        class_names={0: "ground", 1: "building"},
        ignore_raw_id=5,
    )
    assert out.semantic_valid_t1.tolist() == [True, False, True]
    assert out.semantic_valid_t2.tolist() == [True, True, True]
    assert out.change_valid.tolist() == [True, False, True]

    # Existing binary raw-value normalization remains strict.
    raw_bcd = torch.tensor([[0, 255], [255, 0]])
    out = build_canonical_target(
        label_mode="binary",
        change=raw_bcd,
        class_names={0: "unchanged", 255: "changed"},
    )
    assert torch.equal(out.change, torch.tensor([[0, 1], [1, 0]]))
    assert out.change_valid is None
    assert not out.semantic_valid_t1.any()
    assert not out.semantic_valid_t2.any()

    # Ragged 3D topology: N1 != N2 is legal and has no valid masks by default.
    target3d = build_bitemporal_point_target(
        {
            "semantic": torch.tensor([0, 1, 2]),
            "change": torch.tensor([0, 1, 0]),
        },
        {
            "semantic": torch.tensor([0, 3]),
            "change": torch.tensor([1, 0]),
        },
        class_names={0: "ground", 1: "building", 2: "vegetation", 3: "clutter"},
    )
    assert target3d["semantic_t1"].shape == (3,)
    assert target3d["semantic_t2"].shape == (2,)
    assert "semantic_valid_t1" not in target3d
    assert "change_valid_t1" not in target3d

    # Explicit 3D ignore creates masks only then.
    target3d_ignore = build_bitemporal_point_target(
        {
            "semantic": torch.tensor([0, 9, 1]),
            "change": torch.tensor([0, 1, 0]),
        },
        {
            "semantic": torch.tensor([0, 1]),
            "change": torch.tensor([1, 0]),
        },
        class_names={0: "ground", 1: "building"},
        ignore_raw_id=9,
    )
    assert target3d_ignore["semantic_valid_t1"].tolist() == [True, False, True]
    assert target3d_ignore["change_valid_t1"].tolist() == [True, False, True]

    # -100 has no magic meaning unless explicitly configured as ignore.
    try:
        build_bitemporal_point_target(
            {
                "semantic": torch.tensor([0, -100]),
                "change": torch.tensor([0, 1]),
            },
            {
                "semantic": torch.tensor([0]),
                "change": torch.tensor([0]),
            },
            class_names={0: "ground"},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("implicit -100 semantic ignore was accepted")

    assert infer_binary_class_ids({0: "unchanged", 255: "changed"}) == (0, 255)
    assert "changed_raw_id" not in DatasetSpec.__dataclass_fields__
    assert DatasetSpec.__dataclass_fields__["ignore_raw_id"].default is None
    assert infer_unchanged_raw_id({0: "unchanged", 1: "building"}) == 0
    assert infer_unchanged_raw_id({0: "background", 1: "building"}) is None
    assert PAIR_POINT_WINDOW_SIZE_M == 51.2

    print("pair_dataset.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
