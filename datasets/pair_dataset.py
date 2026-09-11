# -*- coding: utf-8 -*-
"""
PAIR standard dataset interface.

Directory is schema. Existing 2D SCD/BCD protocols stay unchanged.

Canonical 3D point files:
    points_t*/sample.npz
        coord      [N,3] mandatory
        rgb        [N,3] optional
        intensity  [N,1] optional

Canonical 3D supervision files:
    semantic_t*/sample.npz
        semantic     [N]
        event        [N]
        event_valid  [N] optional bool

PAIR 3D event taxonomy:
    0 unchanged
    1 added
    2 removed
    3 class_change
    4 height_up
    5 height_down

T1/T2 use independent point topologies. Point and supervision files for one
epoch must preserve exactly the same point order.

Semantic ignore:
    DatasetSpec.ignored_id is explicit. Unknown semantic IDs are never silently
    converted. With ignored_id configured, semantic_valid is generated.

3D train crop:
    Pure 3D training uses a ME-CPT-style change-aware crop sampler:
      1. T2 is grouped into coarse 3D cells;
      2. each cell receives its majority event label;
      3. event classes are sampled with inverse-square-root frequency weights;
      4. a center from that class is sampled;
      5. both epochs receive the same 51.2 x 51.2 m XY crop.

    ME-CPT uses radius=25 m and center cells radius/10=2.5 m. PAIR keeps its
    existing 51.2 m square window, therefore uses half-window/10 = 2.56 m
    center cells. The sampling idea is retained while PAIR's spatial contract
    stays unchanged.

    T2 event labels are used for center selection. A T2 event that is invalid
    only because its temporal support belongs to T1 (e.g. NYC removed) may still
    be used as a crop-center label when the same event is valid on T1. Invalid
    placeholder event=0 points are not promoted into sampling centers.

    If a sampled center yields an empty epoch, another center is tried. A
    jointly-occupied random-window fallback prevents the old one-epoch-empty
    crop failure.

Validation/test:
    never random crop. Oversized evaluation scenes still require deterministic
    tiling; partial random evaluation is intentionally forbidden.

Important separation:
    semantic_valid and event_valid are independent. Semantic ignore must not
    automatically erase a valid event label, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image
from tqdm.auto import tqdm


VALID_MODALITIES = {"image", "point"}

PAIR_POINT_WINDOW_SIZE_M = 51.2
PAIR_CHANGE_CENTER_CELL_M = PAIR_POINT_WINDOW_SIZE_M / 20.0
PAIR_CHANGE_CROP_MAX_TRIES = 32
PAIR_CHANGE_CENTER_CACHE_VERSION = 2

PAIR_EVENT_NAMES = {
    0: "unchanged",
    1: "added",
    2: "removed",
    3: "class_change",
    4: "height_up",
    5: "height_down",
}
PAIR_EVENT_NUM_CLASSES = len(PAIR_EVENT_NAMES)


# =============================================================================
# Dataset schema / label protocol
# =============================================================================

def normalize_class_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def infer_unchanged_raw_id(class_names: Dict[int, str]) -> Optional[int]:
    """Infer a semantic unchanged class by NAME only. Background is not unchanged."""
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
    """Infer physical unchanged/changed raw IDs from binary class_names."""
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
            f"Binary class_names must contain exactly one unchanged/no-change class; "
            f"found raw IDs {unchanged}."
        )
    if len(changed) != 1:
        raise ValueError(
            f"Binary class_names must contain exactly one changed/change class; "
            f"found raw IDs {changed}."
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
    """Internal normalized dataset description built by config_loader."""
    name: str
    modalities: Tuple[str, ...]
    label_mode: str
    class_names: Dict[int, str]
    ignored_id: Optional[int] = None

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
    """Common-topology target retained for the existing 2D path."""
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
    return {int(raw_id) for raw_id in class_names}


def validate_ignore_config(class_names: Dict[int, str], ignored_id: Optional[int]):
    if ignored_id is None:
        return
    ignored_id = int(ignored_id)
    if ignored_id in _declared_class_ids(class_names):
        raise ValueError(
            f"ignored_id={ignored_id} is also present in class_names. "
            "Ignored classes must not occupy a trainable classifier slot."
        )


def validate_semantic_label_values(
    raw: torch.Tensor,
    class_names: Dict[int, str],
    *,
    ignored_id: Optional[int] = None,
    label_name: str = "semantic",
) -> Optional[torch.Tensor]:
    """
    Validate raw semantic IDs.

    Returns None when ignored_id is None; otherwise returns bool semantic_valid.
    """
    raw = _long(raw)
    declared = _declared_class_ids(class_names)
    validate_ignore_config(class_names, ignored_id)

    allowed = torch.zeros_like(raw, dtype=torch.bool)
    for raw_id in declared:
        allowed |= raw == raw_id
    if ignored_id is not None:
        allowed |= raw == int(ignored_id)

    if not allowed.all():
        bad = torch.unique(raw[~allowed]).cpu().tolist()
        suffix = f" plus ignored_id={int(ignored_id)}" if ignored_id is not None else ""
        raise ValueError(
            f"{label_name} contains undeclared raw IDs {bad}. "
            f"class_names declares {sorted(declared)}{suffix}."
        )

    return None if ignored_id is None else raw != int(ignored_id)


def _validate_canonical_change_values(raw: torch.Tensor):
    valid = (raw == 0) | (raw == 1)
    if not valid.all():
        bad = torch.unique(raw[~valid]).cpu().tolist()
        raise ValueError(
            f"Canonical PAIR change targets must contain only 0=unchanged and 1=changed; "
            f"found {bad}"
        )


def _normalize_binary_change_values(
    raw: torch.Tensor,
    class_names: Dict[int, str],
    *,
    ignored_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    unchanged_raw_id, changed_raw_id = infer_binary_class_ids(class_names)

    if ignored_id is not None:
        ignored_id = int(ignored_id)
        if ignored_id in {unchanged_raw_id, changed_raw_id}:
            raise ValueError(
                f"ignored_id={ignored_id} conflicts with binary unchanged/changed class IDs"
            )

    valid = (raw == unchanged_raw_id) | (raw == changed_raw_id)
    if ignored_id is not None:
        valid |= raw == ignored_id

    if not valid.all():
        bad = torch.unique(raw[~valid]).cpu().tolist()
        suffix = f", ignored_id={ignored_id}" if ignored_id is not None else ""
        raise ValueError(
            f"Binary change target contains undeclared raw IDs {bad}; "
            f"class_names declares unchanged={unchanged_raw_id}, changed={changed_raw_id}{suffix}."
        )

    out = torch.zeros_like(raw)
    out[raw == changed_raw_id] = 1
    change_valid = None if ignored_id is None else raw != ignored_id
    return out, change_valid


def build_canonical_target(
    *,
    label_mode: str,
    semantic_t1=None,
    semantic_t2=None,
    change=None,
    class_names: Optional[Dict[int, str]] = None,
    ignored_id: Optional[int] = None,
) -> CanonicalChangeTarget:
    """Build the existing common-topology 2D target."""
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
                f"got semantic_t1={tuple(s1.shape)} and semantic_t2={tuple(s2.shape)}. "
                "Ragged 3D pairs must use per-epoch supervision bundles."
            )

        sv1 = validate_semantic_label_values(
            s1, class_names, ignored_id=ignored_id, label_name="semantic_t1"
        )
        sv2 = validate_semantic_label_values(
            s2, class_names, ignored_id=ignored_id, label_name="semantic_t2"
        )

        if change is None:
            ch = (s1 != s2).long()
        else:
            ch = _long(change)
            if ch.shape != s1.shape:
                raise ValueError("change mask must match semantic label shape")
            _validate_canonical_change_values(ch)

        change_valid = None if ignored_id is None else sv1 & sv2
        return CanonicalChangeTarget(
            change=ch,
            semantic_t1=s1,
            semantic_t2=s2,
            change_valid=change_valid,
            semantic_valid_t1=sv1,
            semantic_valid_t2=sv2,
        )

    if change is None:
        raise ValueError(f"{mode} requires change supervision")
    raw = _long(change)

    if mode == "binary":
        if class_names is None:
            raise ValueError("binary target normalization requires class_names")
        ch, change_valid = _normalize_binary_change_values(
            raw, class_names, ignored_id=ignored_id
        )
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

    _validate_canonical_change_values(raw)
    if semantic_t2 is None:
        raise ValueError("post_semantic requires semantic_t2")
    if class_names is None:
        raise ValueError("post_semantic validation requires class_names")

    s2 = _long(semantic_t2)
    if s2.shape != raw.shape:
        raise ValueError("semantic_t2 must match change mask shape")
    sv2 = validate_semantic_label_values(
        s2, class_names, ignored_id=ignored_id, label_name="semantic_t2"
    )
    s1 = torch.zeros_like(raw)
    return CanonicalChangeTarget(
        change=raw.clone(),
        semantic_t1=s1,
        semantic_t2=s2,
        change_valid=None,
        semantic_valid_t1=torch.zeros_like(raw, dtype=torch.bool),
        semantic_valid_t2=sv2,
    )


def build_default_prompt(spec: DatasetSpec) -> str:
    if spec.has_point:
        text = (
            "Perform 3D semantic change reasoning between Time 1 and Time 2. "
            "Predict the semantic class at both times and the per-point change event."
        )
    else:
        text = (
            "Perform semantic change detection between Time 1 and Time 2. "
            "Identify unchanged and changed regions."
        )
        if spec.label_mode == "semantic_pair":
            text += " For changed regions, infer the semantic class before and after change."
        elif spec.label_mode == "post_semantic":
            text += (
                " The pre-change semantic class may be unknown, while the post-change "
                "class is supervised."
            )
        elif spec.label_mode == "binary":
            text += (
                " The source dataset supervises change only; semantic classes before "
                "and after change may be unknown."
            )

    if spec.class_names:
        classes = ", ".join(f"{raw_id}: {name}" for raw_id, name in spec.class_names.items())
        text += " Valid semantic classes are: " + classes + "."
    return text


# =============================================================================
# Readers
# =============================================================================

def read_label_array(path: Union[str, Path]) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} has multiple arrays {keys}; "
                    "use read_point_supervision() for 3D supervision bundles"
                )
            arr = np.asarray(z[keys[0]])
    else:
        arr = np.asarray(Image.open(path))

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected a single-channel label, got {arr.shape}: {path}")
    return torch.as_tensor(np.array(arr, copy=True), dtype=torch.long)


def _validate_event_values(event: torch.Tensor, label_name: str = "event"):
    event = _long(event)
    valid = (event >= 0) & (event < PAIR_EVENT_NUM_CLASSES)
    if not valid.all():
        bad = torch.unique(event[~valid]).cpu().tolist()
        raise ValueError(
            f"{label_name} contains invalid event IDs {bad}; "
            f"expected 0..{PAIR_EVENT_NUM_CLASSES - 1}"
        )


def read_point_supervision(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Read one epoch's prepared 3D semantic/event bundle."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"PAIR 3D supervision must be NPZ: {path}")

    with np.load(path, allow_pickle=False) as z:
        keys = set(z.keys())
        missing = sorted({"semantic", "event"} - keys)
        if missing:
            raise ValueError(
                f"{path} is missing supervision arrays {missing}; available={sorted(keys)}"
            )
        semantic = np.asarray(z["semantic"])
        event = np.asarray(z["event"])
        event_valid = np.asarray(z["event_valid"]) if "event_valid" in z else None

    if semantic.ndim != 1 or event.ndim != 1:
        raise ValueError(f"{path}: semantic/event must both be 1D [N]")
    if semantic.shape != event.shape:
        raise ValueError(
            f"{path}: semantic/event shapes differ: {semantic.shape} vs {event.shape}"
        )
    if event_valid is not None and event_valid.shape != semantic.shape:
        raise ValueError(
            f"{path}: event_valid shape {event_valid.shape} does not match semantic "
            f"{semantic.shape}"
        )

    out = {
        "semantic": torch.as_tensor(np.array(semantic, copy=True), dtype=torch.long),
        "event": torch.as_tensor(np.array(event, copy=True), dtype=torch.long),
    }
    _validate_event_values(out["event"], f"{path} event")
    if event_valid is not None:
        out["event_valid"] = torch.as_tensor(
            np.array(event_valid, copy=True), dtype=torch.bool
        )
    return out


def _read_geotiff(path: Path):
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    try:
        import rasterio
    except ImportError:
        return None

    with rasterio.open(path) as src:
        arr = src.read()
        meta = {
            "crs": str(src.crs) if src.crs is not None else None,
            "transform": tuple(src.transform),
            "bounds": tuple(src.bounds),
            "width": int(src.width),
            "height": int(src.height),
            "gsd_x": abs(float(src.transform.a)),
            "gsd_y": abs(float(src.transform.e)),
        }
    return np.moveaxis(arr, 0, -1), meta


def read_image(path: Union[str, Path]):
    path = Path(path)
    geo = _read_geotiff(path)

    if geo is not None:
        arr, meta = geo
    else:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im)
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

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if vmax > 1.5:
            arr /= 255.0 if vmax <= 255.0 else 65535.0

    tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(arr, -1, 0))).float()
    return tensor, meta


def same_geo_grid(meta1: Dict[str, Any], meta2: Dict[str, Any], atol=1e-7) -> bool:
    crs1, crs2 = meta1.get("crs"), meta2.get("crs")
    tr1, tr2 = meta1.get("transform"), meta2.get("transform")
    if crs1 is None and crs2 is None and tr1 is None and tr2 is None:
        return True
    if crs1 != crs2 or tr1 is None or tr2 is None:
        return False
    return (
        np.allclose(np.asarray(tr1), np.asarray(tr2), atol=atol, rtol=0.0)
        and meta1["width"] == meta2["width"]
        and meta1["height"] == meta2["height"]
    )


TargetBuilder = Callable[[Dict[str, Any], DatasetSpec], Dict[str, Any]]


def _read_las(path: Path):
    try:
        import laspy
    except ImportError as exc:
        raise ImportError("LAS/LAZ reading requires laspy") from exc

    las = laspy.read(path)
    attrs: Dict[str, Any] = {
        "coord": np.stack(
            [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1
        ).astype(np.float32)
    }
    available = set(las.point_format.dimension_names)
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
            attrs[name] = np.asarray(getattr(las, name))

    try:
        crs = las.header.parse_crs()
        attrs["_crs"] = None if crs is None else str(crs)
    except Exception:
        attrs["_crs"] = None
    return attrs


def _optional_point_tensor(array, *, n: int, width: int, name: str, path: Path):
    arr = np.asarray(array)
    if width == 1 and arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape != (n, width):
        raise ValueError(f"{path}: {name} must be [{n},{width}], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: {name} contains NaN/Inf")
    return torch.from_numpy(
        np.ascontiguousarray(arr.astype(np.float32, copy=False))
    ).float()


def read_point_cloud(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read PAIR point input: coord mandatory, rgb/intensity optional.

    Optional fields are never synthesized and are not normalized here.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    crs = None
    rgb = None
    intensity = None

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.keys())
            if "coord" not in keys:
                raise ValueError(f"{path} must contain coord [N,3]")
            coord = np.asarray(z["coord"], dtype=np.float32)
            if "feat" in keys:
                raise ValueError(
                    f"{path} contains legacy 'feat'. "
                    "Re-prepare using coord + optional rgb/intensity."
                )
            if "rgb" in keys:
                rgb = np.asarray(z["rgb"])
            if "intensity" in keys:
                intensity = np.asarray(z["intensity"])
            if "crs" in keys:
                value = np.asarray(z["crs"])
                crs = str(value.item() if value.ndim == 0 else value)

    elif suffix == ".npy":
        coord = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"{path}: canonical NPY point input must be [N,3]")

    elif suffix in {".las", ".laz"}:
        attrs = _read_las(path)
        coord = attrs["coord"]
        crs = attrs.get("_crs")
        if all(name in attrs for name in ("red", "green", "blue")):
            rgb = np.column_stack((attrs["red"], attrs["green"], attrs["blue"]))
        if "intensity" in attrs:
            intensity = np.asarray(attrs["intensity"])

    else:
        raise ValueError(f"Unsupported point-cloud format: {path}")

    if coord.ndim != 2 or coord.shape[1] != 3 or coord.shape[0] == 0:
        raise ValueError(f"{path}: coord must be non-empty [N,3], got {coord.shape}")
    if not np.isfinite(coord).all():
        raise ValueError(f"{path}: coord contains NaN/Inf")

    n = coord.shape[0]
    out: Dict[str, Any] = {
        "coord": torch.from_numpy(np.ascontiguousarray(coord)).float(),
        "crs": crs,
    }
    if rgb is not None:
        out["rgb"] = _optional_point_tensor(
            rgb, n=n, width=3, name="rgb", path=path
        )
    if intensity is not None:
        out["intensity"] = _optional_point_tensor(
            intensity, n=n, width=1, name="intensity", path=path
        )
    return out


# =============================================================================
# Point spatial helpers
# =============================================================================

def point_xy_bounds(
    p1: Dict[str, Any], p2: Dict[str, Any]
) -> Tuple[float, float, float, float]:
    if p1["coord"].shape[0] == 0 or p2["coord"].shape[0] == 0:
        raise ValueError("Cannot compute bounds of an empty temporal point pair")

    xmin = min(
        float(p1["coord"][:, 0].min().item()),
        float(p2["coord"][:, 0].min().item()),
    )
    ymin = min(
        float(p1["coord"][:, 1].min().item()),
        float(p2["coord"][:, 1].min().item()),
    )
    xmax = max(
        float(p1["coord"][:, 0].max().item()),
        float(p2["coord"][:, 0].max().item()),
    )
    ymax = max(
        float(p1["coord"][:, 1].max().item()),
        float(p2["coord"][:, 1].max().item()),
    )
    return xmin, ymin, xmax, ymax


def _window_start(low: float, high: float, size: float, *, randomize: bool) -> float:
    extent = high - low
    if extent <= size:
        return 0.5 * (low + high) - 0.5 * size
    if not randomize:
        return 0.5 * (low + high - size)
    u = float(torch.rand((), dtype=torch.float64).item())
    return low + u * (extent - size)


def centered_pair_window(center_xy, *, size=PAIR_POINT_WINDOW_SIZE_M):
    if size <= 0:
        raise ValueError("Point spatial window size must be > 0")
    x, y = float(center_xy[0]), float(center_xy[1])
    half = 0.5 * float(size)
    return x - half, y - half, x + half, y + half


def make_random_pair_window(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    *,
    size: float = PAIR_POINT_WINDOW_SIZE_M,
) -> Tuple[float, float, float, float]:
    """Legacy unbiased common XY window retained as a helper."""
    if size <= 0:
        raise ValueError("Point spatial window size must be > 0")
    xmin, ymin, xmax, ymax = point_xy_bounds(p1, p2)
    x0 = _window_start(xmin, xmax, size, randomize=True)
    y0 = _window_start(ymin, ymax, size, randomize=True)
    return x0, y0, x0 + size, y0 + size


def make_joint_occupied_pair_window(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    *,
    size: float = PAIR_POINT_WINDOW_SIZE_M,
    shift_tries: int = 8,
) -> Tuple[float, float, float, float]:
    """
    Random fixed-size window guaranteed to contain >=1 point from each epoch.

    Used only as a defensive fallback when change-aware center retries fail.
    """
    if size <= 0:
        raise ValueError("Point spatial window size must be > 0")

    c1 = p1["coord"]
    c2 = p2["coord"]
    if c1.shape[0] == 0 or c2.shape[0] == 0:
        raise ValueError("Cannot crop an empty temporal point pair")

    xmin, ymin, _, _ = point_xy_bounds(p1, p2)

    for _ in range(max(int(shift_tries), 1)):
        shift_x = float(torch.rand((), dtype=torch.float64).item()) * size
        shift_y = float(torch.rand((), dtype=torch.float64).item()) * size
        ox = xmin - shift_x
        oy = ymin - shift_y

        def occupied(coord):
            ix = torch.floor((coord[:, 0].double() - ox) / size).long()
            iy = torch.floor((coord[:, 1].double() - oy) / size).long()
            return {tuple(v) for v in torch.unique(torch.stack((ix, iy), 1), dim=0).cpu().tolist()}

        common = list(occupied(c1) & occupied(c2))
        if not common:
            continue

        cx, cy = common[int(torch.randint(len(common), (1,)).item())]
        x0 = ox + cx * size
        y0 = oy + cy * size
        return x0, y0, x0 + size, y0 + size

    raise ValueError(
        f"T1/T2 have no jointly occupied {size:.1f}m grid cell after "
        f"{shift_tries} random grid shifts"
    )


def crop_points_xy(point_data: Dict[str, Any], bounds):
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"Invalid XY bounds: {bounds}")

    coord = point_data["coord"]
    keep = (
        (coord[:, 0] >= xmin)
        & (coord[:, 0] < xmax)
        & (coord[:, 1] >= ymin)
        & (coord[:, 1] < ymax)
    )

    out = {**point_data, "coord": coord[keep]}
    for key in ("rgb", "intensity"):
        if key in point_data:
            out[key] = point_data[key][keep]

    out["_crop_mask"] = keep
    out["_source_indices"] = torch.nonzero(keep, as_tuple=False).flatten()
    return out


def crop_point_supervision(
    target: Dict[str, torch.Tensor], keep: torch.Tensor
) -> Dict[str, torch.Tensor]:
    if keep.dtype != torch.bool or keep.ndim != 1:
        raise ValueError("Point crop mask must be bool [N]")

    out = {}
    for key in ("semantic", "event", "event_valid"):
        if key not in target:
            continue
        value = target[key]
        if value.ndim != 1 or value.shape[0] != keep.shape[0]:
            raise ValueError(
                f"Cannot crop {key}: target shape={tuple(value.shape)}, "
                f"mask={tuple(keep.shape)}"
            )
        out[key] = value[keep]
    return out


def build_bitemporal_point_target(
    supervision_t1: Dict[str, torch.Tensor],
    supervision_t2: Dict[str, torch.Tensor],
    *,
    class_names: Dict[int, str],
    ignored_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build ragged semantic + event targets.

    semantic_valid and event_valid intentionally remain independent.
    """
    s1 = _long(supervision_t1["semantic"])
    s2 = _long(supervision_t2["semantic"])
    e1 = _long(supervision_t1["event"])
    e2 = _long(supervision_t2["event"])

    if s1.ndim != 1 or e1.ndim != 1 or s1.shape != e1.shape:
        raise ValueError("3D T1 semantic/event must be same-length 1D [N1]")
    if s2.ndim != 1 or e2.ndim != 1 or s2.shape != e2.shape:
        raise ValueError("3D T2 semantic/event must be same-length 1D [N2]")

    sv1 = validate_semantic_label_values(
        s1, class_names, ignored_id=ignored_id, label_name="3D semantic_t1"
    )
    sv2 = validate_semantic_label_values(
        s2, class_names, ignored_id=ignored_id, label_name="3D semantic_t2"
    )
    _validate_event_values(e1, "3D event_t1")
    _validate_event_values(e2, "3D event_t2")

    ev1 = supervision_t1.get("event_valid")
    ev2 = supervision_t2.get("event_valid")
    if ev1 is None:
        ev1 = torch.ones_like(e1, dtype=torch.bool)
    else:
        ev1 = ev1.bool()
    if ev2 is None:
        ev2 = torch.ones_like(e2, dtype=torch.bool)
    else:
        ev2 = ev2.bool()

    if ev1.shape != e1.shape or ev2.shape != e2.shape:
        raise ValueError("3D event_valid must match its epoch event topology")

    target: Dict[str, torch.Tensor] = {
        "semantic_t1": s1,
        "semantic_t2": s2,
        "event_t1": e1,
        "event_t2": e2,
        "event_valid_t1": ev1,
        "event_valid_t2": ev2,
    }
    if sv1 is not None:
        target["semantic_valid_t1"] = sv1
    if sv2 is not None:
        target["semantic_valid_t2"] = sv2
    return target


def make_point_dict(
    point_data: Dict[str, Any], *, shared_xyz_origin=None
) -> Dict[str, torch.Tensor]:
    """Translate one epoch into a shared local XYZ frame."""
    coord = point_data["coord"].clone()
    if shared_xyz_origin is None:
        shared_xyz_origin = tuple(float(v) for v in coord.amin(dim=0).tolist())

    origin = torch.tensor(
        shared_xyz_origin, dtype=coord.dtype, device=coord.device
    ).view(1, 3)
    out: Dict[str, torch.Tensor] = {"coord": coord - origin}
    for key in ("rgb", "intensity"):
        if key in point_data:
            out[key] = point_data[key]
    return out


# =============================================================================
# Manifest utilities
# =============================================================================

def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}") from exc
    return records


def resolve_dataset_path(
    path: Union[str, Path], dataset_root: Optional[Union[str, Path]]
) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if dataset_root is None:
        raise ValueError(f"Relative path {path} requires dataset_root")
    return Path(dataset_root).expanduser().resolve() / path


def _infer_split_from_records(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    found = set()
    for record in records:
        sample_id = str(record.get("id", "")).lower()
        for split in ("train", "val", "test"):
            if sample_id.startswith(split + "_"):
                found.add(split)
                break
    return next(iter(found)) if len(found) == 1 else None


# =============================================================================
# ME-CPT-style global change-aware center index
# =============================================================================

@dataclass(frozen=True)
class ChangeSamplingCenter:
    record_index: int
    xyz: Tuple[float, float, float]
    event: int


class EmptyTemporalCropError(ValueError):
    pass


def _center_majority_index(
    coord: torch.Tensor,
    event: torch.Tensor,
    candidate: torch.Tensor,
    *,
    cell_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return coarse-cell mean XYZ and majority event label."""
    if cell_size <= 0:
        raise ValueError("Change center cell size must be > 0")

    mask = candidate.reshape(-1).bool().cpu().numpy()
    if not mask.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
        )

    xyz = coord.detach().cpu().numpy().astype(np.float64, copy=False)[mask]
    evt = event.detach().cpu().numpy().astype(np.int64, copy=False)[mask]

    # ME-CPT: round(pos / r), cluster, majority change label, mean position.
    quantized = np.rint(xyz / float(cell_size)).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    n_cells = int(inverse.max()) + 1

    cell_count = np.bincount(inverse, minlength=n_cells).astype(np.float64)
    centers = np.empty((n_cells, 3), dtype=np.float32)
    for d in range(3):
        centers[:, d] = (
            np.bincount(inverse, weights=xyz[:, d], minlength=n_cells)
            / np.maximum(cell_count, 1.0)
        ).astype(np.float32)

    class_count = np.bincount(
        inverse * PAIR_EVENT_NUM_CLASSES + evt,
        minlength=n_cells * PAIR_EVENT_NUM_CLASSES,
    ).reshape(n_cells, PAIR_EVENT_NUM_CLASSES)
    majority = class_count.argmax(axis=1).astype(np.int8)
    return centers, majority


# =============================================================================
# Dataset
# =============================================================================

class UnifiedPAIRDataset(Dataset):
    """
    Manifest-driven PAIR reader.

    Pure 3D train datasets use global change-aware center sampling. Dataset
    length remains the manifest length, so the existing multi-dataset scheduler
    keeps exactly the same per-epoch dataset quota.
    """

    def __init__(
        self,
        records: Union[Sequence[Dict[str, Any]], str, Path],
        spec: DatasetSpec,
        *,
        target_builder: Optional[TargetBuilder] = None,
    ):
        super().__init__()

        self.manifest_path = None
        self.dataset_root = None
        if isinstance(records, (str, Path)):
            self.manifest_path = Path(records).expanduser().resolve()
            self.dataset_root = self.manifest_path.parent.parent
            self.records = load_jsonl(self.manifest_path)
            name = self.manifest_path.stem.lower()
            self.split = name if name in {"train", "val", "test"} else None
        else:
            self.records = list(records)
            self.split = _infer_split_from_records(self.records)

        self.spec = spec
        validate_ignore_config(self.spec.class_names, self.spec.ignored_id)
        self.route = spec.route
        self.target_builder = target_builder
        self.prompt = build_default_prompt(spec)

        if not self.records:
            raise ValueError(f"Dataset {spec.name} contains no records: {self.manifest_path}")
        if self.spec.has_point and self.target_builder is not None:
            raise NotImplementedError(
                "Custom target_builder is not supported for point routes because "
                "point supervision must be cropped with the same per-point mask."
            )

        self._change_centers = None
        self._change_center_events = None
        self._change_center_records = None
        self._change_labels = None
        self._change_label_probabilities = None
        self._change_indices_by_label = {}

        if self.route == "3d" and self.split == "train":
            self._initialize_change_aware_sampling()

    def __len__(self):
        return len(self.records)

    def _path(self, value: Union[str, Path]) -> Path:
        return resolve_dataset_path(value, self.dataset_root)

    # -------------------------------------------------------------------------
    # Change-aware sampling index
    # -------------------------------------------------------------------------

    def _sampling_cache_path(self):
        if self.manifest_path is None:
            return None
        token = str(PAIR_POINT_WINDOW_SIZE_M).replace(".", "p")
        return self.manifest_path.parent / (
            f".{self.manifest_path.stem}_change_centers_{token}_v"
            f"{PAIR_CHANGE_CENTER_CACHE_VERSION}.npz"
        )

    def _sampling_fingerprint(self):
        h = hashlib.sha256()
        h.update(f"v={PAIR_CHANGE_CENTER_CACHE_VERSION}".encode())
        h.update(f"window={PAIR_POINT_WINDOW_SIZE_M}".encode())
        h.update(f"cell={PAIR_CHANGE_CENTER_CELL_M}".encode())
        h.update(f"ignored={self.spec.ignored_id}".encode())

        if self.manifest_path is not None:
            h.update(self.manifest_path.read_bytes())

        for record in self.records:
            for key in ("point_t1", "point_t2", "semantic_t1", "semantic_t2"):
                if key not in record:
                    continue
                path = self._path(record[key])
                stat = path.stat()
                h.update(str(path).encode())
                h.update(str(stat.st_size).encode())
                h.update(str(stat.st_mtime_ns).encode())
        return h.hexdigest()

    def _load_sampling_cache(self, path: Path, fingerprint: str) -> bool:
        if path is None or not path.is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as z:
                cached_fingerprint = str(np.asarray(z["fingerprint"]).item())
                if cached_fingerprint != fingerprint:
                    return False
                centers = np.asarray(z["centers"], dtype=np.float32)
                events = np.asarray(z["events"], dtype=np.int8)
                records = np.asarray(z["records"], dtype=np.int32)
        except Exception:
            return False

        if centers.ndim != 2 or centers.shape[1] != 3:
            return False
        if events.shape != (centers.shape[0],) or records.shape != (centers.shape[0],):
            return False
        if centers.shape[0] == 0:
            return False

        self._set_sampling_arrays(centers, events, records)
        return True

    def _save_sampling_cache(
        self,
        path: Path,
        fingerprint: str,
        centers: np.ndarray,
        events: np.ndarray,
        records: np.ndarray,
    ):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                fingerprint=np.asarray(fingerprint),
                centers=centers.astype(np.float32, copy=False),
                events=events.astype(np.int8, copy=False),
                records=records.astype(np.int32, copy=False),
            )
        os.replace(tmp, path)

    def _sampling_candidate_mask(self, supervision_t1, supervision_t2):
        """
        Select T2 points allowed to define coarse sampling centers.

        Normally event_valid=True is required. For temporally asymmetric labels,
        a nonzero T2 event with event_valid=False can still define a center when
        that same event is valid on T1 in the same source pair. This recovers
        NYC demolition/removed centers without treating invalid event=0
        placeholders as unchanged.
        """
        semantic2 = supervision_t2["semantic"]
        event1 = supervision_t1["event"]
        event2 = supervision_t2["event"]

        semantic_valid2 = torch.ones_like(semantic2, dtype=torch.bool)
        if self.spec.ignored_id is not None:
            semantic_valid2 &= semantic2 != int(self.spec.ignored_id)

        valid1 = supervision_t1.get("event_valid")
        valid2 = supervision_t2.get("event_valid")
        if valid1 is None:
            valid1 = torch.ones_like(event1, dtype=torch.bool)
        else:
            valid1 = valid1.bool()
        if valid2 is None:
            valid2 = torch.ones_like(event2, dtype=torch.bool)
        else:
            valid2 = valid2.bool()

        active_t1 = torch.unique(event1[valid1])
        asymmetric = torch.zeros_like(valid2)
        for event_id in active_t1.tolist():
            event_id = int(event_id)
            if event_id != 0:
                asymmetric |= (~valid2) & (event2 == event_id)

        return semantic_valid2 & (valid2 | asymmetric)

    def _build_sampling_arrays(self):
        all_centers = []
        all_events = []
        all_records = []
        center_count = 0
        event_counts = np.zeros(PAIR_EVENT_NUM_CLASSES, dtype=np.int64)

        progress = tqdm(
            enumerate(self.records),
            total=len(self.records),
            desc=f"{self.spec.name} build change centers",
            dynamic_ncols=True,
            leave=True,
            unit="scene",
        )

        for record_index, record in progress:
            if "point_t2" not in record or "semantic_t1" not in record or "semantic_t2" not in record:
                raise KeyError(
                    "3D change-aware sampling requires point_t2, semantic_t1 and "
                    "semantic_t2 in every train manifest record"
                )

            p2 = read_point_cloud(self._path(record["point_t2"]))
            s1 = read_point_supervision(self._path(record["semantic_t1"]))
            s2 = read_point_supervision(self._path(record["semantic_t2"]))

            if s2["event"].shape[0] != p2["coord"].shape[0]:
                raise ValueError(
                    f"{self.spec.name}/{record.get('id')}: T2 point/event length mismatch "
                    f"during change-center construction"
                )

            candidate = self._sampling_candidate_mask(s1, s2)
            centers, events = _center_majority_index(
                p2["coord"],
                s2["event"],
                candidate,
                cell_size=PAIR_CHANGE_CENTER_CELL_M,
            )
            if centers.shape[0] == 0:
                progress.set_postfix_str(f"centers={center_count:,}")
                continue

            all_centers.append(centers)
            all_events.append(events)
            all_records.append(
                np.full(centers.shape[0], record_index, dtype=np.int32)
            )

            center_count += int(centers.shape[0])
            event_counts += np.bincount(
                events.astype(np.int64, copy=False),
                minlength=PAIR_EVENT_NUM_CLASSES,
            )[:PAIR_EVENT_NUM_CLASSES]
            visible = ", ".join(
                f"{PAIR_EVENT_NAMES[i]}={int(event_counts[i]):,}"
                for i in range(PAIR_EVENT_NUM_CLASSES)
                if event_counts[i] > 0
            )
            progress.set_postfix_str(
                f"centers={center_count:,}" + (f" | {visible}" if visible else "")
            )

        if not all_centers:
            raise RuntimeError(
                f"{self.spec.name}: no valid change-aware sampling centers were built"
            )

        centers = np.concatenate(all_centers, axis=0)
        events = np.concatenate(all_events, axis=0)
        records = np.concatenate(all_records, axis=0)

        valid = (events >= 0) & (events < PAIR_EVENT_NUM_CLASSES)
        centers, events, records = centers[valid], events[valid], records[valid]
        if centers.shape[0] == 0:
            raise RuntimeError(
                f"{self.spec.name}: all change-aware centers had invalid event labels"
            )
        return centers, events, records

    def _set_sampling_arrays(self, centers, events, records):
        self._change_centers = torch.from_numpy(
            np.ascontiguousarray(centers.astype(np.float32, copy=False))
        )
        self._change_center_events = torch.from_numpy(
            np.ascontiguousarray(events.astype(np.int64, copy=False))
        ).long()
        self._change_center_records = torch.from_numpy(
            np.ascontiguousarray(records.astype(np.int64, copy=False))
        ).long()

        labels, counts = torch.unique(
            self._change_center_events, sorted=True, return_counts=True
        )
        counts_np = counts.double().numpy()

        # Same weighting rule used by ME-CPT:
        # sqrt(mean_count / class_count), then normalize.
        weights = np.sqrt(counts_np.mean() / counts_np)
        probabilities = weights / weights.sum()

        self._change_labels = labels.long()
        self._change_label_probabilities = torch.as_tensor(
            probabilities, dtype=torch.double
        )
        self._change_indices_by_label = {
            int(label.item()): torch.nonzero(
                self._change_center_events == label, as_tuple=False
            ).flatten()
            for label in labels
        }

    def _initialize_change_aware_sampling(self):
        fingerprint = self._sampling_fingerprint()
        cache = self._sampling_cache_path()

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0

        def build_and_cache():
            print(
                f"{self.spec.name}: change-aware crop cache is missing or stale."
            )
            print(
                f"{self.spec.name}: scanning {len(self.records):,} training scenes to "
                "build ME-CPT-style sampling centers. This is a first-run preprocessing "
                "step; later runs will load the cached centers directly."
            )
            if distributed:
                print(
                    f"{self.spec.name}: rank 0 is building the cache; other ranks wait "
                    "at the synchronization barrier."
                )
            start = time.time()
            centers, events, records = self._build_sampling_arrays()
            self._set_sampling_arrays(centers, events, records)
            self._save_sampling_cache(
                cache, fingerprint, centers, events, records
            )
            elapsed = time.time() - start
            print(
                f"{self.spec.name}: change-aware crop cache built in {elapsed:.1f}s -> "
                f"{cache}"
            )

        if distributed:
            if rank == 0:
                loaded = self._load_sampling_cache(cache, fingerprint)
                if loaded:
                    print(
                        f"{self.spec.name}: loaded cached change-aware sampling centers "
                        f"from {cache}"
                    )
                else:
                    build_and_cache()
            dist.barrier()
            if rank != 0:
                if not self._load_sampling_cache(cache, fingerprint):
                    raise RuntimeError(
                        f"{self.spec.name}: rank 0 did not produce a valid "
                        f"change-aware sampling cache: {cache}"
                    )
        else:
            loaded = self._load_sampling_cache(cache, fingerprint)
            if loaded:
                print(
                    f"{self.spec.name}: loaded cached change-aware sampling centers "
                    f"from {cache}"
                )
            else:
                build_and_cache()

        if rank == 0:
            counts = {
                PAIR_EVENT_NAMES.get(int(label), str(int(label))): int(
                    self._change_indices_by_label[int(label)].numel()
                )
                for label in self._change_labels.tolist()
            }
            probs = {
                PAIR_EVENT_NAMES.get(int(label), str(int(label))): float(prob)
                for label, prob in zip(
                    self._change_labels.tolist(),
                    self._change_label_probabilities.tolist(),
                )
            }
            print(
                f"{self.spec.name}: ME-CPT-style change-aware crop | "
                f"centers={self._change_centers.shape[0]:,} | "
                f"cell={PAIR_CHANGE_CENTER_CELL_M:.2f}m | "
                f"window={PAIR_POINT_WINDOW_SIZE_M:.1f}m"
            )
            print(f"  center counts: {counts}")
            print(
                "  sampling prob: "
                + str({k: round(v, 4) for k, v in probs.items()})
            )

    def _sample_change_center(self) -> ChangeSamplingCenter:
        label_slot = int(
            torch.multinomial(
                self._change_label_probabilities, 1, replacement=True
            ).item()
        )
        event = int(self._change_labels[label_slot].item())
        candidates = self._change_indices_by_label[event]
        center_index = int(
            candidates[int(torch.randint(candidates.numel(), (1,)).item())].item()
        )

        xyz = self._change_centers[center_index]
        return ChangeSamplingCenter(
            record_index=int(self._change_center_records[center_index].item()),
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            event=event,
        )

    # -------------------------------------------------------------------------
    # 2D
    # -------------------------------------------------------------------------

    def _load_2d(self, record):
        im1, geo1 = read_image(self._path(record["image_t1"]))
        im2, geo2 = read_image(self._path(record["image_t2"]))

        if not same_geo_grid(geo1, geo2):
            raise ValueError(
                "T1/T2 rasters are not on one geospatial grid. "
                "PAIR data preparation must align CRS/GSD/affine first."
            )
        if im1.shape[-2:] != im2.shape[-2:]:
            raise ValueError(
                f"T1/T2 image sizes differ: {tuple(im1.shape[-2:])} vs "
                f"{tuple(im2.shape[-2:])}. Prepare aligned pairs before training."
            )
        return {
            "images_t1": im1,
            "images_t2": im2,
            "geo_t1": geo1,
            "geo_t2": geo2,
        }

    # -------------------------------------------------------------------------
    # 3D
    # -------------------------------------------------------------------------

    def _load_point_supervision_pair(self, record, p1, p2):
        if self.spec.label_mode != "semantic_pair":
            raise NotImplementedError(
                "The current point route supports semantic_pair supervision with "
                "per-epoch semantic/event bundles."
            )
        if "semantic_t1" not in record or "semantic_t2" not in record:
            raise KeyError(
                "3D semantic_pair manifest requires semantic_t1 and semantic_t2 paths"
            )

        t1 = read_point_supervision(self._path(record["semantic_t1"]))
        t2 = read_point_supervision(self._path(record["semantic_t2"]))

        validate_semantic_label_values(
            t1["semantic"],
            self.spec.class_names,
            ignored_id=self.spec.ignored_id,
            label_name=f"{self.spec.name}/{record.get('id')} semantic_t1",
        )
        validate_semantic_label_values(
            t2["semantic"],
            self.spec.class_names,
            ignored_id=self.spec.ignored_id,
            label_name=f"{self.spec.name}/{record.get('id')} semantic_t2",
        )

        n1, n2 = p1["coord"].shape[0], p2["coord"].shape[0]
        if t1["semantic"].shape[0] != n1:
            raise ValueError(
                f"{self.spec.name}/{record.get('id')}: T1 point/supervision "
                f"length mismatch: points={n1}, labels={t1['semantic'].shape[0]}"
            )
        if t2["semantic"].shape[0] != n2:
            raise ValueError(
                f"{self.spec.name}/{record.get('id')}: T2 point/supervision "
                f"length mismatch: points={n2}, labels={t2['semantic'].shape[0]}"
            )
        return t1, t2

    def _choose_point_bounds(
        self,
        record,
        p1,
        p2,
        *,
        reference_bounds=None,
        sampling_center=None,
        force_joint_random=False,
    ):
        """
        Common XY crop priority:
          1. 2D reference footprint
          2. explicit manifest bounds
          3. change-aware center for pure 3D train
          4. jointly occupied random fallback for pure 3D train
          5. full source if <=51.2m
        """
        if reference_bounds is not None:
            if len(reference_bounds) != 4:
                raise ValueError("reference_bounds must be [xmin,ymin,xmax,ymax]")
            return tuple(float(v) for v in reference_bounds)

        explicit = record.get("bounds")
        if explicit is not None:
            if len(explicit) != 4:
                raise ValueError("bounds must be [xmin,ymin,xmax,ymax]")
            return tuple(float(v) for v in explicit)

        if self.route == "2d3d":
            raise ValueError(
                "2D+3D point cropping requires a world-coordinate image footprint "
                "or manifest bounds. PAIR will not independently random-crop points "
                "away from paired images."
            )

        if sampling_center is not None:
            return centered_pair_window(
                sampling_center[:2], size=PAIR_POINT_WINDOW_SIZE_M
            )

        xmin, ymin, xmax, ymax = point_xy_bounds(p1, p2)
        extent_x, extent_y = xmax - xmin, ymax - ymin
        if extent_x <= PAIR_POINT_WINDOW_SIZE_M and extent_y <= PAIR_POINT_WINDOW_SIZE_M:
            return None

        if self.split == "train":
            if force_joint_random or self._change_centers is None:
                return make_joint_occupied_pair_window(
                    p1, p2, size=PAIR_POINT_WINDOW_SIZE_M
                )
            return make_random_pair_window(
                p1, p2, size=PAIR_POINT_WINDOW_SIZE_M
            )

        if self.split in {"val", "test"}:
            raise NotImplementedError(
                f"{self.spec.name}/{record.get('id')}: evaluation source extent "
                f"{extent_x:.2f}m x {extent_y:.2f}m exceeds PAIR's "
                f"{PAIR_POINT_WINDOW_SIZE_M:.1f}m window. Validation/test must use "
                "deterministic tiling; random evaluation cropping is forbidden."
            )

        raise RuntimeError(
            f"{self.spec.name}/{record.get('id')}: cannot infer train/val/test split "
            "for an oversized 3D source."
        )

    def _load_3d(
        self,
        record,
        *,
        reference_bounds=None,
        sampling_center=None,
        sampling_event=None,
        force_joint_random=False,
    ):
        p1 = read_point_cloud(self._path(record["point_t1"]))
        p2 = read_point_cloud(self._path(record["point_t2"]))
        if p1["coord"].shape[0] == 0 or p2["coord"].shape[0] == 0:
            raise ValueError("Empty source T1/T2 point cloud")

        supervision_t1, supervision_t2 = self._load_point_supervision_pair(
            record, p1, p2
        )
        bounds = self._choose_point_bounds(
            record,
            p1,
            p2,
            reference_bounds=reference_bounds,
            sampling_center=sampling_center,
            force_joint_random=force_joint_random,
        )

        source_n1 = int(p1["coord"].shape[0])
        source_n2 = int(p2["coord"].shape[0])

        if bounds is not None:
            p1 = crop_points_xy(p1, bounds)
            p2 = crop_points_xy(p2, bounds)
            supervision_t1 = crop_point_supervision(
                supervision_t1, p1["_crop_mask"]
            )
            supervision_t2 = crop_point_supervision(
                supervision_t2, p2["_crop_mask"]
            )
        else:
            p1["_source_indices"] = torch.arange(source_n1, dtype=torch.long)
            p2["_source_indices"] = torch.arange(source_n2, dtype=torch.long)

        n1 = int(p1["coord"].shape[0])
        n2 = int(p2["coord"].shape[0])
        if n1 == 0 or n2 == 0:
            error = EmptyTemporalCropError if self.split == "train" else ValueError
            raise error(
                f"{self.spec.name}/{record.get('id')}: empty T1/T2 point cloud after "
                f"common spatial crop; bounds={bounds}, N1={n1}, N2={n2}"
            )

        if (
            supervision_t1["semantic"].shape[0] != n1
            or supervision_t1["event"].shape[0] != n1
        ):
            raise RuntimeError("T1 crop broke point/label topology")
        if (
            supervision_t2["semantic"].shape[0] != n2
            or supervision_t2["event"].shape[0] != n2
        ):
            raise RuntimeError("T2 crop broke point/label topology")

        common_z0 = min(
            float(p1["coord"][:, 2].min().item()),
            float(p2["coord"][:, 2].min().item()),
        )
        if bounds is not None:
            shared_xyz_origin = float(bounds[0]), float(bounds[1]), common_z0
        else:
            common_min = torch.minimum(
                p1["coord"].amin(dim=0), p2["coord"].amin(dim=0)
            )
            shared_xyz_origin = tuple(float(v) for v in common_min.tolist())

        target = build_bitemporal_point_target(
            supervision_t1,
            supervision_t2,
            class_names=self.spec.class_names,
            ignored_id=self.spec.ignored_id,
        )

        return {
            "point_dict_t1": make_point_dict(
                p1, shared_xyz_origin=shared_xyz_origin
            ),
            "point_dict_t2": make_point_dict(
                p2, shared_xyz_origin=shared_xyz_origin
            ),
            "point_crs_t1": p1.get("crs"),
            "point_crs_t2": p2.get("crs"),
            "point_source_indices_t1": p1["_source_indices"],
            "point_source_indices_t2": p2["_source_indices"],
            "point_source_count_t1": source_n1,
            "point_source_count_t2": source_n2,
            "bounds": bounds,
            "shared_xyz_origin": shared_xyz_origin,
            "sampling_event": sampling_event,
            "sampling_center": sampling_center,
            "_point_target": target,
        }

    # -------------------------------------------------------------------------
    # Existing 2D target path
    # -------------------------------------------------------------------------

    def _load_targets(self, record):
        if self.target_builder is not None:
            return self.target_builder(record, self.spec)

        s1 = (
            read_label_array(self._path(record["semantic_t1"]))
            if "semantic_t1" in record
            else None
        )
        s2 = (
            read_label_array(self._path(record["semantic_t2"]))
            if "semantic_t2" in record
            else None
        )
        ch = (
            read_label_array(self._path(record["change"]))
            if "change" in record
            else None
        )

        target = build_canonical_target(
            label_mode=self.spec.label_mode,
            semantic_t1=s1,
            semantic_t2=s2,
            change=ch,
            class_names=self.spec.class_names,
            ignored_id=self.spec.ignored_id,
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

    # -------------------------------------------------------------------------
    # Item assembly
    # -------------------------------------------------------------------------

    def _getitem_record(
        self,
        record_index,
        *,
        sampling_center=None,
        sampling_event=None,
        force_joint_random=False,
    ):
        record = self.records[int(record_index)]
        sample: Dict[str, Any] = {
            "sample_id": record.get("id", str(record_index)),
            "dataset_name": self.spec.name,
            "route": self.route,
            "task_mode": self.route,
            "prompt": self.prompt,
            "class_names": dict(self.spec.class_names),
        }

        if self.spec.has_image:
            sample.update(self._load_2d(record))

        point_target = None
        if self.spec.has_point:
            reference_bounds = None
            if self.route == "2d3d":
                geo_t1 = sample.get("geo_t1")
                geo_t2 = sample.get("geo_t2")
                bounds_t1 = None if geo_t1 is None else geo_t1.get("bounds")
                bounds_t2 = None if geo_t2 is None else geo_t2.get("bounds")

                if bounds_t1 is not None and bounds_t2 is not None:
                    if not np.allclose(
                        np.asarray(bounds_t1, dtype=np.float64),
                        np.asarray(bounds_t2, dtype=np.float64),
                        atol=1e-6,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            "T1/T2 image world bounds differ in a 2D+3D sample"
                        )
                    reference_bounds = tuple(float(v) for v in bounds_t1)
                elif record.get("bounds") is not None:
                    reference_bounds = tuple(float(v) for v in record["bounds"])

            point_part = self._load_3d(
                record,
                reference_bounds=reference_bounds,
                sampling_center=sampling_center,
                sampling_event=sampling_event,
                force_joint_random=force_joint_random,
            )
            point_target = point_part.pop("_point_target")
            sample.update(point_part)

        sample["target"] = point_target if self.spec.has_point else self._load_targets(record)

        if self.spec.has_image and not self.spec.has_point:
            h, w = sample["images_t1"].shape[-2:]
            for key in ("change", "semantic_t1", "semantic_t2"):
                target = sample["target"][key]
                if target.ndim >= 2 and tuple(target.shape[-2:]) != (h, w):
                    raise ValueError(
                        f"{self.spec.name}/{sample['sample_id']}: {key} shape "
                        f"{tuple(target.shape)} does not match image {(h, w)}"
                    )

        if self.spec.has_point:
            n1 = sample["point_dict_t1"]["coord"].shape[0]
            n2 = sample["point_dict_t2"]["coord"].shape[0]

            for key in ("semantic_t1", "event_t1"):
                if sample["target"][key].shape[0] != n1:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: {key} length "
                        f"{sample['target'][key].shape[0]} does not match T1 points {n1}"
                    )
            for key in ("semantic_t2", "event_t2"):
                if sample["target"][key].shape[0] != n2:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: {key} length "
                        f"{sample['target'][key].shape[0]} does not match T2 points {n2}"
                    )
            for key in ("semantic_valid_t1", "event_valid_t1"):
                if key in sample["target"] and sample["target"][key].shape[0] != n1:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: {key} length mismatch"
                    )
            for key in ("semantic_valid_t2", "event_valid_t2"):
                if key in sample["target"] and sample["target"][key].shape[0] != n2:
                    raise RuntimeError(
                        f"{self.spec.name}/{sample['sample_id']}: {key} length mismatch"
                    )

        if self.route == "2d3d":
            raster_crs = (
                sample["geo_t1"].get("crs")
                if sample.get("geo_t1") is not None
                else None
            )
            known_crs = [
                c
                for c in (
                    raster_crs,
                    sample.get("point_crs_t1"),
                    sample.get("point_crs_t2"),
                )
                if c
            ]
            if known_crs and any(c != known_crs[0] for c in known_crs[1:]):
                raise ValueError(
                    "2D+3D sample contains mismatched CRS metadata. "
                    "Prepare all modalities in one CRS before training."
                )

        sample["spatial_meta"] = {
            "bounds": sample.get("bounds", record.get("bounds")),
            "raster_geo_t1": sample.get("geo_t1"),
            "raster_geo_t2": sample.get("geo_t2"),
            "point_crs_t1": sample.get("point_crs_t1"),
            "point_crs_t2": sample.get("point_crs_t2"),
            "point_window_size_m": PAIR_POINT_WINDOW_SIZE_M,
            "shared_xyz_origin": sample.get("shared_xyz_origin"),
            "point_source_count_t1": sample.get("point_source_count_t1"),
            "point_source_count_t2": sample.get("point_source_count_t2"),
            "sampling_mode": (
                "change_aware"
                if sampling_center is not None
                else ("joint_random_fallback" if force_joint_random else None)
            ),
            "sampling_event": sample.get("sampling_event"),
            "sampling_event_name": (
                PAIR_EVENT_NAMES.get(int(sample["sampling_event"]))
                if sample.get("sampling_event") is not None
                else None
            ),
            "sampling_center": sample.get("sampling_center"),
            "change_center_cell_m": (
                PAIR_CHANGE_CENTER_CELL_M
                if self._change_centers is not None
                else None
            ),
        }
        return sample

    def __getitem__(self, index):
        # Existing 2D / 2D3D / val / test behavior.
        if self._change_centers is None:
            return self._getitem_record(index)

        # ME-CPT-style global change-aware sampling. The requested DataLoader
        # index does not select the scene; it only contributes to epoch length.
        # This mirrors ME-CPT's getitemrandom() while preserving PAIR's existing
        # multi-dataset update quota.
        last_error = None
        for _ in range(PAIR_CHANGE_CROP_MAX_TRIES):
            center = self._sample_change_center()
            try:
                return self._getitem_record(
                    center.record_index,
                    sampling_center=center.xyz,
                    sampling_event=center.event,
                )
            except EmptyTemporalCropError as exc:
                last_error = exc

        # Defensive fallback: use the DataLoader-selected record and find a
        # jointly occupied random square rather than crashing because one epoch
        # is empty.
        try:
            return self._getitem_record(
                int(index) % len(self.records),
                force_joint_random=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{self.spec.name}: change-aware crop failed "
                f"{PAIR_CHANGE_CROP_MAX_TRIES} times and joint fallback also failed. "
                f"Last change-aware error: {last_error}"
            ) from exc


# =============================================================================
# Lightweight self-test
# =============================================================================

def _self_test():
    classes = {0: "ground", 1: "building", 2: "vegetation", 3: "clutter"}

    # Existing 2D target behavior.
    s1 = torch.tensor([[0, 1], [2, 3]])
    s2 = torch.tensor([[0, 2], [2, 3]])
    out = build_canonical_target(
        label_mode="semantic_pair",
        semantic_t1=s1,
        semantic_t2=s2,
        class_names=classes,
    )
    assert torch.equal(out.change, torch.tensor([[0, 1], [0, 0]]))
    assert out.change_valid is None

    # Semantic ignore remains explicit.
    out = build_canonical_target(
        label_mode="semantic_pair",
        semantic_t1=torch.tensor([0, -1, 1]),
        semantic_t2=torch.tensor([0, 1, 1]),
        class_names={0: "ground", 1: "building"},
        ignored_id=-1,
    )
    assert out.semantic_valid_t1.tolist() == [True, False, True]
    assert out.change_valid.tolist() == [True, False, True]

    # 3D semantic/event topology and independence of valid masks.
    target = build_bitemporal_point_target(
        {
            "semantic": torch.tensor([0, -1, 2]),
            "event": torch.tensor([0, 2, 0]),
            "event_valid": torch.tensor([True, True, False]),
        },
        {
            "semantic": torch.tensor([0, 3]),
            "event": torch.tensor([1, 0]),
            "event_valid": torch.tensor([True, True]),
        },
        class_names=classes,
        ignored_id=-1,
    )
    assert target["semantic_valid_t1"].tolist() == [True, False, True]
    assert target["event_valid_t1"].tolist() == [True, True, False]
    assert target["event_t1"].tolist() == [0, 2, 0]
    assert target["event_t2"].tolist() == [1, 0]
    assert "change_t1" not in target and "change_t2" not in target

    # Optional point attributes remain optional.
    point = {
        "coord": torch.tensor(
            [[10.0, 20.0, 1.0], [11.0, 21.0, 2.0]]
        ),
        "rgb": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ),
        "intensity": torch.tensor([[7.0], [8.0]]),
    }
    packed = make_point_dict(
        point, shared_xyz_origin=(10.0, 20.0, 1.0)
    )
    assert set(packed) == {"coord", "rgb", "intensity"}
    assert torch.equal(
        packed["coord"],
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )

    geometry_only = make_point_dict(
        {"coord": point["coord"]},
        shared_xyz_origin=(10.0, 20.0, 1.0),
    )
    assert set(geometry_only) == {"coord"}

    # Center majority logic: coarse cells preserve event majority.
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.1, 0.0],
            [10.2, 0.1, 0.0],
        ]
    )
    event = torch.tensor([0, 0, 1, 1, 2])
    centers, labels = _center_majority_index(
        coord,
        event,
        torch.ones(5, dtype=torch.bool),
        cell_size=1.0,
    )
    assert centers.shape[0] == 2
    assert sorted(labels.tolist()) == [0, 1]

    # Joint fallback always contains both epochs.
    p1 = {"coord": torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]])}
    p2 = {"coord": torch.tensor([[0.1, 0.1, 0.0], [200.0, 200.0, 0.0]])}
    bounds = make_joint_occupied_pair_window(p1, p2, size=10.0, shift_tries=32)
    c1 = crop_points_xy(p1, bounds)["coord"]
    c2 = crop_points_xy(p2, bounds)["coord"]
    assert c1.shape[0] > 0 and c2.shape[0] > 0

    assert DatasetSpec.__dataclass_fields__["ignored_id"].default is None
    assert PAIR_EVENT_NUM_CLASSES == 6
    assert PAIR_POINT_WINDOW_SIZE_M == 51.2
    assert abs(PAIR_CHANGE_CENTER_CELL_M - 2.56) < 1e-9

    print("pair_dataset.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
