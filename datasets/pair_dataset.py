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

Canonical labels inside PAIR:
    change      : 0 unchanged, 1 changed, -100 ignore
    semantic    : raw dataset class ID, -100 ignore
    UNKNOWN=-1  : internal only, meaning semantic class is not supervised

For binary datasets, physical change-mask values are inferred from class_names.
Example:
    {"0": "unchanged", "255": "changed"}
means disk labels 0/255 are mapped internally to 0/1.

The training JSON does not repeat modality, manifest path, label mode, image
size, changed/unchanged values, ignore values, alignment policy, or prompt.
Those are inferred from the directory, class_names, or fixed internal protocol.
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


UNKNOWN_CLASS_ID = -1
IGNORE_CLASS_ID = -100
VALID_MODALITIES = {"image", "point"}


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
    change: torch.Tensor
    semantic_t1: torch.Tensor
    semantic_t2: torch.Tensor
    change_valid: torch.Tensor
    semantic_valid_t1: torch.Tensor
    semantic_valid_t2: torch.Tensor


def _long(x):
    if torch.is_tensor(x):
        return x.long()
    return torch.as_tensor(np.asarray(x), dtype=torch.long)


def _validate_canonical_change_values(raw: torch.Tensor):
    valid_values = (raw == 0) | (raw == 1) | (raw == IGNORE_CLASS_ID)
    if not valid_values.all():
        bad = torch.unique(raw[~valid_values]).cpu().tolist()
        raise ValueError(
            "Canonical PAIR change targets must use only "
            f"0=unchanged, 1=changed, {IGNORE_CLASS_ID}=ignore; found {bad}"
        )


def _normalize_binary_change_values(
    raw: torch.Tensor,
    class_names: Dict[int, str],
) -> torch.Tensor:
    """
    Convert physical binary labels to PAIR internal 0/1 using class_names.
    """
    unchanged_raw_id, changed_raw_id = infer_binary_class_ids(class_names)

    valid_values = (
        (raw == unchanged_raw_id)
        | (raw == changed_raw_id)
        | (raw == IGNORE_CLASS_ID)
    )
    if not valid_values.all():
        bad = torch.unique(raw[~valid_values]).cpu().tolist()
        raise ValueError(
            f"Binary change target contains undeclared raw IDs {bad}; "
            f"class_names declares unchanged={unchanged_raw_id}, "
            f"changed={changed_raw_id}."
        )

    out = torch.full_like(raw, IGNORE_CLASS_ID)
    out[raw == unchanged_raw_id] = 0
    out[raw == changed_raw_id] = 1
    return out


def build_canonical_target(
    *,
    label_mode: str,
    semantic_t1=None,
    semantic_t2=None,
    change=None,
    class_names: Optional[Dict[int, str]] = None,
) -> CanonicalChangeTarget:
    """
    Convert PAIR-standard supervision into the common semantic-change target.

    No dataset-specific raw-value mapping happens here. Public datasets should
    be converted to PAIR's canonical 0/1/-100 convention during preparation.
    """
    mode = str(label_mode).lower().strip()
    if mode not in {"semantic_pair", "post_semantic", "binary"}:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    if mode == "semantic_pair":
        if semantic_t1 is None or semantic_t2 is None:
            raise ValueError("semantic_pair requires semantic_t1 and semantic_t2")

        s1 = _long(semantic_t1)
        s2 = _long(semantic_t2)
        if s1.shape != s2.shape:
            raise ValueError(
                f"semantic_t1 and semantic_t2 shapes differ: "
                f"{tuple(s1.shape)} vs {tuple(s2.shape)}"
            )

        semantic_invalid = (s1 == IGNORE_CLASS_ID) | (s2 == IGNORE_CLASS_ID)

        if change is None:
            ch = (s1 != s2).long()
            change_invalid = semantic_invalid.clone()
        else:
            raw = _long(change)
            if raw.shape != s1.shape:
                raise ValueError("change mask must match semantic label shape")
            _validate_canonical_change_values(raw)
            ch = raw.clone()
            change_invalid = raw == IGNORE_CLASS_ID

        invalid = semantic_invalid | change_invalid
        s1 = s1.clone()
        s2 = s2.clone()
        ch = ch.clone()
        s1[invalid] = IGNORE_CLASS_ID
        s2[invalid] = IGNORE_CLASS_ID
        ch[invalid] = IGNORE_CLASS_ID

        return CanonicalChangeTarget(
            change=ch,
            semantic_t1=s1,
            semantic_t2=s2,
            change_valid=~invalid,
            semantic_valid_t1=~invalid,
            semantic_valid_t2=~invalid,
        )

    if change is None:
        raise ValueError(f"{mode} requires change supervision")

    raw = _long(change)

    if mode == "binary":
        if class_names is None:
            raise ValueError("binary target normalization requires class_names")
        ch = _normalize_binary_change_values(raw, class_names)
        invalid = ch == IGNORE_CLASS_ID
        s1 = torch.full_like(raw, UNKNOWN_CLASS_ID)
        s2 = torch.full_like(raw, UNKNOWN_CLASS_ID)
        s1[invalid] = IGNORE_CLASS_ID
        s2[invalid] = IGNORE_CLASS_ID
        sem_valid = torch.zeros_like(raw, dtype=torch.bool)

        return CanonicalChangeTarget(
            change=ch,
            semantic_t1=s1,
            semantic_t2=s2,
            change_valid=~invalid,
            semantic_valid_t1=sem_valid,
            semantic_valid_t2=sem_valid.clone(),
        )

    # post_semantic class_names describe semantic classes, so its explicit
    # change target remains canonical 0/1 internally.
    _validate_canonical_change_values(raw)
    invalid = raw == IGNORE_CLASS_ID
    ch = raw.clone()

    # post_semantic = Unknown -> B
    if semantic_t2 is None:
        raise ValueError("post_semantic requires semantic_t2")

    s2_src = _long(semantic_t2)
    if s2_src.shape != raw.shape:
        raise ValueError("semantic_t2 must match change mask shape")

    sem2_invalid = s2_src == IGNORE_CLASS_ID
    s1 = torch.full_like(raw, UNKNOWN_CLASS_ID)
    s2 = s2_src.clone()

    sem2_valid = (~invalid) & (~sem2_invalid)
    s1[invalid] = IGNORE_CLASS_ID
    s2[invalid] = IGNORE_CLASS_ID
    s2[(~invalid) & sem2_invalid] = UNKNOWN_CLASS_ID

    return CanonicalChangeTarget(
        change=ch,
        semantic_t1=s1,
        semantic_t2=s2,
        change_valid=~invalid,
        semantic_valid_t1=torch.zeros_like(raw, dtype=torch.bool),
        semantic_valid_t2=sem2_valid,
    )


def build_default_prompt(spec: DatasetSpec) -> str:
    text = (
        "Perform semantic change detection between Time 1 and Time 2. "
        "Identify unchanged and changed regions."
    )

    if spec.label_mode == "semantic_pair":
        text += (
            " For changed regions, infer the semantic class before and after change."
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


# -----------------------------------------------------------------------------
# Generic file readers
# -----------------------------------------------------------------------------

def read_label_array(path: Union[str, Path]) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        keys = list(z.keys())
        if len(keys) != 1:
            raise ValueError(
                f"{path} has multiple arrays {keys}; prepare one label array per file"
            )
        arr = z[keys[0]]
    else:
        arr = np.asarray(Image.open(path))

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Expected a single-channel label, got {arr.shape}: {path}"
            )

    return torch.as_tensor(np.array(arr, copy=True), dtype=torch.long)


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
        denom = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / denom
    else:
        arr = arr.astype(np.float32)
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if vmax > 1.5:
            arr /= 255.0 if vmax <= 255.0 else 65535.0

    tensor = torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(arr, -1, 0))
    ).float()
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


PointFeatureBuilder = Callable[[Dict[str, np.ndarray]], np.ndarray]
TargetBuilder = Callable[[Dict[str, Any], DatasetSpec], Dict[str, Any]]


def _read_las(path: Path):
    try:
        import laspy
    except ImportError as exc:
        raise ImportError("LAS/LAZ reading requires laspy") from exc

    las = laspy.read(path)
    attrs: Dict[str, Any] = {
        "coord": np.stack(
            [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)],
            axis=1,
        ).astype(np.float32)
    }

    available = set(las.point_format.dimension_names)
    for name in (
        "intensity", "red", "green", "blue", "nir", "classification",
        "return_number", "number_of_returns",
    ):
        if name in available:
            attrs[name] = np.asarray(getattr(las, name))

    try:
        crs = las.header.parse_crs()
        attrs["_crs"] = None if crs is None else str(crs)
    except Exception:
        attrs["_crs"] = None

    return attrs


def read_point_cloud(
    path: Union[str, Path],
    *,
    feature_builder: Optional[PointFeatureBuilder] = None,
) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    crs = None

    if suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        if "coord" not in z or "feat" not in z:
            raise ValueError(
                f"{path} must contain arrays named 'coord' [N,3] and 'feat' [N,C]"
            )
        coord = np.asarray(z["coord"], dtype=np.float32)
        feat = np.asarray(z["feat"], dtype=np.float32)
        if "crs" in z:
            crs = str(z["crs"])

    elif suffix == ".npy":
        arr = np.asarray(np.load(path, allow_pickle=False))
        if arr.ndim != 2 or arr.shape[1] <= 3:
            raise ValueError(f"{path} must be [N,3+C]")
        coord = arr[:, :3].astype(np.float32)
        feat = arr[:, 3:].astype(np.float32)

    elif suffix in {".las", ".laz"}:
        attrs = _read_las(path)
        coord = attrs["coord"]
        crs = attrs.get("_crs")
        if feature_builder is None:
            raise ValueError(
                "LAS/LAZ input requires a feature_builder. For the final PAIR "
                "standard, prefer preparing point files as NPZ with coord+feat."
            )
        feat = np.asarray(feature_builder(attrs), dtype=np.float32)

    else:
        raise ValueError(f"Unsupported point-cloud format: {path}")

    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord must be [N,3], got {coord.shape}")
    if feat.ndim != 2 or feat.shape[0] != coord.shape[0]:
        raise ValueError(
            f"feat must be [N,C] matching coord, got coord={coord.shape}, feat={feat.shape}"
        )

    return {
        "coord": torch.from_numpy(np.ascontiguousarray(coord)).float(),
        "feat": torch.from_numpy(np.ascontiguousarray(feat)).float(),
        "crs": crs,
    }


def crop_points_xy(point_data: Dict[str, Any], bounds):
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    c = point_data["coord"]
    keep = (
        (c[:, 0] >= xmin) & (c[:, 0] < xmax)
        & (c[:, 1] >= ymin) & (c[:, 1] < ymax)
    )
    return {
        **point_data,
        "coord": c[keep],
        "feat": point_data["feat"][keep],
        "_crop_mask": keep,
    }


def make_ptv3_dict(
    point_data: Dict[str, Any],
    *,
    grid_size: float,
    shared_xyz_origin=None,
) -> Dict[str, torch.Tensor]:
    if grid_size <= 0:
        raise ValueError("point grid size must be > 0")

    coord = point_data["coord"].clone()
    if shared_xyz_origin is None:
        shared_xyz_origin = tuple(
            float(v) for v in coord.amin(dim=0).tolist()
        )

    origin = torch.tensor(
        shared_xyz_origin, dtype=coord.dtype, device=coord.device
    ).view(1, 3)

    coord = coord - origin
    grid_coord = torch.floor(coord / float(grid_size)).long()

    return {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": point_data["feat"],
        "batch": torch.zeros(coord.shape[0], dtype=torch.long),
    }


# -----------------------------------------------------------------------------
# Manifest / dataset
# -----------------------------------------------------------------------------

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
    path: Union[str, Path],
    dataset_root: Optional[Union[str, Path]],
) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if dataset_root is None:
        raise ValueError(f"Relative path {path} requires dataset_root")
    return Path(dataset_root).expanduser().resolve() / path


class UnifiedPAIRDataset(Dataset):
    """
    Manifest-driven reader for an already prepared PAIR-standard dataset.

    The directory layout decides route and supervision; manifest records only
    map sample IDs to concrete files.
    """

    def __init__(
        self,
        records: Union[Sequence[Dict[str, Any]], str, Path],
        spec: DatasetSpec,
        *,
        point_feature_builder: Optional[PointFeatureBuilder] = None,
        target_builder: Optional[TargetBuilder] = None,
    ):
        super().__init__()

        self.manifest_path = None
        self.dataset_root = None

        if isinstance(records, (str, Path)):
            self.manifest_path = Path(records).expanduser().resolve()
            self.dataset_root = self.manifest_path.parent.parent
            self.records = load_jsonl(self.manifest_path)
        else:
            self.records = list(records)

        self.spec = spec
        self.route = spec.route
        self.point_feature_builder = point_feature_builder
        self.target_builder = target_builder
        self.prompt = build_default_prompt(spec)

        if not self.records:
            raise ValueError(
                f"Dataset {spec.name} contains no records: {self.manifest_path}"
            )

    def __len__(self):
        return len(self.records)

    def _path(self, value: Union[str, Path]) -> Path:
        return resolve_dataset_path(value, self.dataset_root)

    def _load_2d(self, record):
        im1, geo1 = read_image(self._path(record["image_t1"]))
        im2, geo2 = read_image(self._path(record["image_t2"]))

        # PAIR-standard data must already be temporally aligned.
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

    def _load_3d(self, record):
        p1 = read_point_cloud(
            self._path(record["point_t1"]),
            feature_builder=self.point_feature_builder,
        )
        p2 = read_point_cloud(
            self._path(record["point_t2"]),
            feature_builder=self.point_feature_builder,
        )

        bounds = record.get("bounds")
        if bounds is not None:
            if len(bounds) != 4:
                raise ValueError("bounds must be [xmin,ymin,xmax,ymax]")
            p1 = crop_points_xy(p1, bounds)
            p2 = crop_points_xy(p2, bounds)

        if p1["coord"].shape[0] == 0 or p2["coord"].shape[0] == 0:
            raise ValueError("Empty T1/T2 point cloud after common spatial crop")

        common_z0 = min(
            float(p1["coord"][:, 2].min().item()),
            float(p2["coord"][:, 2].min().item()),
        )

        if bounds is not None:
            shared_xyz_origin = (
                float(bounds[0]), float(bounds[1]), common_z0
            )
        else:
            common_min = torch.minimum(
                p1["coord"].amin(dim=0),
                p2["coord"].amin(dim=0),
            )
            shared_xyz_origin = tuple(
                float(v) for v in common_min.tolist()
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
            "point_crs_t1": p1.get("crs"),
            "point_crs_t2": p2.get("crs"),
            "bounds": bounds,
            "shared_xyz_origin": shared_xyz_origin,
        }

    def _load_targets(self, record):
        if self.target_builder is not None:
            return self.target_builder(record, self.spec)

        s1 = (
            read_label_array(self._path(record["semantic_t1"]))
            if "semantic_t1" in record else None
        )
        s2 = (
            read_label_array(self._path(record["semantic_t2"]))
            if "semantic_t2" in record else None
        )
        ch = (
            read_label_array(self._path(record["change"]))
            if "change" in record else None
        )

        target = build_canonical_target(
            label_mode=self.spec.label_mode,
            semantic_t1=s1,
            semantic_t2=s2,
            change=ch,
            class_names=self.spec.class_names,
        )

        return {
            "change": target.change,
            "semantic_t1": target.semantic_t1,
            "semantic_t2": target.semantic_t2,
            "change_valid": target.change_valid,
            "semantic_valid_t1": target.semantic_valid_t1,
            "semantic_valid_t2": target.semantic_valid_t2,
        }

    def __getitem__(self, index):
        record = self.records[index]

        sample: Dict[str, Any] = {
            "sample_id": record.get("id", str(index)),
            "dataset_name": self.spec.name,
            "route": self.route,
            # Internal backward-compatible alias. Users no longer configure it.
            "task_mode": self.route,
            "prompt": self.prompt,
            "class_names": dict(self.spec.class_names),
        }

        if self.spec.has_image:
            sample.update(self._load_2d(record))

        if self.spec.has_point:
            sample.update(self._load_3d(record))

        sample["target"] = self._load_targets(record)

        # Prepared raster labels must already match the raster topology.
        if self.spec.has_image:
            h, w = sample["images_t1"].shape[-2:]
            for key in ("change", "semantic_t1", "semantic_t2"):
                target = sample["target"][key]
                if target.ndim >= 2 and tuple(target.shape[-2:]) != (h, w):
                    raise ValueError(
                        f"{self.spec.name}/{sample['sample_id']}: {key} shape "
                        f"{tuple(target.shape)} does not match image {(h, w)}"
                    )

        if self.route == "2d3d":
            raster_crs = (
                sample["geo_t1"].get("crs")
                if sample.get("geo_t1") is not None else None
            )
            known_crs = [
                c for c in (
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
            "bounds": record.get("bounds"),
            "raster_geo_t1": sample.get("geo_t1"),
            "raster_geo_t2": sample.get("geo_t2"),
            "point_crs_t1": sample.get("point_crs_t1"),
            "point_crs_t2": sample.get("point_crs_t2"),
            "point_grid_size": self.spec.point_grid_size,
            "shared_xyz_origin": sample.get("shared_xyz_origin"),
        }
        return sample


def _self_test():
    s1 = torch.tensor([[0, 1], [2, 3]])
    s2 = torch.tensor([[0, 4], [2, 5]])
    out = build_canonical_target(
        label_mode="semantic_pair",
        semantic_t1=s1,
        semantic_t2=s2,
    )
    assert torch.equal(out.change, torch.tensor([[0, 1], [0, 1]]))

    ch = torch.tensor([[0, 1], [1, 0]])
    out = build_canonical_target(
        label_mode="post_semantic",
        change=ch,
        semantic_t2=s2,
    )
    assert (out.semantic_t1 == UNKNOWN_CLASS_ID).all()

    raw_bcd = torch.tensor([[0, 255], [255, 0]])
    out = build_canonical_target(
        label_mode="binary",
        change=raw_bcd,
        class_names={0: "unchanged", 255: "changed"},
    )
    assert torch.equal(out.change, torch.tensor([[0, 1], [1, 0]]))
    assert (out.semantic_t1 == UNKNOWN_CLASS_ID).all()

    assert infer_binary_class_ids(
        {0: "unchanged", 255: "changed"}
    ) == (0, 255)

    assert "changed_raw_id" not in DatasetSpec.__dataclass_fields__

    assert infer_unchanged_raw_id(
        {0: "unchanged", 1: "building"}
    ) == 0
    assert infer_unchanged_raw_id(
        {0: "background", 1: "building"}
    ) is None

    print("pair_dataset.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
