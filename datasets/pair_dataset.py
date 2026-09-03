"""
Unified dataset interface for PAIR.

Canonical semantic-change target
--------------------------------
PAIR treats every change dataset as semantic change detection, but source
supervision can have different information completeness:

    A -> B              full semantic change
    Unknown -> B        only post-change semantics are known
    Unknown -> Unknown  binary/all-category change
    Unchanged           no change

Instead of creating a huge transition-class vocabulary, every target is
factorized into:

    change       : 0 / 1
    semantic_t1  : class id or UNKNOWN_CLASS_ID
    semantic_t2  : class id or UNKNOWN_CLASS_ID

This lets datasets have different class vocabularies and avoids a global
num_classes requirement.

Spatial policy
--------------
2D:
    T1/T2 must describe the same footprint. Georeferenced rasters should be
    aligned to one CRS/GSD/affine grid before training. Optional resizing is
    applied to the PAIR, never independently.

3D:
    point clouds are not resized. T1/T2 are cropped by the same metric XY
    bounds and translated into a shared local coordinate frame.

2D+3D:
    all four inputs must refer to the same world-space bounds. Spatial
    metadata is returned for future pixel<->point projection/fusion.

This is a generic first-pass layer. Individual public datasets should use thin
record builders and, when needed, dataset-specific target/feature builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


UNKNOWN_CLASS_ID = -1
IGNORE_CLASS_ID = -100


@dataclass
class DatasetSpec:
    name: str
    task_mode: str                 # 2d | 3d | 2d3d
    label_mode: str = "semantic_pair"  # semantic_pair | post_semantic | binary | custom
    class_names: Optional[Sequence[str]] = None

    # Optional paired raster output size (H, W). This is not registration.
    image_size: Optional[Tuple[int, int]] = None

    # Metric PTv3 grid size.
    point_grid_size: float = 0.10

    changed_value: int = 1
    unchanged_value: int = 0
    ignore_value: Optional[int] = None

    # Reject misregistered GeoTIFF pairs instead of silently stretching them.
    strict_geo_alignment: bool = True

    prompt: Optional[str] = None


@dataclass
class CanonicalChangeTarget:
    """
    Canonical change target definition.

    Semantic class IDs used in this example:

        UNKNOWN_CLASS_ID = -1
        IGNORE_CLASS_ID  = -100

        UNCHANGED    = 0
        BUILDING = 1
        ROAD     = 2
        WATER    = 3
        GROUND   = 4
        TREE     = 5

    Each spatial location (pixel / point) is represented by six fields:

    1. change
    Binary change state:
        0 = Unchanged
        1 = Changed
        -100 = Ignore / invalid

    2. semantic_t1
    Semantic class ID at Time 1.
    If the source dataset does not provide the T1 semantic class,
    use UNKNOWN_CLASS_ID = -1.

    3. semantic_t2
    Semantic class ID at Time 2.
    If the source dataset does not provide the T2 semantic class,
    use UNKNOWN_CLASS_ID = -1.

    4. change_valid
    Whether the change label is valid and can be used for change loss.

    5. semantic_valid_t1
    Whether semantic_t1 is a known ground-truth semantic class and can
    participate in T1 semantic loss.

    6. semantic_valid_t2
    Whether semantic_t2 is a known ground-truth semantic class and can
    participate in T2 semantic loss.


    Examples
    ========

    Example 1: Tree -> Building

        T1 = TREE     = 5
        T2 = BUILDING = 1

        change             = 1
        semantic_t1        = 5
        semantic_t2        = 1

        change_valid       = True
        semantic_valid_t1  = True
        semantic_valid_t2  = True


    Example 2: Unknown -> Building

        T1 class is unavailable.
        T2 = BUILDING = 1

        change             = 1
        semantic_t1        = -1
        semantic_t2        = 1

        change_valid       = True
        semantic_valid_t1  = False
        semantic_valid_t2  = True


    Example 3: Unknown -> Unknown

        The source dataset only provides a binary change mask.
        It tells us that this location changed, but does not provide either
        the T1 or T2 semantic class.

        change             = 1
        semantic_t1        = -1
        semantic_t2        = -1

        change_valid       = True
        semantic_valid_t1  = False
        semantic_valid_t2  = False


    Example 4: Unchanged Tree

        T1 = TREE = 5
        T2 = TREE = 5

        change             = 0
        semantic_t1        = 5
        semantic_t2        = 5

        change_valid       = True
        semantic_valid_t1  = True
        semantic_valid_t2  = True


    Example 5: Unchanged but semantic classes are unknown

        The dataset only tells us that this location is unchanged.

        change             = 0
        semantic_t1        = -1
        semantic_t2        = -1

        change_valid       = True
        semantic_valid_t1  = False
        semantic_valid_t2  = False


    Example 6: Invalid / ignored location

        change             = -100
        semantic_t1        = -100
        semantic_t2        = -100

        change_valid       = False
        semantic_valid_t1  = False
        semantic_valid_t2  = False


    Important:
        UNKNOWN_CLASS_ID = -1 means:
            "the semantic class exists, but this dataset does not provide it."

        IGNORE_CLASS_ID = -100 means:
            "this spatial location itself should not participate in training."

        Therefore UNKNOWN and IGNORE are different concepts.
    """
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


def _ignore_mask(x: torch.Tensor, ignore_value: Optional[int]):
    if ignore_value is None:
        return torch.zeros_like(x, dtype=torch.bool)
    return x == int(ignore_value)


def build_canonical_target(
    *,
    label_mode: str,
    semantic_t1=None,
    semantic_t2=None,
    change=None,
    changed_value: int = 1,
    unchanged_value: int = 0,
    ignore_value: Optional[int] = 255,
) -> CanonicalChangeTarget:
    """Convert heterogeneous source supervision to PAIR's common SCD form."""

    mode = label_mode.lower().strip()
    if mode not in {"semantic_pair", "post_semantic", "binary"}:
        raise ValueError(f"Unsupported generic label_mode: {label_mode}")

    if mode == "semantic_pair":
        if semantic_t1 is None or semantic_t2 is None:
            raise ValueError("semantic_pair requires semantic_t1 and semantic_t2")

        s1 = _long(semantic_t1)
        s2 = _long(semantic_t2)
        if s1.shape != s2.shape:
            raise ValueError("semantic_t1 and semantic_t2 must have the same shape")

        invalid = _ignore_mask(s1, ignore_value) | _ignore_mask(s2, ignore_value)

        if change is None:
            ch = (s1 != s2).long()
        else:
            raw = _long(change)
            if raw.shape != s1.shape:
                raise ValueError("change mask must match semantic label shape")
            ch = torch.full_like(raw, IGNORE_CLASS_ID)
            ch[raw == unchanged_value] = 0
            ch[raw == changed_value] = 1
            invalid |= _ignore_mask(raw, ignore_value)
            invalid |= ~((raw == unchanged_value) | (raw == changed_value))

        s1, s2, ch = s1.clone(), s2.clone(), ch.clone()
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
        raise ValueError(f"{mode} requires a change mask")

    raw = _long(change)
    invalid = _ignore_mask(raw, ignore_value)
    known = (raw == unchanged_value) | (raw == changed_value)
    invalid |= ~known

    ch = torch.full_like(raw, IGNORE_CLASS_ID)
    ch[raw == unchanged_value] = 0
    ch[raw == changed_value] = 1
    ch[invalid] = IGNORE_CLASS_ID

    if mode == "binary":
        # changed = Unknown -> Unknown; unchanged remains simply Unchanged.
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

    # post_semantic: Unknown -> B
    if semantic_t2 is None:
        raise ValueError("post_semantic requires semantic_t2")

    s2_src = _long(semantic_t2)
    if s2_src.shape != raw.shape:
        raise ValueError("semantic_t2 must match change mask shape")

    sem2_invalid = _ignore_mask(s2_src, ignore_value)
    s1 = torch.full_like(raw, UNKNOWN_CLASS_ID)
    s2 = s2_src.clone()

    sem2_valid = (~invalid) & (~sem2_invalid)
    s2[~sem2_valid] = UNKNOWN_CLASS_ID
    s1[invalid] = IGNORE_CLASS_ID
    s2[invalid] = IGNORE_CLASS_ID

    return CanonicalChangeTarget(
        change=ch,
        semantic_t1=s1,
        semantic_t2=s2,
        change_valid=~invalid,
        semantic_valid_t1=torch.zeros_like(raw, dtype=torch.bool),
        semantic_valid_t2=sem2_valid,
    )


def build_default_prompt(spec: DatasetSpec) -> str:
    if spec.prompt:
        return spec.prompt

    text = (
        "Perform semantic change detection between Time 1 and Time 2. "
        "Identify unchanged and changed regions."
    )

    if spec.label_mode == "semantic_pair":
        text += " For changed regions, infer the semantic class before and after change."
    elif spec.label_mode == "post_semantic":
        text += " The pre-change semantic class may be unknown, while the post-change class is supervised."
    elif spec.label_mode == "binary":
        text += " The source dataset only supervises change; semantic classes before and after change are unknown."

    if spec.class_names:
        text += " Valid semantic classes are: " + ", ".join(spec.class_names) + "."

    return text


def read_label_array(path: Union[str, Path]) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=False)
        keys = list(z.keys())
        if len(keys) != 1:
            raise ValueError(f"{path} has multiple arrays {keys}; use a custom target_builder")
        arr = z[keys[0]]
    else:
        arr = np.asarray(Image.open(path))

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected single-channel label, got {arr.shape}: {path}")

    return torch.as_tensor(
        np.array(arr, copy=True),
        dtype=torch.long,
    )


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

    # Convert before losing integer dtype information.
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


def resize_image(image: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        image[None], size=size_hw, mode="bilinear", align_corners=False
    )[0]


def resize_label(label: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        label[None, None].float(), size=size_hw, mode="nearest"
    )[0, 0].long()


PointFeatureBuilder = Callable[[Dict[str, np.ndarray]], np.ndarray]
TargetBuilder = Callable[[Dict[str, Any], DatasetSpec], Dict[str, Any]]


def _read_las(path: Path):
    try:
        import laspy
    except ImportError as exc:
        raise ImportError("LAS/LAZ reading requires laspy") from exc

    las = laspy.read(path)
    attrs: Dict[str, Any] = {
        "coord": np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1).astype(np.float32)
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
    path: Union[str, Path], *, feature_builder: Optional[PointFeatureBuilder] = None
) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    crs = None

    if suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        if "coord" not in z or "feat" not in z:
            raise ValueError(f"{path} must contain coord and feat")
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
                "LAS/LAZ requires a dataset-specific feature_builder. "
                "PAIR will not guess whether RGB/NIR/intensity/etc. are features."
            )
        feat = np.asarray(feature_builder(attrs), dtype=np.float32)

    else:
        raise ValueError(f"Unsupported point-cloud format: {path}")

    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("coord must be [N,3]")
    if feat.ndim != 2 or feat.shape[0] != coord.shape[0]:
        raise ValueError("feat must be [N,C] matching coord")

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
    point_data: Dict[str, Any], *, grid_size: float, shared_xyz_origin=None
) -> Dict[str, torch.Tensor]:
    if grid_size <= 0:
        raise ValueError("point_grid_size must be > 0")

    coord = point_data["coord"].clone()
    if shared_xyz_origin is None:
        shared_xyz_origin = tuple(float(v) for v in coord.amin(dim=0).tolist())

    origin = torch.tensor(
        shared_xyz_origin, dtype=coord.dtype, device=coord.device
    ).view(1, 3)

    # Both time steps use the SAME metric origin. This is important for
    # temporal voxel/grid correspondence and future 2D<->3D projection.
    coord = coord - origin
    grid_coord = torch.floor(coord / float(grid_size)).long()

    return {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": point_data["feat"],
        "batch": torch.zeros(coord.shape[0], dtype=torch.long),
    }


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
    """
    Resolve a manifest file path.

    Canonical PAIR manifests store portable relative paths, e.g.:

        images_t1/train_000001.png

    When the dataset is created from:
        <dataset_root>/manifests/train.jsonl

    the loader automatically resolves those paths against <dataset_root>.
    Absolute paths remain supported for backward compatibility.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if dataset_root is None:
        raise ValueError(
            f"Relative path {path} requires dataset_root."
        )

    return (
        Path(dataset_root)
        .expanduser()
        .resolve()
        / path
    )


class UnifiedPAIRDataset(Dataset):
    """
    Manifest-driven PAIR dataset.

    Standard record keys
    --------------------
    2D:   image_t1, image_t2
    3D:   point_t1, point_t2
    2D3D: all four

    Optional common spatial key:
        bounds = [xmin, ymin, xmax, ymax]

    Generic label keys:
        semantic_pair: semantic_t1, semantic_t2, optional change
        post_semantic: change, semantic_t2
        binary:        change

    Irregular 3D labels/non-corresponding point sets can use label_mode=custom
    with a small dataset-specific target_builder.
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
            self.manifest_path = (
                Path(records)
                .expanduser()
                .resolve()
            )

            # Canonical layout:
            # <dataset_root>/manifests/train.jsonl
            self.dataset_root = (
                self.manifest_path
                .parent
                .parent
            )

            self.records = load_jsonl(
                self.manifest_path
            )
        else:
            self.records = list(records)

        self.spec = spec
        self.point_feature_builder = point_feature_builder
        self.target_builder = target_builder
        self.task_mode = spec.task_mode.lower().strip()

        if self.task_mode not in {"2d", "3d", "2d3d"}:
            raise ValueError("task_mode must be 2d, 3d, or 2d3d")
        if spec.label_mode == "custom" and target_builder is None:
            raise ValueError("label_mode=custom requires target_builder")

        self.prompt = build_default_prompt(spec)

    def __len__(self):
        return len(self.records)

    def _path(self, value: Union[str, Path]) -> Path:
        return resolve_dataset_path(
            value,
            self.dataset_root,
        )

    def _load_2d(self, record):
        im1, geo1 = read_image(self._path(record["image_t1"]))
        im2, geo2 = read_image(self._path(record["image_t2"]))

        if self.spec.strict_geo_alignment and not same_geo_grid(geo1, geo2):
            raise ValueError(
                "T1/T2 rasters are not on one geospatial grid. Reproject them "
                "to a common CRS/GSD/affine grid before PAIR training."
            )

        if im1.shape[-2:] != im2.shape[-2:] and self.spec.image_size is None:
            raise ValueError("T1/T2 image sizes differ; align or set a paired image_size")

        if self.spec.image_size is not None:
            im1 = resize_image(im1, self.spec.image_size)
            im2 = resize_image(im2, self.spec.image_size)

        return {"images_t1": im1, "images_t2": im2, "geo_t1": geo1, "geo_t2": geo2}

    def _load_3d(self, record):
        p1 = read_point_cloud(self._path(record["point_t1"]), feature_builder=self.point_feature_builder)
        p2 = read_point_cloud(self._path(record["point_t2"]), feature_builder=self.point_feature_builder)

        bounds = record.get("bounds")
        if bounds is not None:
            if len(bounds) != 4:
                raise ValueError("bounds must be [xmin,ymin,xmax,ymax]")
            p1 = crop_points_xy(p1, bounds)
            p2 = crop_points_xy(p2, bounds)
        if p1["coord"].shape[0] == 0 or p2["coord"].shape[0] == 0:
            raise ValueError("Empty T1/T2 point cloud after common spatial crop")

        # Shared XYZ origin for both time steps. If a common tile bound exists,
        # use its XY origin and the common minimum Z. Otherwise use the common
        # XYZ minimum of the two clouds.
        common_z0 = min(
            float(p1["coord"][:, 2].min().item()),
            float(p2["coord"][:, 2].min().item()),
        )
        if bounds is not None:
            shared_xyz_origin = (float(bounds[0]), float(bounds[1]), common_z0)
        else:
            common_min = torch.minimum(
                p1["coord"].amin(dim=0), p2["coord"].amin(dim=0)
            )
            shared_xyz_origin = tuple(float(v) for v in common_min.tolist())

        return {
            "point_dict_t1": make_ptv3_dict(
                p1, grid_size=self.spec.point_grid_size, shared_xyz_origin=shared_xyz_origin
            ),
            "point_dict_t2": make_ptv3_dict(
                p2, grid_size=self.spec.point_grid_size, shared_xyz_origin=shared_xyz_origin
            ),
            "point_crs_t1": p1.get("crs"),
            "point_crs_t2": p2.get("crs"),
            "bounds": bounds,
            "shared_xyz_origin": shared_xyz_origin,
        }

    def _load_targets(self, record, raster_size=None):
        if self.spec.label_mode == "custom":
            return self.target_builder(record, self.spec)

        s1 = read_label_array(self._path(record["semantic_t1"])) if "semantic_t1" in record else None
        s2 = read_label_array(self._path(record["semantic_t2"])) if "semantic_t2" in record else None
        ch = read_label_array(self._path(record["change"])) if "change" in record else None

        if raster_size is not None:
            def fit(x):
                if x is None or tuple(x.shape[-2:]) == tuple(raster_size):
                    return x
                return resize_label(x, raster_size)
            s1, s2, ch = fit(s1), fit(s2), fit(ch)

        target = build_canonical_target(
            label_mode=self.spec.label_mode,
            semantic_t1=s1,
            semantic_t2=s2,
            change=ch,
            changed_value=self.spec.changed_value,
            unchanged_value=self.spec.unchanged_value,
            ignore_value=self.spec.ignore_value,
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
            "task_mode": self.task_mode,
            "prompt": self.prompt,
            "class_names": list(self.spec.class_names or []),
        }

        raster_size = None
        if self.task_mode in {"2d", "2d3d"}:
            d2 = self._load_2d(record)
            sample.update(d2)
            raster_size = tuple(d2["images_t1"].shape[-2:])

        if self.task_mode in {"3d", "2d3d"}:
            sample.update(self._load_3d(record))

        if self.spec.label_mode == "custom" or any(
            k in record for k in ("semantic_t1", "semantic_t2", "change")
        ):
            sample["target"] = self._load_targets(
                record,
                raster_size=raster_size if self.task_mode in {"2d", "2d3d"} else None,
            )

        if self.task_mode == "2d3d":
            raster_crs = None
            if sample.get("geo_t1") is not None:
                raster_crs = sample["geo_t1"].get("crs")
            point_crs1 = sample.get("point_crs_t1")
            point_crs2 = sample.get("point_crs_t2")

            # When CRS metadata exists, do not silently fuse incompatible
            # coordinate systems. Missing CRS is allowed for already prepared
            # local benchmark data, but then alignment responsibility belongs
            # to the dataset-specific adapter.
            known_crs = [c for c in (raster_crs, point_crs1, point_crs2) if c]
            if known_crs and any(c != known_crs[0] for c in known_crs[1:]):
                raise ValueError(
                    "2D+3D sample contains mismatched CRS metadata. Reproject "
                    "imagery and point clouds to one common CRS before training."
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
    # Full A->B
    s1 = torch.tensor([[0, 1], [2, 3]])
    s2 = torch.tensor([[0, 4], [2, 5]])
    out = build_canonical_target(
        label_mode="semantic_pair", semantic_t1=s1, semantic_t2=s2, ignore_value=None
    )
    assert torch.equal(out.change, torch.tensor([[0, 1], [0, 1]]))

    # Unknown->B
    ch = torch.tensor([[0, 1], [1, 0]])
    out = build_canonical_target(
        label_mode="post_semantic", change=ch, semantic_t2=s2, ignore_value=None
    )
    assert (out.semantic_t1 == UNKNOWN_CLASS_ID).all()

    # Unknown->Unknown
    out = build_canonical_target(label_mode="binary", change=ch, ignore_value=None)
    assert (out.semantic_t1 == UNKNOWN_CLASS_ID).all()
    assert (out.semantic_t2 == UNKNOWN_CLASS_ID).all()
    assert not out.semantic_valid_t1.any()
    assert not out.semantic_valid_t2.any()

    print("pair_dataset.py canonical target self-test: PASS")


if __name__ == "__main__":
    _self_test()
