"""
PAIR experiment + dataset configuration.

The user JSON is intentionally minimal for each dataset:

    "SECOND": {
        "root": "/home/sht/Datasets/SECONDpair",
        "per_gpu_batch_size": 4,
        "class_names": {
            "0": "unchanged",
            "1": "water",
            ...
        }
    }

Everything structural is inferred from the prepared directory:
    images_t1/images_t2 -> image modality
    points_t1/points_t2 -> point modality
    semantic_t1/semantic_t2 -> semantic_pair
    change + semantic_t2 -> post_semantic
    change -> binary
    manifests/{train,val,test}.jsonl -> fixed split locations
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from datasets.pair_dataset import (
    DatasetSpec,
    infer_unchanged_raw_id,
    route_from_modalities,
)


IMAGE_DIRS = ("images_t1", "images_t2")
POINT_DIRS = ("points_t1", "points_t2")
SEMANTIC_DIRS = ("semantic_t1", "semantic_t2")


def _require_dict(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _normalize_class_names(value, dataset_name) -> Dict[int, str]:
    if not isinstance(value, dict) or not value:
        raise TypeError(
            f"{dataset_name}.class_names must be a non-empty object"
        )

    result = {}
    for raw_id, class_name in value.items():
        try:
            raw_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_name}: class id {raw_id!r} is not an integer"
            ) from exc

        if raw_id in result:
            raise ValueError(
                f"{dataset_name}: duplicate class id {raw_id}"
            )

        name = str(class_name).strip()
        if not name:
            raise ValueError(
                f"{dataset_name}: empty class name for id {raw_id}"
            )
        result[raw_id] = name

    return dict(sorted(result.items()))


def _paired_directory(root: Path, left: str, right: str) -> bool:
    a = (root / left).is_dir()
    b = (root / right).is_dir()
    if a != b:
        raise ValueError(
            f"{root}: {left}/ and {right}/ must either both exist or both be absent"
        )
    return a and b


def infer_dataset_schema(root: Path) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    has_image = _paired_directory(root, "images_t1", "images_t2")
    has_point = _paired_directory(root, "points_t1", "points_t2")

    modalities = []
    if has_image:
        modalities.append("image")
    if has_point:
        modalities.append("point")
    if not modalities:
        raise ValueError(
            f"{root}: no PAIR input modality found. Expected images_t1/images_t2 "
            "and/or points_t1/points_t2."
        )

    sem1 = (root / "semantic_t1").is_dir()
    sem2 = (root / "semantic_t2").is_dir()
    change = (root / "change").is_dir()

    if sem1 != sem2 and not (change and sem2 and not sem1):
        raise ValueError(
            f"{root}: incomplete semantic supervision directories. "
            "PAIR expects semantic_t1+semantic_t2, or change+semantic_t2, "
            "or change only."
        )

    if sem1 and sem2:
        label_mode = "semantic_pair"
    elif change and sem2:
        label_mode = "post_semantic"
    elif change:
        label_mode = "binary"
    else:
        raise ValueError(
            f"{root}: no PAIR supervision found. Expected "
            "semantic_t1/semantic_t2 or change/."
        )

    manifest_dir = root / "manifests"
    train_manifest = manifest_dir / "train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(
            f"{root}: required manifest missing: {train_manifest}"
        )

    val_manifest = manifest_dir / "val.jsonl"
    test_manifest = manifest_dir / "test.jsonl"

    return {
        "modalities": tuple(modalities),
        "route": route_from_modalities(modalities),
        "label_mode": label_mode,
        "train_manifest": train_manifest.resolve(),
        "val_manifest": val_manifest.resolve() if val_manifest.is_file() else None,
        "test_manifest": test_manifest.resolve() if test_manifest.is_file() else None,
        "has_change_directory": change,
    }


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    per_gpu_batch_size: int
    sampling_weight: float
    class_names: Dict[int, str]

    modalities: Tuple[str, ...]
    route: str
    label_mode: str
    unchanged_raw_id: Optional[int]

    train_manifest: Path
    val_manifest: Optional[Path]
    test_manifest: Optional[Path]

    spec: DatasetSpec

    def resolved_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "per_gpu_batch_size": self.per_gpu_batch_size,
            "sampling_weight": self.sampling_weight,
            "class_names": {
                str(k): v for k, v in self.class_names.items()
            },

            # Inferred PAIR schema:
            "modalities": list(self.modalities),
            "route": self.route,
            "label_mode": self.label_mode,
            "unchanged_raw_id": self.unchanged_raw_id,
            "manifests": {
                "train": str(self.train_manifest),
                "val": (
                    None if self.val_manifest is None
                    else str(self.val_manifest)
                ),
                "test": (
                    None if self.test_manifest is None
                    else str(self.test_manifest)
                ),
            },
            "point_grid_size": self.spec.point_grid_size,
        }


@dataclass
class ExperimentConfig:
    path: Path
    raw: Dict[str, Any]
    selected_names: Tuple[str, ...]
    datasets: Dict[str, DatasetConfig]

    @property
    def experiment(self):
        return self.raw["experiment"]

    @property
    def model(self):
        return self.raw["model"]

    @property
    def optimizer(self):
        return self.raw["optimizer"]

    @property
    def training(self):
        return self.raw["training"]

    @property
    def validation(self):
        return self.raw["validation"]

    @property
    def logging(self):
        return self.raw["logging"]

    def resolved_dict(
        self,
        runtime: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved = {
            "experiment": deepcopy(self.experiment),
            "model": deepcopy(self.model),
            "optimizer": deepcopy(self.optimizer),
            "training": deepcopy(self.training),
            "validation": deepcopy(self.validation),
            "logging": deepcopy(self.logging),
            "selected_datasets": list(self.selected_names),
            "datasets": {
                name: self.datasets[name].resolved_dict()
                for name in self.selected_names
            },
        }
        if runtime is not None:
            resolved["runtime"] = deepcopy(runtime)
        return resolved

    @staticmethod
    def hash_resolved(resolved: Dict[str, Any]) -> str:
        payload = json.dumps(
            resolved,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _resolve_path(value, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _global_point_grid_size(raw: Dict[str, Any]) -> float:
    point_cfg = raw.get("model", {}).get("point_encoder", {})
    grid_size = float(point_cfg.get("grid_size", 0.10))
    if grid_size <= 0:
        raise ValueError("model.point_encoder.grid_size must be > 0")
    return grid_size


def _dataset_from_json(
    name: str,
    data: Mapping[str, Any],
    config_dir: Path,
    *,
    point_grid_size: float,
) -> DatasetConfig:
    data = _require_dict(dict(data), f"datasets.{name}")

    allowed = {
        "root",
        "per_gpu_batch_size",
        "sampling_weight",
        "class_names",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise KeyError(
            f"datasets.{name} contains unsupported/redundant fields {unknown}. "
            "PAIR infers modality, manifests and label mode from the directory."
        )

    if "root" not in data:
        raise KeyError(f"datasets.{name}.root is required")

    root = _resolve_path(data["root"], config_dir)
    schema = infer_dataset_schema(root)

    class_names = _normalize_class_names(
        data.get("class_names"), name
    )
    unchanged_raw_id = infer_unchanged_raw_id(class_names)

    batch = int(data.get("per_gpu_batch_size", 1))
    if batch <= 0:
        raise ValueError(
            f"{name}.per_gpu_batch_size must be > 0"
        )

    weight = float(data.get("sampling_weight", 1.0))
    if weight <= 0:
        raise ValueError(
            f"{name}.sampling_weight must be > 0"
        )

    spec = DatasetSpec(
        name=name,
        modalities=schema["modalities"],
        label_mode=schema["label_mode"],
        class_names=class_names,
        point_grid_size=point_grid_size,
        unchanged_raw_id=unchanged_raw_id,
    )

    return DatasetConfig(
        name=name,
        root=root,
        per_gpu_batch_size=batch,
        sampling_weight=weight,
        class_names=class_names,
        modalities=schema["modalities"],
        route=schema["route"],
        label_mode=schema["label_mode"],
        unchanged_raw_id=unchanged_raw_id,
        train_manifest=schema["train_manifest"],
        val_manifest=schema["val_manifest"],
        test_manifest=schema["test_manifest"],
        spec=spec,
    )


def load_experiment_config(
    path: Path,
    selected_names: Optional[Sequence[str]] = None,
) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    for section in (
        "experiment",
        "model",
        "optimizer",
        "training",
        "validation",
        "logging",
        "datasets",
    ):
        if section not in raw:
            raise KeyError(
                f"Missing top-level config section: {section}"
            )
        _require_dict(raw[section], section)

    catalog = raw["datasets"]
    if not catalog:
        raise ValueError("Config contains no datasets")

    if selected_names:
        names = tuple(
            dict.fromkeys(str(x) for x in selected_names)
        )
    else:
        defaults = raw["experiment"].get("default_datasets")
        names = (
            tuple(defaults)
            if defaults
            else tuple(catalog.keys())
        )

    missing = [
        name for name in names
        if name not in catalog
    ]
    if missing:
        raise KeyError(
            f"Unknown datasets {missing}. "
            f"Available: {list(catalog.keys())}"
        )

    if not names:
        raise ValueError("No datasets selected")

    point_grid_size = _global_point_grid_size(raw)

    datasets = {
        name: _dataset_from_json(
            name,
            catalog[name],
            path.parent,
            point_grid_size=point_grid_size,
        )
        for name in names
    }

    if int(raw["experiment"].get("epochs", 0)) <= 0:
        raise ValueError("experiment.epochs must be > 0")

    if int(raw["training"].get("grad_accum", 0)) <= 0:
        raise ValueError("training.grad_accum must be > 0")

    if int(raw["training"].get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers must be >= 0")

    return ExperimentConfig(
        path=path,
        raw=raw,
        selected_names=names,
        datasets=datasets,
    )
