#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NYC-SCD -> PAIR 3D semantic + event preparation (v4).

Source layout (official NYC-SCD version_training):
    version_training/
        Train200/
            <sample>/
                pointCloud0.ply
                pointCloud1.ply
        Val40/
            <sample>/
                pointCloud0.ply
                pointCloud1.ply
        Test40/                         # optional, if present
            <sample>/
                pointCloud0.ply
                pointCloud1.ply

Official PLY fields used here:
    pointCloud0.ply:
        x, y, z, label_mono
    pointCloud1.ply:
        x, y, z, label_mono, label_ch

PAIR output:
    points_t1/<id>.npz
        coord [N1,3] mandatory
        rgb [N1,3] optional
        intensity [N1,1] optional

    points_t2/<id>.npz
        coord [N2,3] mandatory
        rgb [N2,3] optional
        intensity [N2,1] optional

    semantic_t1/<id>.npz
        semantic [N1]
        event [N1]
        event_valid [N1]

    semantic_t2/<id>.npz
        semantic [N2]
        event [N2]
        event_valid [N2]

    manifests/train.jsonl
    manifests/val.jsonl
    manifests/test.jsonl                # only when requested

PAIR 3D semantic taxonomy:
    0 Ground
    1 Building
    2 Vegetation
    3 Clutter
   -1 ignore / unlabeled

NYC-SCD source files may contain semantic IDs outside 0..3 (for example 9).
The official ME-CPT loader converts every semantic ID >3 to -1. This script
does the same canonicalization, so PAIR must use ignored_id=-1 for NYC-SCD.

PAIR 3D event taxonomy:
    0 unchanged
    1 added
    2 removed
    3 class_change
    4 height_up
    5 height_down

NYC-SCD source label_ch:
    0 Unchanged
    1 Newly Built
    2 Demolition
    3 New Clutter

Orthogonalized source -> PAIR event mapping:
    0 -> 0 unchanged
    1 -> 1 added
    2 -> 2 removed
    3 -> 1 added

Temporal support asymmetry:
    T1:
        nearest T2 label_ch is transferred by XY nearest neighbor
        within --nn-radius (default 1.0 m):
            mapped 0 -> unchanged, event=0, valid
            mapped 2 -> removed,   event=2, valid
            mapped 1/3 -> event=0 placeholder, invalid
            mapped -1  -> event=0 placeholder, invalid
            unmatched  -> event=0 placeholder, invalid

    T2:
        source 0   -> unchanged, event=0, valid
        source 1/3 -> added,     event=1, valid
        source 2   -> event=0 placeholder, invalid
        source -1  -> event=0 placeholder, invalid

Thus stored supervision is temporally clean:
    T1 valid event classes = {0 unchanged, 2 removed}
    T2 valid event classes = {0 unchanged, 1 added}

All event_valid=False entries store event=0. Sampling metadata must be derived
from valid per-epoch supervision, not from nonzero invalid event placeholders.

Important:
    - No grid pseudo labels.
    - No pseudo class_change / height_up / height_down.
    - No majority-semantic heuristic.
    - No fake rgb/intensity arrays.
    - No per-tile intensity min-max normalization.
    - Source semantic IDs outside 0..3 are canonicalized to -1 (ignore).
    - Source change IDs outside 0..3 are treated as invalid/unsupervised.
    - event_protocol.json is not produced and an old copy is removed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


SEMANTIC_NAMES = {
    0: "ground",
    1: "building",
    2: "vegetation",
    3: "clutter",
}

EVENT_NAMES = {
    0: "unchanged",
    1: "added",
    2: "removed",
    3: "class_change",
    4: "height_up",
    5: "height_down",
}

SOURCE_CHANGE_NAMES = {
    0: "unchanged",
    1: "newly_built",
    2: "demolition",
    3: "new_clutter",
}

SOURCE_TO_EVENT = np.asarray([0, 1, 2, 1], dtype=np.int8)

SPLIT_DIRS = {
    "train": "Train200",
    "val": "Val40",
    "test": "Test40",
}

DEFAULT_OUTPUT_ROOT = (
    Path(r"E:\home\sht\Datasets\NYC-SCDpair")
    if os.name == "nt"
    else Path("/home/sht/Datasets/NYC-SCDpair")
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare official NYC-SCD version_training data for PAIR 3D."
    )
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="NYC-SCD version_training directory.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"PAIR output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val"),
        help="Splits to regenerate. Default: train val",
    )
    p.add_argument(
        "--nn-radius",
        type=float,
        default=1.0,
        help="Maximum XY distance for transferring T2 label_ch to T1. Default: 1.0 m",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum complete pairs per split; 0 means all.",
    )
    p.add_argument(
        "--vertex-name",
        default="vertex",
        help="PLY element name. Default: vertex",
    )
    return p.parse_args()


# =============================================================================
# Small utilities
# =============================================================================

def banner(text: str):
    print("=" * 92)
    print(text)
    print("=" * 92)


def atomic_write_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def as_relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def integer_array(array, name: str, path: Path) -> np.ndarray:
    x = np.asarray(array)
    if x.ndim != 1:
        raise ValueError(f"{path}: {name} must be 1D, got {x.shape}")
    if not np.issubdtype(x.dtype, np.integer):
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: {name} contains NaN/Inf")
        rounded = np.rint(x)
        if not np.allclose(x, rounded, atol=1e-6, rtol=0.0):
            raise ValueError(f"{path}: {name} contains non-integer values")
        x = rounded
    return x.astype(np.int64, copy=False)


def hist_dict(values: np.ndarray, names: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    out = {}
    for value, count in zip(unique.tolist(), counts.tolist()):
        key = names.get(int(value), str(int(value))) if names is not None else str(int(value))
        out[key] = int(count)
    return out


def add_counter(counter: Counter, values: np.ndarray):
    unique, counts = np.unique(np.asarray(values).reshape(-1), return_counts=True)
    for value, count in zip(unique.tolist(), counts.tolist()):
        counter[int(value)] += int(count)


def make_sample_id(split: str, directory_name: str) -> str:
    lower = directory_name.lower()
    prefix = split + "_"
    return directory_name if lower.startswith(prefix) else f"{split}_{directory_name}"


# =============================================================================
# Source discovery / PLY reading
# =============================================================================

def resolve_split_dir(source_root: Path, split: str) -> Path:
    expected = SPLIT_DIRS[split]
    direct = source_root / expected
    if direct.is_dir():
        return direct

    matches = [
        path
        for path in source_root.iterdir()
        if path.is_dir() and path.name.lower() == expected.lower()
    ]
    if len(matches) == 1:
        return matches[0]

    available = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    raise FileNotFoundError(
        f"{source_root}: split {split!r} expects directory {expected!r}; "
        f"available directories={available}"
    )


def discover_pairs(split_dir: Path) -> Tuple[List[Tuple[Path, Path, Path]], List[Path]]:
    complete = []
    incomplete = []

    for sample_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        pc0 = sample_dir / "pointCloud0.ply"
        pc1 = sample_dir / "pointCloud1.ply"
        if pc0.is_file() and pc1.is_file():
            complete.append((sample_dir, pc0, pc1))
        else:
            incomplete.append(sample_dir)

    return complete, incomplete


def _property_names(vertex) -> set:
    return set(vertex.data.dtype.names or ())


def _first_property(vertex, candidates: Iterable[str]):
    names = _property_names(vertex)
    for name in candidates:
        if name in names:
            return np.asarray(vertex.data[name])
    return None


def _read_rgb(vertex, path: Path) -> Optional[np.ndarray]:
    names = _property_names(vertex)

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack(
            [
                np.asarray(vertex.data["red"]),
                np.asarray(vertex.data["green"]),
                np.asarray(vertex.data["blue"]),
            ]
        )
    elif {"r", "g", "b"}.issubset(names):
        rgb = np.column_stack(
            [
                np.asarray(vertex.data["r"]),
                np.asarray(vertex.data["g"]),
                np.asarray(vertex.data["b"]),
            ]
        )
    else:
        return None

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"{path}: RGB must be [N,3], got {rgb.shape}")
    if not np.isfinite(rgb).all():
        raise ValueError(f"{path}: RGB contains NaN/Inf")
    return rgb.astype(np.float32, copy=False)


def _read_intensity(vertex, path: Path) -> Optional[np.ndarray]:
    intensity = _first_property(vertex, ("intensity", "reflectance"))
    if intensity is None:
        return None
    intensity = np.asarray(intensity)
    if intensity.ndim != 1:
        raise ValueError(f"{path}: intensity must be [N], got {intensity.shape}")
    if not np.isfinite(intensity).all():
        raise ValueError(f"{path}: intensity contains NaN/Inf")
    return intensity.astype(np.float32, copy=False)[:, None]


def read_nyc_ply(path: Path, *, vertex_name: str, require_change: bool) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        ply = PlyData.read(f)

    if vertex_name not in ply:
        raise KeyError(f"{path}: missing PLY element {vertex_name!r}")
    vertex = ply[vertex_name]
    names = _property_names(vertex)

    required = {"x", "y", "z", "label_mono"}
    if require_change:
        required.add("label_ch")
    missing = sorted(required - names)
    if missing:
        raise KeyError(f"{path}: missing PLY properties {missing}; available={sorted(names)}")

    coord = np.column_stack(
        [
            np.asarray(vertex.data["x"]),
            np.asarray(vertex.data["y"]),
            np.asarray(vertex.data["z"]),
        ]
    ).astype(np.float32, copy=False)

    if coord.ndim != 2 or coord.shape[1] != 3 or coord.shape[0] == 0:
        raise ValueError(f"{path}: coord must be non-empty [N,3], got {coord.shape}")
    if not np.isfinite(coord).all():
        raise ValueError(f"{path}: coord contains NaN/Inf")

    semantic = integer_array(vertex.data["label_mono"], "label_mono", path)
    if semantic.shape[0] != coord.shape[0]:
        raise ValueError(f"{path}: label_mono length does not match coord")

    out = {
        "coord": np.ascontiguousarray(coord),
        "semantic": np.ascontiguousarray(semantic),
    }

    if require_change:
        label_ch = integer_array(vertex.data["label_ch"], "label_ch", path)
        if label_ch.shape[0] != coord.shape[0]:
            raise ValueError(f"{path}: label_ch length does not match coord")
        out["label_ch"] = np.ascontiguousarray(label_ch)

    rgb = _read_rgb(vertex, path)
    intensity = _read_intensity(vertex, path)
    if rgb is not None:
        out["rgb"] = np.ascontiguousarray(rgb)
    if intensity is not None:
        out["intensity"] = np.ascontiguousarray(intensity)

    return out


# =============================================================================
# Label validation / event construction
# =============================================================================

def canonicalize_semantic(semantic: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int, ...]]:
    """
    NYC-SCD officially has four semantic classes, but the released PLY files can
    contain sentinel/unlabeled IDs such as 9. The official ME-CPT loader maps
    every semantic label >3 to -1 before CE. We mirror that behavior exactly.

    Negative labels are also treated as ignore to keep the PAIR semantic protocol
    unambiguous.
    """
    semantic = np.asarray(semantic, dtype=np.int64).copy()
    invalid = (semantic < 0) | (semantic > 3)
    raw_ignored = tuple(sorted(int(x) for x in np.unique(semantic[invalid]).tolist()))
    ignored_count = int(invalid.sum())
    semantic[invalid] = -1
    return semantic, ignored_count, raw_ignored


def canonicalize_source_change(label_ch: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int, ...]]:
    """
    Keep official NYC source change IDs 0..3. Any other source value is
    unsupervised and becomes -1. It is never converted into a PAIR event class.
    """
    label_ch = np.asarray(label_ch, dtype=np.int64).copy()
    invalid = (label_ch < 0) | (label_ch > 3)
    raw_ignored = tuple(sorted(int(x) for x in np.unique(label_ch[invalid]).tolist()))
    ignored_count = int(invalid.sum())
    label_ch[invalid] = -1
    return label_ch, ignored_count, raw_ignored


def map_source_event(label_ch: np.ndarray) -> np.ndarray:
    """
    Map valid NYC source change IDs to the global PAIR event taxonomy.

    Invalid source IDs (-1 after canonicalization) stay event=0 with
    event_valid=False; they are not interpreted as unchanged.
    """
    label_ch = np.asarray(label_ch, dtype=np.int64)
    event = np.zeros(label_ch.shape, dtype=np.int8)
    valid_source = (label_ch >= 0) & (label_ch <= 3)
    if valid_source.any():
        event[valid_source] = SOURCE_TO_EVENT[label_ch[valid_source]]
    return event


def build_t2_event(label_ch_t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build temporally valid T2 event supervision.

    T2 supports only:
        source 0   -> event 0 unchanged, valid
        source 1/3 -> event 1 added, valid

    Demolition belongs to T1 support. Source-invalid labels are unsupervised.
    Every invalid T2 entry stores the neutral placeholder event=0.
    """
    label_ch_t2 = np.asarray(label_ch_t2, dtype=np.int64)
    event = np.zeros(label_ch_t2.shape, dtype=np.int8)
    event_valid = np.isin(label_ch_t2, (0, 1, 3))

    event[label_ch_t2 == 1] = 1
    event[label_ch_t2 == 3] = 1
    return event, np.asarray(event_valid, dtype=np.bool_)

def nearest_t2_change_for_t1(
    coord_t1: np.ndarray,
    coord_t2: np.ndarray,
    label_ch_t2: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transfer T2 source label_ch to T1 by XY nearest neighbor.

    Returns:
        mapped_source [N1]  -1 when unmatched
        distance [N1]       inf when unmatched
        matched [N1]
    """
    if radius <= 0:
        raise ValueError("nn radius must be > 0")
    if coord_t2.shape[0] == 0:
        raise ValueError("T2 point cloud is empty")

    tree = cKDTree(np.asarray(coord_t2[:, :2], dtype=np.float64))
    try:
        distance, index = tree.query(
            np.asarray(coord_t1[:, :2], dtype=np.float64),
            k=1,
            distance_upper_bound=float(radius),
            workers=-1,
        )
    except TypeError:
        distance, index = tree.query(
            np.asarray(coord_t1[:, :2], dtype=np.float64),
            k=1,
            distance_upper_bound=float(radius),
        )

    matched = np.isfinite(distance) & (index < coord_t2.shape[0])
    # -2 = no T2 neighbor within radius; -1 = matched source point whose
    # label_ch itself is ignored/invalid after canonicalization.
    mapped_source = np.full(coord_t1.shape[0], -2, dtype=np.int16)
    mapped_source[matched] = label_ch_t2[index[matched]].astype(np.int16, copy=False)
    return mapped_source, distance.astype(np.float32, copy=False), matched


def build_t1_event(
    coord_t1: np.ndarray,
    coord_t2: np.ndarray,
    label_ch_t2: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build temporally valid T1 event supervision.

    T1 supports only:
        mapped source 0 -> event 0 unchanged, valid
        mapped source 2 -> event 2 removed, valid

    Added/new-clutter belong to T2 support. Matched source-invalid and unmatched
    T1 points are unsupervised. Every invalid T1 entry stores event=0.
    """
    mapped_source, distance, matched = nearest_t2_change_for_t1(
        coord_t1, coord_t2, label_ch_t2, radius
    )

    event = np.zeros(coord_t1.shape[0], dtype=np.int8)
    event_valid = matched & np.isin(mapped_source, (0, 2))

    removed = event_valid & (mapped_source == 2)
    event[removed] = 2

    return event, np.asarray(event_valid, dtype=np.bool_), mapped_source, distance


# =============================================================================
# PAIR file writing
# =============================================================================

def save_point_npz(path: Path, point: Dict[str, np.ndarray]):
    payload = {"coord": point["coord"].astype(np.float32, copy=False)}
    if "rgb" in point:
        payload["rgb"] = point["rgb"].astype(np.float32, copy=False)
    if "intensity" in point:
        payload["intensity"] = point["intensity"].astype(np.float32, copy=False)

    # Deliberately no feat/grid_coord/batch and no fabricated optional fields.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_supervision_npz(
    path: Path,
    semantic: np.ndarray,
    event: np.ndarray,
    event_valid: np.ndarray,
):
    if not (semantic.shape == event.shape == event_valid.shape):
        raise ValueError(
            f"{path}: semantic/event/event_valid shapes differ: "
            f"{semantic.shape}, {event.shape}, {event_valid.shape}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        semantic=semantic.astype(np.int16, copy=False),
        event=event.astype(np.int8, copy=False),
        event_valid=event_valid.astype(np.bool_, copy=False),
    )


def prepare_one(
    split: str,
    sample_dir: Path,
    pc0_path: Path,
    pc1_path: Path,
    output_root: Path,
    nn_radius: float,
    vertex_name: str,
):
    sample_id = make_sample_id(split, sample_dir.name)

    t1 = read_nyc_ply(pc0_path, vertex_name=vertex_name, require_change=False)
    t2 = read_nyc_ply(pc1_path, vertex_name=vertex_name, require_change=True)

    t1["semantic"], ignored_sem_t1, raw_ignored_sem_t1 = canonicalize_semantic(t1["semantic"])
    t2["semantic"], ignored_sem_t2, raw_ignored_sem_t2 = canonicalize_semantic(t2["semantic"])
    t2["label_ch"], ignored_change_t2, raw_ignored_change_t2 = canonicalize_source_change(t2["label_ch"])

    event_t2, valid_t2 = build_t2_event(t2["label_ch"])
    event_t1, valid_t1, mapped_source_t1, nn_distance_t1 = build_t1_event(
        t1["coord"], t2["coord"], t2["label_ch"], nn_radius
    )

    # Canonical temporal-support contract:
    #   T1 valid={0,2}, T2 valid={0,1}; every invalid entry stores event=0.
    if np.any(event_t1[~valid_t1] != 0):
        raise RuntimeError(f"{sample_id}: T1 invalid event entries must be 0")
    if np.any(event_t2[~valid_t2] != 0):
        raise RuntimeError(f"{sample_id}: T2 invalid event entries must be 0")
    if np.any(~np.isin(event_t1[valid_t1], (0, 2))):
        raise RuntimeError(f"{sample_id}: T1 valid event outside {{0,2}}")
    if np.any(~np.isin(event_t2[valid_t2], (0, 1))):
        raise RuntimeError(f"{sample_id}: T2 valid event outside {{0,1}}")

    point_t1_path = output_root / "points_t1" / f"{sample_id}.npz"
    point_t2_path = output_root / "points_t2" / f"{sample_id}.npz"
    semantic_t1_path = output_root / "semantic_t1" / f"{sample_id}.npz"
    semantic_t2_path = output_root / "semantic_t2" / f"{sample_id}.npz"

    save_point_npz(point_t1_path, t1)
    save_point_npz(point_t2_path, t2)
    save_supervision_npz(semantic_t1_path, t1["semantic"], event_t1, valid_t1)
    save_supervision_npz(semantic_t2_path, t2["semantic"], event_t2, valid_t2)

    matched = mapped_source_t1 != -2
    record = {
        "id": sample_id,
        "point_t1": as_relative_posix(point_t1_path, output_root),
        "point_t2": as_relative_posix(point_t2_path, output_root),
        "semantic_t1": as_relative_posix(semantic_t1_path, output_root),
        "semantic_t2": as_relative_posix(semantic_t2_path, output_root),
    }

    stats = {
        "n1": int(t1["coord"].shape[0]),
        "n2": int(t2["coord"].shape[0]),
        "semantic_t1": t1["semantic"],
        "semantic_t2": t2["semantic"],
        "source_t2": t2["label_ch"],
        "event_t1": event_t1,
        "event_t2": event_t2,
        "valid_t1": valid_t1,
        "valid_t2": valid_t2,
        "mapped_source_t1": mapped_source_t1,
        "matched_t1": matched,
        "nn_distance_t1": nn_distance_t1,
        "point_keys_t1": tuple(k for k in ("coord", "rgb", "intensity") if k in t1),
        "point_keys_t2": tuple(k for k in ("coord", "rgb", "intensity") if k in t2),
        "ignored_sem_t1": ignored_sem_t1,
        "ignored_sem_t2": ignored_sem_t2,
        "raw_ignored_sem_t1": raw_ignored_sem_t1,
        "raw_ignored_sem_t2": raw_ignored_sem_t2,
        "ignored_change_t2": ignored_change_t2,
        "raw_ignored_change_t2": raw_ignored_change_t2,
    }
    return record, stats


# =============================================================================
# Split preparation
# =============================================================================

def prepare_split(
    split: str,
    source_root: Path,
    output_root: Path,
    nn_radius: float,
    max_samples: int,
    vertex_name: str,
):
    split_dir = resolve_split_dir(source_root, split)
    complete, incomplete = discover_pairs(split_dir)

    if max_samples > 0:
        complete = complete[:max_samples]

    banner(f"{split.upper()}  source={split_dir}")
    print("complete pairs :", len(complete))
    print("incomplete dirs:", len(incomplete))
    if max_samples > 0:
        print("max samples    :", max_samples)

    if not complete:
        raise RuntimeError(f"{split_dir}: no complete pointCloud0/pointCloud1 pairs found")

    records = []
    semantic_t1_total = Counter()
    semantic_t2_total = Counter()
    source_t2_total = Counter()
    event_t1_valid_total = Counter()
    event_t2_valid_total = Counter()

    points_t1_total = 0
    points_t2_total = 0
    valid_t1_total = 0
    valid_t2_total = 0
    matched_t1_total = 0
    ignored_sem_t1_total = 0
    ignored_sem_t2_total = 0
    ignored_change_t2_total = 0
    ignored_sem_raw = Counter()
    ignored_change_raw = Counter()

    split_start = time.time()

    for index, (sample_dir, pc0_path, pc1_path) in enumerate(complete, start=1):
        t0 = time.time()
        record, stats = prepare_one(
            split=split,
            sample_dir=sample_dir,
            pc0_path=pc0_path,
            pc1_path=pc1_path,
            output_root=output_root,
            nn_radius=nn_radius,
            vertex_name=vertex_name,
        )
        records.append(record)

        points_t1_total += stats["n1"]
        points_t2_total += stats["n2"]
        valid_t1_total += int(stats["valid_t1"].sum())
        valid_t2_total += int(stats["valid_t2"].sum())
        matched_t1_total += int(stats["matched_t1"].sum())
        ignored_sem_t1_total += int(stats["ignored_sem_t1"])
        ignored_sem_t2_total += int(stats["ignored_sem_t2"])
        ignored_change_t2_total += int(stats["ignored_change_t2"])
        for value in stats["raw_ignored_sem_t1"]:
            ignored_sem_raw[int(value)] += 1
        for value in stats["raw_ignored_sem_t2"]:
            ignored_sem_raw[int(value)] += 1
        for value in stats["raw_ignored_change_t2"]:
            ignored_change_raw[int(value)] += 1

        add_counter(semantic_t1_total, stats["semantic_t1"])
        add_counter(semantic_t2_total, stats["semantic_t2"])
        add_counter(source_t2_total, stats["source_t2"])
        add_counter(event_t1_valid_total, stats["event_t1"][stats["valid_t1"]])
        add_counter(event_t2_valid_total, stats["event_t2"][stats["valid_t2"]])

        valid1_pct = 100.0 * float(stats["valid_t1"].mean())
        valid2_pct = 100.0 * float(stats["valid_t2"].mean())
        match_pct = 100.0 * float(stats["matched_t1"].mean())
        finite_dist = stats["nn_distance_t1"][stats["matched_t1"]]
        nn_mean = float(finite_dist.mean()) if finite_dist.size else float("nan")

        print(
            f"[{index:4d}/{len(complete):4d}] {record['id']} | "
            f"N1={stats['n1']:,} N2={stats['n2']:,} | "
            f"T1 matched={match_pct:5.1f}% valid={valid1_pct:5.1f}% | "
            f"T2 valid={valid2_pct:5.1f}% | "
            f"NN mean={nn_mean:.3f}m | {time.time() - t0:.2f}s"
        )
        print(f"    point T1 keys : {stats['point_keys_t1']}")
        print(f"    point T2 keys : {stats['point_keys_t2']}")
        print(f"    source T2     : {hist_dict(stats['source_t2'], SOURCE_CHANGE_NAMES)}")
        if stats["ignored_sem_t1"] or stats["ignored_sem_t2"] or stats["ignored_change_t2"]:
            print(
                f"    ignored raw   : semantic T1={stats['ignored_sem_t1']} {stats['raw_ignored_sem_t1']} | "
                f"T2={stats['ignored_sem_t2']} {stats['raw_ignored_sem_t2']} | "
                f"change T2={stats['ignored_change_t2']} {stats['raw_ignored_change_t2']}"
            )
        print(f"    event T1 valid: {hist_dict(stats['event_t1'][stats['valid_t1']], EVENT_NAMES)}")
        print(f"    event T2 valid: {hist_dict(stats['event_t2'][stats['valid_t2']], EVENT_NAMES)}")

    manifest_path = output_root / "manifests" / f"{split}.jsonl"
    atomic_write_jsonl(manifest_path, records)

    print()
    print("-" * 92)
    print(f"{split.upper()} SUMMARY")
    print("-" * 92)
    print("samples        :", len(records))
    print("points T1      :", f"{points_t1_total:,}")
    print("points T2      :", f"{points_t2_total:,}")
    print(
        "T1 NN matched  :",
        f"{matched_t1_total:,}/{points_t1_total:,}",
        f"({100.0 * matched_t1_total / max(points_t1_total, 1):.2f}%)",
    )
    print(
        "event valid T1 :",
        f"{valid_t1_total:,}/{points_t1_total:,}",
        f"({100.0 * valid_t1_total / max(points_t1_total, 1):.2f}%)",
    )
    print(
        "event valid T2 :",
        f"{valid_t2_total:,}/{points_t2_total:,}",
        f"({100.0 * valid_t2_total / max(points_t2_total, 1):.2f}%)",
    )
    print("semantic T1    :", {str(k): semantic_t1_total[k] for k in sorted(semantic_t1_total)})
    print("semantic T2    :", {str(k): semantic_t2_total[k] for k in sorted(semantic_t2_total)})
    print(
        "semantic ignore:",
        f"T1={ignored_sem_t1_total:,}, T2={ignored_sem_t2_total:,}, "
        f"raw IDs seen={sorted(ignored_sem_raw)}",
    )
    print("source change  :", {str(k): source_t2_total[k] for k in sorted(source_t2_total)})
    print(
        "change ignore  :",
        f"T2={ignored_change_t2_total:,}, raw IDs seen={sorted(ignored_change_raw)}",
    )
    print(
        "event T1 valid :",
        {EVENT_NAMES[k]: event_t1_valid_total[k] for k in sorted(event_t1_valid_total)},
    )
    print(
        "event T2 valid :",
        {EVENT_NAMES[k]: event_t2_valid_total[k] for k in sorted(event_t2_valid_total)},
    )
    print("manifest       :", manifest_path)
    print("elapsed        :", f"{time.time() - split_start:.2f}s")

    # Protocol sanity: NYC must never produce supervised event 3/4/5.
    #
    # T2 has an exact source-level identity:
    #   valid unchanged == source label_ch 0
    #   valid added     == source label_ch 1 + 3
    #   source label_ch 2 and -1 are both invalid on T2
    expected_t2_unchanged = int(source_t2_total[0])
    expected_t2_added = int(source_t2_total[1] + source_t2_total[3])
    expected_t2_valid = expected_t2_unchanged + expected_t2_added
    if valid_t2_total != expected_t2_valid:
        raise RuntimeError(
            f"{split}: T2 event_valid count mismatch: got {valid_t2_total:,}, "
            f"expected {expected_t2_valid:,} = source0 + source1 + source3. "
            "Source label_ch 2 and ignored -1 must both be invalid on T2."
        )
    if int(event_t2_valid_total[0]) != expected_t2_unchanged:
        raise RuntimeError(
            f"{split}: T2 valid unchanged mismatch: got "
            f"{int(event_t2_valid_total[0]):,}, expected source0="
            f"{expected_t2_unchanged:,}"
        )
    if int(event_t2_valid_total[1]) != expected_t2_added:
        raise RuntimeError(
            f"{split}: T2 valid added mismatch: got "
            f"{int(event_t2_valid_total[1]):,}, expected source1+source3="
            f"{expected_t2_added:,}"
        )

    if any(event_t1_valid_total[k] for k in (1, 3, 4, 5)):
        raise RuntimeError(
            f"{split}: T1 valid events violate NYC active classes [0,2]: "
            f"{dict(event_t1_valid_total)}"
        )
    if any(event_t2_valid_total[k] for k in (2, 3, 4, 5)):
        raise RuntimeError(
            f"{split}: T2 valid events violate NYC active classes [0,1]: "
            f"{dict(event_t2_valid_total)}"
        )

    return records


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"NYC-SCD source root not found: {source_root}")
    if args.nn_radius <= 0:
        raise ValueError("--nn-radius must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    for name in ("points_t1", "points_t2", "semantic_t1", "semantic_t2", "manifests"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    # Old experimental versions wrote this metadata file. PAIR no longer
    # depends on it; directory structure + manifests are the protocol.
    old_protocol = output_root / "event_protocol.json"
    if old_protocol.exists():
        old_protocol.unlink()

    banner("NYC-SCD -> PAIR semantic + 3D event preparation v4.3")
    print("source root      :", source_root)
    print("output root      :", output_root)
    print("splits           :", list(args.splits))
    print("semantic classes :", SEMANTIC_NAMES, "+ ignore=-1")
    print("PAIR config      : NYC-SCD ignored_id must be -1")
    print("source change    :", SOURCE_CHANGE_NAMES)
    print("source event map : 0->0, 1->1, 2->2, 3->1")
    print("stored T1 events : valid {0 unchanged, 2 removed}; invalid -> event 0")
    print("stored T2 events : valid {0 unchanged, 1 added}; invalid -> event 0")
    print("T1 active event  : [0, 2]")
    print("T2 active event  : [0, 1]")
    print("XY NN radius     :", f"{args.nn_radius} m")
    print("max samples      :", args.max_samples if args.max_samples > 0 else "all")
    print("event_protocol   : not written")

    total_start = time.time()
    all_records = {}

    for split in args.splits:
        all_records[split] = prepare_split(
            split=split,
            source_root=source_root,
            output_root=output_root,
            nn_radius=float(args.nn_radius),
            max_samples=int(args.max_samples),
            vertex_name=args.vertex_name,
        )

    banner("DONE")
    for split, records in all_records.items():
        print(f"{split:5s}: {len(records):,} samples -> {output_root / 'manifests' / f'{split}.jsonl'}")
    print("total elapsed:", f"{time.time() - total_start:.2f}s")
    print()
    print("PAIR output is ready for config_loader directory inference.")


# =============================================================================
# Lightweight logic self-test
# =============================================================================

def _self_test():
    semantic, ignored_n, ignored_raw = canonicalize_semantic(
        np.asarray([0, 1, 2, 3, 9, -5], dtype=np.int64)
    )
    assert semantic.tolist() == [0, 1, 2, 3, -1, -1]
    assert ignored_n == 2
    assert ignored_raw == (-5, 9)

    change, ignored_n, ignored_raw = canonicalize_source_change(
        np.asarray([0, 1, 2, 3, 9], dtype=np.int64)
    )
    assert change.tolist() == [0, 1, 2, 3, -1]
    assert ignored_n == 1
    assert ignored_raw == (9,)

    source = np.asarray([0, 1, 2, 3, -1], dtype=np.int64)
    assert map_source_event(source).tolist() == [0, 1, 2, 1, 0]

    event2, valid2 = build_t2_event(source)
    assert event2.tolist() == [0, 1, 0, 1, 0]
    assert valid2.tolist() == [True, True, False, True, False]
    assert np.all(event2[~valid2] == 0)

    coord2 = np.asarray(
        [[0.0, 0.0, 1.0], [10.0, 0.0, 2.0], [20.0, 0.0, 3.0], [30.0, 0.0, 4.0]],
        dtype=np.float32,
    )
    coord1 = np.asarray(
        [[0.1, 0.0, 1.0], [10.1, 0.0, 2.0], [20.1, 0.0, 3.0], [30.1, 0.0, 4.0], [50.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    source_t1_test = np.asarray([0, 1, 2, 3], dtype=np.int64)
    event1, valid1, mapped, _ = build_t1_event(
        coord1, coord2, source_t1_test, radius=1.0
    )
    assert mapped.tolist() == [0, 1, 2, 3, -2]
    assert event1.tolist() == [0, 0, 2, 0, 0]
    assert valid1.tolist() == [True, False, True, False, False]
    assert np.all(event1[~valid1] == 0)

    assert not np.isin(event1[valid1], (1, 3, 4, 5)).any()
    assert not np.isin(event2[valid2], (2, 3, 4, 5)).any()
    print("prepare_nycs_event_v4 logic self-test: PASS")


if __name__ == "__main__":
    main()
