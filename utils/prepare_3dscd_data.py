#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare original NYC-SCD PLY pairs for PAIR.

Input
=====
ROOT/
└── version_training/
    ├── Train200/
    ├── Val40/
    └── Test40/
        └── sample_id/
            ├── pointCloud0.ply
            └── pointCloud1.ply

Output
======
NYC-SCDpair/
├── points_t1/
├── points_t2/
├── semantic_t1/
├── semantic_t2/
└── manifests/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl

Point files
===========
points_t1/sample.npz:
    coord : float32 [N1, 3]
    feat  : float32 [N1, 3]

points_t2/sample.npz:
    coord : float32 [N2, 3]
    feat  : float32 [N2, 3]

Supervision bundles
===================
semantic_t1/sample.npz:
    semantic : int64 [N1]
    change   : int64 [N1]

semantic_t2/sample.npz:
    semantic : int64 [N2]
    change   : int64 [N2]

Definitions
===========
- pointCloud0 -> PAIR T1
- pointCloud1 -> PAIR T2

semantic:
    Comes from official NYC-SCD label_mono.
    Raw IDs 0..3 are preserved.
    Other values -> -100 ignore.

T2 change:
    Comes from official pointCloud1.label_ch.
        raw 0       -> 0 unchanged
        raw 1/2/3   -> 1 changed
        other       -> -100 ignore

T1 change:
    NYC-SCD does not provide an official T1 change label.
    PAIR derives binary changed spatial support on the T1 topology by
    projecting official T2 label_ch to T1 using nearest-neighbor XY
    correspondence.

    If nearest T2 point is within --change-projection-radius:
        raw T2 0       -> T1 change 0
        raw T2 1/2/3   -> T1 change 1
        other          -> T1 change -100

    If no T2 point lies within the radius:
        T1 change -> -100

Why XY instead of XYZ?
======================
For ALS change detection, removed geometry can differ greatly in Z after
change. For example, a T1 building point and a T2 exposed-ground point may
share the same horizontal footprint while having very different heights.

Rules
=====
- Folders missing either pointCloud0.ply or pointCloud1.ply are skipped.
- N1 and N2 are independent and are NOT required to match.
- Original point order within each epoch is preserved.
- No crop, voxelization, normalization, resampling, or registration is done.
- Original PLY files are never modified or deleted.
- All complete source pairs and PLY headers are validated before writing.
- Existing output roots are never overwritten.
- No DatasetSpec or dataset_meta.json is generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IGNORE = -100
VALID_SEMANTIC = (0, 1, 2, 3)

RAW_SPLITS = {
    "Train200": "train",
    "Val40": "val",
    "Test40": "test",
}

REQ_T1 = {"x", "y", "z", "label_mono"}
REQ_T2 = {"x", "y", "z", "label_mono", "label_ch"}


@dataclass(frozen=True)
class Plan:
    raw_split: str
    split: str
    source_id: str
    sample_id: str
    src_t1: Path
    src_t2: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare NYC-SCD for PAIR bi-temporal 3D SCD."
    )

    parser.add_argument(
        "root",
        type=Path,
        help="NYC-SCD root or version_training root.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root. Default: sibling NYC-SCDpair.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train200", "Val40", "Test40"],
        choices=list(RAW_SPLITS),
        help="Raw split folders to prepare.",
    )

    parser.add_argument(
        "--change-projection-radius",
        type=float,
        default=0.75,
        help=(
            "Maximum XY nearest-neighbor distance in meters when deriving "
            "T1 binary change from official T2 label_ch. Default: 0.75."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate structure and headers only. Write nothing.",
    )

    return parser.parse_args()


def resolve_roots(root: Path, output: Path | None):
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}"
        )

    if (root / "version_training").is_dir():
        dataset_root = root
        source_root = root / "version_training"

    elif root.name == "version_training":
        dataset_root = root.parent
        source_root = root

    elif any((root / split).is_dir() for split in RAW_SPLITS):
        dataset_root = root
        source_root = root

    else:
        raise RuntimeError(
            f"Cannot locate version_training splits under: {root}"
        )

    if output is None:
        output_root = dataset_root.parent / f"{dataset_root.name}pair"
    else:
        output_root = output.expanduser().resolve()

    return source_root.resolve(), output_root.resolve()


def ply_properties(path: Path):
    props = []
    current_element = None

    with path.open("rb") as f:
        first = f.readline().decode(
            "ascii",
            errors="strict",
        ).strip()

        if first != "ply":
            raise RuntimeError(
                f"Not a valid PLY file: {path}"
            )

        while True:
            raw = f.readline()

            if not raw:
                raise RuntimeError(
                    f"PLY has no end_header: {path}"
                )

            line = raw.decode(
                "ascii",
                errors="strict",
            ).strip()

            if not line:
                continue

            parts = line.split()

            if parts[0] == "element" and len(parts) >= 3:
                current_element = parts[1]

            elif (
                parts[0] == "property"
                and current_element == "vertex"
            ):
                props.append(parts[-1])

            elif parts[0] == "end_header":
                return props


def scan_split(source_root: Path, raw_split: str):
    split_root = source_root / raw_split

    stats = {
        "exists": split_root.is_dir(),
        "total": 0,
        "usable": 0,
        "skipped": 0,
        "missing_t1": 0,
        "missing_t2": 0,
        "missing_both": 0,
        "examples": [],
    }

    if not split_root.is_dir():
        return [], stats

    split = RAW_SPLITS[raw_split]

    sample_dirs = sorted(
        path
        for path in split_root.iterdir()
        if path.is_dir()
    )

    stats["total"] = len(sample_dirs)

    plans = []

    for sample_dir in sample_dirs:
        point_t1 = sample_dir / "pointCloud0.ply"
        point_t2 = sample_dir / "pointCloud1.ply"

        has_t1 = point_t1.is_file()
        has_t2 = point_t2.is_file()

        if not has_t1 or not has_t2:
            stats["skipped"] += 1

            if not has_t1 and not has_t2:
                stats["missing_both"] += 1
                reason = (
                    "missing pointCloud0.ply "
                    "+ pointCloud1.ply"
                )

            elif not has_t1:
                stats["missing_t1"] += 1
                reason = "missing pointCloud0.ply"

            else:
                stats["missing_t2"] += 1
                reason = "missing pointCloud1.ply"

            if len(stats["examples"]) < 20:
                stats["examples"].append(
                    (sample_dir.name, reason)
                )

            continue

        plans.append(
            Plan(
                raw_split=raw_split,
                split=split,
                source_id=sample_dir.name,
                sample_id=f"{split}_{sample_dir.name}",
                src_t1=point_t1,
                src_t2=point_t2,
            )
        )

    stats["usable"] = len(plans)

    return plans, stats


def scan_all(source_root: Path, requested_splits):
    plans_by_split = {}
    stats_by_split = {}

    for raw_split in requested_splits:
        plans, stats = scan_split(
            source_root,
            raw_split,
        )

        split = RAW_SPLITS[raw_split]
        stats_by_split[split] = stats

        if plans:
            plans_by_split[split] = plans

    if not plans_by_split:
        raise RuntimeError(
            "No complete pointCloud0/pointCloud1 pairs found."
        )

    return plans_by_split, stats_by_split


def validate_headers(plans_by_split):
    first_t1_props = None
    first_t2_props = None

    for plans in plans_by_split.values():
        for plan in plans:
            props_t1 = ply_properties(plan.src_t1)
            props_t2 = ply_properties(plan.src_t2)

            missing_t1 = REQ_T1 - set(props_t1)
            missing_t2 = REQ_T2 - set(props_t2)

            if missing_t1:
                raise RuntimeError(
                    f"{plan.src_t1} missing "
                    f"{sorted(missing_t1)}; "
                    f"properties={props_t1}"
                )

            if missing_t2:
                raise RuntimeError(
                    f"{plan.src_t2} missing "
                    f"{sorted(missing_t2)}; "
                    f"properties={props_t2}"
                )

            if first_t1_props is None:
                first_t1_props = props_t1

            if first_t2_props is None:
                first_t2_props = props_t2

    return first_t1_props, first_t2_props


def require_packages():
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'plyfile'. Install with:\n"
            "  pip install plyfile"
        ) from exc

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'scipy'. Install with:\n"
            "  pip install scipy"
        ) from exc

    return PlyData, cKDTree


def int_field(vertex, name: str, path: Path):
    values = np.asarray(
        vertex[name]
    )

    if values.ndim != 1:
        raise RuntimeError(
            f"{path}: {name} must be 1D"
        )

    if np.issubdtype(
        values.dtype,
        np.floating,
    ):
        rounded = np.rint(
            values
        )

        if not np.allclose(
            values,
            rounded,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{path}: {name} contains "
                "non-integer values"
            )

        values = rounded

    return values.astype(
        np.int64,
        copy=False,
    )


def canonical_semantic(raw):
    out = np.full(
        raw.shape,
        IGNORE,
        dtype=np.int64,
    )

    valid = np.isin(
        raw,
        VALID_SEMANTIC,
    )

    out[valid] = raw[valid]

    return out


def canonical_change(raw):
    out = np.full(
        raw.shape,
        IGNORE,
        dtype=np.int64,
    )

    out[raw == 0] = 0

    out[
        np.isin(
            raw,
            (1, 2, 3),
        )
    ] = 1

    return out


def read_t1(path: Path, PlyData):
    with path.open("rb") as f:
        ply = PlyData.read(f)

    if "vertex" not in ply:
        raise RuntimeError(
            f"{path}: no vertex element"
        )

    vertex = ply["vertex"].data

    coord = np.column_stack(
        [
            vertex["x"],
            vertex["y"],
            vertex["z"],
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    semantic = canonical_semantic(
        int_field(
            vertex,
            "label_mono",
            path,
        )
    )

    if coord.shape[0] != semantic.shape[0]:
        raise RuntimeError(
            f"{path}: coord/semantic length mismatch"
        )

    if (
        coord.ndim != 2
        or coord.shape[1] != 3
        or not np.isfinite(coord).all()
    ):
        raise RuntimeError(
            f"{path}: invalid XYZ array "
            f"shape={coord.shape}"
        )

    return (
        np.ascontiguousarray(coord),
        np.ascontiguousarray(semantic),
    )


def read_t2(path: Path, PlyData):
    with path.open("rb") as f:
        ply = PlyData.read(f)

    if "vertex" not in ply:
        raise RuntimeError(
            f"{path}: no vertex element"
        )

    vertex = ply["vertex"].data

    coord = np.column_stack(
        [
            vertex["x"],
            vertex["y"],
            vertex["z"],
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    semantic = canonical_semantic(
        int_field(
            vertex,
            "label_mono",
            path,
        )
    )

    raw_change = int_field(
        vertex,
        "label_ch",
        path,
    )

    if (
        coord.shape[0] != semantic.shape[0]
        or coord.shape[0] != raw_change.shape[0]
    ):
        raise RuntimeError(
            f"{path}: coord/label length mismatch"
        )

    if (
        coord.ndim != 2
        or coord.shape[1] != 3
        or not np.isfinite(coord).all()
    ):
        raise RuntimeError(
            f"{path}: invalid XYZ array "
            f"shape={coord.shape}"
        )

    return (
        np.ascontiguousarray(coord),
        np.ascontiguousarray(semantic),
        np.ascontiguousarray(raw_change),
    )


def derive_change_t1(
    coord_t1,
    semantic_t1,
    coord_t2,
    raw_change_t2,
    radius,
    cKDTree,
):
    if coord_t1.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.float64),
        )

    if coord_t2.shape[0] == 0:
        return (
            np.full(
                coord_t1.shape[0],
                IGNORE,
                dtype=np.int64,
            ),
            np.zeros(
                coord_t1.shape[0],
                dtype=bool,
            ),
            np.full(
                coord_t1.shape[0],
                np.inf,
                dtype=np.float64,
            ),
        )

    tree = cKDTree(
        coord_t2[
            :,
            :2,
        ].astype(
            np.float64,
            copy=False,
        )
    )

    distance, index = tree.query(
        coord_t1[
            :,
            :2,
        ].astype(
            np.float64,
            copy=False,
        ),
        k=1,
        distance_upper_bound=radius,
        workers=-1,
    )

    covered = np.isfinite(
        distance
    )

    projected_raw = np.full(
        coord_t1.shape[0],
        IGNORE,
        dtype=np.int64,
    )

    projected_raw[covered] = (
        raw_change_t2[
            index[covered]
        ]
    )

    change_t1 = canonical_change(
        projected_raw
    )

    change_t1[
        semantic_t1 == IGNORE
    ] = IGNORE

    return (
        np.ascontiguousarray(change_t1),
        covered,
        distance,
    )


def build_change_t2(
    raw_change_t2,
    semantic_t2,
):
    change_t2 = canonical_change(
        raw_change_t2
    )

    change_t2[
        semantic_t2 == IGNORE
    ] = IGNORE

    return np.ascontiguousarray(
        change_t2
    )


def paths_for(plan: Plan):
    sample_id = plan.sample_id

    return {
        "point_t1": (
            Path("points_t1")
            / f"{sample_id}.npz"
        ),
        "point_t2": (
            Path("points_t2")
            / f"{sample_id}.npz"
        ),
        "semantic_t1": (
            Path("semantic_t1")
            / f"{sample_id}.npz"
        ),
        "semantic_t2": (
            Path("semantic_t2")
            / f"{sample_id}.npz"
        ),
    }


def manifest_record(plan: Plan):
    paths = paths_for(
        plan
    )

    return {
        "id": plan.sample_id,
        **{
            key: value.as_posix()
            for key, value
            in paths.items()
        },
    }


def save_point_npz(
    path: Path,
    coord,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feat = np.ascontiguousarray(
        coord.copy()
    )

    with path.open("wb") as f:
        np.savez(
            f,
            coord=np.ascontiguousarray(coord),
            feat=feat,
        )


def save_supervision_npz(
    path: Path,
    semantic,
    change,
):
    if semantic.shape != change.shape:
        raise RuntimeError(
            f"semantic/change shape mismatch "
            f"for {path}: "
            f"{semantic.shape} vs {change.shape}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("wb") as f:
        np.savez(
            f,
            semantic=np.ascontiguousarray(
                semantic,
            ),
            change=np.ascontiguousarray(
                change,
            ),
        )


def add_histogram(
    counter: Counter,
    array,
):
    values, counts = np.unique(
        array,
        return_counts=True,
    )

    for value, count in zip(
        values.tolist(),
        counts.tolist(),
    ):
        counter[int(value)] += int(count)


def write_manifest(
    path: Path,
    plans,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for plan in plans:
            f.write(
                json.dumps(
                    manifest_record(plan),
                    ensure_ascii=False,
                )
                + "\n"
            )


def convert(
    staging_root: Path,
    plans_by_split,
    radius: float,
):
    PlyData, cKDTree = require_packages()

    result = {}

    for split, plans in plans_by_split.items():
        semantic_t1_hist = Counter()
        semantic_t2_hist = Counter()
        change_t1_hist = Counter()
        change_t2_hist = Counter()

        n_t1_total = 0
        n_t2_total = 0

        covered_total = 0
        query_total = 0

        distance_sum = 0.0
        distance_count = 0

        print()
        print(
            f"[{split}] "
            f"{len(plans)} complete pairs"
        )

        for index, plan in enumerate(
            plans,
            start=1,
        ):
            (
                coord_t1,
                semantic_t1,
            ) = read_t1(
                plan.src_t1,
                PlyData,
            )

            (
                coord_t2,
                semantic_t2,
                raw_change_t2,
            ) = read_t2(
                plan.src_t2,
                PlyData,
            )

            change_t2 = build_change_t2(
                raw_change_t2,
                semantic_t2,
            )

            (
                change_t1,
                covered,
                distances,
            ) = derive_change_t1(
                coord_t1,
                semantic_t1,
                coord_t2,
                raw_change_t2,
                radius,
                cKDTree,
            )

            paths = paths_for(
                plan
            )

            save_point_npz(
                staging_root
                / paths["point_t1"],
                coord_t1,
            )

            save_point_npz(
                staging_root
                / paths["point_t2"],
                coord_t2,
            )

            save_supervision_npz(
                staging_root
                / paths["semantic_t1"],
                semantic_t1,
                change_t1,
            )

            save_supervision_npz(
                staging_root
                / paths["semantic_t2"],
                semantic_t2,
                change_t2,
            )

            n_t1_total += int(
                coord_t1.shape[0]
            )

            n_t2_total += int(
                coord_t2.shape[0]
            )

            query_total += int(
                covered.shape[0]
            )

            covered_total += int(
                covered.sum()
            )

            valid_distances = distances[
                covered
            ]

            if valid_distances.size:
                distance_sum += float(
                    valid_distances.sum()
                )

                distance_count += int(
                    valid_distances.size
                )

            add_histogram(
                semantic_t1_hist,
                semantic_t1,
            )

            add_histogram(
                semantic_t2_hist,
                semantic_t2,
            )

            add_histogram(
                change_t1_hist,
                change_t1,
            )

            add_histogram(
                change_t2_hist,
                change_t2,
            )

            if (
                index == 1
                or index % 20 == 0
                or index == len(plans)
            ):
                coverage = (
                    100.0
                    * float(
                        covered.mean()
                    )
                    if covered.size
                    else 0.0
                )

                print(
                    f"  {index:5d}/"
                    f"{len(plans):5d} "
                    f"{plan.source_id} | "
                    f"N1={coord_t1.shape[0]:,} "
                    f"N2={coord_t2.shape[0]:,} | "
                    f"T1 XY-covered="
                    f"{coverage:.2f}%"
                )

        write_manifest(
            staging_root
            / "manifests"
            / f"{split}.jsonl",
            plans,
        )

        result[split] = {
            "samples": len(plans),
            "n_t1": n_t1_total,
            "n_t2": n_t2_total,
            "semantic_t1": dict(
                sorted(
                    semantic_t1_hist.items()
                )
            ),
            "semantic_t2": dict(
                sorted(
                    semantic_t2_hist.items()
                )
            ),
            "change_t1": dict(
                sorted(
                    change_t1_hist.items()
                )
            ),
            "change_t2": dict(
                sorted(
                    change_t2_hist.items()
                )
            ),
            "coverage": (
                covered_total / query_total
                if query_total
                else 0.0
            ),
            "mean_distance": (
                distance_sum / distance_count
                if distance_count
                else float("nan")
            ),
        }

    return result


def main():
    args = parse_args()

    if args.change_projection_radius <= 0:
        raise ValueError(
            "--change-projection-radius "
            "must be > 0"
        )

    (
        source_root,
        output_root,
    ) = resolve_roots(
        args.root,
        args.output,
    )

    (
        plans_by_split,
        scan_stats,
    ) = scan_all(
        source_root,
        args.splits,
    )

    (
        t1_properties,
        t2_properties,
    ) = validate_headers(
        plans_by_split
    )

    print("=" * 80)
    print(
        "PAIR NYC-SCD "
        "BI-TEMPORAL PREPARATION"
    )
    print("=" * 80)

    print(
        f"source  : {source_root}"
    )

    print(
        f"output  : {output_root}"
    )

    print(
        "T1 change projection radius: "
        f"{args.change_projection_radius:.3f} m"
    )

    print()

    for split in (
        "train",
        "val",
        "test",
    ):
        stats = scan_stats.get(
            split
        )

        if (
            stats is None
            or not stats["exists"]
        ):
            print(
                f"{split:8s}: "
                "missing split folder -> skipped"
            )
            continue

        print(
            f"{split:8s}: "
            f"{stats['usable']} usable / "
            f"{stats['total']} folders | "
            f"skipped={stats['skipped']}"
        )

        if stats["skipped"]:
            print(
                "          "
                f"missing pointCloud0="
                f"{stats['missing_t1']} | "
                f"missing pointCloud1="
                f"{stats['missing_t2']} | "
                f"missing both="
                f"{stats['missing_both']}"
            )

            for (
                sample_id,
                reason,
            ) in stats["examples"][:5]:
                print(
                    f"          skip: "
                    f"{sample_id} "
                    f"({reason})"
                )

            if stats["skipped"] > 5:
                print(
                    "          ... "
                    f"{stats['skipped'] - 5} more"
                )

    print()

    print(
        "pointCloud0 properties:",
        ", ".join(
            t1_properties
        ),
    )

    print(
        "pointCloud1 properties:",
        ", ".join(
            t2_properties
        ),
    )

    print()

    print(
        "Supervision file format:"
    )

    print(
        "  semantic_t1/*.npz "
        "-> semantic + change"
    )

    print(
        "  semantic_t2/*.npz "
        "-> semantic + change"
    )

    print()

    print(
        "Validation: PASS"
    )

    if args.dry_run:
        print(
            "DRY RUN COMPLETE -- "
            "no files were written."
        )
        return

    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: "
            f"{output_root}"
        )

    staging_root = (
        output_root.parent
        / f".{output_root.name}.prepare_tmp"
    )

    if staging_root.exists():
        raise FileExistsError(
            "Staging directory already exists: "
            f"{staging_root}"
        )

    staging_root.mkdir(
        parents=True
    )

    try:
        stats = convert(
            staging_root,
            plans_by_split,
            args.change_projection_radius,
        )

        staging_root.rename(
            output_root
        )

    except Exception:
        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )
        raise

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    for split, item in stats.items():
        print(
            f"{split:8s}: "
            f"{item['samples']} samples | "
            f"T1={item['n_t1']:,} | "
            f"T2={item['n_t2']:,}"
        )

        print(
            "          "
            f"semantic T1="
            f"{item['semantic_t1']} | "
            f"change T1="
            f"{item['change_t1']}"
        )

        print(
            "          "
            f"semantic T2="
            f"{item['semantic_t2']} | "
            f"change T2="
            f"{item['change_t2']}"
        )

        print(
            "          "
            f"T1 XY-covered="
            f"{100.0 * item['coverage']:.2f}% | "
            f"mean NN="
            f"{item['mean_distance']:.3f} m"
        )

    print()
    print(
        "Final dataset layout:"
    )
    print(
        "  points_t1/"
    )
    print(
        "  points_t2/"
    )
    print(
        "  semantic_t1/"
    )
    print(
        "  semantic_t2/"
    )
    print(
        "  manifests/"
    )
    print()

    print(
        "Each semantic_*.npz contains:"
    )
    print(
        "  semantic [N]"
    )
    print(
        "  change   [N]"
    )

    print()

    print(
        "Original NYC-SCD PLY files "
        "were kept unchanged."
    )

    print(
        "DatasetSpec is intentionally "
        "NOT generated."
    )


if __name__ == "__main__":
    main()
