#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NYC-SCD -> PAIR three-state event remapping v21.

v21 keeps the complete v20 pipeline and changes exactly one dataset-policy
decision: vegetation is excluded from direct event supervision.

    T1: unchanged / removed, but vegetation is always unchanged
    T2: unchanged / added,   but vegetation is always unchanged

This follows the source NYC-SCD protocol, whose change annotation does not
attempt to exhaustively label vegetation dynamics.  Raw ``label_ch`` still
defines the exact candidate XY cells.  Semantic ``label_mono`` remains
point-for-point immutable.  There is no morphology or object expansion.

Examples:

    vegetation -> building : T1 vegetation unchanged, T2 building added
    building -> vegetation : T1 building removed, T2 vegetation unchanged
    building -> clutter    : T1 building removed, T2 clutter added

The CLI is intentionally the same as v20, including ``--test true``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import remap_nycs_pair_events_v20 as base


REMAPPER_VERSION = "v21_three_state_ignore_vegetation_events"
EVENT_SEMANTIC_IDS = (1, 3)  # building, clutter; vegetation (2) is excluded


def parse_args():
    parser = argparse.ArgumentParser(
        description="NYC-SCD -> PAIR three-state remapping v21 (vegetation ignored)."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=base.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val")
    )
    parser.add_argument("--footprint-grid", type=float, default=0.50)
    parser.add_argument("--match-xy", type=float, default=0.75)
    parser.add_argument("--match-z", type=float, default=1.00)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--vertex-name", default="vertex")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=(),
        help="Optional exact sample IDs.",
    )
    parser.add_argument(
        "--focus",
        type=base.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Process the three agreed inspection samples only.",
    )
    parser.add_argument(
        "--test",
        type=base.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Process only the ten known inspection samples.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def remap_pair_events(
    coord_t1: np.ndarray,
    semantic_t1: np.ndarray,
    coord_t2: np.ndarray,
    semantic_t2: np.ndarray,
    source_change_t2: np.ndarray,
    cfg: base.RemapConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Apply v20 correspondence logic while excluding vegetation events."""
    coord_t1 = np.asarray(coord_t1, dtype=np.float32)
    coord_t2 = np.asarray(coord_t2, dtype=np.float32)
    semantic_t1 = np.asarray(semantic_t1, dtype=np.int64).reshape(-1)
    semantic_t2 = np.asarray(semantic_t2, dtype=np.int64).reshape(-1)
    source_change_t2 = np.asarray(source_change_t2, dtype=np.int64).reshape(-1)
    if coord_t1.shape != (semantic_t1.size, 3):
        raise ValueError("T1 coord/semantic topology mismatch")
    if coord_t2.shape != (semantic_t2.size, 3):
        raise ValueError("T2 coord/semantic topology mismatch")
    if source_change_t2.size != semantic_t2.size:
        raise ValueError("T2 label_ch topology mismatch")

    valid_semantic_t1 = (semantic_t1 >= 0) & (semantic_t1 <= 3)
    valid_semantic_t2 = (semantic_t2 >= 0) & (semantic_t2 <= 3)
    valid_t1 = valid_semantic_t1.copy()
    valid_source_t2 = (source_change_t2 >= 0) & (source_change_t2 <= 3)
    valid_t2 = valid_semantic_t2 & valid_source_t2

    grid = base.build_footprint_grid(coord_t1, coord_t2, cfg.footprint_grid)
    flat_t1 = base.point_cells(coord_t1, grid)
    flat_t2 = base.point_cells(coord_t2, grid)
    changed_source_points = valid_source_t2 & (source_change_t2 != 0)
    candidate_cells = np.zeros(grid.count, dtype=np.bool_)
    candidate_cells[flat_t2[changed_source_points]] = True

    # This is the only intentional difference from v20: semantic 2 is absent.
    candidate_t1 = candidate_cells[flat_t1] & np.isin(semantic_t1, EVENT_SEMANTIC_IDS)
    candidate_t2 = (
        candidate_cells[flat_t2]
        & np.isin(semantic_t2, EVENT_SEMANTIC_IDS)
        & valid_t2
    )

    support_t1 = base.same_semantic_support(
        coord_t1,
        semantic_t1,
        coord_t2,
        semantic_t2,
        candidate_t1,
        match_xy=cfg.match_xy,
        match_z=cfg.match_z,
    )
    support_t2 = base.same_semantic_support(
        coord_t2,
        semantic_t2,
        coord_t1,
        semantic_t1,
        candidate_t2,
        match_xy=cfg.match_xy,
        match_z=cfg.match_z,
    )

    event_t1 = np.zeros(semantic_t1.size, dtype=np.int8)
    event_t2 = np.zeros(semantic_t2.size, dtype=np.int8)
    event_t1[candidate_t1 & ~support_t1] = 2
    event_t2[candidate_t2 & ~support_t2] = 1
    event_t1[~valid_t1] = 0
    event_t2[~valid_t2] = 0

    if np.any((event_t1 != 0) & ~candidate_cells[flat_t1]):
        raise RuntimeError("v21 invariant failed: T1 event escaped exact source footprint")
    if np.any((event_t2 != 0) & ~candidate_cells[flat_t2]):
        raise RuntimeError("v21 invariant failed: T2 event escaped exact source footprint")
    if np.any(event_t1[semantic_t1 == 0] != 0) or np.any(event_t2[semantic_t2 == 0] != 0):
        raise RuntimeError("v21 invariant failed: ground received a non-zero event")
    if np.any(event_t1[semantic_t1 == 2] != 0) or np.any(event_t2[semantic_t2 == 2] != 0):
        raise RuntimeError("v21 invariant failed: vegetation received a non-zero event")
    if not set(np.unique(event_t1)).issubset({0, 2}):
        raise RuntimeError("v21 invariant failed: T1 may contain only unchanged/removed")
    if not set(np.unique(event_t2)).issubset({0, 1}):
        raise RuntimeError("v21 invariant failed: T2 may contain only unchanged/added")

    debug = {
        "remap_version": REMAPPER_VERSION,
        "semantic_policy": "immutable_label_mono",
        "vegetation_event_policy": "semantic_2_always_unchanged",
        "event_protocol": base.EVENT_NAMES,
        "config": asdict(cfg),
        "grid": {
            "x0": grid.x0,
            "y0": grid.y0,
            "size": grid.size,
            "nx": grid.nx,
            "ny": grid.ny,
            "cells": grid.count,
        },
        "source_changed_points_t2": int(changed_source_points.sum()),
        "ignored_semantic_points_t1": int((~valid_semantic_t1).sum()),
        "ignored_semantic_points_t2": int((~valid_semantic_t2).sum()),
        "candidate_cells": int(candidate_cells.sum()),
        "candidate_points_t1": int(candidate_t1.sum()),
        "candidate_points_t2": int(candidate_t2.sum()),
        "same_semantic_supported_t1": int((candidate_t1 & support_t1).sum()),
        "same_semantic_supported_t2": int((candidate_t2 & support_t2).sum()),
        "event_t1": base.hist_dict(event_t1[valid_t1], base.EVENT_NAMES),
        "event_t2": base.hist_dict(event_t2[valid_t2], base.EVENT_NAMES),
        "quality_flag": "ok" if np.any(valid_t1) and np.any(valid_t2) else "invalid",
    }
    return event_t1, valid_t1, event_t2, valid_t2, debug


def write_protocol(output_root: Path, cfg: base.RemapConfig):
    base.atomic_write_json(
        output_root / "event_protocol.json",
        {
            "version": REMAPPER_VERSION,
            "semantic": base.SEMANTIC_NAMES,
            "semantic_policy": "label_mono is immutable absolute ground truth",
            "event": base.EVENT_NAMES,
            "epoch_constraints": {
                "t1": ["unchanged", "removed"],
                "t2": ["unchanged", "added"],
            },
            "candidate_policy": "exact XY cells containing raw T2 label_ch != 0; no morphology or object expansion",
            "ground_policy": "ground is always unchanged",
            "vegetation_policy": "semantic class 2 is always unchanged in both epochs",
            "event_semantic_classes": {
                "1": "building",
                "3": "clutter",
            },
            "matching_policy": "same-semantic 3-D support; tolerance only suppresses false events",
            "derived_only": ["semantic/class transition", "height up", "height down"],
            "config": asdict(cfg),
        },
    )


def run_self_test():
    cfg = base.RemapConfig(footprint_grid=0.5, match_xy=0.2, match_z=0.2)

    # A vegetation -> building transition: vegetation is deliberately not
    # supervised, while the new building remains added.
    coord_t1 = np.array([[0.1, 0.1, 2.0]], dtype=np.float32)
    sem_t1 = np.array([2], dtype=np.int64)
    coord_t2 = np.array([[0.1, 0.1, 5.0]], dtype=np.float32)
    sem_t2 = np.array([1], dtype=np.int64)
    event_t1, _, event_t2, _, _ = remap_pair_events(
        coord_t1, sem_t1, coord_t2, sem_t2, np.array([1]), cfg
    )
    assert event_t1.tolist() == [0]
    assert event_t2.tolist() == [1]

    # A building -> vegetation transition: old building is removed, new
    # vegetation is deliberately not supervised.
    coord_t1 = np.array([[1.1, 0.1, 5.0]], dtype=np.float32)
    sem_t1 = np.array([1], dtype=np.int64)
    coord_t2 = np.array([[1.1, 0.1, 2.0]], dtype=np.float32)
    sem_t2 = np.array([2], dtype=np.int64)
    event_t1, _, event_t2, _, _ = remap_pair_events(
        coord_t1, sem_t1, coord_t2, sem_t2, np.array([2]), cfg
    )
    assert event_t1.tolist() == [2]
    assert event_t2.tolist() == [0]

    # Vegetation dynamics alone never become an event, even inside a raw
    # changed cell.
    coord_t1 = np.array([[2.1, 0.1, 1.0]], dtype=np.float32)
    coord_t2 = np.array([[2.1, 0.1, 6.0]], dtype=np.float32)
    semantic = np.array([2], dtype=np.int64)
    event_t1, _, event_t2, _, _ = remap_pair_events(
        coord_t1, semantic, coord_t2, semantic, np.array([3]), cfg
    )
    assert event_t1.tolist() == [0] and event_t2.tolist() == [0]
    print("v21 self-test passed: vegetation events ignored; building add/remove retained")


def main():
    args = parse_args()
    cfg = base.RemapConfig(
        footprint_grid=float(args.footprint_grid),
        match_xy=float(args.match_xy),
        match_z=float(args.match_z),
    )
    base.validate_config(cfg)
    if args.self_test:
        run_self_test()
        return
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    selection_modes = int(bool(args.test)) + int(bool(args.focus)) + int(bool(args.ids))
    if selection_modes > 1:
        raise ValueError("use only one of --test, --focus, or --ids")
    selected_ids = tuple(
        base.KNOWN_TEST_SAMPLE_IDS
        if args.test
        else base.FOCUS_SAMPLE_IDS
        if args.focus
        else args.ids
    )
    run_splits = (
        tuple(sorted({sample_id.split("_", 1)[0] for sample_id in selected_ids}))
        if selected_ids
        else tuple(args.splits)
    )
    manifest_suffix = "_test" if args.test else ""

    for name in (
        "points_t1",
        "points_t2",
        "semantic_t1",
        "semantic_t2",
        "manifests",
        "remap_debug",
        "qa",
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    write_protocol(output_root, cfg)

    # remap_one lives in the stable v20 I/O framework and resolves this global
    # at runtime.  Replacing it here keeps all file handling identical while
    # applying only the v21 decision function above.
    base.remap_pair_events = remap_pair_events

    base.banner("NYC-SCD -> PAIR THREE-STATE EVENT REMAPPING v21")
    print("source root       :", source_root)
    print("output root       :", output_root)
    print("semantic          : immutable", base.SEMANTIC_NAMES)
    print("event             :", base.EVENT_NAMES)
    print("T1 valid events   : unchanged / removed")
    print("T2 valid events   : unchanged / added")
    print("candidate region  : exact raw changed XY cells; no expansion")
    print("ground            : always unchanged")
    print("vegetation        : always unchanged in both epochs")
    print("same-semantic 3-D :", f"xy<={cfg.match_xy}m, z<={cfg.match_z}m")
    print("test mode         :", bool(args.test))
    print("selected IDs      :", list(selected_ids) if selected_ids else "all")

    start = time.time()
    all_records = []
    for split in run_splits:
        all_records.extend(
            base.remap_split(
                split,
                source_root,
                output_root,
                cfg,
                selected_ids,
                int(args.max_samples),
                args.vertex_name,
                manifest_suffix,
            )
        )
    base.banner("V21 REMAPPING DONE")
    print("samples:", len(all_records))
    print("elapsed:", f"{time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
