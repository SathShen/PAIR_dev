#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic 颜色

-1 ignore：深灰色
0 ground：米黄色 / 土黄色
1 building：红色
2 vegetation：绿色
3 clutter：淡紫色
未知 semantic ID：亮紫红色

NEW Event 颜色

0 unchanged：浅灰色
1 added：青蓝色
2 removed：橙红色
3 class_change：紫红色
4 height_up：黄色
5 height_down：蓝色
event_valid=False：深灰色 / 接近黑色

OLD RAW T2 Change 颜色

-1 ignore：深灰色
0 unchanged：浅灰色
1 newly_built：青蓝色
2 demolition：橙红色
3 new_clutter：紫红色
Visualize NEW vs OLD NYC-SCD/PAIR labels from TWO ROOT PATHS.

One sample = one Open3D GUI window with 6 panels:

    NEW T1 Semantic | NEW T2 Semantic
    NEW T1 Event    | NEW T2 Event
    OLD T1 Event    | OLD T2 Event

Default old-mode="auto":
1) If OLD_ROOT is a PAIR-style prepared dataset:
       points_t1/
       points_t2/
       semantic_t1/
       semantic_t2/
   bottom row shows OLD T1 Event / OLD T2 Event.

2) If OLD_ROOT is the official NYC-SCD version_training root:
       Train200/<sample>/pointCloud0.ply, pointCloud1.ply
       Val40/<sample>/...
       Test40/<sample>/...
   bottom row shows:
       OLD T1 Source Semantic(label_mono)
       OLD T2 Source Change(label_ch)

This lets you run directly with two paths, no manifest required.

Examples
--------
Compare new remapping against the old prepared PAIR labels:

python visualize_new_old_6panels.py \
    --new-root /home/sht/Datasets/NYC-SCDpair_remapped \
    --old-root /home/sht/Datasets/NYC-SCDpair \
    --split train \
    --num-samples 10

Compare new remapping against the OFFICIAL raw NYC-SCD source:

python visualize_new_old_6panels.py \
    --new-root /home/sht/Datasets/NYC-SCDpair_remapped \
    --old-root /home/sht/Datasets/NYC-SCD/version_training \
    --old-mode raw \
    --split train \
    --num-samples 10

Randomized spot-check batch:

python visualize_new_old_6panels.py \
    --new-root /home/sht/Datasets/NYC-SCDpair_remapped \
    --old-root /home/sht/Datasets/NYC-SCD/version_training \
    --old-mode raw \
    --split train \
    --num-samples 10 \
    --shuffle true \
    --seed 42

Change --seed to inspect a different reproducible batch.

Dependencies
------------
pip install numpy open3d
pip install plyfile   # only needed for --old-mode raw
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError as exc:
    raise SystemExit(
        "Open3D is required.\nInstall with:\n    pip install open3d\n"
    ) from exc


# =============================================================================
# Labels / colors
# =============================================================================

SEMANTIC_NAMES = {
    -1: "ignore",
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
    -1: "ignore",
    0: "unchanged",
    1: "newly_built",
    2: "demolition",
    3: "new_clutter",
}

SEMANTIC_COLORS = {
    -1: (0.20, 0.20, 0.20),
    0: (0.82, 0.77, 0.62),
    1: (0.93, 0.30, 0.30),
    2: (0.28, 0.82, 0.32),
    3: (0.52, 0.50, 0.88),
}

EVENT_COLORS = {
    0: (0.76, 0.76, 0.76),  # unchanged
    1: (0.10, 0.82, 0.95),  # added
    2: (1.00, 0.36, 0.04),  # removed
    3: (0.86, 0.28, 0.86),  # class_change
    4: (0.98, 0.82, 0.08),  # height_up
    5: (0.20, 0.42, 1.00),  # height_down
}

SOURCE_CHANGE_COLORS = {
    -1: (0.15, 0.15, 0.15),
    0: (0.76, 0.76, 0.76),
    1: (0.10, 0.82, 0.95),  # newly built
    2: (1.00, 0.36, 0.04),  # demolition
    3: (0.86, 0.28, 0.86),  # new clutter
}

INVALID_EVENT_COLOR = (0.10, 0.10, 0.10)
BACKGROUND = (0.025, 0.025, 0.030, 1.0)


# =============================================================================
# CLI
# =============================================================================

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got {value!r}. "
        "Use true/false, yes/no, 1/0, or on/off."
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Six-panel comparison of NEW and OLD NYC-SCD/PAIR labels."
    )
    p.add_argument("--new-root", type=Path, required=True)
    p.add_argument("--old-root", type=Path, required=True)
    p.add_argument(
        "--old-mode",
        choices=("auto", "prepared", "raw"),
        default="auto",
        help=(
            "prepared = old PAIR-style root; "
            "raw = official NYC-SCD version_training; "
            "auto = detect automatically"
        ),
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )
    p.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Explicit NEW sample ids. Example: train_18TWK....",
    )
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument(
        "--shuffle",
        type=str2bool,
        default=False,
        metavar="BOOL",
        help=(
            "Shuffle comparable sample ids before taking --num-samples. "
            "Use --seed for reproducible spot-check batches. "
            "Default: false."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-points",
        type=int,
        default=300_000,
        help="Maximum displayed points per epoch/panel.",
    )
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--window-width", type=int, default=1400)
    p.add_argument("--window-height", type=int, default=1000)
    return p.parse_args()


# =============================================================================
# NPZ / PAIR-style loading
# =============================================================================

def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def is_prepared_root(root: Path) -> bool:
    return all(
        (root / name).is_dir()
        for name in ("points_t1", "points_t2", "semantic_t1", "semantic_t2")
    )


def is_raw_root(root: Path) -> bool:
    return any((root / name).is_dir() for name in ("Train200", "Val40", "Test40"))


def detect_old_mode(root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if is_prepared_root(root):
        return "prepared"
    if is_raw_root(root):
        return "raw"
    raise FileNotFoundError(
        f"Cannot detect old dataset type from {root}.\n"
        "Expected either PAIR-style points_t1/... directories or "
        "official Train200/Val40/Test40 directories."
    )


def discover_prepared_ids(root: Path) -> List[str]:
    dirs = [
        root / "points_t1",
        root / "points_t2",
        root / "semantic_t1",
        root / "semantic_t2",
    ]
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(d)

    sets = [{p.stem for p in d.glob("*.npz")} for d in dirs]
    ids = sorted(set.intersection(*sets))
    if not ids:
        raise RuntimeError(f"No complete prepared samples under {root}")
    return ids


def load_prepared_epoch(root: Path, sample_id: str, epoch: str):
    points = load_npz(root / f"points_{epoch}" / f"{sample_id}.npz")
    labels = load_npz(root / f"semantic_{epoch}" / f"{sample_id}.npz")

    if "coord" not in points:
        raise KeyError(f"{sample_id}/{epoch}: point file missing coord")
    for key in ("semantic", "event", "event_valid"):
        if key not in labels:
            raise KeyError(f"{sample_id}/{epoch}: label file missing {key}")

    coord = np.asarray(points["coord"], dtype=np.float64)
    semantic = np.asarray(labels["semantic"]).reshape(-1).astype(np.int64)
    event = np.asarray(labels["event"]).reshape(-1).astype(np.int64)
    event_valid = np.asarray(labels["event_valid"]).reshape(-1).astype(bool)

    n = coord.shape[0]
    if semantic.shape[0] != n or event.shape[0] != n or event_valid.shape[0] != n:
        raise ValueError(
            f"{sample_id}/{epoch}: topology mismatch: "
            f"coord={n}, semantic={len(semantic)}, event={len(event)}, "
            f"event_valid={len(event_valid)}"
        )

    # NEW protocol: either epoch may legitimately contain event ids 0..5.
    bad = sorted(set(np.unique(event[event_valid]).tolist()) - set(range(6)))
    if bad:
        print(f"[WARN] {sample_id}/{epoch}: unknown valid event ids {bad}")

    return {
        "coord": coord,
        "semantic": semantic,
        "event": event,
        "event_valid": event_valid,
    }


# =============================================================================
# Official raw NYC-SCD loading
# =============================================================================

RAW_SPLIT_DIR = {
    "train": "Train200",
    "val": "Val40",
    "test": "Test40",
}


def strip_split_prefix(sample_id: str) -> str:
    lower = sample_id.lower()
    for prefix in ("train_", "val_", "test_"):
        if lower.startswith(prefix):
            return sample_id[len(prefix):]
    return sample_id


def raw_sample_dir(old_root: Path, sample_id: str, split: str) -> Path:
    split_dir = old_root / RAW_SPLIT_DIR[split]
    raw_name = strip_split_prefix(sample_id)
    direct = split_dir / raw_name
    if direct.is_dir():
        return direct

    # Defensive fallback for case differences.
    matches = [
        p for p in split_dir.iterdir()
        if p.is_dir() and p.name.lower() == raw_name.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Raw sample for {sample_id!r} not found under {split_dir}"
    )


def read_raw_ply(path: Path, require_change: bool):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError(
            "Raw NYC-SCD visualization needs plyfile:\n"
            "    pip install plyfile"
        ) from exc

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])

    for k in ("x", "y", "z", "label_mono"):
        if k not in names:
            raise KeyError(f"{path}: missing field {k}")
    if require_change and "label_ch" not in names:
        raise KeyError(f"{path}: missing field label_ch")

    coord = np.column_stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ]
    )
    semantic = np.asarray(vertex["label_mono"], dtype=np.int64)
    out = {"coord": coord, "semantic": semantic}
    if "label_ch" in names:
        out["label_ch"] = np.asarray(vertex["label_ch"], dtype=np.int64)
    return out


def load_raw_pair(old_root: Path, sample_id: str, split: str):
    d = raw_sample_dir(old_root, sample_id, split)
    t1 = read_raw_ply(d / "pointCloud0.ply", require_change=False)
    t2 = read_raw_ply(d / "pointCloud1.ply", require_change=True)
    return t1, t2


# =============================================================================
# Sampling / coloring
# =============================================================================

def keep_changed_first(event, valid, max_points, seed):
    n = event.shape[0]
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    changed = np.flatnonzero(valid & (event != 0))
    if changed.size >= max_points:
        out = rng.choice(changed, size=max_points, replace=False)
        return np.sort(out.astype(np.int64))

    remain_budget = max_points - changed.size
    mask = np.ones(n, dtype=bool)
    mask[changed] = False
    remain = np.flatnonzero(mask)
    if remain.size > remain_budget:
        remain = rng.choice(remain, size=remain_budget, replace=False)

    out = np.concatenate([changed, remain])
    out.sort()
    return out.astype(np.int64)


def random_indices(n: int, max_points: int, seed: int):
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = rng.choice(n, size=max_points, replace=False)
    out.sort()
    return out.astype(np.int64)


def colors_from_ids(ids, lut, default=(1.0, 0.0, 1.0)):
    ids = np.asarray(ids).reshape(-1)
    out = np.tile(np.asarray(default, dtype=np.float64), (ids.shape[0], 1))
    for k, color in lut.items():
        out[ids == k] = np.asarray(color, dtype=np.float64)
    return out


def semantic_colors(semantic):
    return colors_from_ids(semantic, SEMANTIC_COLORS)


def event_colors(event, valid):
    event = np.asarray(event).reshape(-1)
    valid = np.asarray(valid).reshape(-1).astype(bool)

    out = np.tile(
        np.asarray(INVALID_EVENT_COLOR, dtype=np.float64),
        (event.shape[0], 1),
    )
    for k, color in EVENT_COLORS.items():
        out[valid & (event == k)] = np.asarray(color, dtype=np.float64)
    return out


def source_change_colors(label_ch):
    label_ch = np.asarray(label_ch).reshape(-1).astype(np.int64)
    canonical = label_ch.copy()
    canonical[(canonical < 0) | (canonical > 3)] = -1
    return colors_from_ids(canonical, SOURCE_CHANGE_COLORS)


def hist_text(values, names, valid=None):
    values = np.asarray(values).reshape(-1)
    if valid is not None:
        valid = np.asarray(valid).reshape(-1).astype(bool)
        values = values[valid]

    if values.size == 0:
        return "none"

    u, c = np.unique(values, return_counts=True)
    return " ".join(
        f"{names.get(int(k), str(int(k)))}={int(v):,}"
        for k, v in zip(u, c)
    )


# =============================================================================
# Open3D GUI
# =============================================================================

def make_point_cloud(coord, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(coord, dtype=np.float64)
    )
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64)
    )
    return pcd


def make_material(point_size):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)
    return mat


def bbox_from_points(points):
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    hi = np.where((hi - lo) < 1e-6, lo + 1e-3, hi)
    return o3d.geometry.AxisAlignedBoundingBox(lo, hi)


def set_oblique_camera(widget, points):
    bbox = bbox_from_points(points)
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = max(float(np.linalg.norm(extent)), 1.0)
    eye = center + np.asarray([0.72, -0.72, 0.58]) * radius
    up = np.asarray([0.0, 0.0, 1.0])
    widget.setup_camera(60.0, bbox, center)
    widget.look_at(center, eye, up)


class SixPanelWindow:
    def __init__(
        self,
        title,
        panels,
        summary,
        point_size,
        width,
        height,
    ):
        self.window = gui.Application.instance.create_window(
            title,
            width,
            height,
        )

        em = self.window.theme.font_size
        self.margin = int(0.40 * em)
        self.gap = int(0.28 * em)
        self.header_h = int(4.5 * em)

        self.header = gui.Vert(
            0.10 * em,
            gui.Margins(
                self.margin,
                self.margin,
                self.margin,
                self.margin,
            ),
        )
        title_label = gui.Label(title)
        title_label.text_color = gui.Color(0.96, 0.96, 0.96)
        self.header.add_child(title_label)

        summary_label = gui.Label(summary)
        summary_label.text_color = gui.Color(0.72, 0.72, 0.72)
        self.header.add_child(summary_label)

        self.window.add_child(self.header)
        self.widgets = []
        self.labels = []

        material = make_material(point_size)

        for panel_name, coord, colors in panels:
            widget = gui.SceneWidget()
            widget.scene = rendering.Open3DScene(self.window.renderer)
            widget.scene.set_background(BACKGROUND)
            widget.scene.show_axes(False)
            widget.scene.show_skybox(False)

            pcd = make_point_cloud(coord, colors)
            widget.scene.add_geometry("points", pcd, material)
            set_oblique_camera(widget, coord)

            label = gui.Label(panel_name)
            label.text_color = gui.Color(0.96, 0.96, 0.96)

            self.window.add_child(widget)
            self.window.add_child(label)
            self.widgets.append(widget)
            self.labels.append(label)

        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, _):
        r = self.window.content_rect

        header_h = min(self.header_h, max(1, r.height // 6))
        self.header.frame = gui.Rect(
            r.x,
            r.y,
            r.width,
            header_h,
        )

        body_y = r.y + header_h
        body_h = max(1, r.height - header_h)

        gap = self.gap
        col_w = max(1, (r.width - gap) // 2)
        row_h = max(1, (body_h - 2 * gap) // 3)

        frames = []
        for row in range(3):
            y = body_y + row * (row_h + gap)
            h = (
                row_h
                if row < 2
                else body_h - 2 * (row_h + gap)
            )
            frames.append(gui.Rect(r.x, y, col_w, h))
            frames.append(
                gui.Rect(
                    r.x + col_w + gap,
                    y,
                    r.width - col_w - gap,
                    h,
                )
            )

        label_h = int(1.25 * self.window.theme.font_size)
        for widget, label, frame in zip(
            self.widgets,
            self.labels,
            frames,
        ):
            widget.frame = frame
            label.frame = gui.Rect(
                frame.x + self.margin,
                frame.y + self.margin,
                max(1, frame.width - 2 * self.margin),
                label_h,
            )


# =============================================================================
# Selection / assembly
# =============================================================================

def choose_ids(
    new_root,
    old_root,
    old_mode,
    split,
    explicit,
    num_samples,
    shuffle,
    seed,
):
    new_ids = discover_prepared_ids(new_root)
    new_ids = [
        x
        for x in new_ids
        if x.lower().startswith(split.lower() + "_")
    ]
    new_id_set = set(new_ids)

    if explicit:
        missing = [x for x in explicit if x not in new_id_set]
        if missing:
            raise FileNotFoundError(
                "Missing NEW ids:\n  " + "\n  ".join(missing)
            )
        candidates = list(explicit)
    elif old_mode == "prepared":
        old_ids = set(discover_prepared_ids(old_root))
        candidates = [x for x in new_ids if x in old_ids]
    else:
        candidates = []
        for x in new_ids:
            try:
                raw_sample_dir(old_root, x, split)
                candidates.append(x)
            except FileNotFoundError:
                pass

    if not candidates:
        raise RuntimeError(
            f"No comparable {split} samples found between NEW and OLD roots."
        )

    if num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    return candidates[:num_samples]


def build_sample(new_root, old_root, old_mode, sample_id, split, max_points, seed):
    nt1 = load_prepared_epoch(new_root, sample_id, "t1")
    nt2 = load_prepared_epoch(new_root, sample_id, "t2")

    ni1 = keep_changed_first(
        nt1["event"], nt1["event_valid"], max_points, seed
    )
    ni2 = keep_changed_first(
        nt2["event"], nt2["event_valid"], max_points, seed + 1
    )

    panels = [
        (
            "NEW T1 Semantic",
            nt1["coord"][ni1],
            semantic_colors(nt1["semantic"][ni1]),
        ),
        (
            "NEW T2 Semantic",
            nt2["coord"][ni2],
            semantic_colors(nt2["semantic"][ni2]),
        ),
        (
            "NEW T1 Event",
            nt1["coord"][ni1],
            event_colors(
                nt1["event"][ni1],
                nt1["event_valid"][ni1],
            ),
        ),
        (
            "NEW T2 Event",
            nt2["coord"][ni2],
            event_colors(
                nt2["event"][ni2],
                nt2["event_valid"][ni2],
            ),
        ),
    ]

    if old_mode == "prepared":
        ot1 = load_prepared_epoch(old_root, sample_id, "t1")
        ot2 = load_prepared_epoch(old_root, sample_id, "t2")

        oi1 = keep_changed_first(
            ot1["event"], ot1["event_valid"], max_points, seed + 2
        )
        oi2 = keep_changed_first(
            ot2["event"], ot2["event_valid"], max_points, seed + 3
        )

        panels.extend(
            [
                (
                    "OLD T1 Event",
                    ot1["coord"][oi1],
                    event_colors(
                        ot1["event"][oi1],
                        ot1["event_valid"][oi1],
                    ),
                ),
                (
                    "OLD T2 Event",
                    ot2["coord"][oi2],
                    event_colors(
                        ot2["event"][oi2],
                        ot2["event_valid"][oi2],
                    ),
                ),
            ]
        )

        old_summary = (
            "OLD T1 "
            + hist_text(
                ot1["event"],
                EVENT_NAMES,
                ot1["event_valid"],
            )
            + " | OLD T2 "
            + hist_text(
                ot2["event"],
                EVENT_NAMES,
                ot2["event_valid"],
            )
        )
    else:
        rt1, rt2 = load_raw_pair(old_root, sample_id, split)
        ri1 = random_indices(
            rt1["coord"].shape[0],
            max_points,
            seed + 2,
        )
        ri2 = random_indices(
            rt2["coord"].shape[0],
            max_points,
            seed + 3,
        )

        panels.extend(
            [
                (
                    "OLD RAW T1 Semantic (label_mono)",
                    rt1["coord"][ri1],
                    semantic_colors(rt1["semantic"][ri1]),
                ),
                (
                    "OLD RAW T2 Change (label_ch)",
                    rt2["coord"][ri2],
                    source_change_colors(rt2["label_ch"][ri2]),
                ),
            ]
        )

        old_summary = (
            "RAW T2 "
            + hist_text(
                rt2["label_ch"],
                SOURCE_CHANGE_NAMES,
            )
        )

    new_summary = (
        "NEW T1 "
        + hist_text(
            nt1["event"],
            EVENT_NAMES,
            nt1["event_valid"],
        )
        + " | NEW T2 "
        + hist_text(
            nt2["event"],
            EVENT_NAMES,
            nt2["event_valid"],
        )
    )

    return panels, new_summary + " || " + old_summary


def print_legend(old_mode):
    print("\nEVENT colors")
    for k in range(6):
        print(f"  {k}: {EVENT_NAMES[k]}")
    print("  invalid: dark gray")

    if old_mode == "raw":
        print("\nRAW label_ch colors")
        for k in (-1, 0, 1, 2, 3):
            print(f"  {k}: {SOURCE_CHANGE_NAMES[k]}")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    old_mode = detect_old_mode(args.old_root, args.old_mode)

    print("=" * 92)
    print("NEW vs OLD NYC-SCD / PAIR 6-PANEL VISUALIZER")
    print("=" * 92)
    print("NEW root :", args.new_root)
    print("OLD root :", args.old_root)
    print("OLD mode :", old_mode)
    print("split    :", args.split)
    print("shuffle  :", args.shuffle)
    print("seed     :", args.seed)

    if old_mode == "prepared":
        print(
            "layout   : NEW semantics / NEW events / OLD events"
        )
    else:
        print(
            "layout   : NEW semantics / NEW events / "
            "RAW T1 semantic + RAW T2 label_ch"
        )

    print_legend(old_mode)

    sample_ids = choose_ids(
        args.new_root,
        args.old_root,
        old_mode,
        args.split,
        args.ids,
        args.num_samples,
        args.shuffle,
        args.seed,
    )

    print(f"Selected {len(sample_ids)} sample(s):")
    for i, sid in enumerate(sample_ids, 1):
        print(f"  {i:02d}. {sid}")

    prepared = []
    for i, sid in enumerate(sample_ids):
        panels, summary = build_sample(
            args.new_root,
            args.old_root,
            old_mode,
            sid,
            args.split,
            args.max_points,
            args.seed + i * 100,
        )
        prepared.append((sid, panels, summary))
        print(f"[{i+1:02d}/{len(sample_ids):02d}] loaded {sid}")

    app = gui.Application.instance
    app.initialize()

    windows = []
    for sid, panels, summary in prepared:
        windows.append(
            SixPanelWindow(
                title=sid,
                panels=panels,
                summary=summary,
                point_size=args.point_size,
                width=args.window_width,
                height=args.window_height,
            )
        )

    app.run()


if __name__ == "__main__":
    main()
