#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a conventional 2D binary change-detection dataset for PAIR.

Expected input
==============
ROOT/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/                   # optional
│   ├── A/
│   ├── B/
│   └── label/
└── test/                  # optional
    ├── A/
    ├── B/
    └── label/

Output
======
ROOT/
├── images_t1/
├── images_t2/
├── change/
└── manifests/
    ├── train.jsonl
    ├── val.jsonl          # if val exists or is split from train
    └── test.jsonl         # if test exists

PAIR BCD preparation rules
==========================
- A     -> images_t1
- B     -> images_t2
- label -> change
- Original filename stems are preserved exactly.
- Physical BCD masks remain 0/255:
      0   = unchanged
      255 = changed
- No label remapping is performed here.
- A / B / label are paired by filename stem.
- Files are MOVED, not copied.
- Manifest paths are relative to ROOT.
- If val/ is missing, a deterministic subset is split from train/.
- ALL splits, labels, image sizes, manifests, destinations, and cross-split
  collisions are validated before ANY filesystem modification.

Example
=======
python prepare_bcd_for_pair.py /home/sht/Datasets/LEVIR-CDpair --dry-run
python prepare_bcd_for_pair.py /home/sht/Datasets/LEVIR-CDpair
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

SOURCE_TO_TARGET = {
    "A": "images_t1",
    "B": "images_t2",
    "label": "change",
}

MANIFEST_KEYS = {
    "A": "image_t1",
    "B": "image_t2",
    "label": "change",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare 2D BCD data for PAIR."
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Dataset root, e.g. /home/sht/Datasets/LEVIR-CDpair",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.00,
        help=(
            "If val/ does not exist, split this fraction from train. "
            "Default: 0.20. Use 0 to disable automatic validation split."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split. Default: 42.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview only; do not modify files.",
    )

    parser.add_argument(
        "--keep-empty-split-folders",
        action="store_true",
        help="Keep empty train/val/test folders after moving files.",
    )

    return parser.parse_args()


def scan_by_stem(folder: Path):
    """
    Return:
        stem -> file path

    Duplicate stems are rejected because pairing would be ambiguous.
    """
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Missing required folder: {folder}"
        )

    result = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        stem = path.stem

        if stem in result:
            raise RuntimeError(
                f"Duplicate stem {stem!r} in {folder}:\n"
                f"  {result[stem]}\n"
                f"  {path}"
            )

        result[stem] = path

    if not result:
        raise RuntimeError(
            f"No supported files found in: {folder}"
        )

    return result


def validate_split(root: Path, split: str):
    """
    Validate A/B/label one-to-one pairing for one source split.
    """
    split_root = root / split

    maps = {
        source_name: scan_by_stem(
            split_root / source_name
        )
        for source_name in SOURCE_TO_TARGET
    }

    id_sets = {
        key: set(mapping)
        for key, mapping in maps.items()
    }

    all_ids = set().union(
        *id_sets.values()
    )

    common_ids = set.intersection(
        *id_sets.values()
    )

    if all_ids != common_ids:
        lines = [
            f"Split {split!r} is not one-to-one matched:"
        ]

        for key, ids in id_sets.items():
            missing = sorted(
                all_ids - ids
            )

            if missing:
                lines.append(
                    f"  {key}: missing {len(missing)} files; "
                    f"examples={missing[:10]}"
                )

        raise RuntimeError(
            "\n".join(lines)
        )

    if not common_ids:
        raise RuntimeError(
            f"No valid samples found in split {split!r}."
        )

    return sorted(common_ids), maps


def validate_label_0255(path: Path):
    """
    BCD physical labels must remain 0/255.
    """
    arr = np.asarray(
        Image.open(path)
    )

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                "BCD label must be single-channel; "
                f"got shape={arr.shape}: {path}"
            )

    values = set(
        int(v)
        for v in np.unique(arr).tolist()
    )

    if not values.issubset({0, 255}):
        raise ValueError(
            f"Unexpected BCD label values {sorted(values)} in {path}. "
            "Expected only 0=unchanged and 255=changed."
        )

    return values


def validate_spatial_match(
    image_t1: Path,
    image_t2: Path,
    label: Path,
):
    """
    T1, T2 and change mask must have identical width/height.
    """
    with Image.open(image_t1) as a:
        size_t1 = a.size

    with Image.open(image_t2) as b:
        size_t2 = b.size

    with Image.open(label) as y:
        size_label = y.size

    if not (
        size_t1 == size_t2 == size_label
    ):
        raise ValueError(
            "Spatial size mismatch:\n"
            f"  T1    {size_t1}: {image_t1}\n"
            f"  T2    {size_t2}: {image_t2}\n"
            f"  label {size_label}: {label}"
        )


def split_train_val(
    sample_ids,
    val_ratio: float,
    seed: int,
):
    """
    Deterministically partition train IDs while preserving original stems.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(
            "--val-ratio must satisfy 0 <= ratio < 1"
        )

    sample_ids = list(
        sample_ids
    )

    if val_ratio == 0.0:
        return sorted(sample_ids), []

    shuffled = list(
        sample_ids
    )

    rng = random.Random(
        seed
    )
    rng.shuffle(
        shuffled
    )

    n_val = int(
        round(
            len(shuffled)
            * val_ratio
        )
    )

    if len(shuffled) > 1:
        n_val = max(
            1,
            min(
                n_val,
                len(shuffled) - 1,
            ),
        )
    else:
        n_val = 0

    val_set = set(
        shuffled[:n_val]
    )

    train_ids = sorted(
        sample_id
        for sample_id in sample_ids
        if sample_id not in val_set
    )

    val_ids = sorted(
        sample_id
        for sample_id in sample_ids
        if sample_id in val_set
    )

    return train_ids, val_ids


def relative_to_root(
    path: Path,
    root: Path,
):
    return (
        path.resolve()
        .relative_to(
            root.resolve()
        )
        .as_posix()
    )


def build_plan(
    root: Path,
    source_split: str,
    output_split: str,
    sample_ids,
    maps,
):
    """
    Build one complete split plan without modifying the filesystem.

    IMPORTANT:
    The original sample stem is preserved exactly.
    """
    moves = []
    records = []
    observed_values = set()

    for sample_id in sample_ids:
        src_t1 = maps["A"][sample_id]
        src_t2 = maps["B"][sample_id]
        src_change = maps["label"][sample_id]

        validate_spatial_match(
            src_t1,
            src_t2,
            src_change,
        )

        observed_values.update(
            validate_label_0255(
                src_change
            )
        )

        record = {
            "id": sample_id,
        }

        for source_name, target_folder in SOURCE_TO_TARGET.items():
            src = maps[source_name][sample_id]

            # Keep the original filename stem and extension.
            dst = (
                root
                / target_folder
                / src.name
            )

            moves.append(
                (src, dst)
            )

            record[
                MANIFEST_KEYS[source_name]
            ] = relative_to_root(
                dst,
                root,
            )

        records.append(
            record
        )

    manifest_path = (
        root
        / "manifests"
        / f"{output_split}.jsonl"
    )

    return {
        "source_split": source_split,
        "output_split": output_split,
        "sample_ids": list(sample_ids),
        "moves": moves,
        "records": records,
        "manifest_path": manifest_path,
        "observed_values": observed_values,
    }


def validate_all_destinations(plans):
    """
    Validate all destination paths globally before moving any file.

    This catches:
    - already prepared target files
    - duplicate stems across train/val/test after flattening
    - existing manifests
    """
    destinations = {}

    for plan in plans:
        manifest_path = plan["manifest_path"]

        if manifest_path.exists():
            raise FileExistsError(
                "Manifest already exists; refusing to overwrite:\n"
                f"{manifest_path}"
            )

        for src, dst in plan["moves"]:
            if dst.exists():
                raise FileExistsError(
                    "Destination already exists; refusing to overwrite:\n"
                    f"{dst}"
                )

            key = dst.resolve()

            if key in destinations:
                other_src = destinations[key]

                raise RuntimeError(
                    "Cross-split filename collision after flattening:\n"
                    f"  {other_src}\n"
                    f"  {src}\n"
                    f"would both become:\n"
                    f"  {dst}\n"
                    "Source filename stems must be globally unique."
                )

            destinations[key] = src


def remove_empty_tree(path: Path):
    """
    Remove only empty directories.
    Never removes files.
    """
    if not path.exists():
        return

    for child in sorted(
        path.rglob("*"),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass

    try:
        path.rmdir()
    except OSError:
        pass


def main():
    args = parse_args()

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}"
        )

    if not (
        root / "train"
    ).is_dir():
        raise FileNotFoundError(
            f"Required train/ folder does not exist: {root / 'train'}"
        )

    print("=" * 78)
    print("PAIR 2D BINARY CHANGE DATASET PREPARATION")
    print("=" * 78)
    print(f"root      : {root}")
    print(f"dry run   : {args.dry_run}")
    print(f"val ratio : {args.val_ratio}")
    print(f"seed      : {args.seed}")
    print()

    # --------------------------------------------------------------
    # Scan source splits.
    # --------------------------------------------------------------
    train_ids, train_maps = validate_split(
        root,
        "train",
    )

    assignments = []

    if (
        root / "val"
    ).is_dir():
        val_ids, val_maps = validate_split(
            root,
            "val",
        )

        assignments.append(
            (
                "train",
                "train",
                train_ids,
                train_maps,
            )
        )

        assignments.append(
            (
                "val",
                "val",
                val_ids,
                val_maps,
            )
        )

        print(
            "Validation source: existing val/"
        )

    else:
        out_train_ids, out_val_ids = split_train_val(
            train_ids,
            args.val_ratio,
            args.seed,
        )

        assignments.append(
            (
                "train",
                "train",
                out_train_ids,
                train_maps,
            )
        )

        if out_val_ids:
            assignments.append(
                (
                    "train",
                    "val",
                    out_val_ids,
                    train_maps,
                )
            )

        print(
            "Validation source: deterministic split from train/ "
            f"({len(out_val_ids)}/{len(train_ids)} samples)"
        )

    if (
        root / "test"
    ).is_dir():
        test_ids, test_maps = validate_split(
            root,
            "test",
        )

        assignments.append(
            (
                "test",
                "test",
                test_ids,
                test_maps,
            )
        )

    # --------------------------------------------------------------
    # Validate EVERYTHING and build ALL plans before moving anything.
    # --------------------------------------------------------------
    plans = []
    observed_values = set()

    for (
        source_split,
        output_split,
        sample_ids,
        maps,
    ) in assignments:
        if not sample_ids:
            continue

        plan = build_plan(
            root,
            source_split,
            output_split,
            sample_ids,
            maps,
        )

        plans.append(
            plan
        )

        observed_values.update(
            plan["observed_values"]
        )

        print(
            f"{output_split:8s}: "
            f"{len(sample_ids):5d} samples "
            f"(source={source_split})"
        )

        print(
            "  example:",
            json.dumps(
                plan["records"][0],
                ensure_ascii=False,
            ),
        )

    validate_all_destinations(
        plans
    )

    print()
    print(
        "Observed physical label values:",
        sorted(
            observed_values
        ),
    )
    print(
        "Output physical label values  : unchanged, kept exactly as 0/255"
    )
    print(
        "Validation: PASS"
    )

    if args.dry_run:
        print(
            "DRY RUN COMPLETE -- no files were changed."
        )
        return

    # --------------------------------------------------------------
    # Create canonical PAIR folders.
    # --------------------------------------------------------------
    for folder in (
        "images_t1",
        "images_t2",
        "change",
        "manifests",
    ):
        (
            root / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------------
    # Move files and write manifests.
    # --------------------------------------------------------------
    for plan in plans:
        for src, dst in plan["moves"]:
            shutil.move(
                str(src),
                str(dst),
            )

        with plan["manifest_path"].open(
            "w",
            encoding="utf-8",
        ) as f:
            for record in plan["records"]:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"[{plan['output_split']}] "
            f"prepared {len(plan['records'])} samples -> "
            f"{plan['manifest_path'].name}"
        )

    # --------------------------------------------------------------
    # Remove now-empty source split trees.
    # --------------------------------------------------------------
    if not args.keep_empty_split_folders:
        for split in (
            "train",
            "val",
            "test",
        ):
            remove_empty_tree(
                root / split
            )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print("Final dataset layout:")
    print("  images_t1/")
    print("  images_t2/")
    print("  change/       # physical masks remain 0/255")
    print("  manifests/")


if __name__ == "__main__":
    main()
