#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a conventional SECOND-style 2D semantic change-detection dataset
for PAIR.

This script ONLY reorganizes files and generates JSONL manifests.
It does NOT generate DatasetSpec. DatasetSpec should be written manually
inside the PAIR project.

Input
=====
ROOT/
├── train/
│   ├── im1/
│   ├── im2/
│   ├── label1/
│   └── label2/
├── val/
│   ├── im1/
│   ├── im2/
│   ├── label1/
│   └── label2/
└── test/                  # optional
    ├── im1/
    ├── im2/
    ├── label1/
    └── label2/

Output
======
ROOT/
├── images_t1/
├── images_t2/
├── semantic_t1/
├── semantic_t2/
└── manifests/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl         # only if test exists

Rules
=====
- Files are MOVED, not copied.
- Manifest paths are relative to ROOT.
- No dataset_meta.json is generated.
- No dataset_spec.py is generated.
- No resizing or label remapping is performed.
- im1 / im2 / label1 / label2 are paired by filename stem.
- ALL splits are validated before ANY file is moved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".npy",
    ".npz",
}

SOURCE_TO_TARGET = {
    "im1": "images_t1",
    "im2": "images_t2",
    "label1": "semantic_t1",
    "label2": "semantic_t2",
}

MANIFEST_KEYS = {
    "im1": "image_t1",
    "im2": "image_t2",
    "label1": "semantic_t1",
    "label2": "semantic_t2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten SECOND-style train/val/test folders "
            "and generate PAIR manifests."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Dataset root, e.g. /data/SECONDBbi",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help=(
            "Split folders to look for. "
            "Missing splits are skipped."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and preview only. "
            "No files will be moved and no manifests will be written."
        ),
    )

    parser.add_argument(
        "--keep-empty-split-folders",
        action="store_true",
        help=(
            "Keep empty train/val/test folders after moving files."
        ),
    )

    return parser.parse_args()


def scan_by_stem(folder: Path):
    """
    Scan one modality folder and return:
        filename stem -> file path

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

    return result


def validate_split(root: Path, split: str):
    """
    Validate one split.

    im1 / im2 / label1 / label2 must contain exactly the same sample IDs.
    """
    split_root = root / split

    maps = {
        source_name: scan_by_stem(
            split_root / source_name
        )
        for source_name in SOURCE_TO_TARGET
    }

    id_sets = {
        key: set(mapping.keys())
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


def relative_to_root(path: Path, root: Path) -> str:
    """
    Convert to portable manifest path, for example:

        images_t1/train_000001.png
    """
    return (
        path.resolve()
        .relative_to(root.resolve())
        .as_posix()
    )


def ensure_destination_free(path: Path):
    if path.exists():
        raise FileExistsError(
            "Destination already exists; refusing to overwrite:\n"
            f"{path}"
        )


def build_split_plan(root: Path, split: str):
    """
    Build all file moves and manifest records for one split
    without changing the filesystem.
    """
    sample_ids, maps = validate_split(
        root,
        split,
    )

    moves = []
    records = []

    for old_id in sample_ids:
        # Prefix with split name to prevent collisions when
        # train/val/test contain identical source filenames.
        sample_id = f"{split}_{old_id}"

        record = {
            "id": sample_id,
        }

        for source_name, target_folder in SOURCE_TO_TARGET.items():
            src = maps[source_name][old_id]

            dst = (
                root
                / target_folder
                / f"{sample_id}{src.suffix.lower()}"
            )

            ensure_destination_free(
                dst
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
        / f"{split}.jsonl"
    )

    if manifest_path.exists():
        raise FileExistsError(
            "Manifest already exists; refusing to overwrite:\n"
            f"{manifest_path}"
        )

    return {
        "sample_ids": sample_ids,
        "moves": moves,
        "records": records,
        "manifest_path": manifest_path,
    }


def remove_empty_tree(path: Path):
    """
    Remove empty directories only.
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

    existing_splits = [
        split
        for split in args.splits
        if (root / split).is_dir()
    ]

    if not existing_splits:
        raise RuntimeError(
            "No split folders were found."
        )

    print("=" * 72)
    print("PAIR SECOND DATASET PREPARATION")
    print("=" * 72)
    print(f"root    : {root}")
    print(f"dry run : {args.dry_run}")
    print(
        "splits  : "
        + ", ".join(existing_splits)
    )
    print()

    # --------------------------------------------------------------
    # Validate ALL splits before moving ANY file.
    # --------------------------------------------------------------
    plans = {}

    for split in existing_splits:
        plan = build_split_plan(
            root,
            split,
        )

        plans[split] = plan

        print(
            f"{split:8s}: "
            f"{len(plan['sample_ids'])} samples, "
            f"{len(plan['moves'])} files"
        )

        print(
            "  example:",
            json.dumps(
                plan["records"][0],
                ensure_ascii=False,
            ),
        )

    print()
    print("Validation: PASS")

    if args.dry_run:
        print(
            "DRY RUN COMPLETE -- no files were changed."
        )
        return

    # --------------------------------------------------------------
    # Create canonical PAIR folders.
    # --------------------------------------------------------------
    for target_folder in SOURCE_TO_TARGET.values():
        (
            root / target_folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    (
        root / "manifests"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Move files and write manifests.
    # --------------------------------------------------------------
    for split in existing_splits:
        plan = plans[split]

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
            f"[{split}] "
            f"moved {len(plan['moves'])} files -> "
            f"{plan['manifest_path'].name}"
        )

    # --------------------------------------------------------------
    # Remove now-empty split folders unless requested otherwise.
    # --------------------------------------------------------------
    if not args.keep_empty_split_folders:
        for split in existing_splits:
            remove_empty_tree(
                root / split
            )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    for split in existing_splits:
        print(
            f"{split:8s}: "
            f"{len(plans[split]['sample_ids'])} samples"
        )

    print()
    print("Final dataset layout:")
    print("  images_t1/")
    print("  images_t2/")
    print("  semantic_t1/")
    print("  semantic_t2/")
    print("  manifests/")
    print()
    print(
        "DatasetSpec is intentionally NOT generated. "
        "Define it manually in the PAIR project."
    )


if __name__ == "__main__":
    main()
