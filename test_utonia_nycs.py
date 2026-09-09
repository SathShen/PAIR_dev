#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test: NYC-SCD prepared data -> frozen Utonia point encoder.

Run from PAIR project root:

    python test_utonia_nycs.py \
        --config configs/pair_train.json \
        --split val \
        --device cuda:0

This test intentionally stops BEFORE PointAdapter/Qwen/UnifiedDecoder.
It validates the current data + Utonia boundary first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.point_encoder import (
    UtoniaPointEncoder,
    UtoniaPointEncoderConfig,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train.json"),
    )
    p.add_argument(
        "--dataset",
        default="NYC-SCD",
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
    )
    p.add_argument(
        "--device",
        default="cuda:0",
    )
    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Bad JSON at {path}:{line_no}"
                ) from exc

    if not records:
        raise RuntimeError(f"Empty manifest: {path}")

    return records


def resolve(root: Path, value: str):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_point(path: Path):
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.keys())
        if "coord" not in keys:
            raise RuntimeError(
                f"{path}: missing 'coord'; available={sorted(keys)}"
            )
        coord = np.asarray(z["coord"], dtype=np.float32)

        feat = (
            np.asarray(z["feat"], dtype=np.float32)
            if "feat" in keys
            else None
        )

    if coord.ndim != 2 or coord.shape[1] != 3:
        raise RuntimeError(
            f"{path}: coord must be [N,3], got {coord.shape}"
        )

    if not np.isfinite(coord).all():
        raise RuntimeError(f"{path}: coord contains NaN/Inf")

    if feat is not None:
        if feat.ndim != 2 or feat.shape[0] != coord.shape[0]:
            raise RuntimeError(
                f"{path}: feat topology mismatch: "
                f"coord={coord.shape}, feat={feat.shape}"
            )

    return coord, feat


def load_supervision(path: Path):
    if path.suffix.lower() != ".npz":
        raise RuntimeError(
            f"{path}: current 3D supervision must be an NPZ bundle"
        )

    with np.load(path, allow_pickle=False) as z:
        keys = set(z.keys())
        required = {"semantic", "change"}
        missing = required - keys

        if missing:
            raise RuntimeError(
                f"{path}: missing {sorted(missing)}; "
                f"available={sorted(keys)}"
            )

        semantic = np.asarray(z["semantic"], dtype=np.int64)
        change = np.asarray(z["change"], dtype=np.int64)

    if semantic.ndim != 1 or change.ndim != 1:
        raise RuntimeError(
            f"{path}: semantic/change must both be 1D; "
            f"got {semantic.shape}, {change.shape}"
        )

    if semantic.shape != change.shape:
        raise RuntimeError(
            f"{path}: semantic/change shape mismatch: "
            f"{semantic.shape} vs {change.shape}"
        )

    return semantic, change


def unique_counts(x: np.ndarray):
    values, counts = np.unique(x, return_counts=True)
    return {
        int(v): int(n)
        for v, n in zip(values.tolist(), counts.tolist())
    }


def validate_semantic(
    semantic: np.ndarray,
    *,
    class_names: dict[int, str],
    ignored_id,
    label: str,
):
    observed = set(map(int, np.unique(semantic).tolist()))
    allowed = set(class_names)

    if ignored_id is not None:
        allowed.add(int(ignored_id))

    unknown = sorted(observed - allowed)

    if unknown:
        raise RuntimeError(
            f"{label}: unknown semantic IDs {unknown}. "
            f"class_names={sorted(class_names)}, ignored_id={ignored_id}. "
            "Prepared data must not silently contain undeclared labels."
        )


def validate_change(
    change: np.ndarray,
    *,
    ignored_id,
    label: str,
):
    observed = set(map(int, np.unique(change).tolist()))
    allowed = {0, 1}

    if ignored_id is not None:
        allowed.add(int(ignored_id))

    unknown = sorted(observed - allowed)

    if unknown:
        raise RuntimeError(
            f"{label}: invalid change IDs {unknown}. "
            "PAIR binary change must be 0/1"
            + (
                f" plus ignored_id={ignored_id}."
                if ignored_id is not None
                else " and ignored_id is None."
            )
        )


def inspect_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Utonia checkpoint not found: {path}"
        )

    try:
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        ckpt = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            "Unexpected Utonia checkpoint: top-level object is not a dict"
        )

    config = ckpt.get("config")

    if not isinstance(config, dict):
        raise RuntimeError(
            "Utonia checkpoint has no dict-valued 'config'"
        )

    print("\n[Checkpoint]")
    print(" path        :", path)
    print(" in_channels :", config.get("in_channels"))
    print(" enc_channels:", config.get("enc_channels"))
    print(" enc_depths  :", config.get("enc_depths"))
    print(" enc_num_head:", config.get("enc_num_head"))
    print(" enc_mode    :", config.get("enc_mode"))
    print(" enable_flash:", config.get("enable_flash"))

    return config


def to_point_dict(coord_np: np.ndarray, device: torch.device):
    coord = torch.from_numpy(
        np.ascontiguousarray(coord_np)
    ).to(
        device=device,
        dtype=torch.float32,
    )

    return {
        "coord": coord,
        "batch": torch.zeros(
            coord.shape[0],
            dtype=torch.long,
            device=device,
        ),
    }


def run_encoder(
    encoder,
    point_dict,
    *,
    name: str,
    device: torch.device,
):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    output = encoder(point_dict)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    features = output.features

    if features.ndim != 2:
        raise RuntimeError(
            f"{name}: output features must be [N,D], "
            f"got {tuple(features.shape)}"
        )

    if features.shape[0] != point_dict["coord"].shape[0]:
        raise RuntimeError(
            f"{name}: dense topology was not restored: "
            f"input N={point_dict['coord'].shape[0]}, "
            f"output N={features.shape[0]}"
        )

    if features.shape[1] != encoder.output_dim:
        raise RuntimeError(
            f"{name}: output D={features.shape[1]} "
            f"but encoder.output_dim={encoder.output_dim}"
        )

    if not torch.isfinite(features).all():
        raise RuntimeError(
            f"{name}: output contains NaN/Inf"
        )

    print(f"\n[{name} Utonia forward]")
    print(" input points :", point_dict["coord"].shape[0])
    print(" output       :", tuple(features.shape))
    print(" dtype        :", features.dtype)
    print(" elapsed      :", f"{elapsed:.3f} s")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(" peak allocated:", f"{peak:.3f} GiB")
        print(" peak reserved :", f"{reserved:.3f} GiB")

    return output


def main():
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)

    if args.dataset not in config["datasets"]:
        raise KeyError(
            f"Dataset {args.dataset!r} not found in {config_path}"
        )

    ds_cfg = config["datasets"][args.dataset]
    root = Path(ds_cfg["root"]).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}"
        )

    ignored_id = ds_cfg.get("ignored_id", None)
    class_names = {
        int(k): str(v)
        for k, v in ds_cfg["class_names"].items()
    }

    model_cfg = config["model"]["point_encoder"]
    checkpoint = Path(
        model_cfg["checkpoint"]
    ).expanduser().resolve()
    voxel_size = float(
        model_cfg["voxel_size"]
    )

    print("=" * 78)
    print("PAIR NYC-SCD -> Utonia smoke test")
    print("=" * 78)
    print(" config      :", config_path)
    print(" dataset     :", args.dataset)
    print(" root        :", root)
    print(" split       :", args.split)
    print(" ignored_id  :", ignored_id)
    print(" class_names :", class_names)
    print(" voxel_size  :", voxel_size)

    ckpt_cfg = inspect_checkpoint(checkpoint)

    manifest_path = root / "manifests" / f"{args.split}.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    records = load_manifest(manifest_path)

    if not 0 <= args.index < len(records):
        raise IndexError(
            f"--index {args.index} outside [0, {len(records)-1}]"
        )

    record = records[args.index]

    print("\n[Manifest]")
    print(" samples :", len(records))
    print(" selected:", record.get("id", args.index))

    required = (
        "point_t1",
        "point_t2",
        "semantic_t1",
        "semantic_t2",
    )

    missing = [
        key for key in required
        if key not in record
    ]

    if missing:
        raise RuntimeError(
            f"Manifest record missing keys: {missing}"
        )

    point_t1_path = resolve(root, record["point_t1"])
    point_t2_path = resolve(root, record["point_t2"])
    sem_t1_path = resolve(root, record["semantic_t1"])
    sem_t2_path = resolve(root, record["semantic_t2"])

    coord_t1, feat_t1 = load_point(point_t1_path)
    coord_t2, feat_t2 = load_point(point_t2_path)

    semantic_t1, change_t1 = load_supervision(sem_t1_path)
    semantic_t2, change_t2 = load_supervision(sem_t2_path)

    if coord_t1.shape[0] != semantic_t1.shape[0]:
        raise RuntimeError(
            f"T1 topology mismatch: points={coord_t1.shape[0]}, "
            f"labels={semantic_t1.shape[0]}"
        )

    if coord_t2.shape[0] != semantic_t2.shape[0]:
        raise RuntimeError(
            f"T2 topology mismatch: points={coord_t2.shape[0]}, "
            f"labels={semantic_t2.shape[0]}"
        )

    validate_semantic(
        semantic_t1,
        class_names=class_names,
        ignored_id=ignored_id,
        label="semantic_t1",
    )
    validate_semantic(
        semantic_t2,
        class_names=class_names,
        ignored_id=ignored_id,
        label="semantic_t2",
    )
    validate_change(
        change_t1,
        ignored_id=ignored_id,
        label="change_t1",
    )
    validate_change(
        change_t2,
        ignored_id=ignored_id,
        label="change_t2",
    )

    print("\n[Prepared sample]")
    print(
        " T1:",
        f"N={coord_t1.shape[0]:,}",
        f"feat={None if feat_t1 is None else feat_t1.shape}",
    )
    print(
        "     semantic:",
        unique_counts(semantic_t1),
    )
    print(
        "     change  :",
        unique_counts(change_t1),
    )
    print(
        " T2:",
        f"N={coord_t2.shape[0]:,}",
        f"feat={None if feat_t2 is None else feat_t2.shape}",
    )
    print(
        "     semantic:",
        unique_counts(semantic_t2),
    )
    print(
        "     change  :",
        unique_counts(change_t2),
    )

    device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False"
            )
        print("\n[CUDA]")
        print(" torch      :", torch.__version__)
        print(" cuda       :", torch.version.cuda)
        print(
            " gpu        :",
            torch.cuda.get_device_name(device),
        )

    encoder = UtoniaPointEncoder(
        UtoniaPointEncoderConfig(
            checkpoint=str(checkpoint),
            voxel_size=voxel_size,
        )
    ).to(device)

    encoder.eval()

    total_params = sum(
        p.numel()
        for p in encoder.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    print("\n[Encoder]")
    print(" params      :", f"{total_params:,}")
    print(" trainable   :", f"{trainable_params:,}")
    print(" output_dim  :", encoder.output_dim)
    print(" stage dims  :", encoder.stage_channels)

    if trainable_params != 0:
        raise RuntimeError(
            "Utonia is supposed to be frozen, "
            f"but {trainable_params:,} parameters are trainable"
        )

    expected_dim = sum(
        int(x)
        for x in ckpt_cfg["enc_channels"]
    )

    if encoder.output_dim != expected_dim:
        raise RuntimeError(
            f"Encoder output_dim={encoder.output_dim}, "
            f"but checkpoint enc_channels sum to {expected_dim}"
        )

    p1 = to_point_dict(
        coord_t1,
        device,
    )
    p2 = to_point_dict(
        coord_t2,
        device,
    )

    out1 = run_encoder(
        encoder,
        p1,
        name="T1",
        device=device,
    )

    del p1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out2 = run_encoder(
        encoder,
        p2,
        name="T2",
        device=device,
    )

    print("\n" + "=" * 78)
    print("PASS")
    print("=" * 78)
    print(
        "Data topology, label IDs, frozen Utonia loading, "
        "and dense inverse mapping all passed."
    )
    print(
        "T1 dense:",
        tuple(out1.features.shape),
    )
    print(
        "T2 dense:",
        tuple(out2.features.shape),
    )


if __name__ == "__main__":
    main()
