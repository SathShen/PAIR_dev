#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR NYC-SCD -> Utonia -> PointAdapter smoke test.

This test uses a REAL prepared NYC-SCD sample and creates a deterministic
FAKE intensity channel only for testing the optional intensity path.

It checks:
1. Prepared NYC-SCD point/label topology.
2. Frozen Utonia forward for T1/T2.
3. Fake intensity survives PointEncoder unchanged.
4. PointAdapter produces:
       dense_features [N, 256]
       reasoning tokens [K, 2560], K <= 512
5. Geometry-only path also works.
6. With a freshly initialized PointAdapter, fake-intensity and geometry-only
   dense features are initially equal because the residual intensity fusion
   final layer is zero-initialized.

Run from PAIR root:

    CUDA_VISIBLE_DEVICES=2 python test_utonia_adapter_nycs.py \
        --config configs/pair_train.json \
        --split val \
        --device cuda:0
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
from models.point_adapter import (
    PointAdapter,
    PointAdapterConfig,
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
        if "coord" not in z:
            raise RuntimeError(
                f"{path}: missing 'coord'; available={sorted(z.keys())}"
            )
        coord = np.asarray(z["coord"], dtype=np.float32)

    if coord.ndim != 2 or coord.shape[1] != 3:
        raise RuntimeError(
            f"{path}: coord must be [N,3], got {coord.shape}"
        )

    if coord.shape[0] == 0:
        raise RuntimeError(f"{path}: empty point cloud")

    if not np.isfinite(coord).all():
        raise RuntimeError(f"{path}: coord contains NaN/Inf")

    return coord


def load_supervision(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {"semantic", "change"}
        missing = required - set(z.keys())

        if missing:
            raise RuntimeError(
                f"{path}: missing {sorted(missing)}; "
                f"available={sorted(z.keys())}"
            )

        semantic = np.asarray(z["semantic"], dtype=np.int64)
        change = np.asarray(z["change"], dtype=np.int64)

    if semantic.ndim != 1 or change.ndim != 1:
        raise RuntimeError(
            f"{path}: semantic/change must be 1D"
        )

    if semantic.shape != change.shape:
        raise RuntimeError(
            f"{path}: semantic/change topology mismatch"
        )

    return semantic, change


def unique_counts(x: np.ndarray):
    values, counts = np.unique(x, return_counts=True)
    return {
        int(v): int(n)
        for v, n in zip(values.tolist(), counts.tolist())
    }


def validate_labels(
    semantic: np.ndarray,
    change: np.ndarray,
    *,
    class_names: dict[int, str],
    ignored_id,
    name: str,
):
    semantic_ids = set(map(int, np.unique(semantic).tolist()))
    allowed_semantic = set(class_names)

    if ignored_id is not None:
        allowed_semantic.add(int(ignored_id))

    unknown_semantic = sorted(
        semantic_ids - allowed_semantic
    )

    if unknown_semantic:
        raise RuntimeError(
            f"{name}: unknown semantic IDs {unknown_semantic}; "
            f"class_names={sorted(class_names)}, ignored_id={ignored_id}"
        )

    change_ids = set(map(int, np.unique(change).tolist()))
    allowed_change = {0, 1}

    if ignored_id is not None:
        allowed_change.add(int(ignored_id))

    unknown_change = sorted(
        change_ids - allowed_change
    )

    if unknown_change:
        raise RuntimeError(
            f"{name}: invalid change IDs {unknown_change}"
        )


def make_fake_intensity(
    coord: torch.Tensor,
) -> torch.Tensor:
    """
    Deterministic synthetic intensity in [0,1].

    This is NOT intended to model real LiDAR radiometry.
    It only provides a non-constant per-point signal for testing the
    optional PAIR intensity branch.

    We combine normalized x/y/z so the test exercises a varied intensity
    tensor without relying on randomness.
    """
    xyz = coord.float()

    xyz_min = xyz.amin(
        dim=0,
        keepdim=True,
    )

    xyz_max = xyz.amax(
        dim=0,
        keepdim=True,
    )

    span = (
        xyz_max - xyz_min
    ).clamp_min(1e-6)

    norm = (
        xyz - xyz_min
    ) / span

    intensity = (
        0.50 * norm[:, 2:3]
        + 0.30 * norm[:, 0:1]
        + 0.20 * norm[:, 1:2]
    )

    return intensity.clamp(
        0.0,
        1.0,
    )


def point_dict_from_numpy(
    coord_np: np.ndarray,
    *,
    device: torch.device,
    with_fake_intensity: bool,
):
    coord = torch.from_numpy(
        np.ascontiguousarray(coord_np)
    ).to(
        device=device,
        dtype=torch.float32,
    )

    point_dict = {
        "coord": coord,
        "batch": torch.zeros(
            coord.shape[0],
            dtype=torch.long,
            device=device,
        ),
    }

    if with_fake_intensity:
        point_dict["intensity"] = make_fake_intensity(
            coord
        )

    return point_dict


def tensor_stats(x: torch.Tensor):
    x = x.float()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def run_utonia(
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

    encoded = encoder(
        point_dict
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    if encoded.features.shape != (
        point_dict["coord"].shape[0],
        encoder.output_dim,
    ):
        raise RuntimeError(
            f"{name}: unexpected Utonia output "
            f"{tuple(encoded.features.shape)}"
        )

    if not torch.isfinite(
        encoded.features
    ).all():
        raise RuntimeError(
            f"{name}: Utonia produced NaN/Inf"
        )

    print(f"\n[{name} / Utonia]")
    print(" input points :", f"{point_dict['coord'].shape[0]:,}")
    print(" features     :", tuple(encoded.features.shape))
    print(" frozen       :", all(
        not p.requires_grad
        for p in encoder.parameters()
    ))
    print(" elapsed      :", f"{elapsed:.3f} s")

    if encoded.intensity is None:
        print(" intensity    : None")
    else:
        print(" intensity    :", tuple(encoded.intensity.shape))
        print(" intensity stat:", tensor_stats(encoded.intensity))

        source_intensity = point_dict["intensity"]

        if not torch.equal(
            encoded.intensity,
            source_intensity,
        ):
            max_error = (
                encoded.intensity
                - source_intensity
            ).abs().max().item()

            raise RuntimeError(
                f"{name}: PointEncoder altered intensity; "
                f"max error={max_error}"
            )

        if (
            encoded.intensity_mask is None
            or not encoded.intensity_mask.all()
        ):
            raise RuntimeError(
                f"{name}: expected all fake intensity points to be valid"
            )

    if device.type == "cuda":
        peak = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 3)
        )
        print(" peak allocated:", f"{peak:.3f} GiB")

    return encoded


def run_adapter(
    adapter,
    encoded,
    *,
    name: str,
    device: torch.device,
):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    output = adapter.forward_with_metadata(
        encoded
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    n = encoded.coord.shape[0]

    if output.dense_features.shape != (
        n,
        adapter.dense_dim,
    ):
        raise RuntimeError(
            f"{name}: dense feature shape mismatch: "
            f"{tuple(output.dense_features.shape)}"
        )

    tokens = output.tokens

    if not torch.is_tensor(tokens):
        raise RuntimeError(
            f"{name}: single-cloud test expected Tensor tokens"
        )

    if (
        tokens.ndim != 2
        or tokens.shape[1] != adapter.out_dim
        or tokens.shape[0] > adapter.num_tokens
    ):
        raise RuntimeError(
            f"{name}: bad token shape {tuple(tokens.shape)}"
        )

    if not torch.isfinite(
        output.dense_features
    ).all():
        raise RuntimeError(
            f"{name}: dense features contain NaN/Inf"
        )

    if not torch.isfinite(
        tokens
    ).all():
        raise RuntimeError(
            f"{name}: tokens contain NaN/Inf"
        )

    print(f"\n[{name} / PointAdapter]")
    print(" dense        :", tuple(output.dense_features.shape))
    print(" tokens       :", tuple(tokens.shape))
    print(" pooled voxels:", output.pooled_voxel_count)
    print(" token voxel  :", output.effective_voxel_size)
    print(" intensity_used:", output.intensity_used)
    print(" elapsed      :", f"{elapsed:.3f} s")

    if device.type == "cuda":
        peak = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 3)
        )
        print(" peak allocated:", f"{peak:.3f} GiB")

    return output


def main():
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_json(
        config_path
    )

    if args.dataset not in config["datasets"]:
        raise KeyError(
            f"{args.dataset!r} is not configured"
        )

    ds_cfg = config["datasets"][
        args.dataset
    ]

    root = Path(
        ds_cfg["root"]
    ).expanduser().resolve()

    class_names = {
        int(k): str(v)
        for k, v in ds_cfg[
            "class_names"
        ].items()
    }

    ignored_id = ds_cfg.get(
        "ignored_id",
        None,
    )

    model_cfg = config["model"]
    encoder_cfg = model_cfg[
        "point_encoder"
    ]

    checkpoint = Path(
        encoder_cfg["checkpoint"]
    ).expanduser().resolve()

    voxel_size = float(
        encoder_cfg["voxel_size"]
    )

    max_tokens = int(
        model_cfg[
            "max_point_reasoning_tokens"
        ]
    )

    manifest_path = (
        root
        / "manifests"
        / f"{args.split}.jsonl"
    )

    records = load_manifest(
        manifest_path
    )

    if not 0 <= args.index < len(records):
        raise IndexError(
            f"--index {args.index} outside "
            f"[0,{len(records)-1}]"
        )

    record = records[
        args.index
    ]

    required = (
        "point_t1",
        "point_t2",
        "semantic_t1",
        "semantic_t2",
    )

    missing = [
        key
        for key in required
        if key not in record
    ]

    if missing:
        raise RuntimeError(
            f"Manifest record missing keys: {missing}"
        )

    t1_path = resolve(
        root,
        record["point_t1"],
    )

    t2_path = resolve(
        root,
        record["point_t2"],
    )

    s1_path = resolve(
        root,
        record["semantic_t1"],
    )

    s2_path = resolve(
        root,
        record["semantic_t2"],
    )

    coord_t1 = load_point(
        t1_path
    )

    coord_t2 = load_point(
        t2_path
    )

    semantic_t1, change_t1 = (
        load_supervision(
            s1_path
        )
    )

    semantic_t2, change_t2 = (
        load_supervision(
            s2_path
        )
    )

    if (
        coord_t1.shape[0]
        != semantic_t1.shape[0]
    ):
        raise RuntimeError(
            "T1 point/label topology mismatch"
        )

    if (
        coord_t2.shape[0]
        != semantic_t2.shape[0]
    ):
        raise RuntimeError(
            "T2 point/label topology mismatch"
        )

    validate_labels(
        semantic_t1,
        change_t1,
        class_names=class_names,
        ignored_id=ignored_id,
        name="T1",
    )

    validate_labels(
        semantic_t2,
        change_t2,
        class_names=class_names,
        ignored_id=ignored_id,
        name="T2",
    )

    print("=" * 78)
    print("PAIR NYC-SCD -> Utonia -> PointAdapter smoke test")
    print("=" * 78)
    print(" config      :", config_path)
    print(" dataset     :", args.dataset)
    print(" split       :", args.split)
    print(" sample      :", record.get("id", args.index))
    print(" ignored_id  :", ignored_id)
    print(" Utonia ckpt :", checkpoint)
    print(" Utonia voxel:", voxel_size)
    print(" token budget:", max_tokens)

    print("\n[Prepared data]")
    print(
        " T1:",
        f"N={coord_t1.shape[0]:,}",
        "semantic=",
        unique_counts(semantic_t1),
        "change=",
        unique_counts(change_t1),
    )

    print(
        " T2:",
        f"N={coord_t2.shape[0]:,}",
        "semantic=",
        unique_counts(semantic_t2),
        "change=",
        unique_counts(change_t2),
    )

    device = torch.device(
        args.device
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable"
            )

        print("\n[CUDA]")
        print(" torch      :", torch.__version__)
        print(" cuda       :", torch.version.cuda)
        print(
            " gpu        :",
            torch.cuda.get_device_name(
                device
            ),
        )

    encoder = UtoniaPointEncoder(
        UtoniaPointEncoderConfig(
            checkpoint=str(checkpoint),
            voxel_size=voxel_size,
        )
    ).to(device)

    encoder.eval()

    if any(
        p.requires_grad
        for p in encoder.parameters()
    ):
        raise RuntimeError(
            "Utonia must be fully frozen"
        )

    adapter = PointAdapter(
        PointAdapterConfig(
            in_dim=encoder.output_dim,
            dense_dim=int(
                model_cfg[
                    "decoder_dim"
                ]
            ),
            out_dim=2560,
            num_tokens=max_tokens,
        )
    ).to(device)

    adapter.eval()

    print("\n[PointAdapter]")
    print(
        " trainable params:",
        f"{adapter.trainable_parameter_count():,}",
    )
    print(
        " dims:",
        f"{encoder.output_dim}"
        f" -> {adapter.dense_dim}"
        f" -> {adapter.out_dim}",
    )

    # ------------------------------------------------------------------
    # T1: geometry-only reference
    # ------------------------------------------------------------------
    t1_geo_input = point_dict_from_numpy(
        coord_t1,
        device=device,
        with_fake_intensity=False,
    )

    t1_encoded_geo = run_utonia(
        encoder,
        t1_geo_input,
        name="T1 geometry-only",
        device=device,
    )

    t1_adapter_geo = run_adapter(
        adapter,
        t1_encoded_geo,
        name="T1 geometry-only",
        device=device,
    )

    if t1_adapter_geo.intensity_used:
        raise RuntimeError(
            "Geometry-only path incorrectly reports intensity_used=True"
        )

    # ------------------------------------------------------------------
    # T1: same points + deterministic fake intensity
    # ------------------------------------------------------------------
    t1_fake_input = point_dict_from_numpy(
        coord_t1,
        device=device,
        with_fake_intensity=True,
    )

    t1_encoded_fake = run_utonia(
        encoder,
        t1_fake_input,
        name="T1 fake-intensity",
        device=device,
    )

    # Utonia must be identical: fake intensity bypasses Utonia entirely.
    utonia_diff = (
        t1_encoded_geo.features
        - t1_encoded_fake.features
    ).abs().max().item()

    print(
        "\n[T1 Utonia geometry vs fake-I]"
    )
    print(
        " max feature difference:",
        f"{utonia_diff:.9g}",
    )

    if not torch.allclose(
        t1_encoded_geo.features,
        t1_encoded_fake.features,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "Fake intensity changed frozen Utonia features beyond "
            f"floating-point tolerance; max difference={utonia_diff}"
        )

    t1_adapter_fake = run_adapter(
        adapter,
        t1_encoded_fake,
        name="T1 fake-intensity",
        device=device,
    )

    if not t1_adapter_fake.intensity_used:
        raise RuntimeError(
            "Fake intensity path reports intensity_used=False"
        )

    # Fresh adapter starts geometry-only because residual intensity fusion's
    # final layer is zero initialized.
    dense_diff = (
        t1_adapter_geo.dense_features
        - t1_adapter_fake.dense_features
    ).abs().max().item()

    print(
        "\n[T1 adapter initial geometry vs fake-I]"
    )
    print(
        " max dense difference:",
        f"{dense_diff:.9g}",
    )

    if not torch.allclose(
        t1_adapter_geo.dense_features,
        t1_adapter_fake.dense_features,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "Fresh intensity residual branch should start from the "
            "geometry-only representation within floating-point tolerance; "
            f"max dense difference={dense_diff}"
        )

    # ------------------------------------------------------------------
    # T2: full fake-intensity path
    # ------------------------------------------------------------------
    t2_fake_input = point_dict_from_numpy(
        coord_t2,
        device=device,
        with_fake_intensity=True,
    )

    t2_encoded_fake = run_utonia(
        encoder,
        t2_fake_input,
        name="T2 fake-intensity",
        device=device,
    )

    t2_adapter_fake = run_adapter(
        adapter,
        t2_encoded_fake,
        name="T2 fake-intensity",
        device=device,
    )

    if not t2_adapter_fake.intensity_used:
        raise RuntimeError(
            "T2 fake intensity was not used"
        )

    print("\n" + "=" * 78)
    print("PASS")
    print("=" * 78)
    print(
        "Real NYC-SCD -> frozen Utonia -> PointAdapter passed."
    )
    print(
        "Fake intensity was carried correctly and entered only "
        "the PAIR intensity branch, not Utonia."
    )
    print(
        "T1 dense:",
        tuple(t1_adapter_fake.dense_features.shape),
        "tokens:",
        tuple(t1_adapter_fake.tokens.shape),
    )
    print(
        "T2 dense:",
        tuple(t2_adapter_fake.dense_features.shape),
        "tokens:",
        tuple(t2_adapter_fake.tokens.shape),
    )


if __name__ == "__main__":
    main()
