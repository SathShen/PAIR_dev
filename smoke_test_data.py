#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real 2D dataset -> PAIR temporal model smoke test.

Tests
=====
1. Read train/val/test JSONL with UnifiedPAIRDataset.
2. Resolve relative manifest paths correctly.
3. Load real T1/T2 images and semantic labels.
4. Build canonical semantic-change target:
       change
       semantic_t1
       semantic_t2
       change_valid
       semantic_valid_t1
       semantic_valid_t2
5. Check image/label geometry and tensor validity.
6. Convert the dataset image tensor safely to PIL for Qwen preprocessing.
7. Build the real Qwen3-VL backbone + temporal PAIR model.
8. Feed REAL dataset T1/T2 into task_mode="2d".
9. Verify:
       image_hidden_t1
       image_hidden_t2
       task_hidden
       T1/T2 representations differ
10. Optional native language generation.

This test intentionally does NOT resize to 448x448.
It uses the dataset's current spatial size unless DatasetSpec.image_size
is manually set below.

Run example
===========
CUDA_VISIBLE_DEVICES=2 python smoke_test_real_2d_dataset.py \
    --dataset-root /data2/sht/Datasets/SECONDBbi \
    --split train

Optional generation:
CUDA_VISIBLE_DEVICES=2 python smoke_test_real_2d_dataset.py \
    --dataset-root /data2/sht/Datasets/SECONDBbi \
    --split train \
    --generate
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ======================================================================
# PAIR dataset import
# ======================================================================

# Try the likely project locations so this smoke test does not care whether
# pair_dataset.py is currently in datasets/, data_preparation/, or repo root.
try:
    from datasets.pair_dataset import (
        DatasetSpec,
        UnifiedPAIRDataset,
        UNKNOWN_CLASS_ID,
        IGNORE_CLASS_ID,
    )
    DATASET_IMPORT = "datasets.pair_dataset"

except ImportError:
    try:
        from data_preparation.pair_dataset import (
            DatasetSpec,
            UnifiedPAIRDataset,
            UNKNOWN_CLASS_ID,
            IGNORE_CLASS_ID,
        )
        DATASET_IMPORT = "data_preparation.pair_dataset"

    except ImportError:
        from pair_dataset import (
            DatasetSpec,
            UnifiedPAIRDataset,
            UNKNOWN_CLASS_ID,
            IGNORE_CLASS_ID,
        )
        DATASET_IMPORT = "pair_dataset"


from models.pair import PAIRModel
from models.qwen3vl_backbone import Qwen3VLBackbone


DEFAULT_MODEL_DIR = (
    "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"
)


# ======================================================================
# MANUAL DATASET SPEC
# ======================================================================
#
# IMPORTANT:
# Do NOT let the preparation script guess these values.
#
# For the first smoke test, class_names=None is enough to verify the data/model
# pipeline. Once the exact SECOND class-ID mapping is confirmed, fill it here.
#
# Example:
#
# CLASS_NAMES = [
#     "class_0",
#     "class_1",
#     ...
# ]
#
CLASS_NAMES = None

# Leave None to preserve the dataset's native image size.
IMAGE_SIZE = None

# Change this only if the actual source labels use another ignore value.
IGNORE_VALUE = 255


def build_spec() -> DatasetSpec:
    return DatasetSpec(
        name="SECONDBbi",
        task_mode="2d",
        label_mode="semantic_pair",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        ignore_value=IGNORE_VALUE,
        strict_geo_alignment=True,
        prompt=(
            "Perform semantic change detection between Time 1 and Time 2. "
            "Identify unchanged regions and changed regions, and infer the "
            "semantic classes before and after each change when possible."
        ),
    )


# ======================================================================
# Utilities
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Prepared dataset root containing images_t1/, images_t2/, "
            "semantic_t1/, semantic_t2/, manifests/."
        ),
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index inside the selected split.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also test native Qwen language generation.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
    )

    return parser.parse_args()


def gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def summarize_tensor(
    name: str,
    x: torch.Tensor | None,
):
    if x is None:
        print(f"  {name}: None")
        return

    print(
        f"  {name}: "
        f"shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"device={x.device}"
    )


def unique_summary(
    tensor: torch.Tensor,
    *,
    max_values: int = 30,
):
    values, counts = torch.unique(
        tensor.detach().cpu(),
        return_counts=True,
    )

    pairs = list(
        zip(
            values.tolist(),
            counts.tolist(),
        )
    )

    if len(pairs) > max_values:
        return pairs[:max_values] + [
            ("...", "...")
        ]

    return pairs


def chw_float_to_pil(
    image: torch.Tensor,
) -> Image.Image:
    """
    Convert UnifiedPAIRDataset image tensor [C,H,W], float approximately
    in [0,1], back to a normal uint8 PIL image before Qwen processing.

    Why?
    ----
    The dataset loader currently returns normalized torch tensors, while the
    already validated PAIR temporal smoke tests feed PIL images to Qwen.
    Converting explicitly here prevents an image processor from accidentally
    applying an additional 1/255 scaling to an already normalized tensor.

    This is a smoke-test bridge. Later we can decide whether the canonical
    dataset output should keep both raw/PIL and decoder tensor representations.
    """

    if not torch.is_tensor(image):
        raise TypeError(
            f"Expected torch.Tensor image, got {type(image)}"
        )

    if image.ndim != 3:
        raise ValueError(
            f"Expected image [C,H,W], got {tuple(image.shape)}"
        )

    if image.shape[0] not in (1, 3, 4):
        raise ValueError(
            f"Unsupported channel count: {image.shape[0]}"
        )

    x = (
        image.detach()
        .cpu()
        .float()
    )

    if not torch.isfinite(x).all():
        raise ValueError(
            "Image tensor contains NaN/Inf."
        )

    vmin = float(x.min().item())
    vmax = float(x.max().item())

    # Current pair_dataset.py should produce ~[0,1].
    # Fail loudly instead of silently clipping badly normalized data.
    if vmin < -1e-4 or vmax > 1.0001:
        raise ValueError(
            "Dataset image is not in expected [0,1] range: "
            f"min={vmin}, max={vmax}"
        )

    x = (
        x.clamp(0.0, 1.0)
        * 255.0
    ).round().to(
        torch.uint8
    )

    arr = (
        x.permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    if arr.shape[2] == 1:
        arr = arr[:, :, 0]

    return Image.fromarray(arr)


def assert_bool_mask(
    name: str,
    mask: torch.Tensor,
    expected_shape,
):
    assert mask.dtype == torch.bool, (
        f"{name} must be bool, got {mask.dtype}"
    )

    assert tuple(mask.shape) == tuple(expected_shape), (
        f"{name} shape {tuple(mask.shape)} != {tuple(expected_shape)}"
    )


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    manifest = (
        dataset_root
        / "manifests"
        / f"{args.split}.jsonl"
    )

    print("=" * 92)
    print("PAIR REAL 2D DATASET -> TEMPORAL MODEL SMOKE TEST")
    print("=" * 92)
    print()
    print("Dataset module :", DATASET_IMPORT)
    print("Dataset root   :", dataset_root)
    print("Manifest       :", manifest)
    print("Split          :", args.split)
    print("Sample index   :", args.index)
    print("Qwen checkpoint:", args.model_dir)
    print()

    assert dataset_root.is_dir(), (
        f"Dataset root not found: {dataset_root}"
    )

    assert manifest.is_file(), (
        f"Manifest not found: {manifest}"
    )

    assert os.path.isdir(args.model_dir), (
        f"Qwen checkpoint not found: {args.model_dir}"
    )

    # ==================================================================
    # 1. Build real dataset
    # ==================================================================
    print("[1] Building UnifiedPAIRDataset...")

    spec = build_spec()

    dataset = UnifiedPAIRDataset(
        manifest,
        spec,
    )

    assert len(dataset) > 0, (
        "Dataset is empty."
    )

    assert 0 <= args.index < len(dataset), (
        f"--index {args.index} outside dataset length {len(dataset)}"
    )

    print("  dataset length:", len(dataset))
    print("  task mode:", spec.task_mode)
    print("  label mode:", spec.label_mode)
    print("  class_names:", spec.class_names)
    print("  image_size:", spec.image_size)
    print("  DATASET BUILD OK")
    print()

    # ==================================================================
    # 2. Load one REAL sample
    # ==================================================================
    print("[2] Reading real sample...")

    t0 = time.perf_counter()

    sample = dataset[
        args.index
    ]

    print(
        f"  read time: "
        f"{time.perf_counter() - t0:.3f} s"
    )

    print("  sample_id:", sample["sample_id"])
    print("  dataset_name:", sample["dataset_name"])
    print("  task_mode:", sample["task_mode"])
    print("  sample keys:", sorted(sample.keys()))

    assert sample["task_mode"] == "2d"

    assert "images_t1" in sample
    assert "images_t2" in sample
    assert "target" in sample

    image_t1 = sample["images_t1"]
    image_t2 = sample["images_t2"]
    target = sample["target"]

    summarize_tensor(
        "images_t1",
        image_t1,
    )
    summarize_tensor(
        "images_t2",
        image_t2,
    )

    assert image_t1.ndim == 3
    assert image_t2.ndim == 3

    assert image_t1.shape == image_t2.shape, (
        "T1/T2 image tensors have different shapes."
    )

    assert torch.isfinite(
        image_t1
    ).all()

    assert torch.isfinite(
        image_t2
    ).all()

    print(
        "  image T1 range:",
        float(image_t1.min()),
        "to",
        float(image_t1.max()),
    )
    print(
        "  image T2 range:",
        float(image_t2.min()),
        "to",
        float(image_t2.max()),
    )

    print("  REAL IMAGE PAIR OK")
    print()

    # ==================================================================
    # 3. Validate canonical SCD target
    # ==================================================================
    print("[3] Checking canonical semantic-change target...")

    required_target_keys = (
        "change",
        "semantic_t1",
        "semantic_t2",
        "change_valid",
        "semantic_valid_t1",
        "semantic_valid_t2",
    )

    for key in required_target_keys:
        assert key in target, (
            f"Missing target field: {key}"
        )

    change = target["change"]
    sem_t1 = target["semantic_t1"]
    sem_t2 = target["semantic_t2"]

    h, w = image_t1.shape[-2:]
    spatial_shape = (
        h,
        w,
    )

    assert tuple(
        change.shape
    ) == spatial_shape

    assert tuple(
        sem_t1.shape
    ) == spatial_shape

    assert tuple(
        sem_t2.shape
    ) == spatial_shape

    assert_bool_mask(
        "change_valid",
        target["change_valid"],
        spatial_shape,
    )

    assert_bool_mask(
        "semantic_valid_t1",
        target["semantic_valid_t1"],
        spatial_shape,
    )

    assert_bool_mask(
        "semantic_valid_t2",
        target["semantic_valid_t2"],
        spatial_shape,
    )

    print(
        "  semantic_t1 values/counts:",
        unique_summary(
            sem_t1
        ),
    )

    print(
        "  semantic_t2 values/counts:",
        unique_summary(
            sem_t2
        ),
    )

    print(
        "  change values/counts:",
        unique_summary(
            change
        ),
    )

    print(
        "  valid change pixels:",
        int(
            target[
                "change_valid"
            ].sum()
        ),
        "/",
        change.numel(),
    )

    print(
        "  valid T1 semantic pixels:",
        int(
            target[
                "semantic_valid_t1"
            ].sum()
        ),
        "/",
        sem_t1.numel(),
    )

    print(
        "  valid T2 semantic pixels:",
        int(
            target[
                "semantic_valid_t2"
            ].sum()
        ),
        "/",
        sem_t2.numel(),
    )

    valid_change = change[
        target["change_valid"]
    ]

    valid_values = set(
        torch.unique(
            valid_change
        ).tolist()
    )

    assert valid_values.issubset(
        {0, 1}
    ), (
        "Valid canonical change labels must be only 0/1, "
        f"but got {sorted(valid_values)}"
    )

    # For semantic_pair supervision, where both labels are valid,
    # change should exactly equal semantic_t1 != semantic_t2.
    both_semantic_valid = (
        target[
            "semantic_valid_t1"
        ]
        & target[
            "semantic_valid_t2"
        ]
        & target[
            "change_valid"
        ]
    )

    if both_semantic_valid.any():
        derived = (
            sem_t1[
                both_semantic_valid
            ]
            != sem_t2[
                both_semantic_valid
            ]
        ).long()

        actual = change[
            both_semantic_valid
        ]

        assert torch.equal(
            derived,
            actual,
        ), (
            "Canonical change != (semantic_t1 != semantic_t2) "
            "on valid semantic pixels."
        )

    print("  CANONICAL TARGET OK")
    print()

    # ==================================================================
    # 4. Convert dataset images to Qwen-safe PIL
    # ==================================================================
    print("[4] Converting real image tensors to Qwen-safe PIL...")

    pil_t1 = chw_float_to_pil(
        image_t1
    )

    pil_t2 = chw_float_to_pil(
        image_t2
    )

    print(
        "  PIL T1:",
        pil_t1.mode,
        pil_t1.size,
    )

    print(
        "  PIL T2:",
        pil_t2.mode,
        pil_t2.size,
    )

    assert pil_t1.size == (
        image_t1.shape[2],
        image_t1.shape[1],
    )

    assert pil_t2.size == (
        image_t2.shape[2],
        image_t2.shape[1],
    )

    print("  PIL CONVERSION OK")
    print()

    # ==================================================================
    # 5. Check that LOCAL pair.py is the temporal model
    # ==================================================================
    print("[5] Checking local PAIR temporal interface...")

    forward_signature = inspect.signature(
        PAIRModel.forward
    )

    parameter_names = set(
        forward_signature.parameters.keys()
    )

    print(
        "  PAIRModel.forward parameters:",
        list(
            forward_signature.parameters.keys()
        ),
    )

    if (
        "images_t1" not in parameter_names
        or "images_t2" not in parameter_names
    ):
        raise RuntimeError(
            "\nYour local models/pair.py is NOT the temporal PAIR version.\n"
            "Expected PAIRModel.forward(..., images_t1=..., images_t2=...).\n"
            "Do not continue with the old single-image `images=` interface."
        )

    print("  TEMPORAL PAIR INTERFACE OK")
    print()

    # ==================================================================
    # 6. Build real Qwen + 2D PAIR
    # ==================================================================
    print("[6] Building Qwen3VLBackbone + PAIRModel...")

    assert torch.cuda.is_available(), (
        "CUDA is required for the full-model test."
    )

    device = torch.device(
        "cuda"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()

    print(
        "  PyTorch:",
        torch.__version__,
    )
    print(
        "  CUDA:",
        torch.version.cuda,
    )
    print(
        "  GPU:",
        torch.cuda.get_device_name(
            device
        ),
    )

    t0 = time.perf_counter()

    qwen = Qwen3VLBackbone(
        model_dir=args.model_dir,
        dtype=torch.bfloat16,
        device="cuda",
        device_map="cuda",
        local_files_only=True,
    )

    model = PAIRModel(
        qwen_backbone=qwen,
    )

    model.eval()

    torch.cuda.synchronize()

    print(
        f"  model load time: "
        f"{time.perf_counter() - t0:.2f} s"
    )
    print(
        "  Qwen hidden size:",
        qwen.hidden_size,
    )

    print("  MODEL BUILD OK")
    print()

    # ==================================================================
    # 7. REAL DATA -> temporal PAIR forward
    # ==================================================================
    print("[7] Feeding REAL T1/T2 into temporal PAIR...")

    prompt = sample["prompt"]

    print("  prompt:")
    print(
        "   ",
        prompt,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output = model(
            task_mode="2d",
            prompt=prompt,
            images_t1=pil_t1,
            images_t2=pil_t2,
            return_logits=False,
            return_hidden_states=True,
        )

    torch.cuda.synchronize()

    print(
        f"  forward time: "
        f"{time.perf_counter() - t0:.3f} s"
    )

    summarize_tensor(
        "image_hidden_t1",
        output.image_hidden_t1,
    )

    summarize_tensor(
        "image_hidden_t2",
        output.image_hidden_t2,
    )

    summarize_tensor(
        "task_hidden",
        output.task_hidden,
    )

    assert output.image_hidden_t1 is not None
    assert output.image_hidden_t2 is not None
    assert output.task_hidden is not None

    assert output.image_hidden_t1.ndim == 2
    assert output.image_hidden_t2.ndim == 2

    assert output.image_hidden_t1.shape[1] == qwen.hidden_size
    assert output.image_hidden_t2.shape[1] == qwen.hidden_size

    assert output.task_hidden.shape[-1] == qwen.hidden_size

    assert torch.isfinite(
        output.image_hidden_t1
    ).all()

    assert torch.isfinite(
        output.image_hidden_t2
    ).all()

    assert torch.isfinite(
        output.task_hidden
    ).all()

    # Point branch must be inactive.
    assert output.point_hidden_t1 is None
    assert output.point_hidden_t2 is None

    print(
        "  Qwen image token counts T1/T2:",
        output.image_hidden_t1.shape[0],
        output.image_hidden_t2.shape[0],
    )

    # Different native image resolutions may produce dynamic token counts.
    # T1/T2 of the same paired sample should still produce matching counts.
    assert (
        output.image_hidden_t1.shape[0]
        == output.image_hidden_t2.shape[0]
    ), (
        "T1/T2 Qwen image token counts differ."
    )

    image_delta = (
        output.image_hidden_t1.float()
        - output.image_hidden_t2.float()
    ).abs().mean().item()

    print(
        "  mean |image_hidden_t1 - image_hidden_t2|:",
        image_delta,
    )

    if torch.equal(
        image_t1,
        image_t2,
    ):
        print(
            "  NOTE: raw T1/T2 tensors are identical; "
            "representation-difference assertion skipped."
        )
    else:
        assert image_delta > 0.0, (
            "Real T1/T2 images differ but Qwen representations are identical."
        )

    # If temporal PAIR exposes spatial hidden features, validate them too.
    if output.aux is not None:
        spatial_t1 = output.aux.get(
            "image_hidden_2d_t1"
        )
        spatial_t2 = output.aux.get(
            "image_hidden_2d_t2"
        )

        if spatial_t1 is not None:
            summarize_tensor(
                "image_hidden_2d_t1",
                spatial_t1,
            )

        if spatial_t2 is not None:
            summarize_tensor(
                "image_hidden_2d_t2",
                spatial_t2,
            )

        print(
            "  aux image counts T1/T2:",
            output.aux.get(
                "image_count_t1"
            ),
            output.aux.get(
                "image_count_t2"
            ),
        )

    print("  REAL DATA -> PAIR FORWARD OK")
    print()

    # ==================================================================
    # 8. Optional generation
    # ==================================================================
    if args.generate:
        print("[8] Testing native generation with REAL T1/T2...")

        with torch.inference_mode():
            generated = model.generate(
                task_mode="2d",
                prompt=prompt,
                images_t1=pil_t1,
                images_t2=pil_t2,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        print("  generated text:")
        print(
            "   ",
            generated.generated_text,
        )

        assert generated.generated_ids is not None
        assert isinstance(
            generated.generated_text,
            str,
        )

        print("  REAL DATA GENERATION OK")
        print()

    # ==================================================================
    # Final
    # ==================================================================
    torch.cuda.synchronize()

    print("=" * 92)
    print("SUCCESS")
    print("=" * 92)
    print()
    print("The real 2D pipeline is connected:")
    print()
    print("  manifests/*.jsonl")
    print("          ↓")
    print("  UnifiedPAIRDataset")
    print("          ↓")
    print("  real Image T1 / Image T2")
    print("          ↓")
    print("  canonical semantic-change target")
    print("          ↓")
    print("  temporal Qwen3-VL / PAIR")
    print("          ↓")
    print("  image_hidden_t1 / image_hidden_t2 / task_hidden")
    print()
    print(
        "Peak allocated GPU memory:",
        f"{gib(torch.cuda.max_memory_allocated()):.2f} GiB",
    )
    print(
        "Peak reserved GPU memory: ",
        f"{gib(torch.cuda.max_memory_reserved()):.2f} GiB",
    )
    print()
    print("Manifest relative paths:   PASS")
    print("Real paired image loading: PASS")
    print("Canonical target:          PASS")
    print("Real T1/T2 -> PAIR:        PASS")
    print("Dynamic image size:        PASS")
    if args.generate:
        print("Native generation:         PASS")


if __name__ == "__main__":
    main()
