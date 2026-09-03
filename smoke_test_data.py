#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR real multi-class dataset -> config DatasetSpec -> train-mode forward test.

This test intentionally reads DatasetSpec from:

    datasets/config.py

It DOES NOT infer class names or class count from the data.

Pipeline
--------
dataset/config.py
      ↓
manual DatasetSpec
      ↓
manifests/train.jsonl
      ↓
UnifiedPAIRDataset
      ↓
real multi-class T1/T2 semantic labels
      ↓
real T1/T2 images
      ↓
PAIRModel.train()
      ↓
temporal 2D forward with autograd
      ↓
image_hidden_t1 / image_hidden_t2 / task_hidden / logits

Run
---
CUDA_VISIBLE_DEVICES=2 python smoke_test_dataset_forward.py \
    --dataset-root /home/sht/Datasets/SECONDpair \
    --split train \
    --spec SECOND_SPEC \
    --index 0
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path

import torch
from PIL import Image

from datasets.pair_dataset import (
    DatasetSpec,
    UnifiedPAIRDataset,
    IGNORE_CLASS_ID,
    UNKNOWN_CLASS_ID,
)
from models.pair import PAIRModel
from models.qwen3vl_backbone import Qwen3VLBackbone


DEFAULT_MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Prepared dataset root containing manifests/ and flat modality folders.",
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )

    parser.add_argument(
        "--spec",
        type=str,
        default="SECOND_SPEC",
        help="Variable name of DatasetSpec in dataset/config.py.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
    )

    return parser.parse_args()


def load_spec(spec_name: str) -> DatasetSpec:
    """
    Load one manually written DatasetSpec from dataset/config.py.

    No automatic class discovery from labels is performed.
    """
    module = importlib.import_module("datasets.configs")

    if not hasattr(module, spec_name):
        available = {
            name: value
            for name, value in vars(module).items()
            if isinstance(value, DatasetSpec)
        }

        names = ", ".join(sorted(available)) if available else "<none>"

        raise AttributeError(
            f"dataset.config has no DatasetSpec named {spec_name!r}.\n"
            f"Available DatasetSpec objects: {names}"
        )

    spec = getattr(module, spec_name)

    if not isinstance(spec, DatasetSpec):
        raise TypeError(
            f"dataset.config.{spec_name} exists, but is not DatasetSpec: "
            f"{type(spec)}"
        )

    return spec


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """
    UnifiedPAIRDataset currently returns float32 [C,H,W] in [0,1].
    Convert back to uint8 PIL so Qwen owns its native image preprocessing.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], got {tuple(image.shape)}"
        )

    x = image.detach().cpu().float()

    if not torch.isfinite(x).all():
        raise ValueError("Image contains NaN/Inf.")

    vmin = float(x.min())
    vmax = float(x.max())

    if vmin < -1e-4 or vmax > 1.0001:
        raise ValueError(
            f"Expected image in [0,1], got [{vmin}, {vmax}]"
        )

    x = (
        x.clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
    )

    arr = (
        x.permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    if arr.shape[-1] == 1:
        arr = arr[..., 0]

    return Image.fromarray(arr)


def tensor_info(name: str, x):
    if x is None:
        print(f"  {name:26s}: None")
        return

    print(
        f"  {name:26s}: "
        f"shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"device={x.device}, "
        f"requires_grad={x.requires_grad}"
    )


def label_stats(name: str, x: torch.Tensor):
    values, counts = torch.unique(
        x.detach().cpu(),
        return_counts=True,
    )

    pairs = list(
        zip(
            values.tolist(),
            counts.tolist(),
        )
    )

    print(
        f"  {name:26s}: {pairs}"
    )

    return pairs


def validate_multiclass_target(
    spec: DatasetSpec,
    target,
):
    """
    Check the real semantic labels against the MANUALLY written class_names.

    We do not derive class_names from labels.
    """
    if spec.class_names is None:
        raise ValueError(
            "This test is specifically for real multi-class data, but "
            f"{spec.name}.class_names is None in dataset/config.py.\n"
            "Fill class_names manually in the DatasetSpec first."
        )

    class_names = list(spec.class_names)
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(
            f"Need a multi-class DatasetSpec, got {num_classes} class(es)."
        )

    sem1 = target["semantic_t1"]
    sem2 = target["semantic_t2"]

    valid1 = target["semantic_valid_t1"]
    valid2 = target["semantic_valid_t2"]

    ids1 = torch.unique(
        sem1[valid1]
    ).tolist()

    ids2 = torch.unique(
        sem2[valid2]
    ).tolist()

    bad1 = [
        int(v)
        for v in ids1
        if not (0 <= int(v) < num_classes)
    ]

    bad2 = [
        int(v)
        for v in ids2
        if not (0 <= int(v) < num_classes)
    ]

    if bad1 or bad2:
        raise ValueError(
            "Real semantic label IDs do not match the manually written "
            "DatasetSpec.class_names.\n"
            f"num_classes={num_classes}\n"
            f"bad T1 IDs={bad1}\n"
            f"bad T2 IDs={bad2}\n"
            "Fix dataset/config.py or the source label mapping manually."
        )

    observed = sorted(
        set(
            int(v)
            for v in ids1 + ids2
        )
    )

    print("  manual class_names:")
    for i, name in enumerate(class_names):
        suffix = "  [present in this sample]" if i in observed else ""
        print(
            f"    {i:2d}: {name}{suffix}"
        )

    print(
        "  observed valid semantic IDs:",
        observed,
    )

    return num_classes, observed


def gib(value: int) -> float:
    return value / 1024**3


def main():
    args = parse_args()

    root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    manifest = (
        root
        / "manifests"
        / f"{args.split}.jsonl"
    )

    print("=" * 96)
    print("PAIR REAL MULTI-CLASS DATASET -> TRAIN FORWARD")
    print("=" * 96)
    print()

    # ==================================================================
    # 1. Read the ACTUAL manually written DatasetSpec
    # ==================================================================
    print("[1] LOAD DatasetSpec FROM dataset/config.py")

    spec = load_spec(
        args.spec
    )

    print(
        f"  spec variable             : dataset.config.{args.spec}"
    )
    print(
        f"  name                      : {spec.name}"
    )
    print(
        f"  task_mode                 : {spec.task_mode}"
    )
    print(
        f"  label_mode                : {spec.label_mode}"
    )
    print(
        f"  class_names               : {list(spec.class_names) if spec.class_names is not None else None}"
    )
    print(
        f"  image_size                : {spec.image_size}"
    )
    print(
        f"  ignore_value              : {spec.ignore_value}"
    )

    if spec.task_mode.lower().strip() != "2d":
        raise ValueError(
            f"This test expects a 2D spec, got {spec.task_mode!r}."
        )

    if spec.label_mode.lower().strip() != "semantic_pair":
        raise ValueError(
            "This multi-class SECOND test expects label_mode='semantic_pair', "
            f"got {spec.label_mode!r}."
        )

    print("  CONFIG SPEC: PASS")
    print()

    # ==================================================================
    # 2. Read real dataset through UnifiedPAIRDataset
    # ==================================================================
    print("[2] LOAD REAL MULTI-CLASS DATA")

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}"
        )

    dataset = UnifiedPAIRDataset(
        manifest,
        spec,
    )

    print(
        "  dataset length            :",
        len(dataset),
    )

    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"index={args.index}, dataset length={len(dataset)}"
        )

    t0 = time.perf_counter()
    sample = dataset[
        args.index
    ]
    read_time = time.perf_counter() - t0

    print(
        "  sample_id                 :",
        sample["sample_id"],
    )
    print(
        f"  read time                 : {read_time:.3f} s"
    )

    image_t1 = sample["images_t1"]
    image_t2 = sample["images_t2"]
    target = sample["target"]

    tensor_info(
        "images_t1",
        image_t1,
    )

    tensor_info(
        "images_t2",
        image_t2,
    )

    label_stats(
        "semantic_t1",
        target["semantic_t1"],
    )

    label_stats(
        "semantic_t2",
        target["semantic_t2"],
    )

    label_stats(
        "change",
        target["change"],
    )

    assert image_t1.shape == image_t2.shape

    num_classes, observed_ids = validate_multiclass_target(
        spec,
        target,
    )

    # Canonical change must still be binary over valid positions.
    valid_change = target["change"][
        target["change_valid"]
    ]

    change_ids = set(
        int(v)
        for v in torch.unique(
            valid_change
        ).tolist()
    )

    if not change_ids.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"Canonical change contains non-binary valid IDs: {change_ids}"
        )

    print("  REAL MULTI-CLASS DATA: PASS")
    print()

    # ==================================================================
    # 3. Build model
    # ==================================================================
    print("[3] BUILD PAIR")

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"Qwen checkpoint not found: {args.model_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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

    model.train()

    print(
        "  model.training            :",
        model.training,
    )

    print(
        "  qwen hidden size          :",
        qwen.hidden_size,
    )

    print("  MODEL BUILD: PASS")
    print()

    # ==================================================================
    # 4. REAL training-style forward
    # ==================================================================
    print("[4] REAL TRAIN-MODE FORWARD")

    pil_t1 = tensor_to_pil(
        image_t1
    )

    pil_t2 = tensor_to_pil(
        image_t2
    )

    # No no_grad(), no inference_mode().
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    output = model(
        task_mode=spec.task_mode,

        prompt=sample["prompt"],

        images_t1=pil_t1,
        images_t2=pil_t2,

        return_logits=True,
        return_hidden_states=True,

        use_cache=False,
    )

    torch.cuda.synchronize()
    forward_time = (
        time.perf_counter()
        - t0
    )

    print(
        f"  forward time              : {forward_time:.3f} s"
    )

    tensor_info(
        "image_hidden_t1",
        output.image_hidden_t1,
    )

    tensor_info(
        "image_hidden_t2",
        output.image_hidden_t2,
    )

    tensor_info(
        "task_hidden",
        output.task_hidden,
    )

    tensor_info(
        "logits",
        output.logits,
    )

    if output.aux is not None:
        tensor_info(
            "image_hidden_2d_t1",
            output.aux.get(
                "image_hidden_2d_t1"
            ),
        )

        tensor_info(
            "image_hidden_2d_t2",
            output.aux.get(
                "image_hidden_2d_t2"
            ),
        )

    # ------------------------------------------------------------------
    # Autograd must exist in a training forward.
    # ------------------------------------------------------------------
    required_grad_outputs = {
        "image_hidden_t1": output.image_hidden_t1,
        "image_hidden_t2": output.image_hidden_t2,
        "task_hidden": output.task_hidden,
        "logits": output.logits,
    }

    for name, value in required_grad_outputs.items():
        if value is None:
            raise RuntimeError(
                f"{name} is None."
            )

        if not value.requires_grad:
            raise RuntimeError(
                f"{name}.requires_grad=False in train-mode forward."
            )

        if not torch.isfinite(
            value
        ).all():
            raise RuntimeError(
                f"{name} contains NaN/Inf."
            )

    delta = (
        output.image_hidden_t1.float()
        - output.image_hidden_t2.float()
    ).abs().mean()

    print(
        "  mean |T1-T2 hidden|       :",
        float(
            delta.detach().cpu()
        ),
    )

    if (
        not torch.equal(
            image_t1,
            image_t2,
        )
        and float(
            delta.detach().cpu()
        ) <= 0.0
    ):
        raise RuntimeError(
            "T1/T2 images differ but contextual visual representations collapsed."
        )

    print("  TRAIN FORWARD: PASS")
    print()

    # ==================================================================
    # 5. Summary
    # ==================================================================
    print("[5] GPU")

    print(
        "  peak allocated            : "
        f"{gib(torch.cuda.max_memory_allocated()):.2f} GiB"
    )

    print(
        "  peak reserved             : "
        f"{gib(torch.cuda.max_memory_reserved()):.2f} GiB"
    )

    print()
    print("=" * 96)
    print("SUCCESS")
    print("=" * 96)
    print(
        f"DatasetSpec source          : dataset.config.{args.spec}"
    )
    print(
        f"Manual semantic classes     : {num_classes}"
    )
    print(
        f"Classes present in sample   : {observed_ids}"
    )
    print(
        "Real multi-class data       : PASS"
    )
    print(
        "Canonical semantic target   : PASS"
    )
    print(
        "PAIR train-mode forward     : PASS"
    )
    print(
        "Autograd graph              : PASS"
    )
    print()
    print(
        "No class vocabulary was inferred from label files."
    )
    print(
        "The next step after this passes is the real dense semantic-change "
        "decoder/loss, then backward."
    )


if __name__ == "__main__":
    main()
