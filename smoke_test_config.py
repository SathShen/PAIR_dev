#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR three-dataset model smoke.

Current intended trio:
    SECOND      : 2D SCD  -> ValBest F_scd
    LEVIR-CD    : 2D BCD  -> ValBest IoU
    LandsatSCD  : 2D SCD  -> ValBest F_scd

What this smoke checks
======================
1. One config loads all three datasets together.
2. Directory-as-schema inference is correct.
3. One real sample can be loaded from every dataset.
4. LEVIR physical 0/255 labels are normalized internally to 0/1.
5. MultiDatasetScheduler can build one combined three-dataset epoch plan.
6. Qwen + LoRA + unified decoder can initialize.
7. Exactly one real forward/backward/optimizer step runs for EACH dataset,
   using batch size 1 to keep the smoke lightweight.
8. Metrics run on each prediction.
9. Per-dataset checkpoint selection metric is correct.

This script does NOT save a checkpoint and does NOT consume a full epoch.

Run from PAIR repository root:
    CUDA_VISIBLE_DEVICES=2 python smoke_test_three_datasets.py

Optional:
    CUDA_VISIBLE_DEVICES=2 python smoke_test_three_datasets.py \
        --config configs/pair_train.json
"""

from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.config_loader import load_experiment_config
from datasets.multi_dataset import DatasetRegistry, MultiDatasetScheduler, collate_pair
from datasets.pair_dataset import UnifiedPAIRDataset
from loss import PAIRSemanticChangeLoss
from metrics import PAIRMetrics
from train import (
    build_model,
    build_optimizer,
    build_settings,
    forward_loss,
    selection_metric_for_spec,
    set_seed,
)


DEFAULT_DATASETS = (
    "SECOND",
    "LEVIR-CD",
    "LandsatSCD",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train.json"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def make_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model smoke.")

    return {
        "distributed": False,
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "device": torch.device("cuda:0"),
        "is_main": True,
    }


def first_batch(ds_cfg):
    dataset = UnifiedPAIRDataset(
        ds_cfg.train_manifest,
        ds_cfg.spec,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_pair,
    )

    return dataset, next(iter(loader))


def unique_values(x):
    return sorted(
        int(v)
        for v in torch.unique(x.detach().cpu()).tolist()
    )


def main():
    args = parse_args()

    if len(args.datasets) != 3:
        raise ValueError(
            "This smoke is intentionally for exactly three datasets; "
            f"got {args.datasets}"
        )

    runtime = make_runtime()

    experiment = load_experiment_config(
        args.config,
        args.datasets,
    )

    # build_settings only needs these two CLI-like fields.
    class SmokeCLI:
        output_dir = None
        resume = None

    settings = build_settings(
        experiment,
        SmokeCLI(),
    )

    set_seed(
        args.seed,
        runtime["rank"],
    )

    print("=" * 92)
    print("PAIR THREE-DATASET SMOKE")
    print("=" * 92)
    print("Config   :", args.config)
    print("Datasets :", list(experiment.selected_names))
    print("Device   :", runtime["device"])
    print()

    # ------------------------------------------------------------------
    # 1) Directory-as-schema + first-sample checks.
    # ------------------------------------------------------------------
    expected = {
        "SECOND": ("2d", "semantic_pair", "scd/F_scd"),
        "LEVIR-CD": ("2d", "binary", "change/IoU"),
        "LandsatSCD": ("2d", "semantic_pair", "scd/F_scd"),
    }

    smoke_batches = {}

    for name in experiment.selected_names:
        ds_cfg = experiment.datasets[name]
        dataset, samples = first_batch(ds_cfg)
        sample = samples[0]

        metric_key, metric_label = selection_metric_for_spec(
            ds_cfg.spec
        )

        print(f"[{name}]")
        print("  root          :", ds_cfg.root)
        print("  modalities    :", list(ds_cfg.modalities))
        print("  route         :", ds_cfg.route)
        print("  label_mode    :", ds_cfg.label_mode)
        print("  unchanged_raw :", ds_cfg.unchanged_raw_id)
        print("  classes       :", ds_cfg.class_names)
        print("  train samples :", len(dataset))
        print("  first id      :", sample["sample_id"])
        print("  change values :", unique_values(sample["target"]["change"]))
        print("  selection     :", f"{metric_label} ({metric_key})")

        if name in expected:
            want_route, want_mode, want_metric = expected[name]
            assert ds_cfg.route == want_route, (
                name, ds_cfg.route, want_route
            )
            assert ds_cfg.label_mode == want_mode, (
                name, ds_cfg.label_mode, want_mode
            )
            assert metric_key == want_metric, (
                name, metric_key, want_metric
            )

        # Canonical target presented to loss/metrics must always be binary 0/1.
        change_values = set(
            unique_values(
                sample["target"]["change"][
                    sample["target"]["change_valid"]
                ]
            )
        )
        assert change_values.issubset({0, 1}), (
            f"{name}: internal valid change target is not canonical 0/1: "
            f"{sorted(change_values)}"
        )

        smoke_batches[name] = samples
        print()

    # ------------------------------------------------------------------
    # 2) Real three-dataset scheduler construction.
    #    Registry uses the real configured per-dataset batch sizes here,
    #    but no training batches are consumed from it.
    # ------------------------------------------------------------------
    registry = DatasetRegistry(
        experiment,
        runtime,
        num_workers=0,
    )

    scheduler = MultiDatasetScheduler(
        experiment,
        registry,
        grad_accum=1,
    )

    print("Combined epoch scheduler:")
    print("  total updates :", scheduler.updates_per_epoch)
    for name, info in scheduler.summary().items():
        print(
            f"  {name:12s}: "
            f"train_batches={info['train_batches']} "
            f"updates={info['optimizer_updates']} "
            f"fraction={info['update_fraction']:.4f}"
        )
    print()

    # For the smoke itself, exercise the three branches once each.
    smoke_order = list(experiment.selected_names)
    random.Random(args.seed).shuffle(smoke_order)

    print("Smoke update order:", " -> ".join(smoke_order))
    print()

    # ------------------------------------------------------------------
    # 3) Build the actual trainable model.
    # ------------------------------------------------------------------
    print("Building Qwen/LoRA/PAIR decoder...")
    model = build_model(
        settings,
        runtime,
    )
    model.train()

    criterion = PAIRSemanticChangeLoss().to(
        runtime["device"]
    )

    optimizer, _, _ = build_optimizer(
        model,
        settings,
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {trainable / 1e6:.2f} M"
    )
    print()

    # ------------------------------------------------------------------
    # 4) One true optimizer update per dataset.
    # ------------------------------------------------------------------
    metric_results = {}

    for step, name in enumerate(
        smoke_order,
        start=1,
    ):
        ds_cfg = experiment.datasets[name]
        samples = smoke_batches[name]

        optimizer.zero_grad(
            set_to_none=True
        )

        torch.cuda.synchronize()

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            prediction, loss_output, target = forward_loss(
                model,
                criterion,
                samples,
                ds_cfg.spec,
            )

        loss = loss_output.total

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{name}: non-finite loss {float(loss.detach().cpu())}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            settings.max_grad_norm,
        )

        if not torch.isfinite(
            torch.as_tensor(grad_norm)
        ):
            raise RuntimeError(
                f"{name}: non-finite gradient norm {grad_norm}"
            )

        optimizer.step()

        evaluator = PAIRMetrics(
            ds_cfg.class_names,
            runtime["device"],
            settings.change_threshold,
            unchanged_raw_id=ds_cfg.unchanged_raw_id,
        )

        evaluator.update(
            prediction,
            target,
        )

        result = evaluator.compute()
        metric_key, metric_label = selection_metric_for_spec(
            ds_cfg.spec
        )

        metric_value = float(
            result["scalars"][metric_key]
        )

        metric_results[name] = {
            "label": metric_label,
            "key": metric_key,
            "value": metric_value,
        }

        print(
            f"STEP {step}/3 [{name}] PASS | "
            f"loss={float(loss.detach().cpu()):.4f} | "
            f"grad={float(torch.as_tensor(grad_norm).detach().cpu()):.4f} | "
            f"{metric_label}={metric_value:.4f}"
        )

        del prediction, loss_output, target, evaluator
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 92)
    print("THREE-DATASET MODEL SMOKE: PASS")
    print("=" * 92)

    print("Selection metrics:")
    for name in experiment.selected_names:
        item = metric_results[name]
        print(
            f"  {name:12s} -> "
            f"{item['label']:5s} "
            f"({item['key']})"
        )

    print()
    print(
        "This smoke performed exactly 3 optimizer updates: "
        "one SECOND, one LEVIR-CD, one LandsatSCD."
    )


if __name__ == "__main__":
    main()
