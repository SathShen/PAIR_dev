#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR p32 contribution ablation.

Goal
----
Measure whether the 2D reasoning branch carried by the deepest p32 feature
actually contributes to SECOND validation performance.

The script evaluates the SAME checkpoint twice on the SAME validation set:

    1) NORMAL:
       p4 / p8 / p16 / p32 -> CGDecoder

    2) ZERO-P32:
       p4 / p8 / p16 / 0   -> CGDecoder

Only the final p32 inputs presented to CGDecoder are zeroed.
Qwen, Cross-Attn, Unified Decoder, TASK conditioning, p4/p8/p16, prototype
classification, change head, metrics, and checkpoint weights are unchanged.

This is an inference-only diagnostic:
    - no optimizer
    - no backward
    - no source modification
    - no checkpoint modification

Recommended:
    Use the ValBest SECOND checkpoint whose ~0.64 F_scd plateau you want to
    diagnose, and run the full validation set first.

Example
-------
CUDA_VISIBLE_DEVICES=1 python test_p32_ablation.py \
    --config configs/pair_train_qwen4b.json \
    --dataset SECOND \
    --checkpoint outputs/.../ValBest_SECOND_....pt

Quick smoke test:
CUDA_VISIBLE_DEVICES=1 python test_p32_ablation.py \
    --config configs/pair_train_qwen4b.json \
    --dataset SECOND \
    --checkpoint outputs/.../ValBest_SECOND_....pt \
    --max-samples 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import MethodType

import torch
from tqdm.auto import tqdm

import train as train_mod
from datasets.config_loader import load_experiment_config
from datasets.multi_dataset import DatasetRegistry
from loss import PAIRSemanticChangeLoss
from metrics import PAIRMetrics
from models.pair import PAIRModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare normal SECOND validation against p32=0 ablation."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train_qwen4b.json"),
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="SECOND",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PAIR checkpoint to evaluate.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = full validation set; otherwise stop after at least this many samples.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override config training.num_workers.",
    )
    p.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable BF16 autocast.",
    )
    p.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.005,
        help=(
            "Maximum allowed absolute difference between the checkpoint's "
            "saved validation F_scd and the NORMAL rerun before ZERO-P32 is allowed."
        ),
    )
    return p.parse_args()


def model_only_load_checkpoint(model: PAIRModel, path: Path):
    """Load model state through the project's current checkpoint loader."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    legacy_missing_buffers = train_mod.load_model_state_from_checkpoint(
        ckpt,
        model,
    )
    if legacy_missing_buffers:
        raise RuntimeError(
            "This is a legacy checkpoint without saved BatchNorm buffers. "
            "Run recalibrate_old_checkpoint_bn.py first, then use the recovered "
            "*_BNRecal.pt checkpoint for p32 ablation."
        )
    return ckpt



class ZeroP32:
    """
    Temporarily replace only the deepest 1/32 features passed into CGDecoder
    with zeros.

    The original decoder method is restored when leaving the context.
    """

    def __init__(self, model: PAIRModel):
        self.model = model
        self.decoder = model.decoder
        self.original = None
        self.printed_shape = False

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError("ZeroP32 context entered twice")

        self.original = self.decoder.forward_2d_cg
        original_bound_method = self.original
        owner = self

        def zero_p32_forward(
            decoder_self,
            *,
            feat_pyramid_t1,
            feat_pyramid_t2,
            prediction_mode,
            class_names=None,
            qwen_backbone=None,
            detach_qwen_class_encoder=True,
        ):
            if len(feat_pyramid_t1) != 4 or len(feat_pyramid_t2) != 4:
                raise RuntimeError(
                    "p32 ablation expects 4-level 2D pyramids "
                    "[p4,p8,p16,p32], got "
                    f"{len(feat_pyramid_t1)} and {len(feat_pyramid_t2)}"
                )

            p1 = list(feat_pyramid_t1)
            p2 = list(feat_pyramid_t2)

            if not owner.printed_shape:
                print(
                    "[ZERO-P32] deepest feature shapes:",
                    f"T1={tuple(p1[-1].shape)}",
                    f"T2={tuple(p2[-1].shape)}",
                )
                owner.printed_shape = True

            # The ONLY ablation in this test.
            p1[-1] = torch.zeros_like(p1[-1])
            p2[-1] = torch.zeros_like(p2[-1])

            return original_bound_method(
                feat_pyramid_t1=p1,
                feat_pyramid_t2=p2,
                prediction_mode=prediction_mode,
                class_names=class_names,
                qwen_backbone=qwen_backbone,
                detach_qwen_class_encoder=detach_qwen_class_encoder,
            )

        self.decoder.forward_2d_cg = MethodType(
            zero_p32_forward,
            self.decoder,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.decoder.forward_2d_cg = self.original
        self.original = None
        return False


@torch.no_grad()
def evaluate(
    model,
    loader,
    spec,
    device,
    change_threshold,
    use_autocast,
    max_samples,
    label,
):
    model.eval()

    criterion = PAIRSemanticChangeLoss().to(device)
    evaluator = PAIRMetrics(
        spec.class_names,
        device,
        change_threshold,
    )

    count = 0
    start = time.time()

    progress = tqdm(
        loader,
        total=len(loader),
        desc=label,
        dynamic_ncols=True,
        leave=True,
    )

    for samples in progress:
        if use_autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, _, target = train_mod.forward_loss(
                    model,
                    criterion,
                    samples,
                    spec,
                )
        else:
            prediction, _, target = train_mod.forward_loss(
                model,
                criterion,
                samples,
                spec,
            )

        evaluator.update(prediction, target)
        count += len(samples)
        progress.set_postfix(samples=count, refresh=False)

        if max_samples > 0 and count >= max_samples:
            break

    result = evaluator.compute()
    result["seconds"] = time.time() - start
    result["samples"] = count
    return result


def scalar(result, key):
    return float(result["scalars"].get(key, float("nan")))


def verify_normal_matches_checkpoint(ckpt, dataset_name, normal, tolerance):
    selection = (ckpt.get("validation_selection") or {}).get(dataset_name)
    if not selection:
        print(
            "Warning: checkpoint has no saved validation_selection entry; "
            "cannot verify the NORMAL baseline automatically."
        )
        return

    metric_key = selection.get("metric_key")
    saved_value = selection.get("value")
    if metric_key != "scd/F_scd" or saved_value is None:
        print(
            "Warning: checkpoint selection metric is not scd/F_scd; "
            "cannot perform the expected baseline check."
        )
        return

    rerun_value = scalar(normal, "scd/F_scd")
    diff = abs(float(saved_value) - rerun_value)
    print()
    print("BASELINE REPRODUCTION CHECK")
    print("saved F_scd :", f"{float(saved_value):.6f}")
    print("rerun F_scd :", f"{rerun_value:.6f}")
    print("abs diff     :", f"{diff:.6f}")
    print("tolerance    :", f"{float(tolerance):.6f}")

    if diff > float(tolerance):
        raise RuntimeError(
            "NORMAL validation does not reproduce the checkpoint baseline. "
            "ZERO-P32 will NOT be run because the ablation would be invalid."
        )


def print_result(title, result):
    s = result["scalars"]
    keys = (
        "scd/F_scd",
        "scd/SeK",
        "scd/mIoU",
        "scd/IoU_c",
        "change/F1",
        "change/IoU",
        "semantic/mIoU",
        "semantic/mF1",
    )

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print(f"samples : {result['samples']}")
    print(f"seconds : {result['seconds']:.2f}")
    for key in keys:
        if key in s:
            print(f"{key:<24}: {float(s[key]):.6f}")


def print_comparison(normal, zero):
    metrics = (
        "scd/F_scd",
        "scd/SeK",
        "scd/mIoU",
        "scd/IoU_c",
        "change/F1",
        "change/IoU",
        "semantic/mIoU",
        "semantic/mF1",
    )

    print()
    print("=" * 96)
    print("P32 ABLATION COMPARISON")
    print("=" * 96)
    print(
        f"{'metric':<24}"
        f"{'normal':>14}"
        f"{'zero-p32':>14}"
        f"{'delta(zero-normal)':>20}"
    )
    print("-" * 96)

    for key in metrics:
        if key not in normal["scalars"] or key not in zero["scalars"]:
            continue
        a = float(normal["scalars"][key])
        b = float(zero["scalars"][key])
        print(f"{key:<24}{a:>14.6f}{b:>14.6f}{(b-a):>20.6f}")

    if (
        "scd/F_scd" in normal["scalars"]
        and "scd/F_scd" in zero["scalars"]
    ):
        normal_f = scalar(normal, "scd/F_scd")
        zero_f = scalar(zero, "scd/F_scd")
        drop = normal_f - zero_f

        print()
        print(f"F_scd absolute drop after zeroing p32: {drop:.6f}")

        # These are only rough diagnostic bands, not scientific thresholds.
        if drop < 0.005:
            print(
                "Interpretation: p32 contributes very little on this checkpoint; "
                "CGDecoder is likely relying mainly on p4/p8/p16."
            )
        elif drop < 0.02:
            print(
                "Interpretation: p32 contributes, but the contribution is modest."
            )
        else:
            print(
                "Interpretation: p32 is materially important; the reasoning branch "
                "is not being ignored by CGDecoder."
            )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic expects CUDA, matching PAIR validation.")

    device = torch.device("cuda:0")

    experiment = load_experiment_config(
        args.config,
        selected_names=[args.dataset],
    )
    cfg = experiment.datasets[args.dataset]
    spec = cfg.spec

    if spec.route != "2d":
        raise ValueError(
            f"This p32 test is for the 2D route, got route={spec.route!r}"
        )
    if spec.label_mode != "semantic_pair":
        raise ValueError(
            "This diagnostic is intended for 2D SCD semantic_pair datasets "
            f"(e.g. SECOND), got label_mode={spec.label_mode!r}"
        )

    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(experiment.training.get("num_workers", 0))
    )
    change_threshold = float(
        experiment.validation.get("change_threshold", 0.5)
    )

    runtime = {
        "device": device,
        "distributed": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "is_main": True,
    }

    registry = DatasetRegistry(
        experiment,
        runtime,
        num_workers=num_workers,
    )
    handle = registry.handles[args.dataset]

    if handle.val_loader is None:
        raise RuntimeError(
            f"Dataset {args.dataset} has no validation manifest."
        )

    flags = train_mod.active_model_flags(experiment)
    model = PAIRModel.from_config(
        experiment.model,
        device,
        **flags,
    )

    ckpt = model_only_load_checkpoint(
        model,
        args.checkpoint,
    )
    model.to(device)
    model.eval()

    print("=" * 96)
    print("PAIR P32 CONTRIBUTION ABLATION")
    print("=" * 96)
    print("config      :", args.config)
    print("dataset     :", args.dataset)
    print("checkpoint  :", args.checkpoint)
    print("checkpoint epoch:", ckpt.get("epoch", "N/A"))
    print("device      :", device)
    print("autocast    :", "BF16" if not args.no_autocast else "OFF")
    print(
        "samples     :",
        "FULL VAL" if args.max_samples <= 0 else f"up to {args.max_samples}",
    )
    print()
    print("NORMAL   : [p4, p8, p16, p32] -> CGDecoder")
    print("ZERO-P32 : [p4, p8, p16,   0] -> CGDecoder")
    print("Only p32 at the CGDecoder input is changed.")

    normal = evaluate(
        model=model,
        loader=handle.val_loader,
        spec=spec,
        device=device,
        change_threshold=change_threshold,
        use_autocast=not args.no_autocast,
        max_samples=args.max_samples,
        label="VAL NORMAL",
    )

    # Never spend another full validation pass on an invalid ablation.
    # Only compare against saved full-validation metadata when this run also
    # uses the full validation set.
    if args.max_samples <= 0:
        verify_normal_matches_checkpoint(
            ckpt,
            args.dataset,
            normal,
            args.baseline_tolerance,
        )

    with ZeroP32(model):
        zero = evaluate(
            model=model,
            loader=handle.val_loader,
            spec=spec,
            device=device,
            change_threshold=change_threshold,
            use_autocast=not args.no_autocast,
            max_samples=args.max_samples,
            label="VAL ZERO-P32",
        )

    print_result("NORMAL", normal)
    print_result("ZERO-P32", zero)
    print_comparison(normal, zero)


if __name__ == "__main__":
    main()
