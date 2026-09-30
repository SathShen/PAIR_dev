#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR checkpoint p32 ablation.

Runs the SAME saved checkpoint twice on the SAME validation set:

  NORMAL:
      [p4, p8, p16, p32] -> CGDecoder

  ZERO-P32:
      [p4, p8, p16,   0] -> CGDecoder

Only the final 1/32 feature handed to CGDecoder is changed.
Everything upstream still runs normally.

This script requires the checkpoint-buffer fix in train.py
(load_model_state_from_checkpoint + pair_state checkpoints).

No optimizer step, no backward, no checkpoint/source modification.
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
        description="PAIR checkpoint p32 contribution ablation."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train_qwen4b.json"),
    )
    p.add_argument("--dataset", type=str, default="SECOND")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = full validation; >0 = quick partial validation.",
    )
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-autocast", action="store_true")
    p.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.005,
        help="Allowed |rerun F_scd - saved F_scd| before ZERO-P32.",
    )
    return p.parse_args()


def load_checkpoint_model_only(model: PAIRModel, path: Path):
    if not hasattr(train_mod, "load_model_state_from_checkpoint"):
        raise RuntimeError(
            "Current train.py has no load_model_state_from_checkpoint(). "
            "Use the checkpoint-buffer-fixed train.py first."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    legacy_missing_buffers = train_mod.load_model_state_from_checkpoint(
        ckpt,
        model,
    )
    if legacy_missing_buffers:
        raise RuntimeError(
            "This checkpoint is legacy and did not save BatchNorm buffers. "
            "Do not use it for a rigorous p32 ablation. Retrain/save with "
            "the fixed checkpoint format."
        )
    return ckpt


class ZeroP32:
    def __init__(self, model: PAIRModel):
        self.decoder = model.decoder
        self.original = None
        self.printed = False

    def __enter__(self):
        self.original = self.decoder.forward_2d_cg
        original = self.original
        owner = self

        def wrapped(
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
                    "Expected [p4,p8,p16,p32] for both times, got "
                    f"{len(feat_pyramid_t1)} and {len(feat_pyramid_t2)}"
                )

            p1 = list(feat_pyramid_t1)
            p2 = list(feat_pyramid_t2)

            if not owner.printed:
                print(
                    "[ZERO-P32] p32 shapes:",
                    tuple(p1[-1].shape),
                    tuple(p2[-1].shape),
                )
                owner.printed = True

            p1[-1] = torch.zeros_like(p1[-1])
            p2[-1] = torch.zeros_like(p2[-1])

            return original(
                feat_pyramid_t1=p1,
                feat_pyramid_t2=p2,
                prediction_mode=prediction_mode,
                class_names=class_names,
                qwen_backbone=qwen_backbone,
                detach_qwen_class_encoder=detach_qwen_class_encoder,
            )

        self.decoder.forward_2d_cg = MethodType(wrapped, self.decoder)
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
    bar = tqdm(
        loader,
        total=len(loader),
        desc=label,
        dynamic_ncols=True,
    )

    for samples in bar:
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            prediction, _, target = train_mod.forward_loss(
                model,
                criterion,
                samples,
                spec,
            )

        evaluator.update(prediction, target)
        count += len(samples)
        bar.set_postfix(samples=count, refresh=False)

        if max_samples > 0 and count >= max_samples:
            break

    result = evaluator.compute()
    result["seconds"] = time.time() - start
    result["samples"] = count
    return result


def get_scalar(result, key):
    return float(result["scalars"].get(key, float("nan")))


def verify_baseline(ckpt, dataset, normal, tolerance):
    selection = (ckpt.get("validation_selection") or {}).get(dataset)
    if not selection:
        print(
            "Warning: checkpoint has no validation_selection; "
            "cannot automatically verify the NORMAL baseline."
        )
        return

    key = selection.get("metric_key")
    saved = selection.get("value")
    if key != "scd/F_scd" or saved is None:
        print(
            "Warning: checkpoint selection metric is not scd/F_scd; "
            "skipping automatic baseline check."
        )
        return

    rerun = get_scalar(normal, "scd/F_scd")
    diff = abs(float(saved) - rerun)

    print()
    print("=" * 96)
    print("BASELINE REPRODUCTION CHECK")
    print("=" * 96)
    print(f"saved F_scd : {float(saved):.6f}")
    print(f"rerun F_scd : {rerun:.6f}")
    print(f"abs diff    : {diff:.6f}")
    print(f"tolerance   : {float(tolerance):.6f}")

    if diff > float(tolerance):
        raise RuntimeError(
            "NORMAL does not reproduce the saved checkpoint baseline. "
            "ZERO-P32 is aborted because the ablation would be invalid."
        )


def print_result(title, result):
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
        if key in result["scalars"]:
            print(f"{key:<24}: {float(result['scalars'][key]):.6f}")


def print_comparison(normal, zero):
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
    print("P32 ABLATION COMPARISON")
    print("=" * 96)
    print(
        f"{'metric':<24}"
        f"{'normal':>14}"
        f"{'zero-p32':>14}"
        f"{'delta(zero-normal)':>20}"
    )
    print("-" * 96)

    for key in keys:
        if key not in normal["scalars"] or key not in zero["scalars"]:
            continue
        a = float(normal["scalars"][key])
        b = float(zero["scalars"][key])
        print(f"{key:<24}{a:>14.6f}{b:>14.6f}{(b-a):>20.6f}")

    if "scd/F_scd" in normal["scalars"] and "scd/F_scd" in zero["scalars"]:
        normal_f = get_scalar(normal, "scd/F_scd")
        zero_f = get_scalar(zero, "scd/F_scd")
        drop = normal_f - zero_f
        print()
        print(f"F_scd drop after zeroing p32: {drop:.6f}")
        if abs(drop) < 0.005:
            print(
                "Diagnostic: removing p32 barely changes F_scd; "
                "the final 2D prediction does not materially depend on p32."
            )
        elif drop > 0:
            print(
                "Diagnostic: p32 materially helps this checkpoint."
            )
        else:
            print(
                "Diagnostic: zeroing p32 improves F_scd; "
                "p32 is not only unnecessary here, it may be slightly harmful."
            )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda:0")
    experiment = load_experiment_config(
        args.config,
        selected_names=[args.dataset],
    )
    spec = experiment.datasets[args.dataset].spec

    if spec.route != "2d" or spec.label_mode != "semantic_pair":
        raise ValueError(
            "This test is intended for a 2D semantic_pair SCD dataset such as SECOND."
        )

    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(experiment.training.get("num_workers", 0))
    )
    threshold = float(
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
        raise RuntimeError(f"{args.dataset} has no validation loader.")

    flags = train_mod.active_model_flags(experiment)
    model = PAIRModel.from_config(
        experiment.model,
        device,
        **flags,
    )
    ckpt = load_checkpoint_model_only(
        model,
        args.checkpoint.expanduser().resolve(),
    )
    model.to(device)
    model.eval()

    print("=" * 96)
    print("PAIR CHECKPOINT P32 ABLATION")
    print("=" * 96)
    print("config      :", args.config)
    print("dataset     :", args.dataset)
    print("checkpoint  :", args.checkpoint)
    print("epoch       :", ckpt.get("epoch", "N/A"))
    print("autocast    :", "BF16" if not args.no_autocast else "OFF")
    print(
        "samples     :",
        "FULL VAL" if args.max_samples <= 0 else f"up to {args.max_samples}",
    )
    print("NORMAL      : [p4,p8,p16,p32]")
    print("ZERO-P32    : [p4,p8,p16,0]")

    normal = evaluate(
        model,
        handle.val_loader,
        spec,
        device,
        threshold,
        not args.no_autocast,
        args.max_samples,
        "VAL NORMAL",
    )
    print_result("NORMAL", normal)

    if args.max_samples <= 0:
        verify_baseline(
            ckpt,
            args.dataset,
            normal,
            args.baseline_tolerance,
        )

    with ZeroP32(model):
        zero = evaluate(
            model,
            handle.val_loader,
            spec,
            device,
            threshold,
            not args.no_autocast,
            args.max_samples,
            "VAL ZERO-P32",
        )

    print_result("ZERO-P32", zero)
    print_comparison(normal, zero)


if __name__ == "__main__":
    main()
