#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR train-mode pyramid / reasoning gradient diagnostic.

Purpose
-------
Diagnose WHY p32 may be ignored by the 2D CGDecoder.

For real SECOND training batches, this script:
  1) runs the normal current forward in model.train()
  2) captures the exact [p4,p8,p16,p32] tensors handed to CGDecoder
  3) retains their gradients
  4) backpropagates the real current PAIR total loss
  5) reports gradient strength at each pyramid input
  6) reports parameter-gradient strength for the reasoning path and CGDecoder

No optimizer.step() is executed.
No checkpoint or source file is modified.

The most useful numbers are:
  - p32 grad RMS vs p16 / shallow mean
  - p32 |grad * activation| vs shallow scales
  - unified_attention / reasoning_injection / LLM-LoRA gradient RMS
    versus CGDecoder gradient RMS

Example
-------
CUDA_VISIBLE_DEVICES=1 python test_p32_train_grad.py \
  --config configs/pair_train_qwen4b.json \
  --dataset SECOND \
  --checkpoint outputs/.../ValBest_SECOND_....pt \
  --batches 3
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch

import train as train_mod
from datasets.config_loader import load_experiment_config
from datasets.multi_dataset import DatasetRegistry
from loss import PAIRSemanticChangeLoss
from models.pair import PAIRModel


SCALE_NAMES = ("p4", "p8", "p16", "p32")


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure train-mode gradient flow into PAIR p4/p8/p16/p32."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train_qwen4b.json"),
    )
    p.add_argument("--dataset", type=str, default="SECOND")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Recommended. If omitted, gradients are measured at initialization.",
    )
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-autocast", action="store_true")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def tensor_stats(x, g):
    xd = x.detach().float()
    out = {
        "shape": tuple(x.shape),
        "requires_grad": bool(x.requires_grad),
        "act_rms": float(torch.sqrt(torch.mean(xd * xd)).cpu()),
        "act_mean_abs": float(torch.mean(torch.abs(xd)).cpu()),
    }

    if g is None:
        out.update(
            grad_l2=0.0,
            grad_rms=0.0,
            grad_mean_abs=0.0,
            grad_x_act=0.0,
        )
        return out

    gd = g.detach().float()
    out.update(
        grad_l2=float(torch.linalg.vector_norm(gd).cpu()),
        grad_rms=float(torch.sqrt(torch.mean(gd * gd)).cpu()),
        grad_mean_abs=float(torch.mean(torch.abs(gd)).cpu()),
        grad_x_act=float(torch.mean(torch.abs(gd * xd)).cpu()),
    )
    return out


class CapturePyramidGrad:
    """Capture the exact four feature levels handed to CGDecoder."""

    def __init__(self, model: PAIRModel):
        self.decoder = model.decoder
        self.original = None
        self.t1 = None
        self.t2 = None

    def clear(self):
        self.t1 = None
        self.t2 = None

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
                    "Expected [p4,p8,p16,p32] for both times."
                )

            owner.t1 = list(feat_pyramid_t1)
            owner.t2 = list(feat_pyramid_t2)

            for x in owner.t1 + owner.t2:
                if x.requires_grad:
                    x.retain_grad()

            return original(
                feat_pyramid_t1=feat_pyramid_t1,
                feat_pyramid_t2=feat_pyramid_t2,
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


def load_checkpoint_model_only(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if hasattr(train_mod, "load_model_state_from_checkpoint"):
        legacy_missing = train_mod.load_model_state_from_checkpoint(
            ckpt,
            model,
        )
        if legacy_missing:
            print(
                "WARNING: legacy checkpoint lacks saved BN running buffers. "
                "This train-mode gradient test can still run because BatchNorm "
                "uses current batch statistics in train mode, but use a new-format "
                "checkpoint for rigorous validation ablations."
            )
        return ckpt

    raise RuntimeError(
        "Current train.py has no load_model_state_from_checkpoint(). "
        "Use the checkpoint-buffer-fixed train.py."
    )


def parameter_group(name):
    n = name.lower()

    if "pair_lora_" in n:
        if "qwen_backbone" in n and "visual" in n:
            return "qwen_vision_lora"
        if "point_encoder" in n:
            return "utonia_lora"
        return "custom_lora_other"

    if "lora_" in n:
        return "qwen_llm_lora"

    if "decoder.cg_decoder_2d" in n:
        return "cg_decoder_2d"

    if "decoder.classifier_cd" in n:
        return "change_head_2d"

    if "decoder.unified_attention" in n:
        return "unified_attention"

    if "decoder.reasoning_injection" in n:
        return "cross_attention"

    if "decoder.reasoning_projection" in n:
        return "llm_projection"

    if "decoder.task_conditioning" in n:
        return "task_conditioning"

    if "image_adapter" in n:
        return "image_dense_projection"

    if (
        "decoder.class_encoder" in n
        or "decoder.logit_scale" in n
        or "decoder.semantic_head" in n
    ):
        return "semantic_prototype"

    if "decoder" in n:
        return "decoder_other"

    return "main_other"


def grouped_parameter_grads(model):
    sq = defaultdict(float)
    numel = defaultdict(int)
    tensors = defaultdict(int)

    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach().float()
        norm = float(torch.linalg.vector_norm(g).cpu())
        group = parameter_group(name)
        sq[group] += norm * norm
        numel[group] += g.numel()
        tensors[group] += 1

    rows = []
    for group in sorted(sq):
        l2 = math.sqrt(sq[group])
        rms = l2 / math.sqrt(numel[group]) if numel[group] else 0.0
        rows.append(
            {
                "group": group,
                "l2": l2,
                "rms": rms,
                "numel": numel[group],
                "tensors": tensors[group],
            }
        )

    rows.sort(key=lambda x: x["l2"], reverse=True)
    return rows


def global_grad_norm(model):
    sq = 0.0
    active = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n = float(torch.linalg.vector_norm(g).cpu())
        sq += n * n
        active += g.numel()
    norm = math.sqrt(sq)
    rms = norm / math.sqrt(active) if active else 0.0
    return norm, rms, active


def get_loss_components(loss_output):
    items = vars(loss_output) if hasattr(loss_output, "__dict__") else {}
    result = {}
    for k, v in items.items():
        if torch.is_tensor(v) and v.numel() == 1:
            result[k] = float(v.detach().float().cpu())
    return result


def print_pyramid_table(batch_index, capture):
    if capture.t1 is None or capture.t2 is None:
        raise RuntimeError("Did not capture 2D pyramid tensors.")

    scale_means = {}

    print()
    print("PYRAMID INPUT GRADIENTS — gradient from total loss into CGDecoder inputs")
    print("-" * 118)
    print(
        f"{'scale':<7}{'time':<6}{'shape':<24}"
        f"{'act RMS':>12}{'grad L2':>14}{'grad RMS':>14}"
        f"{'mean|g|':>14}{'mean|g*x|':>16}"
    )
    print("-" * 118)

    for i, scale in enumerate(SCALE_NAMES):
        values = []
        for time_name, tensor in (("T1", capture.t1[i]), ("T2", capture.t2[i])):
            s = tensor_stats(tensor, tensor.grad)
            values.append(s)
            print(
                f"{scale:<7}{time_name:<6}{str(s['shape']):<24}"
                f"{s['act_rms']:>12.4e}"
                f"{s['grad_l2']:>14.4e}"
                f"{s['grad_rms']:>14.4e}"
                f"{s['grad_mean_abs']:>14.4e}"
                f"{s['grad_x_act']:>16.4e}"
            )
        scale_means[scale] = {
            key: statistics.mean(v[key] for v in values)
            for key in ("grad_rms", "grad_x_act", "act_rms")
        }

    shallow_grad = statistics.mean(
        scale_means[s]["grad_rms"] for s in ("p4", "p8", "p16")
    )
    shallow_gxa = statistics.mean(
        scale_means[s]["grad_x_act"] for s in ("p4", "p8", "p16")
    )

    p32_grad = scale_means["p32"]["grad_rms"]
    p32_gxa = scale_means["p32"]["grad_x_act"]

    print()
    print("P32 LOCAL-USE RATIOS")
    print("-" * 72)
    print(
        "p32 grad RMS / p16 grad RMS      :",
        f"{p32_grad / max(scale_means['p16']['grad_rms'], 1e-30):.6f}",
    )
    print(
        "p32 grad RMS / shallow mean      :",
        f"{p32_grad / max(shallow_grad, 1e-30):.6f}",
    )
    print(
        "p32 |g*x| / p16 |g*x|            :",
        f"{p32_gxa / max(scale_means['p16']['grad_x_act'], 1e-30):.6f}",
    )
    print(
        "p32 |g*x| / shallow mean         :",
        f"{p32_gxa / max(shallow_gxa, 1e-30):.6f}",
    )

    return scale_means


def print_parameter_table(rows):
    print()
    print("PARAMETER GRADIENT GROUPS")
    print("-" * 92)
    print(
        f"{'group':<28}"
        f"{'||g||':>14}"
        f"{'grad RMS':>14}"
        f"{'active numel':>18}"
        f"{'tensors':>10}"
    )
    print("-" * 92)
    for row in rows:
        print(
            f"{row['group']:<28}"
            f"{row['l2']:>14.6f}"
            f"{row['rms']:>14.4e}"
            f"{row['numel']:>18,d}"
            f"{row['tensors']:>10,d}"
        )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")
    experiment = load_experiment_config(
        args.config,
        selected_names=[args.dataset],
    )
    spec = experiment.datasets[args.dataset].spec

    if spec.route != "2d" or spec.label_mode != "semantic_pair":
        raise ValueError(
            "This diagnostic is intended for 2D semantic_pair SCD, e.g. SECOND."
        )

    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(experiment.training.get("num_workers", 0))
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

    flags = train_mod.active_model_flags(experiment)
    model = PAIRModel.from_config(
        experiment.model,
        device,
        **flags,
    )

    ckpt = None
    if args.checkpoint is not None:
        ckpt = load_checkpoint_model_only(
            model,
            args.checkpoint.expanduser().resolve(),
        )

    model.to(device)
    model.train()

    criterion = PAIRSemanticChangeLoss().to(device)

    configured_clip = float(
        experiment.optimizer.get("max_grad_norm", 0.0)
    )

    print("=" * 118)
    print("PAIR TRAIN-MODE P32 / PYRAMID GRADIENT DIAGNOSTIC")
    print("=" * 118)
    print("config      :", args.config)
    print("dataset     :", args.dataset)
    print("checkpoint  :", args.checkpoint if args.checkpoint else "<initialization>")
    if ckpt is not None:
        print("epoch       :", ckpt.get("epoch", "N/A"))
    print("mode        : model.train(), real train batches, NO optimizer.step()")
    print("autocast    :", "BF16" if not args.no_autocast else "OFF")
    print("batches     :", args.batches)
    print("max_grad_norm:", configured_clip)
    print()

    aggregates = defaultdict(list)

    train_iter = iter(handle.train_loader)

    with CapturePyramidGrad(model) as capture:
        for batch_idx in range(1, args.batches + 1):
            try:
                samples = next(train_iter)
            except StopIteration:
                train_iter = iter(handle.train_loader)
                samples = next(train_iter)

            model.zero_grad(set_to_none=True)
            capture.clear()

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=not args.no_autocast,
            ):
                prediction, loss_output, target = train_mod.forward_loss(
                    model,
                    criterion,
                    samples,
                    spec,
                )
                total = loss_output.total

            total.backward()

            gnorm, grms, active = global_grad_norm(model)

            print()
            print("=" * 118)
            print(f"BATCH {batch_idx}")
            print("=" * 118)
            print("samples              :", len(samples))
            print("total loss           :", f"{float(total.detach().float().cpu()):.8f}")
            print("raw global grad norm :", f"{gnorm:.8f}")
            print("global grad RMS      :", f"{grms:.8e}")
            print("active grad numel    :", f"{active:,}")
            if configured_clip > 0:
                print(
                    "would clip?           :",
                    "YES" if gnorm > configured_clip else "NO",
                )

            components = get_loss_components(loss_output)
            if components:
                print()
                print("LOSS COMPONENTS")
                print("-" * 72)
                for key in sorted(components):
                    print(f"{key:<36}: {components[key]:.8f}")

            scale_means = print_pyramid_table(batch_idx, capture)
            rows = grouped_parameter_grads(model)
            print_parameter_table(rows)

            for scale in SCALE_NAMES:
                aggregates[f"{scale}/grad_rms"].append(
                    scale_means[scale]["grad_rms"]
                )
                aggregates[f"{scale}/grad_x_act"].append(
                    scale_means[scale]["grad_x_act"]
                )

            row_map = {r["group"]: r for r in rows}
            for group in (
                "qwen_llm_lora",
                "qwen_vision_lora",
                "image_dense_projection",
                "llm_projection",
                "cross_attention",
                "unified_attention",
                "task_conditioning",
                "cg_decoder_2d",
                "change_head_2d",
                "semantic_prototype",
            ):
                if group in row_map:
                    aggregates[f"group/{group}/l2"].append(
                        row_map[group]["l2"]
                    )
                    aggregates[f"group/{group}/rms"].append(
                        row_map[group]["rms"]
                    )

            # Release this graph before the next batch.
            model.zero_grad(set_to_none=True)
            capture.clear()
            del prediction, loss_output, target, total
            gc.collect()
            torch.cuda.empty_cache()

    print()
    print("=" * 118)
    print("AGGREGATE")
    print("=" * 118)

    for scale in SCALE_NAMES:
        g = statistics.mean(aggregates[f"{scale}/grad_rms"])
        gx = statistics.mean(aggregates[f"{scale}/grad_x_act"])
        print(
            f"{scale:<6} mean grad RMS={g:.6e}   "
            f"mean |g*x|={gx:.6e}"
        )

    p32_grad = statistics.mean(aggregates["p32/grad_rms"])
    p16_grad = statistics.mean(aggregates["p16/grad_rms"])
    shallow_grad = statistics.mean(
        statistics.mean(aggregates[f"{s}/grad_rms"])
        for s in ("p4", "p8", "p16")
    )

    p32_gxa = statistics.mean(aggregates["p32/grad_x_act"])
    p16_gxa = statistics.mean(aggregates["p16/grad_x_act"])
    shallow_gxa = statistics.mean(
        statistics.mean(aggregates[f"{s}/grad_x_act"])
        for s in ("p4", "p8", "p16")
    )

    print()
    print("KEY P32 RATIOS")
    print("-" * 72)
    print(
        "p32 grad RMS / p16      :",
        f"{p32_grad / max(p16_grad, 1e-30):.6f}",
    )
    print(
        "p32 grad RMS / shallow  :",
        f"{p32_grad / max(shallow_grad, 1e-30):.6f}",
    )
    print(
        "p32 |g*x| / p16         :",
        f"{p32_gxa / max(p16_gxa, 1e-30):.6f}",
    )
    print(
        "p32 |g*x| / shallow     :",
        f"{p32_gxa / max(shallow_gxa, 1e-30):.6f}",
    )

    print()
    print("MEAN PARAMETER GRADIENT GROUPS")
    print("-" * 92)
    print(f"{'group':<28}{'mean ||g||':>16}{'mean grad RMS':>18}")
    print("-" * 92)

    groups = sorted({
        k.split("/")[1]
        for k in aggregates
        if k.startswith("group/") and k.endswith("/l2")
    })
    for group in groups:
        l2s = aggregates[f"group/{group}/l2"]
        rmss = aggregates[f"group/{group}/rms"]
        print(
            f"{group:<28}"
            f"{statistics.mean(l2s):>16.6f}"
            f"{statistics.mean(rmss):>18.6e}"
        )

    print()
    print("HOW TO READ")
    print("-" * 118)
    print(
        "1) If ZERO-P32 barely changes checkpoint F_scd AND p32 grad RMS / shallow is tiny, "
        "the CGDecoder has a real shallow-feature shortcut."
    )
    print(
        "2) If ZERO-P32 barely changes F_scd but p32 input gradient is not tiny, "
        "the decoder is still sending learning signal to p32; inspect whether the upstream "
        "LLM/Cross/Unified modules fail to turn that signal into useful features."
    )
    print(
        "3) Compare unified_attention / cross_attention / qwen_llm_lora RMS against "
        "cg_decoder_2d. This separates local CGDecoder ignoring p32 from upstream gradient starvation."
    )


if __name__ == "__main__":
    main()
