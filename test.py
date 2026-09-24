#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR gradient-scale diagnostic.

Purpose
-------
Measure the *actual* gradient scale produced by the current PAIR training code
without changing train.py / loss.py / model code and without performing an
optimizer step.

The script imports the project's current train.py at runtime, so it follows the
same:
    - config loader
    - DatasetRegistry
    - PAIRModel.from_config(...)
    - forward_loss(...)
    - PAIRSemanticChangeLoss
    - BF16 autocast

For every sampled training batch it reports:
    1. total loss and raw global parameter-gradient norm
    2. whether the configured max_grad_norm would clip the gradient
    3. exact d(total)/d(component) coefficient for each exposed loss component
    4. each component's actual weighted parameter-gradient norm
    5. cosine(component gradient, total gradient)
    6. gradient norms for important parameter groups
    7. aggregate statistics across several batches
    8. for 2D semantic_pair SCD, a SAME-FORWARD legacy control using the old
       full-valid semantic CE/Lovasz mask, so changed-only vs old supervision
       can be compared without confounding model initialization or batch order

Important
---------
No optimizer.step() is executed.
No checkpoint or source file is modified.

Recommended:
    run this on the checkpoint whose F_scd you care about, not only at init.

Examples
--------
# Current random/init state:
CUDA_VISIBLE_DEVICES=1 python test_grad_scale.py \
    --config configs/pair_train.json \
    --dataset SECOND \
    --batches 3

# A trained checkpoint:
CUDA_VISIBLE_DEVICES=1 python test_grad_scale.py \
    --config configs/pair_train.json \
    --dataset SECOND \
    --checkpoint outputs/.../ValBest_SECOND_....pt \
    --batches 3

# More batches, but this is slower because each loss component gets its own
# autograd pass through the same forward graph:
CUDA_VISIBLE_DEVICES=1 python test_grad_scale.py \
    --config configs/pair_train.json \
    --dataset SECOND \
    --checkpoint outputs/.../ValBest_SECOND_....pt \
    --batches 8
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

import train as train_mod
from datasets.config_loader import load_experiment_config
from datasets.multi_dataset import DatasetRegistry, MultiDatasetScheduler
from loss import PAIRSemanticChangeLoss
from models.pair import PAIRModel


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Measure PAIR loss/gradient scales on real training batches."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train.json"),
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="SECOND",
        help="One dataset name from the config. SECOND is recommended first.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional PAIR checkpoint. If omitted, diagnose initialization.",
    )
    p.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Number of training batches to diagnose.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed for batch order.",
    )
    p.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable BF16 autocast. Default matches training and uses BF16 autocast.",
    )
    p.add_argument(
        "--top-groups",
        type=int,
        default=12,
        help="How many detailed parameter groups to print.",
    )
    p.add_argument(
        "--no-legacy-comparison",
        action="store_true",
        help=(
            "Skip the same-forward control that recomputes the old full-valid "
            "semantic CE/Lovasz loss on 2D semantic_pair batches."
        ),
    )
    return p.parse_args()


# =============================================================================
# Small helpers
# =============================================================================

def fmt(x: Optional[float], digits=6):
    if x is None:
        return "N/A"
    if not math.isfinite(float(x)):
        return str(x)
    return f"{float(x):.{digits}g}"


def scalar(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def is_scalar_tensor(x) -> bool:
    return torch.is_tensor(x) and x.numel() == 1


def parameter_group(name: str) -> str:
    """
    Human-readable grouping for the current PAIR model.

    This is diagnostic-only. Unknown trainable parameters are kept as
    'main_other' rather than silently dropped.
    """
    n = name.lower()

    if "lora_" in n:
        return "qwen_lora"

    if "cg_decoder_2d" in n:
        return "2d_cg_decoder"

    if "classifier_cd" in n or "change_head" in n:
        return "2d_change_head"

    if (
        "class_encoder" in n
        or "prototype" in n
        or "logit_scale" in n
        or "semantic_head" in n
    ):
        return "semantic_prototype_head"

    if (
        "vision_intermediate" in n
        or "pyramid" in n
        or "scale_adapter" in n
        or "feature_adapter" in n
        or "vision_adapter" in n
    ):
        return "2d_pyramid_adapter"

    if "point_adapter" in n:
        return "point_adapter"

    if "event_head" in n:
        return "3d_event_head"

    if "decoder" in n:
        return "decoder_other"

    return "main_other"


def trainable_parameters(model):
    named = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    if not named:
        raise RuntimeError("No trainable parameters found.")
    return named


def grad_stats(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grads: Sequence[Optional[torch.Tensor]],
):
    """
    Return global and grouped L2/RMS gradient statistics.
    """
    global_sq = 0.0
    global_active_numel = 0
    group_sq = defaultdict(float)
    group_numel = defaultdict(int)
    group_tensors = defaultdict(int)
    max_abs = 0.0

    for (name, p), g in zip(named_params, grads):
        if g is None:
            continue

        gd = g.detach()
        norm = float(torch.linalg.vector_norm(gd.float()).cpu())
        sq = norm * norm
        global_sq += sq

        numel = gd.numel()
        global_active_numel += numel

        group = parameter_group(name)
        group_sq[group] += sq
        group_numel[group] += numel
        group_tensors[group] += 1

        if numel:
            this_max = float(gd.abs().max().float().cpu())
            max_abs = max(max_abs, this_max)

    global_norm = math.sqrt(max(global_sq, 0.0))
    global_rms = (
        global_norm / math.sqrt(global_active_numel)
        if global_active_numel > 0
        else 0.0
    )

    groups = {}
    for group in sorted(group_sq):
        norm = math.sqrt(max(group_sq[group], 0.0))
        numel = group_numel[group]
        groups[group] = {
            "norm": norm,
            "rms": norm / math.sqrt(numel) if numel else 0.0,
            "active_numel": numel,
            "tensors": group_tensors[group],
        }

    return {
        "norm": global_norm,
        "rms": global_rms,
        "active_numel": global_active_numel,
        "max_abs": max_abs,
        "groups": groups,
    }


def grad_dot(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
) -> float:
    total = 0.0
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        total += float(
            torch.sum(ga.detach().float() * gb.detach().float()).cpu()
        )
    return total


def grad_cosine(
    grads_a,
    grads_b,
    norm_a: float,
    norm_b: float,
):
    if norm_a <= 0.0 or norm_b <= 0.0:
        return float("nan")
    return grad_dot(grads_a, grads_b) / (norm_a * norm_b)


def collect_exposed_components(loss_output):
    """
    Use loss_output.as_dict(), but do not assume any hard-coded weighting.

    The exact coefficient used by total loss will be recovered by:
        d(total) / d(component)
    """
    if hasattr(loss_output, "as_dict"):
        items = loss_output.as_dict()
    else:
        raise TypeError(
            f"Loss output {type(loss_output).__name__} has no as_dict()."
        )

    components = {}
    for key, value in items.items():
        if key == "loss":
            continue
        if is_scalar_tensor(value) and value.requires_grad:
            components[key] = value

    return components


def top_prefix_gradients(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grads: Sequence[Optional[torch.Tensor]],
    depth: int = 2,
):
    """
    Automatic name-prefix view in case a new module is not covered by the
    semantic grouping above.
    """
    sq = defaultdict(float)
    n = defaultdict(int)

    for (name, _), g in zip(named_params, grads):
        if g is None:
            continue
        parts = name.split(".")
        prefix = ".".join(parts[: min(depth, len(parts))])
        gn = float(torch.linalg.vector_norm(g.detach().float()).cpu())
        sq[prefix] += gn * gn
        n[prefix] += g.numel()

    result = []
    for prefix, value in sq.items():
        norm = math.sqrt(value)
        result.append(
            (
                prefix,
                norm,
                norm / math.sqrt(n[prefix]) if n[prefix] else 0.0,
                n[prefix],
            )
        )
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# =============================================================================
# Checkpoint
# =============================================================================

def load_checkpoint_if_requested(
    args,
    model,
    experiment,
    settings,
    registry,
):
    if args.checkpoint is None:
        print("Checkpoint: <none>  [diagnosing initialization]")
        return

    path = args.checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    # Use the project's current loader rather than duplicating checkpoint format.
    dataset_scheduler = MultiDatasetScheduler(
        experiment,
        registry,
        settings.grad_accum,
    )
    total_updates = (
        dataset_scheduler.updates_per_epoch * settings.epochs
    )
    optimizer, _, _ = train_mod.build_optimizer(model, settings)
    scheduler = train_mod.build_scheduler(
        optimizer,
        total_updates,
        settings.warmup_ratio,
        settings.scheduler,
    )

    loaded = train_mod.load_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
    )

    epoch = loaded[0] if len(loaded) > 0 else "?"
    optimizer_step = loaded[2] if len(loaded) > 2 else "?"
    print(f"Checkpoint: {path}")
    print(f"  saved epoch={epoch} optimizer_step={optimizer_step}")

    # We never step the optimizer in this script.
    del optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# One batch
# =============================================================================

def diagnose_batch(
    model,
    criterion,
    samples,
    spec,
    named_params,
    max_grad_norm: float,
    use_autocast: bool,
    batch_index: int,
    compare_legacy_full_semantic: bool,
):
    model.zero_grad(set_to_none=True)

    autocast_ctx = torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=use_autocast,
    )

    with autocast_ctx:
        prediction, loss_output, target = train_mod.forward_loss(
            model,
            criterion,
            samples,
            spec,
        )

    total = loss_output.total
    if not is_scalar_tensor(total):
        raise RuntimeError(
            f"Expected scalar total loss, got shape {tuple(total.shape)}"
        )

    params = [p for _, p in named_params]

    # First: exact total parameter gradient.
    total_grads = torch.autograd.grad(
        total,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total_stats = grad_stats(named_params, total_grads)

    components = collect_exposed_components(loss_output)

    rows = []
    component_group_rows = {}

    # Recover the exact local coefficient that the current loss uses for each
    # exposed component. This avoids assuming CE/BCE/Dice weights in this test.
    active_components = list(components.items())

    for comp_i, (name, comp) in enumerate(active_components):
        coeff_tuple = torch.autograd.grad(
            total,
            comp,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        coeff_tensor = coeff_tuple[0]

        if coeff_tensor is None:
            coeff = 0.0
            weighted = comp * 0.0
        else:
            coeff = scalar(coeff_tensor)
            # Detach coefficient: we want the actual first-order weighted
            # contribution, not higher-order derivatives of the coefficient.
            weighted = comp * coeff_tensor.detach()

        comp_grads = torch.autograd.grad(
            weighted,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        stats = grad_stats(named_params, comp_grads)
        cosine = grad_cosine(
            comp_grads,
            total_grads,
            stats["norm"],
            total_stats["norm"],
        )

        rows.append(
            {
                "name": name,
                "scalar": scalar(comp),
                "coeff": coeff,
                "weighted_scalar": scalar(weighted),
                "grad_norm": stats["norm"],
                "grad_rms": stats["rms"],
                "cos_total": cosine,
            }
        )
        component_group_rows[name] = stats["groups"]

        del comp_grads

    # Explicitly release graph references after all component probes.
    loss_values = (
        loss_output.as_dict()
        if hasattr(loss_output, "as_dict")
        else {"loss": total}
    )

    raw_norm = total_stats["norm"]
    if max_grad_norm > 0 and raw_norm > max_grad_norm:
        clip_factor = max_grad_norm / (raw_norm + 1e-12)
    else:
        clip_factor = 1.0

    change = target.get("change")
    changed_ratio = None
    semantic_mask_stats = None
    if torch.is_tensor(change):
        ch = change.reshape(-1)
        change_valid = target.get("change_valid")
        if torch.is_tensor(change_valid):
            change_valid = change_valid.reshape(-1).bool()
        else:
            change_valid = torch.ones_like(ch, dtype=torch.bool)

        if change_valid.any():
            changed_ratio = float(
                (ch[change_valid] != 0).float().mean().cpu()
            )

        semantic_mask_stats = {}
        for time_id in (1, 2):
            sem = target.get(f"semantic_t{time_id}")
            if not torch.is_tensor(sem):
                continue
            sem_valid = target.get(f"semantic_valid_t{time_id}")
            if torch.is_tensor(sem_valid):
                sem_valid = sem_valid.reshape(-1).bool()
            else:
                sem_valid = torch.ones(sem.numel(), dtype=torch.bool, device=sem.device)
            local_change_valid = change_valid.to(sem_valid.device)
            local_change = ch.to(sem_valid.device)
            changed_sem = sem_valid & local_change_valid & (local_change == 1)
            semantic_mask_stats[time_id] = {
                "full_valid": int(sem_valid.sum().item()),
                "changed_valid": int(changed_sem.sum().item()),
                "fraction_kept": (
                    float(changed_sem.sum().item()) / float(sem_valid.sum().item())
                    if sem_valid.any() else float("nan")
                ),
            }

    # Same-forward control: recompute ONLY the loss graph with the old semantic
    # mask. No second model forward, no optimizer step, no changed batch order.
    legacy_control = None
    if (
        compare_legacy_full_semantic
        and spec.route == "2d"
        and spec.label_mode == "semantic_pair"
    ):
        legacy_loss_output = criterion(
            prediction=prediction,
            target=target,
            class_names=spec.class_names,
            semantic_changed_only=False,
        )
        legacy_total = legacy_loss_output.total
        legacy_total_grads = torch.autograd.grad(
            legacy_total,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        legacy_total_stats = grad_stats(named_params, legacy_total_grads)
        total_cosine = grad_cosine(
            total_grads,
            legacy_total_grads,
            total_stats["norm"],
            legacy_total_stats["norm"],
        )

        legacy_semantic = legacy_loss_output.semantic
        coeff_tuple = torch.autograd.grad(
            legacy_total,
            legacy_semantic,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        legacy_semantic_coeff = (
            0.0 if coeff_tuple[0] is None else scalar(coeff_tuple[0])
        )
        legacy_semantic_grads = torch.autograd.grad(
            legacy_semantic * legacy_semantic_coeff,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        legacy_semantic_stats = grad_stats(
            named_params, legacy_semantic_grads
        )

        legacy_control = {
            "total_loss": scalar(legacy_total),
            "total_grad_norm": legacy_total_stats["norm"],
            "total_cosine_to_changed_only": total_cosine,
            "semantic_loss": scalar(legacy_semantic),
            "semantic_coeff": legacy_semantic_coeff,
            "semantic_grad_norm": legacy_semantic_stats["norm"],
            "groups": legacy_total_stats["groups"],
        }

        del legacy_semantic_grads
        del legacy_total_grads
        del legacy_loss_output
        del legacy_total

    print()
    print("=" * 116)
    print(f"BATCH {batch_index}")
    print("=" * 116)
    print(f"samples                  : {len(samples)}")
    if changed_ratio is not None:
        print(f"changed-pixel ratio      : {changed_ratio:.6f}")
    if semantic_mask_stats:
        for time_id in sorted(semantic_mask_stats):
            s = semantic_mask_stats[time_id]
            print(
                f"semantic T{time_id} supervision  : "
                f"full={s['full_valid']:,} changed-only={s['changed_valid']:,} "
                f"kept={s['fraction_kept']:.4f}"
            )
    print(f"total loss               : {scalar(total):.8f}")
    print(f"raw global grad norm     : {raw_norm:.8f}")
    print(f"global grad RMS          : {total_stats['rms']:.8e}")
    print(f"global max |grad|        : {total_stats['max_abs']:.8e}")
    print(f"active grad parameters   : {total_stats['active_numel']:,}")
    print(f"configured max_grad_norm : {max_grad_norm:.8f}")
    print(f"would clip?              : {'YES' if clip_factor < 1.0 else 'NO'}")
    print(f"global clip factor       : {clip_factor:.8f}")
    print(
        f"implied post-clip norm   : "
        f"{raw_norm * clip_factor:.8f}"
    )

    print()
    print("LOSS COMPONENTS — exact coefficient recovered from d(total)/d(component)")
    print("-" * 116)
    print(
        f"{'component':<28}"
        f"{'value':>13}"
        f"{'coef':>11}"
        f"{'coef*value':>15}"
        f"{'||g||':>13}"
        f"{'g RMS':>13}"
        f"{'cos(g,total)':>15}"
    )
    print("-" * 116)
    for row in rows:
        print(
            f"{row['name']:<28}"
            f"{row['scalar']:>13.6f}"
            f"{row['coeff']:>11.5f}"
            f"{row['weighted_scalar']:>15.6f}"
            f"{row['grad_norm']:>13.6f}"
            f"{row['grad_rms']:>13.3e}"
            f"{row['cos_total']:>15.6f}"
        )

    if legacy_control is not None:
        print()
        print("SAME-FORWARD CONTROL — old FULL-VALID semantic supervision")
        print("-" * 116)
        print(
            f"changed-only total loss / grad : "
            f"{scalar(total):.6f} / {total_stats['norm']:.6f}"
        )
        print(
            f"legacy full-sem total / grad  : "
            f"{legacy_control['total_loss']:.6f} / "
            f"{legacy_control['total_grad_norm']:.6f}"
        )
        print(
            f"cos(total_changed, total_old) : "
            f"{legacy_control['total_cosine_to_changed_only']:.6f}"
        )
        print(
            f"legacy semantic loss / grad   : "
            f"{legacy_control['semantic_loss']:.6f} / "
            f"{legacy_control['semantic_grad_norm']:.6f} "
            f"(coef={legacy_control['semantic_coeff']:.5f})"
        )

    print()
    print("TOTAL-GRAD PARAMETER GROUPS")
    print("-" * 116)
    print(
        f"{'group':<30}"
        f"{'||g|| raw':>15}"
        f"{'||g|| clipped':>17}"
        f"{'RMS raw':>15}"
        f"{'active numel':>16}"
    )
    print("-" * 116)

    sorted_groups = sorted(
        total_stats["groups"].items(),
        key=lambda kv: kv[1]["norm"],
        reverse=True,
    )
    for group, s in sorted_groups:
        print(
            f"{group:<30}"
            f"{s['norm']:>15.6f}"
            f"{s['norm'] * clip_factor:>17.6f}"
            f"{s['rms']:>15.3e}"
            f"{s['active_numel']:>16,}"
        )

    print()
    print("TOP AUTOMATIC NAME PREFIXES — total gradient")
    print("-" * 116)
    print(
        f"{'prefix':<58}"
        f"{'||g||':>15}"
        f"{'RMS':>15}"
        f"{'active numel':>16}"
    )
    print("-" * 116)
    prefix_rows = top_prefix_gradients(
        named_params,
        total_grads,
        depth=2,
    )
    for prefix, norm, rms, n in prefix_rows[:12]:
        print(
            f"{prefix:<58}"
            f"{norm:>15.6f}"
            f"{rms:>15.3e}"
            f"{n:>16,}"
        )

    # Save only plain Python numbers for aggregation.
    record = {
        "loss": scalar(total),
        "raw_grad_norm": raw_norm,
        "grad_rms": total_stats["rms"],
        "clip_factor": clip_factor,
        "changed_ratio": changed_ratio,
        "semantic_mask_stats": semantic_mask_stats,
        "legacy_control": legacy_control,
        "components": rows,
        "groups": total_stats["groups"],
    }

    # Drop all graph-owning references before next batch.
    del total_grads
    del components
    del loss_output
    del total
    del target
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    return record


# =============================================================================
# Aggregate
# =============================================================================

def mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(xs) if xs else float("nan")


def summarize(records, max_grad_norm):
    print()
    print("=" * 116)
    print("AGGREGATE")
    print("=" * 116)

    norms = [r["raw_grad_norm"] for r in records]
    losses = [r["loss"] for r in records]
    factors = [r["clip_factor"] for r in records]
    hit = sum(f < 1.0 for f in factors)

    print(f"batches                   : {len(records)}")
    print(f"mean total loss           : {mean(losses):.8f}")
    print(f"mean raw grad norm        : {mean(norms):.8f}")
    print(f"median raw grad norm      : {median(norms):.8f}")
    print(f"min / max raw grad norm   : {min(norms):.8f} / {max(norms):.8f}")
    print(
        f"clip hits @{max_grad_norm:g}          : "
        f"{hit}/{len(records)} ({100.0 * hit / len(records):.1f}%)"
    )
    print(f"mean clip factor          : {mean(factors):.8f}")

    for threshold in (1.0, 2.0, 5.0, 10.0):
        count = sum(n > threshold for n in norms)
        print(
            f"norm > {threshold:<4g}              : "
            f"{count}/{len(norms)} ({100.0 * count / len(norms):.1f}%)"
        )

    # Aggregate component rows by name.
    by_component = defaultdict(list)
    for record in records:
        for row in record["components"]:
            by_component[row["name"]].append(row)

    if by_component:
        print()
        print("MEAN LOSS-COMPONENT GRADIENT SCALE")
        print("-" * 116)
        print(
            f"{'component':<28}"
            f"{'mean value':>14}"
            f"{'mean coef':>13}"
            f"{'mean ||g||':>15}"
            f"{'||g||/total':>15}"
            f"{'mean cosine':>15}"
        )
        print("-" * 116)

        total_mean = mean(norms)
        for name, rows in by_component.items():
            gn = mean([r["grad_norm"] for r in rows])
            print(
                f"{name:<28}"
                f"{mean([r['scalar'] for r in rows]):>14.6f}"
                f"{mean([r['coeff'] for r in rows]):>13.5f}"
                f"{gn:>15.6f}"
                f"{(gn / total_mean if total_mean > 0 else float('nan')):>15.6f}"
                f"{mean([r['cos_total'] for r in rows]):>15.6f}"
            )

    legacy_rows = [
        r["legacy_control"]
        for r in records
        if r.get("legacy_control") is not None
    ]
    if legacy_rows:
        print()
        print("CHANGED-ONLY vs OLD FULL-VALID SEMANTIC — SAME-FORWARD MEAN")
        print("-" * 116)
        current_total = mean(norms)
        old_total = mean([r["total_grad_norm"] for r in legacy_rows])
        old_sem = mean([r["semantic_grad_norm"] for r in legacy_rows])
        print(f"current changed-only mean total grad : {current_total:.8f}")
        print(f"old full-valid mean total grad       : {old_total:.8f}")
        print(
            f"old/current total-grad ratio         : "
            f"{old_total / current_total if current_total > 0 else float('nan'):.6f}"
        )
        print(
            f"mean cos(current total, old total)   : "
            f"{mean([r['total_cosine_to_changed_only'] for r in legacy_rows]):.6f}"
        )
        print(f"old full-valid mean semantic grad    : {old_sem:.8f}")
        current_sem_rows = by_component.get("loss_semantic", [])
        if current_sem_rows:
            current_sem = mean([r["grad_norm"] for r in current_sem_rows])
            print(f"current changed-only semantic grad   : {current_sem:.8f}")
            print(
                f"old/current semantic-grad ratio      : "
                f"{old_sem / current_sem if current_sem > 0 else float('nan'):.6f}"
            )

    # Aggregate important module groups.
    group_values = defaultdict(list)
    for record in records:
        for group, stats in record["groups"].items():
            group_values[group].append(stats["norm"])

    if group_values:
        print()
        print("MEAN TOTAL-GRAD NORM BY PARAMETER GROUP")
        print("-" * 80)
        print(f"{'group':<35}{'mean ||g||':>18}{'post-clip approx':>22}")
        print("-" * 80)
        mean_factor = mean(factors)
        for group, values in sorted(
            group_values.items(),
            key=lambda kv: mean(kv[1]),
            reverse=True,
        ):
            gn = mean(values)
            print(
                f"{group:<35}"
                f"{gn:>18.6f}"
                f"{gn * mean_factor:>22.6f}"
            )

    print()
    print("HOW TO READ")
    print("-" * 116)
    print(
        "1) raw grad norm >> max_grad_norm and clip hits near 100%: "
        "training is dominated by global clipping."
    )
    print(
        "2) component ||g|| is NOT additive. Use cosine too: two large "
        "components can reinforce or cancel each other."
    )
    print(
        "3) d(total)/d(component) is the exact coefficient used by the "
        "current loss graph at this batch; no loss-weight formula is assumed."
    )
    print(
        "4) Compare qwen_lora vs 2d_cg_decoder / heads. Their parameter counts "
        "differ, so RMS is useful alongside L2 norm."
    )
    print(
        "5) For a decision about the current trained model, rerun with the "
        "actual ValBest checkpoint and preferably 5-10 representative batches."
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.batches < 1:
        raise ValueError("--batches must be >= 1")

    # Single-process diagnostic is deliberate.
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "Run test_grad_scale.py as a single process, not torchrun. "
            "Use CUDA_VISIBLE_DEVICES=... python test_grad_scale.py ..."
        )

    runtime = train_mod.setup_distributed()

    try:
        experiment = load_experiment_config(
            args.config,
            [args.dataset],
        )

        cli_like = SimpleNamespace(
            output_dir=None,
            resume=args.checkpoint,
        )
        settings = train_mod.build_settings(experiment, cli_like)

        if args.seed is not None:
            settings.seed = int(args.seed)

        train_mod.set_seed(settings.seed, runtime["rank"])

        registry = DatasetRegistry(
            experiment,
            runtime,
            num_workers=settings.num_workers,
        )

        if args.dataset not in registry.handles:
            raise KeyError(
                f"Dataset {args.dataset!r} not available. "
                f"Available={sorted(registry.handles)}"
            )

        spec = experiment.datasets[args.dataset].spec

        print("=" * 116)
        print("PAIR GRADIENT-SCALE DIAGNOSTIC")
        print("=" * 116)
        print(f"config                   : {args.config.resolve()}")
        print(f"dataset                  : {args.dataset}")
        print(f"route / label_mode       : {spec.route} / {spec.label_mode}")
        print(f"device                   : {runtime['device']}")
        print(f"autocast                 : {'BF16' if not args.no_autocast else 'OFF'}")
        print(f"batches                  : {args.batches}")
        print(f"configured max_grad_norm : {settings.max_grad_norm}")
        print()

        # Match current train.py exactly.
        model = PAIRModel.from_config(
            experiment.model,
            runtime["device"],
        )
        criterion = PAIRSemanticChangeLoss().to(runtime["device"])

        load_checkpoint_if_requested(
            args,
            model,
            experiment,
            settings,
            registry,
        )

        model.train()
        criterion.train()

        named_params = trainable_parameters(model)
        total_trainable = sum(p.numel() for _, p in named_params)
        lora_trainable = sum(
            p.numel()
            for name, p in named_params
            if "lora_" in name.lower()
        )

        print()
        print(f"trainable parameters     : {total_trainable:,}")
        print(f"LoRA trainable           : {lora_trainable:,}")
        print(f"non-LoRA trainable       : {total_trainable - lora_trainable:,}")

        handle = registry.handles[args.dataset]
        handle.train_cycle.reset(0)

        records = []
        for batch_idx in range(1, args.batches + 1):
            samples = handle.train_cycle.next()

            record = diagnose_batch(
                model=model,
                criterion=criterion,
                samples=samples,
                spec=spec,
                named_params=named_params,
                max_grad_norm=float(settings.max_grad_norm),
                use_autocast=not args.no_autocast,
                batch_index=batch_idx,
                compare_legacy_full_semantic=not args.no_legacy_comparison,
            )
            records.append(record)

        summarize(
            records,
            max_grad_norm=float(settings.max_grad_norm),
        )

    finally:
        train_mod.cleanup_distributed()


if __name__ == "__main__":
    main()
