#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR reasoning-path diagnostic v3.

This script is intentionally diagnostic-only:
  - no optimizer.step()
  - no checkpoint/source modification
  - no gradient clipping is actually applied

It answers three separate questions.

A) Does pure dense<-LLM Cross-Attention destroy / smooth spatial detail?
   For P4/P8/P16/P32 and T1/T2 it measures:
     raw projected dense feature
     exact Cross-Attention query input
     Cross-Attention output
     Unified output
     <TASK>-conditioned native output
     final U-Net pyramid input

   Spatial statistics:
     feature RMS
     spatial std
     adjacent-token delta RMS
     adjacent-token cosine similarity

B) Is sample-specific LLM CONTENT actually useful?
   It runs validation twice:
     NORMAL
     SHUFFLE-LLM

   SHUFFLE-LLM rotates the LLM reasoning tokens AND <TASK> hidden across
   samples in the batch while keeping each sample's dense Vision features
   untouched.  This preserves the overall LLM feature distribution but breaks
   image<->reasoning correspondence.

C) Is max_grad_norm=3 really clipping too often?
   It audits one or more complete training-loader passes at the checkpoint
   WITHOUT optimizer.step().  Gradients are accumulated exactly according to
   training.grad_accum, and raw pre-clip global norms are collected.

   It reports:
     P50/P75/P90/P95/P99/max raw grad norm
     clipping rate and scale coefficient for several candidate thresholds
     configured max_grad_norm behavior
     decoder.logit_scale |grad|
     class_encoder gradient norm

Important
---------
The gradient audit is a "snapshot audit" at the supplied checkpoint.  Because
there is no optimizer.step(), it characterizes the batch-to-batch gradient
distribution of THIS checkpoint.  For an early-training clipping decision,
run the same script on an early checkpoint too.

Example
-------
CUDA_VISIBLE_DEVICES=2 python test3.py \
  --config configs/pair_train_qwen4b.json \
  --dataset SECOND \
  --checkpoint outputs/.../ValBest_SECOND_....pt \
  --feature-batches 3 \
  --shuffle-val true \
  --grad-audit-epochs 1

Two complete gradient-audit passes:
CUDA_VISIBLE_DEVICES=2 python test3.py \
  --config configs/pair_train_qwen4b.json \
  --dataset SECOND \
  --checkpoint outputs/.../ValBest_SECOND_....pt \
  --feature-batches 3 \
  --shuffle-val true \
  --grad-audit-epochs 2
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import train as train_mod
from datasets.config_loader import load_experiment_config
from datasets.multi_dataset import DatasetRegistry
from loss import PAIRSemanticChangeLoss
from metrics import PAIRMetrics
from models.change_decoder import UnifiedTokenSet
from models.pair import PAIRModel


SCALE_NAMES = ("p4", "p8", "p16", "p32")
STAGE_NAMES = (
    "raw_dense",
    "query_input",
    "cross_output",
    "unified_output",
    "task_output",
    "decoder_input",
)


# =============================================================================
# CLI
# =============================================================================


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parse_args():
    p = argparse.ArgumentParser(
        description="PAIR Cross/Unified/LLM-content + gradient-clipping diagnostic."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train_qwen4b.json"),
    )
    p.add_argument("--dataset", type=str, default="SECOND")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-autocast", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--feature-batches",
        type=int,
        default=3,
        help="Validation batches used for spatial-detail + attention diagnostics. 0 disables.",
    )
    p.add_argument(
        "--shuffle-val",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Run NORMAL vs SHUFFLE-LLM validation.",
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="0 = full validation for NORMAL/SHUFFLE-LLM.",
    )
    p.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.005,
        help="Abort SHUFFLE-LLM if NORMAL does not reproduce saved F_scd within this tolerance.",
    )

    p.add_argument(
        "--grad-audit-epochs",
        type=int,
        default=1,
        help="Complete training-loader passes for raw pre-clip grad statistics. 0 disables.",
    )
    p.add_argument(
        "--grad-audit-max-updates",
        type=int,
        default=0,
        help="Optional cap across the whole grad audit. 0 = all updates.",
    )
    p.add_argument(
        "--grad-audit-print-every",
        type=int,
        default=10,
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return p.parse_args()


# =============================================================================
# Common helpers
# =============================================================================


def load_checkpoint_model_only(model: PAIRModel, path: Path):
    if not hasattr(train_mod, "load_model_state_from_checkpoint"):
        raise RuntimeError(
            "Current train.py has no load_model_state_from_checkpoint(). "
            "Use the checkpoint-buffer-fixed train.py."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    legacy_missing_buffers = train_mod.load_model_state_from_checkpoint(
        ckpt,
        model,
    )
    if legacy_missing_buffers:
        raise RuntimeError(
            "This checkpoint is legacy and did not save BatchNorm buffers. "
            "Use a new-format checkpoint for rigorous diagnostics."
        )
    return ckpt


def global_grad_norm(model: torch.nn.Module) -> Tuple[float, float, int]:
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


def named_prefix_grad_norm(model: torch.nn.Module, prefix: str) -> float:
    sq = 0.0
    for name, p in model.named_parameters():
        if not name.startswith(prefix) or p.grad is None:
            continue
        g = p.grad.detach().float()
        n = float(torch.linalg.vector_norm(g).cpu())
        sq += n * n
    return math.sqrt(sq)


def logit_scale_grad_info(model: PAIRModel) -> Dict[str, float]:
    p = getattr(model.decoder, "logit_scale", None)
    if p is None:
        return {
            "log_value": float("nan"),
            "scale": float("nan"),
            "grad_abs": 0.0,
        }

    log_value = float(p.detach().float().cpu())
    scale = float(p.detach().float().exp().clamp(min=1.0, max=100.0).cpu())
    grad_abs = (
        abs(float(p.grad.detach().float().cpu()))
        if p.grad is not None
        else 0.0
    )
    return {
        "log_value": log_value,
        "scale": scale,
        "grad_abs": grad_abs,
    }


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def scalar_metrics(result):
    return {k: float(v) for k, v in result["scalars"].items()}


# =============================================================================
# A) Internal feature / attention capture
# =============================================================================


def _grid_shape_from_positions(
    positions: torch.Tensor,
    expected_n: int,
) -> Optional[Tuple[int, int]]:
    if positions.ndim != 2 or positions.shape[1] < 2:
        return None
    with torch.no_grad():
        x = positions[:, 0].detach().float().cpu()
        y = positions[:, 1].detach().float().cpu()
        w = int(torch.unique(x).numel())
        h = int(torch.unique(y).numel())
    if h > 0 and w > 0 and h * w == int(expected_n):
        return h, w

    root = int(round(math.sqrt(int(expected_n))))
    if root * root == int(expected_n):
        return root, root
    return None


def _spatial_stats_one_map(feat_hwd: torch.Tensor) -> Dict[str, float]:
    x = feat_hwd.detach().float()
    if x.ndim != 3:
        raise ValueError(f"Expected [H,W,D], got {tuple(x.shape)}")

    feature_rms = float(torch.sqrt(torch.mean(x * x)).cpu())
    centered = x - x.mean(dim=(0, 1), keepdim=True)
    spatial_std = float(torch.sqrt(torch.mean(centered * centered)).cpu())

    diffs = []
    cosines = []

    if x.shape[1] > 1:
        a = x[:, :-1, :]
        b = x[:, 1:, :]
        diffs.append((b - a).reshape(-1, x.shape[-1]))
        cosines.append(
            F.cosine_similarity(
                a.reshape(-1, x.shape[-1]),
                b.reshape(-1, x.shape[-1]),
                dim=-1,
                eps=1e-8,
            )
        )

    if x.shape[0] > 1:
        a = x[:-1, :, :]
        b = x[1:, :, :]
        diffs.append((b - a).reshape(-1, x.shape[-1]))
        cosines.append(
            F.cosine_similarity(
                a.reshape(-1, x.shape[-1]),
                b.reshape(-1, x.shape[-1]),
                dim=-1,
                eps=1e-8,
            )
        )

    if diffs:
        d = torch.cat(diffs, dim=0)
        neighbor_delta_rms = float(torch.sqrt(torch.mean(d * d)).cpu())
        neighbor_cosine = float(torch.cat(cosines).mean().cpu())
    else:
        neighbor_delta_rms = 0.0
        neighbor_cosine = 1.0

    return {
        "feature_rms": feature_rms,
        "spatial_std": spatial_std,
        "neighbor_delta_rms": neighbor_delta_rms,
        "neighbor_cosine": neighbor_cosine,
    }


def spatial_stats_flat(
    features: torch.Tensor,
    positions: torch.Tensor,
    batch_ids: torch.Tensor,
) -> Dict[str, float]:
    rows = []
    ids = sorted(int(x) for x in torch.unique(batch_ids).tolist())
    for batch_id in ids:
        mask = batch_ids == batch_id
        f = features[mask]
        p = positions[mask]
        shape = _grid_shape_from_positions(p, f.shape[0])
        if shape is None:
            continue
        h, w = shape
        rows.append(_spatial_stats_one_map(f.reshape(h, w, f.shape[-1])))

    if not rows:
        return {
            "feature_rms": float("nan"),
            "spatial_std": float("nan"),
            "neighbor_delta_rms": float("nan"),
            "neighbor_cosine": float("nan"),
        }

    return {
        key: statistics.mean(row[key] for row in rows)
        for key in rows[0]
    }


def spatial_stats_bchw(x: torch.Tensor) -> Dict[str, float]:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    rows = []
    for b in range(x.shape[0]):
        rows.append(
            _spatial_stats_one_map(
                x[b].permute(1, 2, 0).contiguous()
            )
        )
    return {
        key: statistics.mean(row[key] for row in rows)
        for key in rows[0]
    }


@torch.no_grad()
def attention_stats(
    module,
    dense: torch.Tensor,
    dense_batch_ids: torch.Tensor,
    reasoning: torch.Tensor,
    reasoning_batch_ids: torch.Tensor,
    *,
    use_autocast: bool,
) -> Dict[str, float]:
    """
    Recompute the exact ReasoningInjection attention in eval mode, but request
    weights.  The production forward uses need_weights=False.
    """
    entropy_values = []
    max_values = []
    effective_values = []
    mass_usage_values = []
    top1_coverage_values = []
    key_counts = []

    for batch_id in sorted(int(x) for x in torch.unique(dense_batch_ids).tolist()):
        dmask = dense_batch_ids == batch_id
        rmask = reasoning_batch_ids == batch_id
        d = dense[dmask]
        r = reasoning[rmask]
        if d.shape[0] == 0 or r.shape[0] == 0:
            continue

        kv = module.norm_kv(r).unsqueeze(0)
        k = int(r.shape[0])
        key_counts.append(k)

        total_entropy = 0.0
        total_max = 0.0
        total_effective = 0.0
        total_count = 0
        key_mass = torch.zeros(k, dtype=torch.float64, device=d.device)
        top1_used = torch.zeros(k, dtype=torch.bool, device=d.device)

        for start in range(0, d.shape[0], module.query_chunk_size):
            end = min(start + module.query_chunk_size, d.shape[0])
            q = module.norm_q(d[start:end]).unsqueeze(0)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=bool(use_autocast and dense.is_cuda),
            ):
                _, w = module.attention(
                    q,
                    kv,
                    kv,
                    need_weights=True,
                    average_attn_weights=False,
                )
            # [1, heads, Q, K]
            w = w[0].float().clamp_min(1e-12)
            entropy = -(w * w.log()).sum(dim=-1)
            maxp = w.max(dim=-1).values
            effective = entropy.exp()

            count = int(entropy.numel())
            total_entropy += float(entropy.sum().cpu())
            total_max += float(maxp.sum().cpu())
            total_effective += float(effective.sum().cpu())
            total_count += count

            key_mass += w.double().sum(dim=(0, 1))
            top1 = w.argmax(dim=-1).reshape(-1)
            top1_used[top1.unique()] = True

        if total_count == 0:
            continue

        mean_entropy = total_entropy / total_count
        norm_entropy = (
            mean_entropy / math.log(k)
            if k > 1
            else 0.0
        )
        mean_max = total_max / total_count
        mean_effective = total_effective / total_count

        key_mass = key_mass / key_mass.sum().clamp_min(1e-30)
        mass_entropy = float(
            (-(key_mass * key_mass.clamp_min(1e-30).log()).sum()).cpu()
        )
        effective_mass_keys = math.exp(mass_entropy)
        mass_usage = effective_mass_keys / max(k, 1)
        top1_coverage = float(top1_used.float().mean().cpu())

        entropy_values.append(norm_entropy)
        max_values.append(mean_max)
        effective_values.append(mean_effective)
        mass_usage_values.append(mass_usage)
        top1_coverage_values.append(top1_coverage)

    return {
        "keys": safe_mean(key_counts),
        "norm_entropy": safe_mean(entropy_values),
        "mean_max_prob": safe_mean(max_values),
        "mean_effective_keys_per_query": safe_mean(effective_values),
        "mass_effective_key_fraction": safe_mean(mass_usage_values),
        "top1_key_coverage": safe_mean(top1_coverage_values),
    }


class CaptureReasoningInternals:
    """
    Capture exact 2D reasoning stages without changing production source.

    The wrapper is valid only for the current 2D multiscale route where
    fuse_2d_multiscale_reasoning calls:
        scale0 T1, scale0 T2, ..., scale3 T1, scale3 T2
    then one unified_attention.forward_pair(), then two task_conditioning calls.
    """

    def __init__(self, model: PAIRModel):
        self.model = model
        self.decoder = model.decoder
        self.records: List[Dict] = []
        self.decoder_inputs = None
        self.pending = None
        self._orig_inject = None
        self._orig_ri_forward = None
        self._orig_unified = None
        self._orig_task = None
        self._orig_cg = None
        self._task_call = 0

    def clear(self):
        self.records = []
        self.decoder_inputs = None
        self.pending = None
        self._task_call = 0

    def __enter__(self):
        dec = self.decoder
        owner = self

        self._orig_inject = dec._inject_reasoning_one_time
        orig_inject = self._orig_inject

        def inject_wrapped(
            decoder_self,
            *,
            dense_tokens,
            dense_time_id,
            reasoning_tokens,
            reasoning_time_id,
            scale_id=None,
        ):
            if owner.pending is not None:
                raise RuntimeError("Nested reasoning injection capture is unsupported.")
            owner.pending = {
                "scale_id": int(scale_id) if scale_id is not None else -1,
                "time_id": int(dense_time_id),
                "raw_dense": dense_tokens.features.detach(),
                "positions": dense_tokens.positions.detach(),
                "batch_ids": dense_tokens.batch_ids.detach(),
            }
            try:
                return orig_inject(
                    dense_tokens=dense_tokens,
                    dense_time_id=dense_time_id,
                    reasoning_tokens=reasoning_tokens,
                    reasoning_time_id=reasoning_time_id,
                    scale_id=scale_id,
                )
            finally:
                if owner.pending is not None:
                    # reasoning_injection.forward should consume it.
                    owner.pending = None

        dec._inject_reasoning_one_time = MethodType(inject_wrapped, dec)

        ri = dec.reasoning_injection
        self._orig_ri_forward = ri.forward
        orig_ri_forward = self._orig_ri_forward

        def ri_wrapped(
            module_self,
            *,
            dense,
            dense_batch_ids,
            reasoning,
            reasoning_batch_ids,
        ):
            if owner.pending is None:
                raise RuntimeError("ReasoningInjection capture lost scale/time metadata.")

            meta = owner.pending
            out = orig_ri_forward(
                dense=dense,
                dense_batch_ids=dense_batch_ids,
                reasoning=reasoning,
                reasoning_batch_ids=reasoning_batch_ids,
            )
            record = dict(meta)
            record.update(
                query_input=dense.detach(),
                reasoning_input=reasoning.detach(),
                reasoning_batch_ids=reasoning_batch_ids.detach(),
                cross_output=out.detach(),
            )
            owner.records.append(record)
            owner.pending = None
            return out

        ri.forward = MethodType(ri_wrapped, ri)

        ua = dec.unified_attention
        self._orig_unified = ua.forward_pair
        orig_unified = self._orig_unified

        def unified_wrapped(
            module_self,
            *,
            x1,
            batch_ids1,
            x2,
            batch_ids2,
        ):
            out1, out2 = orig_unified(
                x1=x1,
                batch_ids1=batch_ids1,
                x2=x2,
                batch_ids2=batch_ids2,
            )

            for time_id, out in ((0, out1), (1, out2)):
                recs = sorted(
                    [r for r in owner.records if r["time_id"] == time_id],
                    key=lambda r: r["scale_id"],
                )
                lengths = [int(r["cross_output"].shape[0]) for r in recs]
                if sum(lengths) != int(out.shape[0]):
                    raise RuntimeError(
                        f"Unified capture length mismatch T{time_id + 1}: "
                        f"{sum(lengths)} vs {out.shape[0]}"
                    )
                for r, part in zip(recs, torch.split(out.detach(), lengths, dim=0)):
                    r["unified_output"] = part

            return out1, out2

        ua.forward_pair = MethodType(unified_wrapped, ua)

        tc = dec.task_conditioning
        self._orig_task = tc.forward
        orig_task = self._orig_task

        def task_wrapped(
            module_self,
            *,
            x,
            batch_ids,
            task_hidden,
        ):
            out = orig_task(
                x=x,
                batch_ids=batch_ids,
                task_hidden=task_hidden,
            )
            time_id = owner._task_call
            owner._task_call += 1
            if time_id not in (0, 1):
                raise RuntimeError(
                    "Expected exactly two task-conditioning calls in 2D forward."
                )
            recs = sorted(
                [r for r in owner.records if r["time_id"] == time_id],
                key=lambda r: r["scale_id"],
            )
            lengths = [int(r["cross_output"].shape[0]) for r in recs]
            if sum(lengths) != int(out.shape[0]):
                raise RuntimeError(
                    f"Task capture length mismatch T{time_id + 1}: "
                    f"{sum(lengths)} vs {out.shape[0]}"
                )
            for r, part in zip(recs, torch.split(out.detach(), lengths, dim=0)):
                r["task_output"] = part
            return out

        tc.forward = MethodType(task_wrapped, tc)

        self._orig_cg = dec.forward_2d_cg
        orig_cg = self._orig_cg

        def cg_wrapped(
            decoder_self,
            *,
            feat_pyramid_t1,
            feat_pyramid_t2,
            prediction_mode,
            class_names=None,
            qwen_backbone=None,
            detach_qwen_class_encoder=True,
        ):
            owner.decoder_inputs = (
                tuple(x.detach() for x in feat_pyramid_t1),
                tuple(x.detach() for x in feat_pyramid_t2),
            )
            return orig_cg(
                feat_pyramid_t1=feat_pyramid_t1,
                feat_pyramid_t2=feat_pyramid_t2,
                prediction_mode=prediction_mode,
                class_names=class_names,
                qwen_backbone=qwen_backbone,
                detach_qwen_class_encoder=detach_qwen_class_encoder,
            )

        dec.forward_2d_cg = MethodType(cg_wrapped, dec)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.decoder._inject_reasoning_one_time = self._orig_inject
        self.decoder.reasoning_injection.forward = self._orig_ri_forward
        self.decoder.unified_attention.forward_pair = self._orig_unified
        self.decoder.task_conditioning.forward = self._orig_task
        self.decoder.forward_2d_cg = self._orig_cg
        return False


def collect_feature_batch_stats(
    model: PAIRModel,
    capture: CaptureReasoningInternals,
    *,
    use_autocast: bool,
) -> Dict:
    if len(capture.records) != 8:
        raise RuntimeError(
            f"Expected 8 reasoning records (4 scales x 2 times), got {len(capture.records)}"
        )
    if capture.decoder_inputs is None:
        raise RuntimeError("Did not capture final U-Net pyramid inputs.")

    out = {}
    ri = model.decoder.reasoning_injection

    for rec in capture.records:
        scale_id = rec["scale_id"]
        time_id = rec["time_id"]
        scale = SCALE_NAMES[scale_id]
        time_name = f"T{time_id + 1}"
        key = f"{scale}/{time_name}"

        stage_stats = {}
        for stage in (
            "raw_dense",
            "query_input",
            "cross_output",
            "unified_output",
            "task_output",
        ):
            stage_stats[stage] = spatial_stats_flat(
                rec[stage],
                rec["positions"],
                rec["batch_ids"],
            )

        decoder_tensor = capture.decoder_inputs[time_id][scale_id]
        stage_stats["decoder_input"] = spatial_stats_bchw(decoder_tensor)

        attn = attention_stats(
            ri,
            rec["query_input"],
            rec["batch_ids"],
            rec["reasoning_input"],
            rec["reasoning_batch_ids"],
            use_autocast=use_autocast,
        )

        out[key] = {
            "stages": stage_stats,
            "attention": attn,
        }

    return out


def merge_feature_stats(all_batches: Sequence[Dict]) -> Dict:
    merged = {}
    keys = sorted(all_batches[0].keys())
    for key in keys:
        merged[key] = {"stages": {}, "attention": {}}

        for stage in STAGE_NAMES:
            metric_keys = all_batches[0][key]["stages"][stage].keys()
            merged[key]["stages"][stage] = {
                metric: safe_mean(
                    [
                        b[key]["stages"][stage][metric]
                        for b in all_batches
                        if math.isfinite(b[key]["stages"][stage][metric])
                    ]
                )
                for metric in metric_keys
            }

        attn_keys = all_batches[0][key]["attention"].keys()
        merged[key]["attention"] = {
            metric: safe_mean(
                [
                    b[key]["attention"][metric]
                    for b in all_batches
                    if math.isfinite(b[key]["attention"][metric])
                ]
            )
            for metric in attn_keys
        }

    return merged


def print_feature_report(merged: Dict):
    print()
    print("=" * 132)
    print("A) SPATIAL DETAIL THROUGH RAW -> CROSS -> UNIFIED -> TASK -> U-NET")
    print("=" * 132)
    print(
        "neighbor_delta_rms: larger = more local variation; "
        "neighbor_cosine: closer to 1 = more locally similar/smoothed."
    )
    print(
        "NOTE: decoder_input P4/P8 have already been bilinearly upsampled, so "
        "their Δ/raw ratio is descriptive but not an apples-to-apples native-grid comparison."
    )

    for scale in SCALE_NAMES:
        for time_name in ("T1", "T2"):
            key = f"{scale}/{time_name}"
            row = merged[key]
            raw_delta = row["stages"]["raw_dense"]["neighbor_delta_rms"]

            print()
            print(f"{key}   reasoning K ~= {row['attention']['keys']:.1f}")
            print("-" * 132)
            print(
                f"{'stage':<18}"
                f"{'feat RMS':>13}"
                f"{'spatial std':>15}"
                f"{'neighbor Δ RMS':>18}"
                f"{'Δ/raw':>12}"
                f"{'neighbor cosine':>18}"
            )
            for stage in STAGE_NAMES:
                s = row["stages"][stage]
                ratio = (
                    s["neighbor_delta_rms"] / raw_delta
                    if raw_delta > 0 and math.isfinite(s["neighbor_delta_rms"])
                    else float("nan")
                )
                print(
                    f"{stage:<18}"
                    f"{s['feature_rms']:>13.4e}"
                    f"{s['spatial_std']:>15.4e}"
                    f"{s['neighbor_delta_rms']:>18.4e}"
                    f"{ratio:>12.4f}"
                    f"{s['neighbor_cosine']:>18.6f}"
                )

            a = row["attention"]
            print(
                "attention: "
                f"norm_entropy={a['norm_entropy']:.4f}  "
                f"mean_max_prob={a['mean_max_prob']:.4f}  "
                f"effective_keys/query={a['mean_effective_keys_per_query']:.2f}  "
                f"mass_effective_fraction={a['mass_effective_key_fraction']:.4f}  "
                f"top1_coverage={a['top1_key_coverage']:.4f}"
            )


# =============================================================================
# B) SHUFFLE-LLM validation
# =============================================================================


def rotate_token_features(tokens: UnifiedTokenSet) -> UnifiedTokenSet:
    ids = sorted(int(x) for x in torch.unique(tokens.batch_ids).tolist())
    if len(ids) <= 1:
        return tokens

    new_features = torch.empty_like(tokens.features)
    for i, dest_id in enumerate(ids):
        src_id = ids[(i + 1) % len(ids)]
        dest_mask = tokens.batch_ids == dest_id
        src_mask = tokens.batch_ids == src_id
        dest_n = int(dest_mask.sum().item())
        src_n = int(src_mask.sum().item())
        if dest_n != src_n:
            raise RuntimeError(
                "SHUFFLE-LLM requires equal reasoning-token count per sample "
                f"within a batch; got dest={dest_n}, src={src_n}."
            )
        new_features[dest_mask] = tokens.features[src_mask]

    return UnifiedTokenSet(
        features=new_features,
        positions=tokens.positions,
        modality_ids=tokens.modality_ids,
        batch_ids=tokens.batch_ids,
    )


class ShuffleLLMContent:
    """
    Break sample correspondence while preserving dense Vision inputs.

    Both T1/T2 reasoning-token features and <TASK> hidden are rotated by the
    same one-sample permutation. Positions/batch IDs stay attached to the
    destination sample.
    """

    def __init__(self, model: PAIRModel):
        self.decoder = model.decoder
        self.original = None
        self.shuffled_batches = 0
        self.skipped_singleton_batches = 0

    def __enter__(self):
        self.original = self.decoder.fuse_2d_multiscale_reasoning
        original = self.original
        owner = self

        def wrapped(
            decoder_self,
            *,
            dense_t1,
            dense_t2,
            reasoning_t1,
            reasoning_t2,
            task_hidden,
        ):
            batch_size = int(task_hidden.shape[0])
            if batch_size <= 1:
                owner.skipped_singleton_batches += 1
                return original(
                    dense_t1=dense_t1,
                    dense_t2=dense_t2,
                    reasoning_t1=reasoning_t1,
                    reasoning_t2=reasoning_t2,
                    task_hidden=task_hidden,
                )

            owner.shuffled_batches += 1
            shuffled_r1 = tuple(rotate_token_features(x) for x in reasoning_t1)
            shuffled_r2 = tuple(rotate_token_features(x) for x in reasoning_t2)

            # Destination sample b receives source sample b+1.
            shuffled_task = torch.roll(task_hidden, shifts=-1, dims=0)

            return original(
                dense_t1=dense_t1,
                dense_t2=dense_t2,
                reasoning_t1=shuffled_r1,
                reasoning_t2=shuffled_r2,
                task_hidden=shuffled_task,
            )

        self.decoder.fuse_2d_multiscale_reasoning = MethodType(
            wrapped,
            self.decoder,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.decoder.fuse_2d_multiscale_reasoning = self.original
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


def print_eval_result(title, result):
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
    print("=" * 104)
    print(title)
    print("=" * 104)
    print(f"samples : {result['samples']}")
    print(f"seconds : {result['seconds']:.2f}")
    for key in keys:
        if key in result["scalars"]:
            print(f"{key:<24}: {float(result['scalars'][key]):.6f}")



def verify_saved_baseline(ckpt, dataset, normal, tolerance):
    selection = (ckpt.get("validation_selection") or {}).get(dataset)
    if not selection:
        print(
            "Warning: checkpoint has no validation_selection; "
            "cannot automatically verify NORMAL."
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

    rerun = float(normal["scalars"].get("scd/F_scd", float("nan")))
    diff = abs(float(saved) - rerun)

    print()
    print("=" * 104)
    print("BASELINE REPRODUCTION CHECK")
    print("=" * 104)
    print(f"saved F_scd : {float(saved):.6f}")
    print(f"rerun F_scd : {rerun:.6f}")
    print(f"abs diff    : {diff:.6f}")
    print(f"tolerance   : {float(tolerance):.6f}")

    if not math.isfinite(rerun) or diff > float(tolerance):
        raise RuntimeError(
            "NORMAL does not reproduce the saved checkpoint baseline. "
            "SHUFFLE-LLM is aborted because the comparison would be invalid."
        )


def print_shuffle_comparison(normal, shuffled):
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
    print("=" * 104)
    print("B) NORMAL vs SHUFFLE-LLM")
    print("=" * 104)
    print(
        f"{'metric':<24}"
        f"{'normal':>14}"
        f"{'shuffle-llm':>16}"
        f"{'delta(shuffle-normal)':>24}"
    )
    print("-" * 104)

    for key in keys:
        if key not in normal["scalars"] or key not in shuffled["scalars"]:
            continue
        a = float(normal["scalars"][key])
        b = float(shuffled["scalars"][key])
        print(f"{key:<24}{a:>14.6f}{b:>16.6f}{(b-a):>24.6f}")


# =============================================================================
# C) Full-pass raw gradient / clipping audit
# =============================================================================


def _grad_audit_updates(loader, grad_accum: int):
    """
    Yield lists of microbatches using the same accumulation grouping principle
    as training. The final group may contain fewer microbatches.
    """
    group = []
    for samples in loader:
        group.append(samples)
        if len(group) == grad_accum:
            yield group
            group = []
    if group:
        yield group


def grad_clip_table(norms: Sequence[float], configured_clip: float):
    p90 = percentile(norms, 0.90)
    p95 = percentile(norms, 0.95)
    p99 = percentile(norms, 0.99)

    candidates = [
        configured_clip,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
        p90, p95, p99,
    ]
    cleaned = []
    for x in candidates:
        if not math.isfinite(x) or x <= 0:
            continue
        if not any(abs(x - y) < 1e-8 for y in cleaned):
            cleaned.append(float(x))
    cleaned.sort()

    rows = []
    arr = np.asarray(norms, dtype=np.float64)
    for clip in cleaned:
        clipped = arr > clip
        coef = np.minimum(1.0, clip / np.maximum(arr, 1e-30))
        rows.append(
            {
                "clip": float(clip),
                "clip_fraction": float(clipped.mean()),
                "mean_coef_all": float(coef.mean()),
                "mean_coef_clipped": (
                    float(coef[clipped].mean()) if clipped.any() else 1.0
                ),
                "mean_postclip_norm": float(np.minimum(arr, clip).mean()),
            }
        )
    return rows


def run_grad_audit(
    model,
    handle,
    spec,
    criterion,
    device,
    use_autocast,
    epochs,
    max_updates,
    print_every,
    grad_accum,
    configured_clip,
):
    model.train()

    raw_norms = []
    global_rmss = []
    logit_grads = []
    logit_scales = []
    class_encoder_norms = []
    logit_global_ratios = []
    losses = []

    total_updates = 0

    print()
    print("=" * 118)
    print("C) FULL-PASS RAW GRADIENT / CLIPPING AUDIT")
    print("=" * 118)
    print("mode              : model.train(), NO optimizer.step(), NO clipping applied")
    print("grad accumulation :", grad_accum)
    print("audit epochs      :", epochs)
    print("configured clip   :", configured_clip)
    print()

    for audit_epoch in range(epochs):
        if handle.train_sampler is not None:
            handle.train_sampler.set_epoch(audit_epoch)

        update_iter = _grad_audit_updates(
            handle.train_loader,
            grad_accum=max(1, int(grad_accum)),
        )
        epoch_bar = tqdm(
            update_iter,
            total=math.ceil(len(handle.train_loader) / max(1, grad_accum)),
            desc=f"GRAD AUDIT E{audit_epoch + 1}",
            dynamic_ncols=True,
        )

        for update_in_epoch, microbatches in enumerate(epoch_bar, start=1):
            if max_updates > 0 and total_updates >= max_updates:
                break

            model.zero_grad(set_to_none=True)

            actual_accum = len(microbatches)
            update_loss = 0.0

            for samples in microbatches:
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    _, loss_output, _ = train_mod.forward_loss(
                        model,
                        criterion,
                        samples,
                        spec,
                    )
                    loss = loss_output.total / actual_accum
                update_loss += float(loss_output.total.detach().float().cpu()) / actual_accum
                loss.backward()

            gnorm, grms, active = global_grad_norm(model)
            ls = logit_scale_grad_info(model)
            ce_norm = named_prefix_grad_norm(
                model,
                "decoder.class_encoder.",
            )

            raw_norms.append(gnorm)
            global_rmss.append(grms)
            logit_grads.append(ls["grad_abs"])
            logit_scales.append(ls["scale"])
            class_encoder_norms.append(ce_norm)
            logit_global_ratios.append(
                ls["grad_abs"] / max(gnorm, 1e-30)
            )
            losses.append(update_loss)

            total_updates += 1

            would_clip = (
                configured_clip > 0 and gnorm > configured_clip
            )
            if (
                update_in_epoch == 1
                or update_in_epoch % max(1, print_every) == 0
            ):
                print(
                    f"AUDIT E{audit_epoch + 1:02d} U{update_in_epoch:04d} "
                    f"loss={update_loss:.6f} "
                    f"raw_grad={gnorm:.6f} "
                    f"clip={'YES' if would_clip else 'NO'} "
                    f"logit_scale={ls['scale']:.4f} "
                    f"|g_logit_scale|={ls['grad_abs']:.6f} "
                    f"class_encoder_grad={ce_norm:.6f}"
                )

            epoch_bar.set_postfix(
                raw_grad=f"{gnorm:.3f}",
                logit_g=f"{ls['grad_abs']:.3f}",
                refresh=False,
            )

            model.zero_grad(set_to_none=True)
            del loss_output, loss

        if max_updates > 0 and total_updates >= max_updates:
            break

    if not raw_norms:
        raise RuntimeError("Gradient audit produced no updates.")

    summary = {
        "updates": len(raw_norms),
        "loss_mean": safe_mean(losses),
        "raw_grad_norm": {
            "mean": safe_mean(raw_norms),
            "p50": percentile(raw_norms, 0.50),
            "p75": percentile(raw_norms, 0.75),
            "p90": percentile(raw_norms, 0.90),
            "p95": percentile(raw_norms, 0.95),
            "p99": percentile(raw_norms, 0.99),
            "max": max(raw_norms),
        },
        "global_grad_rms_mean": safe_mean(global_rmss),
        "logit_scale": {
            "scale_mean": safe_mean(logit_scales),
            "grad_abs_mean": safe_mean(logit_grads),
            "grad_abs_p50": percentile(logit_grads, 0.50),
            "grad_abs_p90": percentile(logit_grads, 0.90),
            "grad_abs_p95": percentile(logit_grads, 0.95),
            "grad_abs_p99": percentile(logit_grads, 0.99),
            "grad_abs_max": max(logit_grads),
            "grad_over_global_mean": safe_mean(logit_global_ratios),
        },
        "class_encoder_grad_norm": {
            "mean": safe_mean(class_encoder_norms),
            "p95": percentile(class_encoder_norms, 0.95),
            "max": max(class_encoder_norms),
        },
    }

    rows = grad_clip_table(raw_norms, configured_clip)
    summary["clip_candidates"] = rows

    print()
    print("RAW GLOBAL GRAD NORM DISTRIBUTION")
    print("-" * 84)
    g = summary["raw_grad_norm"]
    print(
        f"updates={summary['updates']}  "
        f"mean={g['mean']:.6f}  "
        f"P50={g['p50']:.6f}  "
        f"P75={g['p75']:.6f}  "
        f"P90={g['p90']:.6f}  "
        f"P95={g['p95']:.6f}  "
        f"P99={g['p99']:.6f}  "
        f"max={g['max']:.6f}"
    )

    print()
    print("LOGIT_SCALE GRADIENT")
    print("-" * 84)
    l = summary["logit_scale"]
    print(
        f"scale(mean)={l['scale_mean']:.6f}  "
        f"|g|(mean)={l['grad_abs_mean']:.6f}  "
        f"P50={l['grad_abs_p50']:.6f}  "
        f"P90={l['grad_abs_p90']:.6f}  "
        f"P95={l['grad_abs_p95']:.6f}  "
        f"P99={l['grad_abs_p99']:.6f}  "
        f"max={l['grad_abs_max']:.6f}"
    )
    print(
        f"mean |g_logit_scale| / global_grad_norm = "
        f"{l['grad_over_global_mean']:.6f}"
    )
    ce = summary["class_encoder_grad_norm"]
    print(
        f"class_encoder ||g||: mean={ce['mean']:.6f}  "
        f"P95={ce['p95']:.6f}  max={ce['max']:.6f}"
    )

    print()
    print("WHAT DIFFERENT CLIP THRESHOLDS WOULD DO")
    print("-" * 100)
    print(
        f"{'clip':>10}"
        f"{'updates clipped':>18}"
        f"{'mean coef(all)':>18}"
        f"{'mean coef(clipped)':>22}"
        f"{'mean postclip norm':>22}"
    )
    for row in rows:
        marker = "  < configured" if abs(row["clip"] - configured_clip) < 1e-8 else ""
        print(
            f"{row['clip']:>10.4f}"
            f"{100.0 * row['clip_fraction']:>17.2f}%"
            f"{row['mean_coef_all']:>18.4f}"
            f"{row['mean_coef_clipped']:>22.4f}"
            f"{row['mean_postclip_norm']:>22.4f}"
            f"{marker}"
        )

    current = None
    for row in rows:
        if abs(row["clip"] - configured_clip) < 1e-8:
            current = row
            break

    print()
    print("CLIP INTERPRETATION")
    print("-" * 100)
    if current is not None:
        frac = current["clip_fraction"]
        if frac >= 0.50:
            print(
                f"Configured clip={configured_clip:g} would clip "
                f"{100.0 * frac:.1f}% of audited updates: this is routine gradient "
                "rescaling, not rare outlier clipping."
            )
        elif frac >= 0.20:
            print(
                f"Configured clip={configured_clip:g} would clip "
                f"{100.0 * frac:.1f}% of audited updates: clipping is fairly frequent."
            )
        elif frac >= 0.05:
            print(
                f"Configured clip={configured_clip:g} would clip "
                f"{100.0 * frac:.1f}% of audited updates: moderate/outlier-oriented clipping."
            )
        else:
            print(
                f"Configured clip={configured_clip:g} would clip only "
                f"{100.0 * frac:.1f}% of audited updates."
            )

    print(
        f"If the goal is to clip only roughly the largest 5% of gradients at "
        f"THIS checkpoint, the empirical P95 is {g['p95']:.4f}. "
        "Do not treat that number as universal; repeat on an early checkpoint "
        "before changing the training config."
    )

    return summary


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.feature_batches < 0:
        raise ValueError("--feature-batches must be >= 0")
    if args.grad_audit_epochs < 0:
        raise ValueError("--grad-audit-epochs must be >= 0")
    if args.max_val_samples < 0:
        raise ValueError("--max-val-samples must be >= 0")
    if args.grad_audit_max_updates < 0:
        raise ValueError("--grad-audit-max-updates must be >= 0")

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
    use_autocast = not args.no_autocast
    configured_clip = float(
        experiment.optimizer.get("max_grad_norm", 0.0)
    )
    grad_accum = int(
        experiment.training.get("grad_accum", 1)
    )
    threshold = float(
        experiment.validation.get("change_threshold", 0.5)
    )
    grid_size = int(
        experiment.model.get("pyramid_reasoning_grid_size", 8)
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

    criterion = PAIRSemanticChangeLoss().to(device)

    print("=" * 132)
    print("PAIR REASONING-PATH DIAGNOSTIC V3")
    print("=" * 132)
    print("config               :", args.config)
    print("dataset              :", args.dataset)
    print("checkpoint           :", args.checkpoint)
    print("checkpoint epoch     :", ckpt.get("epoch", "N/A"))
    print("autocast             :", "BF16" if use_autocast else "OFF")
    print("pyramid grid size    :", grid_size)
    print("configured grad clip :", configured_clip)
    print("grad accumulation    :", grad_accum)
    print()

    report = {
        "config": str(args.config),
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "pyramid_reasoning_grid_size": grid_size,
        "configured_max_grad_norm": configured_clip,
        "grad_accum": grad_accum,
    }

    # ---------------------------------------------------------------------
    # A) Feature / attention diagnostic
    # ---------------------------------------------------------------------
    if args.feature_batches > 0:
        model.eval()
        feature_batches = []
        val_iter = iter(handle.val_loader)

        with CaptureReasoningInternals(model) as capture:
            for batch_idx in range(args.feature_batches):
                try:
                    samples = next(val_iter)
                except StopIteration:
                    break

                capture.clear()
                with torch.no_grad():
                    with torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=use_autocast,
                    ):
                        train_mod.forward_loss(
                            model,
                            criterion,
                            samples,
                            spec,
                        )

                batch_stats = collect_feature_batch_stats(
                    model,
                    capture,
                    use_autocast=use_autocast,
                )
                feature_batches.append(batch_stats)
                print(
                    f"[feature {batch_idx + 1}/{args.feature_batches}] "
                    f"captured {len(samples)} samples"
                )

                capture.clear()
                gc.collect()
                torch.cuda.empty_cache()

        if not feature_batches:
            raise RuntimeError("No feature diagnostic batches were processed.")

        merged = merge_feature_stats(feature_batches)
        print_feature_report(merged)
        report["feature_attention"] = merged

    # ---------------------------------------------------------------------
    # B) Normal vs shuffled LLM content
    # ---------------------------------------------------------------------
    if args.shuffle_val:
        normal = evaluate(
            model,
            handle.val_loader,
            spec,
            device,
            threshold,
            use_autocast,
            args.max_val_samples,
            "VAL NORMAL",
        )
        print_eval_result("VAL NORMAL", normal)
        verify_saved_baseline(
            ckpt,
            args.dataset,
            normal,
            args.baseline_tolerance,
        )

        with ShuffleLLMContent(model) as shuffler:
            shuffled = evaluate(
                model,
                handle.val_loader,
                spec,
                device,
                threshold,
                use_autocast,
                args.max_val_samples,
                "VAL SHUFFLE-LLM",
            )
            shuffled_batches = shuffler.shuffled_batches
            singleton_batches = shuffler.skipped_singleton_batches

        print_eval_result("VAL SHUFFLE-LLM", shuffled)
        print_shuffle_comparison(normal, shuffled)
        print(
            f"SHUFFLE-LLM batches: shuffled={shuffled_batches}, "
            f"singleton/no-op={singleton_batches}"
        )

        report["normal_val"] = {
            "samples": normal["samples"],
            "seconds": normal["seconds"],
            "scalars": scalar_metrics(normal),
        }
        report["shuffle_llm_val"] = {
            "samples": shuffled["samples"],
            "seconds": shuffled["seconds"],
            "scalars": scalar_metrics(shuffled),
            "shuffled_batches": shuffled_batches,
            "singleton_noop_batches": singleton_batches,
        }

    # ---------------------------------------------------------------------
    # C) Grad clipping + logit_scale audit
    # ---------------------------------------------------------------------
    if args.grad_audit_epochs > 0:
        grad_summary = run_grad_audit(
            model=model,
            handle=handle,
            spec=spec,
            criterion=criterion,
            device=device,
            use_autocast=use_autocast,
            epochs=args.grad_audit_epochs,
            max_updates=args.grad_audit_max_updates,
            print_every=args.grad_audit_print_every,
            grad_accum=grad_accum,
            configured_clip=configured_clip,
        )
        report["grad_audit"] = grad_summary

    if args.report is not None:
        path = args.report.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        print("JSON report:", path)

    print()
    print("=" * 132)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 132)
    print(
        "Do not choose a new architecture from one number alone. "
        "Use A to locate spatial-detail loss, B to test whether sample-specific "
        "LLM content matters, and C to decide whether clipping is routine or "
        "outlier-only."
    )


if __name__ == "__main__":
    main()
