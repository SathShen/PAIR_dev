#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR multi-dataset training for native Qwen3-VL DeepStack + 2D/3D adapters.

Box guidance supports JOINT teacher-forced Qwen grounding CE plus Mask Loss.
Weak training boxes come only from training masks/events; evaluation NEVER uses GT boxes.

Model construction lives in models/pair.py. This file owns only:
    config/runtime
    datasets + multi-dataset scheduling
    target collation
    loss / metrics
    optimizer / scheduler
    DDP
    checkpointing
    logging

CLI:
    python train.py --config configs/pair_train.json --datasets SECOND NYC-SCD

3D supervision:
    semantic_t1 / semantic_t2
    event_t1 / event_t2
    event_valid_t1 / event_valid_t2
There is no 3D binary change target/head. Traditional binary change metrics are
derived in metrics.py from event != unchanged.

3D checkpoint selection:
    Joint Semantic-Event F1 (Fjse), metric key: jse/F1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from datasets.config_loader import ExperimentConfig, load_experiment_config
from datasets.multi_dataset import DatasetRegistry, MultiDatasetScheduler
from loss import PAIRSemanticChangeLoss
from metrics import PAIRMetrics
from models.lora import (
    custom_lora_parameter_count,
    custom_lora_state_dict,
    load_custom_lora_state_dict,
    lora_parameter_count,
    lora_state_dict,
    load_lora_state_dict,
)
from models.pair import PAIRModel
from models.change_decoder import build_grounding_supervision


# =============================================================================
# CLI / runtime
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/pair_train_qwen4b.json"))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def build_settings(experiment: ExperimentConfig, cli):
    o = experiment.optimizer
    t = experiment.training
    v = experiment.validation
    lg = experiment.logging

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    folder_name =  f"{timestamp}_{experiment.experiment['name']}"
    if cli.output_dir is None:
        base_output_dir = Path(lg.get("output_dir", "outputs"))
    else:
        base_output_dir = Path(cli.output_dir)
    output_dir = os.path.join(base_output_dir, folder_name)


    return SimpleNamespace(
        lr=float(o.get("lr", 1e-4)),
        llm_lora_lr=float(o.get("llm_lora_lr", 2e-5)),
        vision_lora_lr=float(o.get("vision_lora_lr", 2e-5)),
        point_lora_lr=float(o.get("point_lora_lr", 1e-5)),
        main_weight_decay=float(o.get("main_weight_decay", 0.01)),
        llm_lora_weight_decay=float(o.get("llm_lora_weight_decay", 0.01)),
        vision_lora_weight_decay=float(o.get("vision_lora_weight_decay", 0.01)),
        point_lora_weight_decay=float(o.get("point_lora_weight_decay", 0.01)),
        scheduler=str(o.get("scheduler", "cosine")),
        warmup_ratio=float(o.get("warmup_ratio", 0.03)),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
        epochs=int(experiment.experiment["epochs"]),
        grad_accum=int(t.get("grad_accum", 1)),
        num_workers=int(t.get("num_workers", 4)),
        change_threshold=float(v.get("change_threshold", 0.5)),
        val_every_epochs=int(v.get("every_epochs", 1)),
        val_max_samples=int(v.get("max_samples", 0)),
        log_every=int(lg.get("log_every", 20)),
        save_every_epochs=int(lg.get("save_every_epochs", 1)),
        tensorboard=bool(lg.get("tensorboard", True)),
        output_dir=Path(output_dir),
        resume=cli.resume,
        seed=int(experiment.experiment.get("seed", 42)),
        train_box_mode=str(t.get("box_mode", "none")).strip().lower(),
        val_box_mode=str(v.get("box_mode", "none")).strip().lower(),
        box_ce_weight=float(t.get("box_ce_weight", 0.3)),
        box_gt_max_boxes=int(t.get("box_gt_max_boxes", 16)),
        box_dropout=float(t.get("box_dropout", 0.15)),
    )


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        rank = dist.get_rank()
    else:
        local_rank = rank = 0
        torch.cuda.set_device(0)
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": torch.device("cuda", local_rank),
        "is_main": rank == 0,
    }


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank=0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


# =============================================================================
# Logging
# =============================================================================

class TeeStdout:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, data):
        self.terminal.write(data)
        self.logfile.write(data)
        return len(data)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return self.terminal.isatty()


def make_writer(output_dir, enabled, is_main):
    if not enabled or not is_main:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard logging requires: pip install tensorboard") from exc
    return SummaryWriter(log_dir=str(output_dir / "tensorboard"))


def write_log_only(log_file, line):
    if log_file is None:
        return
    log_file.write(str(line) + "\n")
    log_file.flush()


def format_duration(seconds):
    total = int(round(max(float(seconds), 0.0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# =============================================================================
# Target collation
# =============================================================================

def _cat_target(samples, key):
    missing = [i for i, sample in enumerate(samples) if key not in sample["target"]]
    if missing:
        raise KeyError(f"Target key {key!r} missing from samples {missing}")
    return torch.cat([sample["target"][key].reshape(-1) for sample in samples], dim=0)


def _cat_valid_or_true(samples, value_key, valid_key):
    parts = []
    for sample in samples:
        target = sample["target"]
        value = target[value_key]
        valid = target.get(valid_key)
        if valid is None:
            valid = torch.ones_like(value, dtype=torch.bool)
        parts.append(valid.reshape(-1).bool())
    return torch.cat(parts, dim=0)


def _cat_optional_valid(samples, value_key, valid_key):
    if not any(valid_key in sample["target"] for sample in samples):
        return None
    return _cat_valid_or_true(samples, value_key, valid_key)


def merge_targets(samples, route):
    """
    Collate prepared PAIR targets without inventing supervision.

    2D stays on the existing semantic + binary-change protocol.
    3D uses semantic + event. event_valid is mandatory because T1/T2 temporal
    support is intentionally asymmetric. semantic_valid is emitted only when
    the dataset actually supplies it (i.e. ignored_id is configured).
    """
    if route == "2d":
        return {
            "semantic_t1": _cat_target(samples, "semantic_t1"),
            "semantic_t2": _cat_target(samples, "semantic_t2"),
            "change": _cat_target(samples, "change"),
            "semantic_valid_t1": _cat_valid_or_true(samples, "semantic_t1", "semantic_valid_t1"),
            "semantic_valid_t2": _cat_valid_or_true(samples, "semantic_t2", "semantic_valid_t2"),
            "change_valid": _cat_valid_or_true(samples, "change", "change_valid"),
        }
    if route == "3d":
        target = {
            "semantic_t1": _cat_target(samples, "semantic_t1"),
            "semantic_t2": _cat_target(samples, "semantic_t2"),
            "event_t1": _cat_target(samples, "event_t1").long(),
            "event_t2": _cat_target(samples, "event_t2").long(),
            "event_valid_t1": _cat_target(samples, "event_valid_t1").bool(),
            "event_valid_t2": _cat_target(samples, "event_valid_t2").bool(),
        }
        semantic_valid_t1 = _cat_optional_valid(samples, "semantic_t1", "semantic_valid_t1")
        semantic_valid_t2 = _cat_optional_valid(samples, "semantic_t2", "semantic_valid_t2")
        if semantic_valid_t1 is not None:
            target["semantic_valid_t1"] = semantic_valid_t1
        if semantic_valid_t2 is not None:
            target["semantic_valid_t2"] = semantic_valid_t2
        return target
    raise NotImplementedError("2D+3D target collation is deferred together with world-coordinate decoder wiring")


# =============================================================================
# Optional multi-box proposals (never inferred from validation ground truth)
# =============================================================================

def validate_box_modes(settings):
    if settings.train_box_mode not in {"none", "provided", "joint"}:
        raise ValueError("training.box_mode must be 'none', 'provided' or 'joint'")
    if settings.box_ce_weight < 0 or settings.box_ce_weight > 10:
        raise ValueError("box_ce_weight must be in [0,10]")
    if settings.box_gt_max_boxes < 1 or settings.box_gt_max_boxes > 64:
        raise ValueError("box_gt_max_boxes must be in [1,64]")
    if not 0 <= settings.box_dropout <= 1:
        raise ValueError("box_dropout must be in [0,1]")
    if settings.val_box_mode not in {"none", "provided", "generated"}:
        raise ValueError("validation.box_mode must be 'none', 'provided' or 'generated'")


def collate_box_proposals(samples, route, *, device, max_proposals):
    """Pad per-sample already-normalized 2D / world-XYZ 3D proposals.

    Required top-level sample key: boxes_2d [K,4] or boxes_3d [K,6].
    Optional top-level keys: box_valid [K] and box_scores [K].
    Proposals must be produced independently of *validation targets*.
    """
    key = {"2d": "boxes_2d", "3d": "boxes_3d"}.get(route)
    if key is None:
        raise NotImplementedError(f"Box collation does not support route={route}")
    d = 4 if route == "2d" else 6
    lengths, boxes_list, valid_list, score_list = [], [], [], []
    for i, sample in enumerate(samples):
        if key not in sample:
            raise KeyError(f"Sample {i} has no {key}; 'provided' box_mode requires "
                           "proposals in the dataset sample (not in target)")
        boxes = torch.as_tensor(sample[key], dtype=torch.float32, device=device)
        if boxes.ndim != 2 or boxes.shape[1] != d:
            raise ValueError(f"Sample {i} {key} must have shape [K,{d}]")
        k = boxes.shape[0]
        if k > max_proposals:
            raise ValueError(f"Sample {i} has K={k} boxes, max={max_proposals}; "
                             "select upstream instead of silently truncating")
        valid = torch.as_tensor(sample.get("box_valid", torch.ones(k, dtype=torch.bool)),
                                device=device)
        score = torch.as_tensor(sample.get("box_scores", torch.ones(k)),
                                dtype=torch.float32, device=device)
        if valid.dtype != torch.bool or valid.shape != (k,):
            raise ValueError(f"Sample {i} box_valid must be bool [K]")
        if score.shape != (k,) or not bool(torch.isfinite(score).all()) or \
                bool(((score < 0) | (score > 1)).any()):
            raise ValueError(f"Sample {i} box_scores must be finite [K] in [0,1]")
        if not bool(torch.isfinite(boxes).all()):
            raise ValueError(f"Sample {i} {key} contains nonfinite coordinates")
        if bool(valid.any()):
            kept = boxes[valid]
            if not bool((kept[:,d//2:] > kept[:,:d//2]).all()):
                raise ValueError(f"Sample {i} valid {key} must have min < max")
            if route == "2d" and bool(((kept < 0) | (kept > 1)).any()):
                raise ValueError("2D proposal coordinates must be normalized to [0,1]")
        lengths.append(k)
        boxes_list.append(boxes)
        valid_list.append(valid)
        score_list.append(score)
    max_k = max(lengths, default=0)
    padded = torch.zeros(len(samples), max_k, d, dtype=torch.float32, device=device)
    padded_valid = torch.zeros(len(samples), max_k, dtype=torch.bool, device=device)
    padded_scores = torch.zeros(len(samples), max_k, dtype=torch.float32, device=device)
    for i, k in enumerate(lengths):
        padded[i, :k] = boxes_list[i]
        padded_valid[i, :k] = valid_list[i]
        padded_scores[i, :k] = score_list[i]
    return {key: padded, "box_valid": padded_valid, "box_scores": padded_scores}


# =============================================================================
# Forward / loss / validation
# =============================================================================

def forward_loss(model, criterion, samples, spec, *, box_mode="none",
                 box_ce_weight=0.3, box_gt_max_boxes=16, box_dropout=0.15):
    """One multimodal Qwen forward + mask decoder and joint objective.

    For joint training, Qwen gets target boxes ONLY as causal assistant suffix.
    The TASK and T_mm positions precede GT text so Mask branch cannot read it.
    The mask decoder receives augmented GT regions as a teacher-guided spatial
    prior. This is joint optimization, NOT differentiable box-coordinate decoding.
    """
    if box_mode not in ("none", "provided", "generated", "joint"):
        raise ValueError(f"Invalid box_mode={box_mode!r}")
    if box_mode in ("generated", "joint") and not model.training and box_mode == "joint":
        raise RuntimeError("GT-derived Box supervision is TRAIN ONLY")
    if box_mode == "generated" and model.training:
        raise RuntimeError("Box generation is evaluation-only")
    base = unwrap(model)
    prompts = [sample["prompt"] for sample in samples]
    if box_mode == "provided":
        boxes = collate_box_proposals(
            samples, spec.route, device=next(base.parameters()).device,
            max_proposals=base.box_max_proposals,
        )
        grounding_targets = None
    elif box_mode == "joint":
        if box_gt_max_boxes > base.box_max_proposals:
            raise ValueError("training.box_gt_max_boxes exceeds model.box_guidance.max_proposals")
        guidance = build_grounding_supervision(
            samples, spec.route, max_boxes=box_gt_max_boxes,
            device=next(base.parameters()).device, box_dropout=box_dropout,
        )
        grounding_targets = guidance.pop("grounding_targets")
        guidance.pop("num_boxes")
        boxes = guidance
    else:
        boxes = {}
        grounding_targets = None
    common = {
        "prompts": prompts, "class_names": spec.class_names,
        "generate_boxes": box_mode == "generated",
        "grounding_targets": grounding_targets,
        **boxes,
    }
    if spec.route == "2d":
        output_sizes = [tuple(sample["target"]["change"].shape[-2:]) for sample in samples]
        prediction_mode = "bcd" if spec.label_mode == "binary" else "scd"
        result = model(
            images_t1=[sample["images_t1"] for sample in samples],
            images_t2=[sample["images_t2"] for sample in samples],
            output_sizes=output_sizes,
            prediction_mode=prediction_mode,
            **common,
        )
    elif spec.route == "3d":
        result = model(
            point_dicts_t1=[sample["point_dict_t1"] for sample in samples],
            point_dicts_t2=[sample["point_dict_t2"] for sample in samples],
            **common,
        )
    else:
        raise NotImplementedError("PAIR 2D+3D requires calibrated image/point correspondence")
    if box_mode == "joint":
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Joint PAIRModel must return (PredictionLogits, grounding_ce)")
        prediction, grounding_ce = result
        if grounding_ce.ndim != 0 or not torch.isfinite(grounding_ce):
            raise ValueError("Grounding CE must be a finite scalar")
    else:
        prediction = result
        grounding_ce = None
    target = merge_targets(samples, spec.route)
    semantic_changed_only = spec.route == "2d" and spec.label_mode == "semantic_pair"
    loss_output = criterion(
        prediction=prediction, target=target,
        class_names=spec.class_names,
        semantic_changed_only=semantic_changed_only,
    )
    if grounding_ce is not None:
        loss_output.grounding_ce = grounding_ce
        loss_output.mask_total = loss_output.total
        loss_output.total = loss_output.total + box_ce_weight * grounding_ce
    return prediction, loss_output, target


def all_reduce_loss_sums(sums, count, device):
    keys = sorted(sums)
    tensor = torch.tensor([sums[k] for k in keys] + [count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = max(float(tensor[-1].item()), 1.0)
    return {key: float(tensor[i].item() / count) for i, key in enumerate(keys)}


@torch.no_grad()
def validate(model, criterion, loader, spec, runtime, settings):
    model.eval()
    evaluator = PAIRMetrics(spec.class_names, runtime["device"], settings.change_threshold)
    sums, count = {}, 0
    start = time.time()
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"VAL {spec.name}",
        dynamic_ncols=True,
        leave=True,
        disable=not runtime["is_main"],
    )
    for samples in progress:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, loss_output, merged_target = forward_loss(model, criterion, samples, spec, box_mode=settings.val_box_mode)
        evaluator.update(prediction, merged_target)
        batch_n = len(samples)
        for key, value in loss_output.as_dict().items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_n
        count += batch_n
        if runtime["is_main"]:
            progress.set_postfix(samples=count, refresh=False)
        if settings.val_max_samples > 0 and count >= settings.val_max_samples:
            break
    evaluator.reduce_distributed()
    losses = all_reduce_loss_sums(sums, count, runtime["device"])
    result = evaluator.compute()
    result["losses"] = losses
    result["seconds"] = time.time() - start
    model.train()
    return result


# =============================================================================
# Optimizer / scheduler
# =============================================================================

def _optimizer_group_name(parameter_name: str) -> str:
    name = parameter_name.lower()

    if "pair_lora_" in name:
        if "qwen_backbone" in name and "visual" in name:
            return "vision_lora"
        if "point_encoder" in name:
            return "point_lora"
        raise RuntimeError(
            "Custom PAIR LoRA parameter could not be assigned to a backbone group: "
            f"{parameter_name}"
        )

    # PEFT Qwen LLM adapters.  Custom adapters are handled above.
    if "lora_" in name:
        return "llm_lora"

    return "main"


def active_model_flags(experiment):
    """Derive route-aware model construction from the selected datasets."""
    specs = [experiment.datasets[name].spec for name in experiment.selected_names]
    enable_2d = any(spec.route in {"2d", "2d3d"} for spec in specs)
    enable_3d = any(spec.route in {"3d", "2d3d"} for spec in specs)
    # binary-only datasets do not need semantic prototypes/logits; semantic_pair
    # and post_semantic both do.
    enable_semantic = any(spec.label_mode != "binary" for spec in specs)
    if not (enable_2d or enable_3d):
        raise RuntimeError("Selected datasets do not activate any PAIR route")
    return {
        "enable_2d": enable_2d,
        "enable_3d": enable_3d,
        "enable_semantic": enable_semantic,
    }


def build_optimizer(model, settings):
    named_groups = {
        "main": [],
        "llm_lora": [],
        "vision_lora": [],
        "point_lora": [],
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = _optimizer_group_name(name)
        named_groups[group_name].append((name, parameter))

    group_hparams = {
        "main": (settings.lr, settings.main_weight_decay),
        "llm_lora": (settings.llm_lora_lr, settings.llm_lora_weight_decay),
        "vision_lora": (settings.vision_lora_lr, settings.vision_lora_weight_decay),
        "point_lora": (settings.point_lora_lr, settings.point_lora_weight_decay),
    }

    groups = []
    group_params = {}
    for group_name in ("main", "llm_lora", "vision_lora", "point_lora"):
        entries = named_groups[group_name]
        if not entries:
            continue
        params = [parameter for _, parameter in entries]
        lr, wd = group_hparams[group_name]
        groups.append(
            {
                "params": params,
                "lr": float(lr),
                "weight_decay": float(wd),
                "name": group_name,
            }
        )
        group_params[group_name] = params

    if not groups:
        raise RuntimeError("PAIR has no trainable parameters")

    # Learning rate and weight decay are explicit for every optimizer group.
    optimizer = torch.optim.AdamW(groups)
    return optimizer, group_params, named_groups



def optimizer_group_snapshot(optimizer):
    return {
        group.get("name", str(index)): {
            "lr": float(group["lr"]),
            "base_lr": float(group.get("initial_lr", group["lr"])),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "params": sum(p.numel() for p in group["params"]),
            "tensors": len(group["params"]),
        }
        for index, group in enumerate(optimizer.param_groups)
    }


def _architecture_module_bucket(parameter_name: str) -> str:
    """Group real PAIR modules for the updated Qwen/Adapter/Decoder topology."""
    name = parameter_name.lower()
    if name.startswith("backbone.qwen_backbone."):
        return "Qwen Vision" if ".visual." in name else "Qwen LLM"
    if name.startswith("backbone.point_encoder."):
        return "Utonia"
    if name.startswith("backbone.point_adapter."):
        return "PointAdapter"
    if name.startswith("image_adapter."):
        return "ImageAdapter"
    if name.startswith("decoder.class_encoder."):
        return "Class/Task Encoder"
    if name.startswith("decoder.query_decoder."):
        return "Shared Query Decoder"
    if name.startswith("decoder.image_semantic_head."):
        return "2D Semantic Head"
    if name.startswith("decoder.image_change_head."):
        return "2D Change Head"
    if name.startswith("decoder.point_semantic_head."):
        return "3D Semantic Head"
    if name.startswith("decoder.point_event_head.") or name.startswith("decoder.event_prototypes"):
        return "3D Event Head"
    if name.startswith("decoder.box_guidance."):
        return "Multi-box Guidance"
    if name.startswith("decoder."):
        return "Temporal Fusion"
    return f"Other ({parameter_name.split('.', 1)[0]})"


def _lora_rank_from_entries(entries):
    """Infer LoRA rank from an A matrix when possible."""
    for name, parameter in entries:
        lname = name.lower()
        if "pair_lora_a" in lname or ".lora_a." in lname:
            if parameter.ndim >= 1:
                return int(parameter.shape[0])
    return None


def _format_state(total_params, trainable_params, has_lora):
    frozen_params = total_params - trainable_params
    if trainable_params == 0:
        return "frozen"
    if frozen_params == 0:
        return "trainable"
    if has_lora:
        return "base frozen + LoRA"
    return "partially trainable"


def print_trainable_parameter_report(model, optimizer):
    """
    One consolidated architecture table.

    For every large PAIR module it shows:
      - whether LoRA is present,
      - frozen/trainable state,
      - total/trainable/frozen parameter counts,
      - optimizer group,
      - configured/base LR.

    No individual tensor names are printed.
    """
    base_model = unwrap(model)

    # Optimizer ownership and hyperparameters.
    group_by_id = {}
    group_hparams = {}
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", str(index))
        group_hparams[group_name] = {
            # HuggingFace schedulers preserve the optimizer's original LR in
            # initial_lr.  During warmup current lr can legitimately be 0.
            "base_lr": float(group.get("initial_lr", group["lr"])),
            "current_lr": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
        }
        for parameter in group["params"]:
            pid = id(parameter)
            if pid in group_by_id:
                raise RuntimeError(
                    "A trainable parameter appears in multiple optimizer groups"
                )
            group_by_id[pid] = group_name

    # Exclusive architecture buckets over ALL parameters, not only trainable
    # ones, so total/frozen counts are meaningful.
    modules = {}
    total_model_params = 0
    total_trainable_params = 0

    for name, parameter in base_model.named_parameters():
        module_name = _architecture_module_bucket(name)
        info = modules.setdefault(
            module_name,
            {
                "total": 0,
                "trainable": 0,
                "entries": [],
                "trainable_groups": set(),
                "has_lora": False,
            },
        )

        numel = parameter.numel()
        info["total"] += numel
        info["entries"].append((name, parameter))
        total_model_params += numel

        lname = name.lower()
        if "pair_lora_" in lname or (
            "lora_" in lname and "pair_lora_" not in lname
        ):
            info["has_lora"] = True

        if parameter.requires_grad:
            info["trainable"] += numel
            total_trainable_params += numel

            group_name = group_by_id.get(id(parameter))
            if group_name is None:
                raise RuntimeError(
                    f"Trainable parameter is missing from optimizer: {name}"
                )
            info["trainable_groups"].add(group_name)

    preferred_order = (
        "Qwen Vision", "Utonia", "PointAdapter", "Qwen LLM", "ImageAdapter",
        "Class/Task Encoder", "Shared Query Decoder", "Temporal Fusion",
        "Multi-box Guidance", "2D Semantic Head", "2D Change Head",
        "3D Semantic Head", "3D Event Head",
    )
    ordered_names = [name for name in preferred_order if name in modules]
    ordered_names += sorted(
        name for name in modules
        if name not in preferred_order
    )

    width = 136
    print("=" * width)
    print("PAIR MODEL / TRAINING PARAMETER SUMMARY")
    print("=" * width)
    print(
        f"{'module':<26}"
        f"{'tuning':<14}"
        f"{'state':<20}"
        f"{'total(M)':>11}"
        f"{'train(M)':>11}"
        f"{'frozen(M)':>12}"
        f"{'optimizer':>16}"
        f"{'base_lr':>14}"
    )
    print("-" * width)

    for module_name in ordered_names:
        info = modules[module_name]
        total_params = int(info["total"])
        trainable_params = int(info["trainable"])
        frozen_params = total_params - trainable_params

        rank = _lora_rank_from_entries(info["entries"])
        if info["has_lora"]:
            tuning = f"LoRA(r={rank})" if rank is not None else "LoRA"
        else:
            tuning = "-"

        state = _format_state(
            total_params,
            trainable_params,
            info["has_lora"],
        )

        groups = sorted(info["trainable_groups"])
        if not groups:
            optimizer_name = "-"
            base_lr_text = "-"
        elif len(groups) == 1:
            optimizer_name = groups[0]
            hp = group_hparams[optimizer_name]
            base_lr_text = f"{hp['base_lr']:.3e}"
        else:
            optimizer_name = ",".join(groups)
            base_lr_text = "mixed"

        print(
            f"{module_name:<26}"
            f"{tuning:<14}"
            f"{state:<20}"
            f"{total_params / 1e6:>11.3f}"
            f"{trainable_params / 1e6:>11.3f}"
            f"{frozen_params / 1e6:>12.3f}"
            f"{optimizer_name:>16}"
            f"{base_lr_text:>14}"
        )

    print("-" * width)
    total_frozen_params = total_model_params - total_trainable_params
    print(
        f"{'TOTAL':<26}"
        f"{'-':<14}"
        f"{'':<20}"
        f"{total_model_params / 1e6:>11.3f}"
        f"{total_trainable_params / 1e6:>11.3f}"
        f"{total_frozen_params / 1e6:>12.3f}"
    )
    print("=" * width)
    print()

def build_scheduler(optimizer, total_updates, warmup_ratio, kind):
    warmup = int(round(total_updates * warmup_ratio))
    if kind == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_updates)
    if kind == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    raise ValueError("optimizer.scheduler must be 'cosine' or 'constant'")


# =============================================================================
# Checkpoints
# =============================================================================

CHECKPOINT_FORMAT_VERSION = 3
CHECKPOINT_ARCHITECTURE = "PAIR-Qwen3VL-native-DeepStack-query-multibox-v1"
FOUNDATION_STATE_PREFIXES = (
    "backbone.qwen_backbone.model.",
    "backbone.point_encoder.model.",
)


def _is_foundation_state_name(name):
    return name.startswith(FOUNDATION_STATE_PREFIXES)


def pair_checkpoint_state(model):
    """Save adapters, decoder, and every trainable non-foundation parameter.

    Qwen/Utonia LoRA is saved separately. If full Qwen tuning is enabled,
    include its trainable weights as well; never silently discard them.
    Frozen original foundation weights remain sourced from their checkpoints.
    """
    base = unwrap(model)
    include_full_qwen = getattr(base, "qwen_tuning", "lora") == "full"
    state = base.state_dict()
    trainable = {
        name for name, param in base.named_parameters()
        if param.requires_grad and (not _is_foundation_state_name(name)
                                    or (include_full_qwen and name.startswith("backbone.qwen_backbone.model.")))
    }
    buffers = {
        name for name, _ in base.named_buffers()
        if not _is_foundation_state_name(name)
        or (include_full_qwen and name.startswith("backbone.qwen_backbone.model."))
    }
    return {name: value.detach().cpu() for name, value in state.items()
            if name in trainable or name in buffers}


def load_model_state_from_checkpoint(ckpt, model):
    """Strict resume for this architecture; older PAIR versions need conversion.

    Do not silently accept missing/newly initialized modules while restoring an
    optimizer state, since that would masquerade as a faithful training resume.
    """
    if ckpt.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION or \
            ckpt.get("architecture") != CHECKPOINT_ARCHITECTURE:
        raise RuntimeError(
            "Checkpoint does not match the rebuilt PAIR architecture. "
            "Old Pyramid/decoder checkpoints cannot be resumed with this train.py; "
            "start a new run or write an explicit weight-conversion script."
        )
    base = unwrap(model)
    saved = ckpt.get("pair_state")
    if not isinstance(saved, dict):
        raise RuntimeError("Missing pair_state in checkpoint")
    expected = pair_checkpoint_state(base)
    missing = set(expected) - set(saved)
    unexpected = set(saved) - set(expected)
    bad_shapes = [k for k in expected.keys() & saved.keys()
                  if tuple(expected[k].shape) != tuple(saved[k].shape)]
    if missing or unexpected or bad_shapes:
        raise RuntimeError("PAIR checkpoint state mismatch: "
                           f"missing={sorted(missing)[:12]}, "
                           f"unexpected={sorted(unexpected)[:12]}, "
                           f"shape_mismatch={sorted(bad_shapes)[:12]}")
    current = base.state_dict()
    current.update(saved)
    base.load_state_dict(current, strict=True)
    load_lora_state_dict(base.qwen_backbone.model, ckpt.get("lora", {}))
    load_custom_lora_state_dict(base.qwen_backbone.model,
                               ckpt.get("vision_lora", {}), kind="vision")
    if base.point_encoder is not None:
        load_custom_lora_state_dict(base.point_encoder.model,
                                   ckpt.get("point_lora", {}), kind="utonia")
    return False


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    update_in_epoch,
    optimizer_step,
    experiment,
    resolved_config,
    dataset_best_values=None,
    dataset_best_epochs=None,
    validation_selection=None,
    validation_metrics=None,
):
    base = unwrap(model)
    checkpoint = {
        "epoch": int(epoch),
        "update_in_epoch": int(update_in_epoch),
        "optimizer_step": int(optimizer_step),
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "pair_state": pair_checkpoint_state(model),
        "lora": lora_state_dict(base.qwen_backbone.model),
        "vision_lora": custom_lora_state_dict(
            base.qwen_backbone.model, kind="vision"
        ),
        "point_lora": (
            custom_lora_state_dict(base.point_encoder.model, kind="utonia")
            if base.point_encoder is not None
            else {}
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dataset_best_values": dict(dataset_best_values or {}),
        "dataset_best_epochs": dict(dataset_best_epochs or {}),
        "validation_selection": validation_selection,
        "validation_metrics": validation_metrics,
        "config": resolved_config,
        "config_hash": experiment.hash_resolved(resolved_config),
        "selected_datasets": list(experiment.selected_names),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, *,
                    selected_datasets=None, model_config=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if selected_datasets is not None and tuple(ckpt.get("selected_datasets", ())) != tuple(selected_datasets):
        raise RuntimeError("Resume selected_datasets differ from the checkpoint: "
                           f"saved={ckpt.get('selected_datasets')}, "
                           f"current={list(selected_datasets)}")
    saved_model_cfg = ckpt.get("config", {}).get("model")
    if model_config is not None and saved_model_cfg != dict(model_config):
        raise RuntimeError("Resume model configuration differs from checkpoint; "
                           "architecture/weights or training setup may differ")
    load_model_state_from_checkpoint(ckpt, model)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("update_in_epoch", 0)),
        int(ckpt.get("optimizer_step", 0)),
        {str(k): float(v) for k, v in ckpt.get("dataset_best_values", {}).items()},
        {str(k): int(v) for k, v in ckpt.get("dataset_best_epochs", {}).items()},
    )


def safe_checkpoint_token(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-_.")
    return value or "Dataset"


def selection_metric_for_spec(spec):
    if spec.route == "2d" and spec.label_mode == "semantic_pair":
        return "scd/F_scd", "Fscd"
    if spec.route in {"3d", "2d3d"} and spec.label_mode == "semantic_pair":
        return "jse/F1", "Fjse"
    if spec.label_mode in {"binary", "post_semantic"}:
        return "change/IoU", "IoU"
    raise ValueError(f"No checkpoint selection rule for route={spec.route!r}, label_mode={spec.label_mode!r}")


def selection_from_results(experiment, results_by_dataset):
    selection = {}
    for name in experiment.selected_names:
        if name not in results_by_dataset:
            continue
        spec = experiment.datasets[name].spec
        metric_key, metric_label = selection_metric_for_spec(spec)
        scalars = results_by_dataset[name]["scalars"]
        if metric_key not in scalars:
            raise KeyError(f"{name}: validation metric {metric_key!r} missing; available={sorted(scalars)}")
        selection[name] = {
            "metric_key": metric_key,
            "metric_label": metric_label,
            "value": float(scalars[metric_key]),
        }
    return selection


def epoch_checkpoint_name(epoch, experiment, selection):
    parts = [f"Ep{int(epoch):03d}"]
    for name in experiment.selected_names:
        item = selection.get(name)
        if item is not None:
            parts.append(f"{safe_checkpoint_token(name)}{item['metric_label']}{item['value']:.4f}")
    if len(parts) == 1:
        parts.append("NoVal")
    return "_".join(parts) + ".pt"


def valbest_checkpoint_name(dataset_name, epoch_name):
    return f"ValBest_{safe_checkpoint_token(dataset_name)}_{epoch_name}"


def remove_old_current_checkpoints(output_dir, keep_path):
    keep_path = Path(keep_path).resolve()
    removed = []
    for path in Path(output_dir).glob("Ep*.pt"):
        if path.resolve() == keep_path:
            continue
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def replace_dataset_valbest(output_dir, dataset_name, keep_path):
    keep_path = Path(keep_path).resolve()
    prefix = f"ValBest_{safe_checkpoint_token(dataset_name)}_"
    removed = []
    for path in Path(output_dir).glob(f"{prefix}*.pt"):
        if path.resolve() == keep_path:
            continue
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


# =============================================================================
# Metrics / TensorBoard
# =============================================================================

def validation_metric_layout(spec, scalars):
    if spec.route == "3d":
        candidates = (
            ("Fjse", "jse/F1"),
            ("JSEP", "jse/Precision"),
            ("JSER", "jse/Recall"),
            ("SemOA", "semantic/OA"),
            ("SemIoU", "semantic/mIoU"),
            ("EvtOA", "event/OA"),
            ("EvtF1", "event/mF1"),
            ("EvtIoU", "event/mIoU"),
            ("ChgF1", "change/F1"),
            ("ChgIoU", "change/IoU"),
        )
        return tuple(item for item in candidates if item[1] in scalars)
    if spec.label_mode in {"binary", "post_semantic"}:
        return (
            ("OA", "change/OA"),
            ("IoU", "change/IoU"),
            ("Recall", "change/Recall"),
            ("Precision", "change/Precision"),
            ("F1", "change/F1"),
        )
    if spec.label_mode == "semantic_pair":
        scd_metrics = (
            ("OA", "scd/OA"),
            ("SeK", "scd/SeK"),
            ("F_scd", "scd/F_scd"),
            ("mIoU", "scd/mIoU"),
        )
        if all(key in scalars for _, key in scd_metrics):
            return scd_metrics
        return tuple(
            item
            for item in (("OA", "semantic/OA"), ("mIoU", "semantic/mIoU"))
            if item[1] in scalars
        )
    return ()


def log_tensorboard_train(writer, values, step, dataset_name):
    """
    TensorBoard train logging:
        - only total train loss for each dataset
    """
    if writer is None:
        return

    value = values.get("loss")
    if isinstance(value, (int, float)):
        writer.add_scalar(f"train/{dataset_name}/loss", value, step)


def log_tensorboard_val(writer, result, epoch, dataset_name, spec):
    """
    TensorBoard validation logging is intentionally minimal:

    2D SCD (SECOND / LandsatSCD):
        Fscd

    3D SCD (NYC-SCD):
        SemIoU
        EvtIoU
        Fjse

    2D BCD (LEVIR-CD):
        F1

    Validation loss and all auxiliary metrics remain available in run.log /
    console/checkpoints, but are not written to TensorBoard.
    """
    if writer is None:
        return

    scalars = result["scalars"]

    if spec.route == "3d":
        selected = (
            ("SemIoU", "semantic/mIoU"),
            ("EvtIoU", "event/mIoU"),
            ("Fjse", "jse/F1"),
        )
    elif spec.label_mode in {"binary", "post_semantic"}:
        selected = (
            ("F1", "change/F1"),
        )
    elif spec.label_mode == "semantic_pair":
        selected = (
            ("Fscd", "scd/F_scd"),
        )
    else:
        selected = ()

    for display_name, key in selected:
        value = scalars.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(
                f"val/{dataset_name}/{display_name}",
                value,
                epoch,
            )


def print_val(dataset_name, result, spec):
    scalars = result["scalars"]
    loss = result["losses"].get("loss", float("nan"))
    fields = [
        f"{name}={scalars[key]:.4f}"
        for name, key in validation_metric_layout(spec, scalars)
        if key in scalars
    ]
    suffix = " | " + " ".join(fields) if fields else ""
    print(f"VAL [{dataset_name}] | loss={loss:.4f}{suffix}")


def iteration_loss_string(means, route):
    base = (
        f"loss={means['loss']:.6f} "
        f"sem1={means['loss_semantic_t1']:.6f} "
        f"sem2={means['loss_semantic_t2']:.6f}"
    )
    if route == "3d":
        return (
            base
            + f" event1={means['loss_event_t1']:.6f}"
            + f" event2={means['loss_event_t2']:.6f}"
            + f" event={means['loss_event']:.6f}"
        )
    return (
        base
        + f" bce={means['loss_change_bce']:.6f}"
        + f" dice={means['loss_change_dice']:.6f}"
    )


def window_loss_string(means):
    fields = [
        f"avg_loss={means['loss']:.4f}",
        f"avg_sem1={means['loss_semantic_t1']:.4f}",
        f"avg_sem2={means['loss_semantic_t2']:.4f}",
    ]
    # A mixed 2D/3D window has zeros for inactive branches. Showing both keeps
    # the global window truthful without guessing which route dominated.
    if "loss_change_bce" in means:
        fields.append(f"avg_bce={means['loss_change_bce']:.4f}")
    if "loss_change_dice" in means:
        fields.append(f"avg_dice={means['loss_change_dice']:.4f}")
    if "loss_event" in means:
        fields.append(f"avg_event={means['loss_event']:.4f}")
    return " ".join(fields)


# =============================================================================
# Main
# =============================================================================

def main():
    run_start = time.time()
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    runtime = setup_distributed()
    writer = None
    log_file = None
    original_stdout = sys.stdout
    run_completed = False

    try:
        experiment = load_experiment_config(cli.config, cli.datasets)
        settings = build_settings(experiment, cli)
        if settings.grad_accum < 1:
            raise ValueError("training.grad_accum must be >= 1")
        validate_box_modes(settings)

        set_seed(settings.seed, runtime["rank"])
        if runtime["is_main"]:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            log_path = settings.output_dir / "run.log"
            log_file = log_path.open("a", encoding="utf-8", buffering=1)
            sys.stdout = TeeStdout(original_stdout, log_file)
            print()
            print("=" * 96)
            print("PAIR RUN LOG")
            print("=" * 96)
            print("Started:", time.strftime("%Y-%m-%d %H:%M:%S"))
            print("Log file:", log_path)
            if settings.resume is not None:
                print("Resume:", settings.resume)
            print()

        if any(experiment.datasets[n].spec.route == "2d3d" for n in experiment.selected_names):
            raise NotImplementedError("2D+3D joint training requires calibrated image/point geometry")
        registry = DatasetRegistry(experiment, runtime, num_workers=settings.num_workers)
        dataset_scheduler = MultiDatasetScheduler(experiment, registry, settings.grad_accum)
        updates_per_epoch = dataset_scheduler.updates_per_epoch
        total_updates = updates_per_epoch * settings.epochs

        # Construct only the modality-specific branches required by the
        # selected datasets.  Cross-Attn + Unified Decoder remain shared and
        # are always constructed for every active route.
        model_flags = active_model_flags(experiment)
        model = PAIRModel.from_config(
            experiment.model,
            runtime["device"],
            **model_flags,
        )
        if settings.train_box_mode == "provided" and settings.val_box_mode == "generated":
            print("NOTE: training uses provided proposals but validation uses predicted boxes; "
                  "check proposal distribution shift.")
        if runtime["distributed"]:
            model = DDP(
                model,
                device_ids=[runtime["local_rank"]],
                output_device=runtime["local_rank"],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        criterion = PAIRSemanticChangeLoss().to(runtime["device"])
        optimizer, _, _ = build_optimizer(model, settings)
        scheduler = build_scheduler(optimizer, total_updates, settings.warmup_ratio, settings.scheduler)

        start_epoch = 0
        start_update_in_epoch = 0
        optimizer_step = 0
        dataset_best_values = {name: -float("inf") for name in experiment.selected_names}
        dataset_best_epochs = {name: 0 for name in experiment.selected_names}

        if settings.resume is not None:
            old_cfg = torch.load(settings.resume, map_location="cpu", weights_only=False).get("config", {})
            old_mode = str(old_cfg.get("training", {}).get("box_mode", "none")).lower()
            if (old_mode == "joint") != (settings.train_box_mode == "joint"):
                raise RuntimeError("Refusing resume across joint/non-joint Box training objectives; "
                                   "start a new run or explicitly migrate optimizer state")
            (
                start_epoch,
                start_update_in_epoch,
                optimizer_step,
                loaded_best_values,
                loaded_best_epochs,
            ) = load_checkpoint(
                settings.resume, model, optimizer, scheduler,
                selected_datasets=experiment.selected_names,
                model_config=experiment.model,
            )
            dataset_best_values.update(loaded_best_values)
            dataset_best_epochs.update(loaded_best_epochs)

        dataset_runtime = registry.runtime_summary(settings.grad_accum)
        for name, info in dataset_runtime.items():
            info["effective_global_batch"] = (
                experiment.datasets[name].per_gpu_batch_size
                * runtime["world_size"]
                * settings.grad_accum
            )

        runtime_config = {
            "world_size": runtime["world_size"],
            "updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates": total_updates,
            "grad_accum": settings.grad_accum,
            "dataset_epoch_plan": dataset_scheduler.summary(),
            "datasets": dataset_runtime,
        }
        resolved_config = experiment.resolved_dict(runtime=runtime_config)

        if runtime["is_main"]:
            writer = make_writer(settings.output_dir, settings.tensorboard, True)
            # Preserve user-authored config exactly; resolved config is separate.
            source_config_text = experiment.path.read_text(encoding="utf-8")
            (settings.output_dir / "config.json").write_text(source_config_text, encoding="utf-8")
            (settings.output_dir / "config_resolved.json").write_text(
                json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            base = unwrap(model)
            print("=" * 96)
            print("PAIR MULTI-DATASET TRAINING")
            print("=" * 96)
            print("Experiment:", experiment.experiment["name"])
            print(f"Box guidance: train={settings.train_box_mode}, val={settings.val_box_mode}")
            if settings.train_box_mode == "none":
                print("NOTE: Box-guided Decoder weights receive no Box-specific "
                      "training signal without provided proposals.")
            if settings.train_box_mode == "joint":
                print(f"JOINT Grounding CE + Mask Loss: weight={settings.box_ce_weight}, "
                      f"max_boxes={settings.box_gt_max_boxes}, box_dropout={settings.box_dropout}")
                print("NOTE: mask decoder uses teacher-guided regions during training; "
                      "validation must use generated boxes or no boxes, never GT")
            print("Datasets:", ", ".join(experiment.selected_names))
            print(
                "Active model routes:",
                f"2D={model_flags['enable_2d']}",
                f"3D={model_flags['enable_3d']}",
                f"semantic={model_flags['enable_semantic']}",
            )
            print()
            print_trainable_parameter_report(model, optimizer)
            print("GPUs:", runtime["world_size"])
            print("Gradient accumulation:", settings.grad_accum)
            print("Updates/epoch:", updates_per_epoch)
            print("Total optimizer updates:", total_updates)
            print("Automatic dataset epoch plan:", dataset_scheduler.summary())
            print()
            for name in experiment.selected_names:
                info = dataset_runtime[name]
                print(
                    f"  {name}: route={info['route']} "
                    f"train={info['train_samples']} val={info['val_samples']} "
                    f"per_gpu_batch={info['per_gpu_batch_size']} "
                    f"effective_global_batch={info['effective_global_batch']} "
                    f"updates={info['optimizer_updates_per_epoch']} "
                    f"fraction={info['update_fraction']:.3f}"
                )
            if writer is not None:
                print("TensorBoard:", settings.output_dir / "tensorboard")
            print()

        model.train()
        log_window_dataset_counts = Counter()
        log_window_sums = {}
        log_window_count = 0
        log_window_dataset_sums = {}

        for epoch in range(start_epoch, settings.epochs):
            epoch_start = time.time()
            registry.reset_epoch(epoch)
            schedule = dataset_scheduler.epoch_schedule(epoch, runtime)
            schedule_counts = {
                name: sum(1 for item in schedule if item.dataset_name == name)
                for name in experiment.selected_names
            }
            first_update = start_update_in_epoch if epoch == start_epoch else 0
            if first_update > 0:
                registry.consume_updates(schedule[:first_update])

            if runtime["is_main"]:
                state = optimizer_group_snapshot(optimizer)
                group_text = " ".join(
                    f"{name}:lr={info['lr']:.8e},wd={info['weight_decay']:.8e}"
                    for name, info in state.items()
                )
                print(f"EPOCH {epoch+1:03d} OPTIMIZER | {group_text}")
                write_log_only(log_file, f"EPOCH {epoch+1:03d} OPTIMIZER | {group_text}")

            for update_idx in range(first_update, updates_per_epoch):
                update = schedule[update_idx]
                dataset_name = update.dataset_name
                accumulation_steps = update.microbatches
                handle = registry.handles[dataset_name]
                spec = handle.config.spec
                optimizer.zero_grad(set_to_none=True)
                update_sums = {}
                update_samples = 0
                update_start = time.time()

                for micro_idx in range(accumulation_steps):
                    samples = registry.next_train_batch(dataset_name)
                    update_samples += len(samples)
                    sync_context = contextlib.nullcontext()
                    if isinstance(model, DDP) and micro_idx + 1 < accumulation_steps:
                        sync_context = model.no_sync()
                    with sync_context:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, loss_output, _ = forward_loss(
                                model, criterion, samples, spec,
                                box_mode=settings.train_box_mode,
                                box_ce_weight=settings.box_ce_weight,
                                box_gt_max_boxes=settings.box_gt_max_boxes,
                                box_dropout=settings.box_dropout,
                            )
                            loss = loss_output.total / accumulation_steps
                        loss.backward()
                    for key, value in loss_output.as_dict().items():
                        update_sums[key] = update_sums.get(key, 0.0) + float(value.detach().cpu())

                trainable = [p for p in model.parameters() if p.requires_grad]
                if settings.max_grad_norm > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(trainable, settings.max_grad_norm).detach().cpu()
                    )
                else:
                    grad_norm = float("nan")

                optimizer.step()
                scheduler.step()
                optimizer_step += 1

                means = {key: value / accumulation_steps for key, value in update_sums.items()}
                optimizer_state = optimizer_group_snapshot(optimizer)
                lrs = {name: info["lr"] for name, info in optimizer_state.items()}
                wds = {name: info["weight_decay"] for name, info in optimizer_state.items()}
                elapsed = time.time() - update_start
                samples_per_sec = update_samples * runtime["world_size"] / max(elapsed, 1e-6)

                if runtime["is_main"]:
                    write_log_only(
                        log_file,
                        (
                            f"ITER E{epoch+1:03d} U{optimizer_step:06d} "
                            f"dataset={dataset_name} route={handle.config.route} "
                            f"update_in_epoch={update_idx+1} accu={accumulation_steps} "
                            f"{iteration_loss_string(means, spec.route)} "
                            f"grad={grad_norm:.6f} "
                            f"lr_main={lrs.get('main', 0.0):.8e} wd_main={wds.get('main', 0.0):.8e} "
                            f"lr_llm_lora={lrs.get('llm_lora', 0.0):.8e} wd_llm_lora={wds.get('llm_lora', 0.0):.8e} "
                            f"lr_vision_lora={lrs.get('vision_lora', 0.0):.8e} wd_vision_lora={wds.get('vision_lora', 0.0):.8e} "
                            f"lr_point_lora={lrs.get('point_lora', 0.0):.8e} wd_point_lora={wds.get('point_lora', 0.0):.8e} "
                            f"sample_per_s={samples_per_sec:.4f} "
                            f"gpu_alloc_GiB={torch.cuda.memory_allocated() / 1024**3:.4f} "
                            f"gpu_reserved_GiB={torch.cuda.memory_reserved() / 1024**3:.4f}"
                        ),
                    )

                log_window_dataset_counts[dataset_name] += 1
                log_window_count += 1
                for key, value in means.items():
                    log_window_sums[key] = log_window_sums.get(key, 0.0) + float(value)

                dataset_sums = log_window_dataset_sums.setdefault(dataset_name, {})
                for key, value in means.items():
                    dataset_sums[key] = dataset_sums.get(key, 0.0) + float(value)

                if optimizer_step % settings.log_every == 0:
                    if runtime["is_main"]:
                        window_means = {
                            key: value / max(log_window_count, 1)
                            for key, value in log_window_sums.items()
                        }
                        mix = " ".join(
                            f"{name} x{log_window_dataset_counts[name]}"
                            for name in experiment.selected_names
                            if log_window_dataset_counts[name] > 0
                        )
                        lr_text = " ".join(
                            f"{name}={info['lr']:.3e}/wd={info['weight_decay']:.2e}"
                            for name, info in optimizer_state.items()
                        )
                        print(
                            f"E{epoch+1:03d} U{optimizer_step:06d} [{mix}] | "
                            f"{window_loss_string(window_means)} | {lr_text}"
                        )
                        if writer is not None:
                            for group_name, info in optimizer_state.items():
                                writer.add_scalar(
                                    f"train/optimizer/{group_name}_lr",
                                    info["lr"],
                                    optimizer_step,
                                )
                                writer.add_scalar(
                                    f"train/optimizer/{group_name}_weight_decay",
                                    info["weight_decay"],
                                    optimizer_step,
                                )
                        for name in experiment.selected_names:
                            count = log_window_dataset_counts[name]
                            if count <= 0:
                                continue
                            dataset_means = {
                                key: value / count
                                for key, value in log_window_dataset_sums.get(name, {}).items()
                            }
                            log_tensorboard_train(writer, dataset_means, optimizer_step, name)
                    log_window_dataset_counts.clear()
                    log_window_sums.clear()
                    log_window_count = 0
                    log_window_dataset_sums.clear()

            start_update_in_epoch = 0

            if runtime["is_main"]:
                text = ", ".join(f"{name}={schedule_counts[name]}" for name in experiment.selected_names)
                print(f"Epoch {epoch+1} dataset updates: {text}")

            # ------------------------------------------------------------------
            # Validation
            # ------------------------------------------------------------------
            results_by_dataset = {}
            selection = {}

            if (epoch + 1) % settings.val_every_epochs == 0:
                if runtime["distributed"]:
                    dist.barrier()

                for dataset_name in experiment.selected_names:
                    handle = registry.handles[dataset_name]
                    if handle.val_loader is None:
                        continue
                    result = validate(
                        model,
                        criterion,
                        handle.val_loader,
                        handle.config.spec,
                        runtime,
                        settings,
                    )
                    results_by_dataset[dataset_name] = result
                    if runtime["is_main"]:
                        print_val(dataset_name, result, handle.config.spec)
                        log_tensorboard_val(
                            writer,
                            result,
                            epoch + 1,
                            dataset_name,
                            handle.config.spec,
                        )

                selection = selection_from_results(experiment, results_by_dataset)

                if runtime["is_main"] and results_by_dataset:
                    improved = []
                    for dataset_name, item in selection.items():
                        value = float(item["value"])
                        if value > dataset_best_values.get(dataset_name, -float("inf")):
                            dataset_best_values[dataset_name] = value
                            dataset_best_epochs[dataset_name] = epoch + 1
                            improved.append(dataset_name)

                    epoch_name = epoch_checkpoint_name(epoch + 1, experiment, selection)
                    full_validation_metrics = {
                        name: result["scalars"] for name, result in results_by_dataset.items()
                    }

                    for dataset_name in improved:
                        best_path = settings.output_dir / valbest_checkpoint_name(dataset_name, epoch_name)
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            scheduler,
                            epoch + 1,
                            0,
                            optimizer_step,
                            experiment,
                            resolved_config,
                            dataset_best_values=dataset_best_values,
                            dataset_best_epochs=dataset_best_epochs,
                            validation_selection=selection,
                            validation_metrics=full_validation_metrics,
                        )
                        removed = replace_dataset_valbest(settings.output_dir, dataset_name, best_path)
                        item = selection[dataset_name]
                        print(
                            f"ValBest [{dataset_name}] "
                            f"{item['metric_label']}={item['value']:.4f} @ Ep{epoch+1:03d}"
                        )
                        for old_name in removed:
                            print(f"  removed old ValBest: {old_name}")

                if runtime["distributed"]:
                    dist.barrier()

            # ------------------------------------------------------------------
            # Current checkpoint
            # ------------------------------------------------------------------
            save_current = (
                (epoch + 1) % settings.save_every_epochs == 0
                or (epoch + 1) == settings.epochs
            )

            if runtime["is_main"] and save_current:
                current_name = epoch_checkpoint_name(epoch + 1, experiment, selection)
                current_path = settings.output_dir / current_name
                full_validation_metrics = {
                    name: result["scalars"] for name, result in results_by_dataset.items()
                }
                save_checkpoint(
                    current_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    0,
                    optimizer_step,
                    experiment,
                    resolved_config,
                    dataset_best_values=dataset_best_values,
                    dataset_best_epochs=dataset_best_epochs,
                    validation_selection=selection,
                    validation_metrics=full_validation_metrics,
                )
                removed = remove_old_current_checkpoints(settings.output_dir, current_path)
                print(f"Current checkpoint: {current_name}")
                for old_name in removed:
                    print(f"  removed old current: {old_name}")

            if writer is not None:
                writer.flush()

            if runtime["is_main"]:
                epoch_elapsed = time.time() - epoch_start
                total_elapsed = time.time() - run_start
                print(
                    f"Epoch {epoch+1} time: {format_duration(epoch_elapsed)} ({epoch_elapsed:.1f} s) | "
                    f"Total time: {format_duration(total_elapsed)} ({total_elapsed:.1f} s)"
                )
                print()

        if runtime["is_main"]:
            print("Training complete.")
            for dataset_name in experiment.selected_names:
                handle = registry.handles[dataset_name]
                if handle.val_loader is None:
                    continue
                metric_key, metric_label = selection_metric_for_spec(handle.config.spec)
                best_value = dataset_best_values.get(dataset_name, -float("inf"))
                best_epoch = dataset_best_epochs.get(dataset_name, 0)
                if best_epoch > 0:
                    print(
                        f"  {dataset_name}: ValBest {metric_label}={best_value:.4f} "
                        f"@ Ep{best_epoch:03d}"
                    )
                else:
                    print(f"  {dataset_name}: no validation best recorded ({metric_key})")

            run_completed = True
            total_elapsed = time.time() - run_start
            print()
            print(f"Total run time: {format_duration(total_elapsed)} ({total_elapsed:.1f} s)")
            print("Finished:", time.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 96)

    finally:
        if runtime.get("is_main", False) and log_file is not None and not run_completed:
            elapsed = time.time() - run_start
            print()
            print(f"Run stopped after: {format_duration(elapsed)} ({elapsed:.1f} s)")
            print("Stopped:", time.strftime("%Y-%m-%d %H:%M:%S"))
        if writer is not None:
            writer.close()
        if log_file is not None:
            sys.stdout.flush()
            sys.stdout = original_stdout
            log_file.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
