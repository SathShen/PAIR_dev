#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR multi-dataset unified training.

CLI:
    python train.py --config configs/pair_train.json --datasets SECOND Estonia3D

Dataset JSON entries contain only training-level values that cannot be inferred
from the prepared data directory. Modality, route, supervision mode and
manifest locations are inferred automatically by datasets/config_loader.py.

Each experiment epoch consumes exactly one pass of every selected dataset.
Per-dataset optimizer-update counts are derived automatically from DataLoader
length and gradient accumulation, then shuffled into one deterministic epoch
plan. DDP ranks share exactly the same plan.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from datasets.config_loader import ExperimentConfig, load_experiment_config
from datasets.multi_dataset import DatasetRegistry, MultiDatasetScheduler

from loss import PAIRSemanticChangeLoss
from metrics import PAIRMetrics, normalized_confusion_image
from models.change_decoder import UnifiedChangeDecoder, UnifiedTokenSet, build_identity_temporal_links
from models.lora import apply_qwen_lora, is_peft_model, lora_parameter_count, lora_state_dict, load_lora_state_dict
from models.pair import PAIRModel
from models.qwen3vl_backbone import Qwen3VLBackbone


DEFAULT_MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


# =============================================================================
# CLI / runtime
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path("configs/pair_train.json"),
        help="Full PAIR experiment JSON config",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset names from config. Example: --datasets SECOND Estonia3D",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional override for logging.output_dir",
    )
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def build_settings(experiment: ExperimentConfig, cli):
    """Flatten JSON sections for existing model/optimizer helpers."""
    m = experiment.model
    lora = m.get("lora", {})
    o = experiment.optimizer
    t = experiment.training
    v = experiment.validation
    lg = experiment.logging

    output_dir = (
        cli.output_dir if cli.output_dir is not None
        else Path(lg.get("output_dir", f"outputs/{experiment.experiment['name']}"))
    )
    targets = lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if isinstance(targets, str):
        targets_string = targets
    else:
        targets_string = ",".join(str(x) for x in targets)

    from types import SimpleNamespace
    return SimpleNamespace(
        # model
        model_dir=str(m["qwen_model"]),
        qwen_tuning=str(m.get("qwen_tuning", "lora")),
        decoder_dim=int(m.get("decoder_dim", 256)),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_target_modules=targets_string,

        # optimizer
        lr=float(o.get("lr", 1e-4)),
        lora_lr=float(o.get("lora_lr", 2e-5)),
        weight_decay=float(o.get("weight_decay", 0.01)),
        scheduler=str(o.get("scheduler", "cosine")),
        warmup_ratio=float(o.get("warmup_ratio", 0.03)),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),

        # training
        epochs=int(experiment.experiment["epochs"]),
        grad_accum=int(t.get("grad_accum", 1)),
        num_workers=int(t.get("num_workers", 4)),

        # validation/logging
        change_threshold=float(v.get("change_threshold", 0.5)),
        val_every_epochs=int(v.get("every_epochs", 1)),
        val_max_samples=int(v.get("max_samples", 0)),
        log_every=int(lg.get("log_every", 20)),
        save_every_epochs=int(lg.get("save_every_epochs", 1)),
        output_dir=Path(output_dir),
        resume=cli.resume,
        seed=int(experiment.experiment.get("seed", 42)),
    )

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        local_rank, rank = 0, 0
        torch.cuda.set_device(0)

    return {
        "distributed": distributed, "world_size": world_size, "rank": rank,
        "local_rank": local_rank, "device": torch.device("cuda", local_rank),
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


def make_writer(output_dir, is_main):
    if not is_main:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard logging requires: pip install tensorboard") from exc
    return SummaryWriter(log_dir=str(output_dir / "tensorboard"))


# =============================================================================
# Image bridge
# =============================================================================

def tensor_to_pil(image):
    x = image.detach().cpu().float()
    if x.ndim != 3 or x.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("Image contains NaN/Inf")
    if float(x.min()) < -1e-4 or float(x.max()) > 1.0001:
        raise ValueError(f"Expected image in [0,1], got [{float(x.min())},{float(x.max())}]")

    x = x.clamp(0, 1).mul(255).round().to(torch.uint8)
    arr = x.permute(1, 2, 0).contiguous().numpy()
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


# =============================================================================
# Thin 2D topology adapters
# =============================================================================

class ImageDenseAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        return self.proj(x)


def make_grid_positions(shape, device):
    if shape is None:
        raise RuntimeError("Missing image token shape")
    t, h, w = shape
    if t != 1:
        raise NotImplementedError(f"Current 2D training expects T=1, got {shape}")
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1), torch.zeros(h * w, device=device)), 1)


def make_batched_image_token_set(features, shapes, batch_ids):
    if features is None or batch_ids is None:
        raise RuntimeError("Missing batched image features/batch IDs")

    positions = []
    expected_ids = []
    for b, shape in enumerate(shapes):
        pos = make_grid_positions(shape, features.device)
        positions.append(pos)
        expected_ids.append(torch.full(
            (pos.shape[0],), b, dtype=torch.long, device=features.device
        ))

    positions = torch.cat(positions, 0)
    expected_ids = torch.cat(expected_ids, 0)
    batch_ids = batch_ids.to(features.device).long()

    if features.shape[0] != positions.shape[0]:
        raise RuntimeError(
            f"Feature/position count mismatch: {features.shape[0]} vs {positions.shape[0]}"
        )
    if not torch.equal(batch_ids, expected_ids):
        raise RuntimeError("PAIR image token order/batch IDs do not match expected sample-major layout")

    n = features.shape[0]
    return UnifiedTokenSet(
        features=features, positions=positions,
        modality_ids=torch.zeros(n, dtype=torch.long, device=features.device),
        batch_ids=batch_ids,
    )


def restore_2d_prediction_batch(prediction, shapes_t1, shapes_t2, output_sizes):
    def restore_semantic(logits, shapes):
        chunks, cursor = [], 0
        k = logits.shape[1]
        for shape, output_size in zip(shapes, output_sizes):
            t, h, w = shape
            n = t * h * w
            if t != 1:
                raise RuntimeError(f"Expected T=1, got {shape}")
            part = logits[cursor:cursor+n]
            if part.shape[0] != n:
                raise RuntimeError("Semantic token split mismatch")
            x = part.T.reshape(1, k, h, w)
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
            chunks.append(x[0].permute(1, 2, 0).reshape(-1, k))
            cursor += n
        if cursor != logits.shape[0]:
            raise RuntimeError("Unconsumed semantic logits after batch topology restore")
        return torch.cat(chunks, 0)

    def restore_change(logits, shapes):
        chunks, cursor = [], 0
        for shape, output_size in zip(shapes, output_sizes):
            t, h, w = shape
            n = t * h * w
            if t != 1:
                raise RuntimeError(f"Expected T=1, got {shape}")
            part = logits[cursor:cursor+n]
            if part.numel() != n:
                raise RuntimeError("Change token split mismatch")
            x = F.interpolate(
                part.reshape(1, 1, h, w), size=output_size,
                mode="bilinear", align_corners=False
            )
            chunks.append(x[0, 0].reshape(-1))
            cursor += n
        if cursor != logits.numel():
            raise RuntimeError("Unconsumed change logits after batch topology restore")
        return torch.cat(chunks, 0)

    prediction.semantic_logits_t1 = restore_semantic(
        prediction.semantic_logits_t1, shapes_t1
    )
    prediction.semantic_logits_t2 = restore_semantic(
        prediction.semantic_logits_t2, shapes_t2
    )
    prediction.change_logits_t1 = restore_change(
        prediction.change_logits_t1, shapes_t1
    )
    prediction.change_logits_t2 = restore_change(
        prediction.change_logits_t2, shapes_t2
    )
    return prediction


def merge_targets(samples):
    keys = (
        "change", "semantic_t1", "semantic_t2",
        "change_valid", "semantic_valid_t1", "semantic_valid_t2",
    )
    merged = {}
    for key in keys:
        merged[key] = torch.cat([
            sample["target"][key].reshape(-1) for sample in samples
        ], 0)

    # Optional topology-specific change targets for future 3D.
    for key in ("change_t1", "change_t2", "change_valid_t1", "change_valid_t2"):
        if all(key in sample["target"] for sample in samples):
            merged[key] = torch.cat([
                sample["target"][key].reshape(-1) for sample in samples
            ], 0)
    return merged


# =============================================================================
# Trainable PAIR wrapper
# =============================================================================

class PAIRTrainModel(nn.Module):
    def __init__(self, pair, decoder, image_adapter, qwen_tuning):
        super().__init__()
        self.pair = pair
        self.decoder = decoder
        self.image_adapter = image_adapter
        self.qwen_tuning = qwen_tuning

    @property
    def qwen_requires_graph(self):
        return self.qwen_tuning in ("lora", "full")

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if self.qwen_tuning in ("frozen", "lora"):
                self.pair.visual_module().eval()
            if self.qwen_tuning == "frozen":
                self.pair.qwen_backbone.model.eval()
        return self

    def forward_2d(self, images_t1, images_t2, prompts, class_names, output_sizes):
        if not (len(images_t1) == len(images_t2) == len(prompts) == len(output_sizes)):
            raise ValueError("Batched 2D inputs have inconsistent lengths")

        kwargs = dict(
            task_mode="2d", prompt=prompts,
            images_t1=images_t1, images_t2=images_t2,
            return_logits=False, return_hidden_states=True,
            return_dense_features=True, use_cache=False,
        )

        if self.qwen_requires_graph:
            out = self.pair(**kwargs)
        else:
            with torch.no_grad():
                out = self.pair(**kwargs)

        if out.image_dense_t1 is None or out.image_dense_t2 is None:
            raise RuntimeError("PAIR did not expose pre-LLM ViT dense features")
        if out.image_hidden_t1 is None or out.image_hidden_t2 is None or out.task_hidden is None:
            raise RuntimeError("PAIR did not expose LLM reasoning features")

        shapes1 = out.aux["image_token_shapes_t1"]
        shapes2 = out.aux["image_token_shapes_t2"]
        if len(shapes1) != len(prompts) or len(shapes2) != len(prompts):
            raise RuntimeError("PAIR returned incorrect number of image token grids")

        for b, (s1, s2) in enumerate(zip(shapes1, shapes2)):
            if s1 != s2:
                raise RuntimeError(
                    f"Current aligned 2D temporal links require identical T1/T2 grids; "
                    f"batch {b}: {s1} vs {s2}"
                )

        dense1 = self.image_adapter(out.image_dense_t1)
        dense2 = self.image_adapter(out.image_dense_t2)

        dense_t1 = make_batched_image_token_set(
            dense1, shapes1, out.aux["image_dense_batch_ids_t1"]
        )
        dense_t2 = make_batched_image_token_set(
            dense2, shapes2, out.aux["image_dense_batch_ids_t2"]
        )
        reasoning_t1 = make_batched_image_token_set(
            out.image_hidden_t1, shapes1, out.aux["image_reasoning_batch_ids_t1"]
        )
        reasoning_t2 = make_batched_image_token_set(
            out.image_hidden_t2, shapes2, out.aux["image_reasoning_batch_ids_t2"]
        )

        # Because both streams are concatenated sample-major and each pair has
        # the same grid, global identity indices stay within each sample.
        if dense1.shape[0] != dense2.shape[0]:
            raise RuntimeError("Aligned 2D batch has different total T1/T2 token counts")

        prediction = self.decoder(
            dense_t1=dense_t1, dense_t2=dense_t2,
            reasoning_t1=reasoning_t1, reasoning_t2=reasoning_t2,
            task_hidden=out.task_hidden,
            links_t1_to_t2=build_identity_temporal_links(
                dense1.shape[0], device=dense1.device
            ),
            links_t2_to_t1=build_identity_temporal_links(
                dense2.shape[0], device=dense2.device
            ),
            class_names=class_names,
            qwen_backbone=self.pair.qwen_backbone,
            detach_qwen_class_encoder=True,
        )
        return restore_2d_prediction_batch(
            prediction, shapes1, shapes2, output_sizes
        )

    def forward(self, task_mode, **kwargs):
        if task_mode == "2d":
            return self.forward_2d(**kwargs)
        raise NotImplementedError(
            f"Training adapter for {task_mode!r} is not connected yet. "
            "The unified decoder itself is batch/ragged aware."
        )


# =============================================================================
# Model / optimizer / scheduler
# =============================================================================

def configure_qwen(qwen, args):
    if args.qwen_tuning == "frozen":
        qwen.freeze()
    elif args.qwen_tuning == "full":
        qwen.unfreeze()
    else:
        targets = tuple(x.strip() for x in args.lora_target_modules.split(",") if x.strip())
        apply_qwen_lora(
            qwen, r=args.lora_r, alpha=args.lora_alpha,
            dropout=args.lora_dropout, target_modules=targets,
        )


def build_model(args, runtime):
    device_str = f"cuda:{runtime['local_rank']}"
    qwen = Qwen3VLBackbone(
        model_dir=args.model_dir, dtype=torch.bfloat16,
        device=device_str, device_map=device_str, local_files_only=True,
    )
    configure_qwen(qwen, args)

    pair = PAIRModel(qwen_backbone=qwen)
    image_adapter = ImageDenseAdapter(qwen.hidden_size, args.decoder_dim).to(runtime["device"])
    decoder = UnifiedChangeDecoder(qwen_dim=qwen.hidden_size, decoder_dim=args.decoder_dim).to(runtime["device"])
    return PAIRTrainModel(pair, decoder, image_adapter, args.qwen_tuning).to(runtime["device"])


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def build_optimizer(model, args):
    base = unwrap(model)
    lora, main = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora.append(p)
        else:
            main.append(p)

    groups = []
    if main:
        groups.append({"params": main, "lr": args.lr, "name": "pair"})
    if lora:
        groups.append({"params": lora, "lr": args.lora_lr, "name": "lora"})

    return torch.optim.AdamW(groups, weight_decay=args.weight_decay), main, lora


def build_scheduler(optimizer, total_updates, warmup_ratio, kind):
    warmup = int(round(total_updates * warmup_ratio))
    if kind == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=total_updates
        )
    return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)


# =============================================================================
# Checkpoints
# =============================================================================

def non_qwen_pair_state(model):
    pair = unwrap(model).pair
    state = pair.state_dict()
    names = {
        name for name, p in pair.named_parameters()
        if p.requires_grad and not name.startswith("qwen_backbone.model.")
    }
    return {k: v.detach().cpu() for k, v in state.items() if k in names}


def save_checkpoint(
    path, model, optimizer, scheduler,
    epoch, update_in_epoch, optimizer_step,
    settings, experiment, resolved_config,
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
        "decoder": base.decoder.state_dict(),
        "image_adapter": base.image_adapter.state_dict(),
        "lora": lora_state_dict(base.pair.qwen_backbone.model),
        "pair_trainable": non_qwen_pair_state(model),
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


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    base = unwrap(model)
    base.decoder.load_state_dict(ckpt["decoder"])
    base.image_adapter.load_state_dict(ckpt["image_adapter"])
    load_lora_state_dict(base.pair.qwen_backbone.model, ckpt.get("lora", {}))
    pair_state = ckpt.get("pair_trainable", {})
    if pair_state:
        state = base.pair.state_dict()
        state.update(pair_state)
        base.pair.load_state_dict(state, strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("update_in_epoch", 0)),
        int(ckpt.get("optimizer_step", 0)),
        {str(k): float(v) for k, v in ckpt.get("dataset_best_values", {}).items()},
        {str(k): int(v) for k, v in ckpt.get("dataset_best_epochs", {}).items()},
    )


def selection_metric_for_spec(spec):
    """Automatic per-dataset ValBest metric selection."""
    route = str(spec.route)
    label_mode = str(spec.label_mode)

    if route == "2d" and label_mode == "semantic_pair":
        return "scd/F_scd", "Fscd"
    if route in {"3d", "2d3d"} and label_mode == "semantic_pair":
        return "semantic/mIoU", "mIoU"
    if label_mode in {"binary", "post_semantic"}:
        return "change/IoU", "IoU"

    raise ValueError(
        f"No checkpoint selection rule for route={route!r}, "
        f"label_mode={label_mode!r}"
    )


def selection_from_results(experiment, results_by_dataset):
    selection = {}
    for name in experiment.selected_names:
        if name not in results_by_dataset:
            continue
        spec = experiment.datasets[name].spec
        metric_key, metric_label = selection_metric_for_spec(spec)
        scalars = results_by_dataset[name]["scalars"]
        if metric_key not in scalars:
            raise KeyError(
                f"{name}: required validation metric {metric_key!r} missing; "
                f"available={sorted(scalars)}"
            )
        selection[name] = {
            "metric_key": metric_key,
            "metric_label": metric_label,
            "value": float(scalars[metric_key]),
        }
    return selection


def safe_checkpoint_token(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    value = value.strip("-_.")
    return value or "Dataset"


def epoch_checkpoint_name(epoch, experiment, selection):
    parts = [f"Ep{int(epoch):03d}"]
    for name in experiment.selected_names:
        item = selection.get(name)
        if item is None:
            continue
        parts.append(
            f"{safe_checkpoint_token(name)}"
            f"{item['metric_label']}{item['value']:.4f}"
        )
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
# Forward / validation / logging
# =============================================================================

def forward_loss(model, criterion, samples, spec):
    if spec.route != "2d":
        raise NotImplementedError("Current training adapter closes batched 2D first")

    images_t1 = [tensor_to_pil(s["images_t1"]) for s in samples]
    images_t2 = [tensor_to_pil(s["images_t2"]) for s in samples]
    prompts = [s["prompt"] for s in samples]
    output_sizes = [tuple(s["target"]["change"].shape[-2:]) for s in samples]

    prediction = model(
        task_mode="2d",
        images_t1=images_t1, images_t2=images_t2,
        prompts=prompts, class_names=spec.class_names,
        output_sizes=output_sizes,
    )
    target = merge_targets(samples)
    loss_output = criterion(
        prediction=prediction, target=target, class_names=spec.class_names
    )
    return prediction, loss_output, target


def all_reduce_loss_sums(sums, count, device):
    keys = sorted(sums)
    tensor = torch.tensor([sums[k] for k in keys] + [count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = max(float(tensor[-1].item()), 1.0)
    return {k: float(tensor[i].item() / count) for i, k in enumerate(keys)}


@torch.no_grad()
def validate(model, criterion, loader, spec, runtime, args):
    model.eval()
    evaluator = PAIRMetrics(
        spec.class_names, runtime["device"], args.change_threshold,
        unchanged_raw_id=spec.unchanged_raw_id,
    )
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
            prediction, loss_output, merged_target = forward_loss(model, criterion, samples, spec)

        evaluator.update(prediction, merged_target)
        batch_n = len(samples)
        for key, value in loss_output.as_dict().items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_n
        count += batch_n
        if runtime["is_main"]:
            progress.set_postfix(samples=count, refresh=False)

        if args.val_max_samples > 0 and count >= args.val_max_samples:
            break

    evaluator.reduce_distributed()
    losses = all_reduce_loss_sums(sums, count, runtime["device"])
    result = evaluator.compute()
    result["losses"] = losses
    result["seconds"] = time.time() - start
    model.train()
    return result


def log_tensorboard_train(writer, values, step, dataset_name):
    """
    Keep TensorBoard train logs intentionally minimal.

    Only log the training losses. Do not log epoch/update counters, gradient
    norm, learning rates, throughput, or GPU memory; those remain available
    in train_log.jsonl and the terminal output.
    """
    if writer is None:
        return

    loss_keys = (
        "loss",
        "loss_semantic_t1",
        "loss_semantic_t2",
        "loss_change_bce",
        "loss_change_dice",
    )

    for key in loss_keys:
        value = values.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(
                f"train/{dataset_name}/{key}",
                value,
                step,
            )


def log_tensorboard_val(writer, result, step, dataset_name):
    """
    Keep TensorBoard validation logs compact:
      - one total validation loss
      - every aggregate scalar metric produced by PAIRMetrics

    Per-class metrics and confusion-matrix images are intentionally omitted
    from TensorBoard. They are still preserved in val_log.jsonl.
    """
    if writer is None:
        return

    total_loss = result["losses"].get("loss")
    if isinstance(total_loss, (int, float)):
        writer.add_scalar(
            f"val/{dataset_name}/loss",
            total_loss,
            step,
        )

    for key, value in result["scalars"].items():
        if isinstance(value, (int, float)):
            writer.add_scalar(
                f"val/{dataset_name}/{key}",
                value,
                step,
            )


def log_tensorboard_macro(writer, macro, step):
    if writer is None:
        return
    for key, value in macro.items():
        writer.add_scalar(f"val/{key}", value, step)


def compute_macro_metrics(results_by_dataset):
    buckets = {}
    for result in results_by_dataset.values():
        for key, value in result["scalars"].items():
            buckets.setdefault(key, []).append(float(value))

    return {
        f"macro/{key}": sum(values) / len(values)
        for key, values in buckets.items()
        if values
    }


def print_val(dataset_name, result):
    s = result["scalars"]
    loss = result["losses"].get("loss", float("nan"))
    print(
        f"VAL [{dataset_name}] | loss={loss:.4f} | "
        f"change F1={s['change/F1']:.4f} IoU={s['change/IoU']:.4f} | "
        f"sem mIoU={s['semantic/mIoU']:.4f} mF1={s['semantic/mF1']:.4f}",
        end="",
    )
    if "scd/F_scd" in s:
        print(
            f" | F_scd={s['scd/F_scd']:.4f} SeK={s['scd/SeK']:.4f} "
            f"mIoU_scd={s['scd/mIoU']:.4f} OA={s['scd/OA']:.4f}"
        )
    else:
        print()


def write_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    runtime = setup_distributed()
    writer = None

    try:
        experiment = load_experiment_config(cli.config, cli.datasets)
        settings = build_settings(experiment, cli)

        if settings.grad_accum < 1:
            raise ValueError("training.grad_accum must be >= 1")

        set_seed(settings.seed, runtime["rank"])

        registry = DatasetRegistry(
            experiment, runtime, num_workers=settings.num_workers
        )
        dataset_scheduler = MultiDatasetScheduler(
            experiment, registry, settings.grad_accum
        )

        updates_per_epoch = dataset_scheduler.updates_per_epoch
        total_updates = updates_per_epoch * settings.epochs

        model = build_model(settings, runtime)
        if runtime["distributed"]:
            # Mixed 2D / 3D / 2D3D training intentionally leaves modality
            # branches unused on some optimizer updates.
            model = DDP(
                model,
                device_ids=[runtime["local_rank"]],
                output_device=runtime["local_rank"],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        criterion = PAIRSemanticChangeLoss().to(runtime["device"])
        optimizer, _, _ = build_optimizer(model, settings)
        scheduler = build_scheduler(
            optimizer, total_updates,
            settings.warmup_ratio, settings.scheduler,
        )

        start_epoch = 0
        start_update_in_epoch = 0
        optimizer_step = 0
        dataset_best_values = {
            name: -float("inf")
            for name in experiment.selected_names
        }
        dataset_best_epochs = {
            name: 0
            for name in experiment.selected_names
        }

        if settings.resume is not None:
            (
                start_epoch,
                start_update_in_epoch,
                optimizer_step,
                loaded_best_values,
                loaded_best_epochs,
            ) = load_checkpoint(
                settings.resume, model, optimizer, scheduler
            )
            dataset_best_values.update(loaded_best_values)
            dataset_best_epochs.update(loaded_best_epochs)

        # Runtime-resolved config is deliberately separate from the source JSON.
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
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            writer = make_writer(
                settings.output_dir,
                bool(experiment.logging.get("tensorboard", True)),
            ) if experiment.logging.get("tensorboard", True) else None

            # Preserve the user-authored source config unchanged.
            source_config_text = experiment.path.read_text(encoding="utf-8")
            (settings.output_dir / "config.json").write_text(
                source_config_text, encoding="utf-8"
            )
            (settings.output_dir / "config_resolved.json").write_text(
                json.dumps(resolved_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            base = unwrap(model)
            lora_trainable, _ = lora_parameter_count(
                base.pair.qwen_backbone.model
            )
            total_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            print("=" * 96)
            print("PAIR MULTI-DATASET TRAINING")
            print("=" * 96)
            print("Experiment:", experiment.experiment["name"])
            print("Datasets:", ", ".join(experiment.selected_names))
            print("Qwen tuning:", settings.qwen_tuning)
            print("LoRA trainable:", f"{lora_trainable / 1e6:.2f} M")
            print("Total trainable:", f"{total_trainable / 1e6:.2f} M")
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
            print("TensorBoard:", settings.output_dir / "tensorboard")
            print()

        train_json = settings.output_dir / "train_log.jsonl"
        val_json = settings.output_dir / "val_log.jsonl"
        model.train()

        log_window_dataset_counts = Counter()

        for epoch in range(start_epoch, settings.epochs):
            registry.reset_epoch(epoch)
            schedule = dataset_scheduler.epoch_schedule(
                epoch, runtime
            )
            schedule_counts = {
                name: sum(
                    1 for item in schedule
                    if item.dataset_name == name
                )
                for name in experiment.selected_names
            }

            first_update = (
                start_update_in_epoch
                if epoch == start_epoch else 0
            )

            # Reconstruct exact loader positions for mid-epoch resume.
            if first_update > 0:
                registry.consume_updates(schedule[:first_update])

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
                    if (
                        isinstance(model, DDP)
                        and micro_idx + 1 < accumulation_steps
                    ):
                        sync_context = model.no_sync()

                    with sync_context:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, loss_output, _ = forward_loss(
                                model, criterion, samples, spec
                            )
                            loss = loss_output.total / accumulation_steps
                        loss.backward()

                    for key, value in loss_output.as_dict().items():
                        update_sums[key] = (
                            update_sums.get(key, 0.0)
                            + float(value.detach().cpu())
                        )

                trainable = [
                    p for p in model.parameters() if p.requires_grad
                ]
                grad_norm = (
                    float(torch.nn.utils.clip_grad_norm_(
                        trainable, settings.max_grad_norm
                    ).detach().cpu())
                    if settings.max_grad_norm > 0
                    else float("nan")
                )

                optimizer.step()
                scheduler.step()
                optimizer_step += 1
                log_window_dataset_counts[dataset_name] += 1

                means = {
                    key: value / accumulation_steps
                    for key, value in update_sums.items()
                }
                lrs = {
                    g.get("name", str(i)): g["lr"]
                    for i, g in enumerate(optimizer.param_groups)
                }
                elapsed = time.time() - update_start

                update_log = {
                    **means,
                    "dataset": dataset_name,
                    "route": handle.config.route,
                    "epoch": epoch + 1,
                    "update_in_epoch": update_idx + 1,
                    "optimizer_step": optimizer_step,
                    "accumulation_steps": accumulation_steps,
                    "grad_norm": grad_norm,
                    "samples_per_sec": (
                        update_samples * runtime["world_size"]
                        / max(elapsed, 1e-6)
                    ),
                    "gpu_alloc_GiB": (
                        torch.cuda.memory_allocated() / 1024**3
                    ),
                    "gpu_reserved_GiB": (
                        torch.cuda.memory_reserved() / 1024**3
                    ),
                    "lr_pair": lrs.get("pair", 0.0),
                    "lr_lora": lrs.get("lora", 0.0),
                }

                if optimizer_step % settings.log_every == 0:
                    if runtime["is_main"]:
                        mix = " ".join(
                            f"{name} x{log_window_dataset_counts[name]}"
                            for name in experiment.selected_names
                            if log_window_dataset_counts[name] > 0
                        )
                        print(
                            f"E{epoch+1:03d} U{optimizer_step:06d} "
                            f"[{mix}] | "
                            f"loss={means['loss']:.4f} "
                            f"sem1={means['loss_semantic_t1']:.4f} "
                            f"sem2={means['loss_semantic_t2']:.4f} "
                            f"bce={means['loss_change_bce']:.4f} "
                            f"dice={means['loss_change_dice']:.4f} | "
                            f"grad={grad_norm:.3f} "
                            f"lr={lrs.get('pair', 0):.2e}/"
                            f"{lrs.get('lora', 0):.2e} | "
                            f"{update_log['samples_per_sec']:.2f} sample/s"
                        )
                        write_jsonl(train_json, update_log)
                        log_tensorboard_train(
                            writer, update_log, optimizer_step, dataset_name
                        )
                    log_window_dataset_counts.clear()

            start_update_in_epoch = 0

            if runtime["is_main"]:
                print(
                    f"Epoch {epoch+1} dataset updates: "
                    + ", ".join(
                        f"{name}={schedule_counts[name]}"
                        for name in experiment.selected_names
                    )
                )

            # ----------------------------------------------------------
            # Validation: each dataset owns its own metric and ValBest.
            # ----------------------------------------------------------
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
                        model, criterion,
                        handle.val_loader,
                        handle.config.spec,
                        runtime, settings,
                    )
                    results_by_dataset[dataset_name] = result

                    if runtime["is_main"]:
                        print_val(dataset_name, result)
                        log_tensorboard_val(
                            writer, result, optimizer_step, dataset_name
                        )

                macro = compute_macro_metrics(results_by_dataset)
                selection = selection_from_results(
                    experiment, results_by_dataset
                )

                if runtime["is_main"] and results_by_dataset:
                    record = {
                        "epoch": epoch + 1,
                        "optimizer_step": optimizer_step,
                        "datasets": {
                            name: {
                                "seconds": result["seconds"],
                                "losses": result["losses"],
                                "metrics": result["scalars"],
                                "selection": selection.get(name),
                                "per_class": result["per_class"],
                                "confusion": {
                                    key: value.tolist()
                                    for key, value
                                    in result["confusion"].items()
                                },
                            }
                            for name, result in results_by_dataset.items()
                        },
                        "macro": macro,
                    }
                    write_jsonl(val_json, record)

                    improved = []
                    for dataset_name, item in selection.items():
                        value = float(item["value"])
                        if value > dataset_best_values.get(
                            dataset_name, -float("inf")
                        ):
                            dataset_best_values[dataset_name] = value
                            dataset_best_epochs[dataset_name] = epoch + 1
                            improved.append(dataset_name)

                    epoch_name = epoch_checkpoint_name(
                        epoch + 1, experiment, selection
                    )
                    full_validation_metrics = {
                        name: result["scalars"]
                        for name, result in results_by_dataset.items()
                    }

                    for dataset_name in improved:
                        best_path = (
                            settings.output_dir
                            / valbest_checkpoint_name(dataset_name, epoch_name)
                        )
                        save_checkpoint(
                            best_path,
                            model, optimizer, scheduler,
                            epoch + 1, 0, optimizer_step,
                            settings, experiment, resolved_config,
                            dataset_best_values=dataset_best_values,
                            dataset_best_epochs=dataset_best_epochs,
                            validation_selection=selection,
                            validation_metrics=full_validation_metrics,
                        )
                        removed = replace_dataset_valbest(
                            settings.output_dir, dataset_name, best_path
                        )
                        item = selection[dataset_name]
                        print(
                            f"ValBest [{dataset_name}] "
                            f"{item['metric_label']}={item['value']:.4f} "
                            f"@ Ep{epoch+1:03d}"
                        )
                        for old_name in removed:
                            print(f"  removed old ValBest: {old_name}")

                if runtime["distributed"]:
                    dist.barrier()

            # ----------------------------------------------------------
            # Current/resume checkpoint.
            # No last.pt. Only one ordinary EpXXX_*.pt checkpoint is kept.
            # ----------------------------------------------------------
            save_current = (
                (epoch + 1) % settings.save_every_epochs == 0
                or (epoch + 1) == settings.epochs
            )

            if runtime["is_main"] and save_current:
                current_name = epoch_checkpoint_name(
                    epoch + 1, experiment, selection
                )
                current_path = settings.output_dir / current_name
                full_validation_metrics = {
                    name: result["scalars"]
                    for name, result in results_by_dataset.items()
                }

                save_checkpoint(
                    current_path,
                    model, optimizer, scheduler,
                    epoch + 1, 0, optimizer_step,
                    settings, experiment, resolved_config,
                    dataset_best_values=dataset_best_values,
                    dataset_best_epochs=dataset_best_epochs,
                    validation_selection=selection,
                    validation_metrics=full_validation_metrics,
                )
                removed = remove_old_current_checkpoints(
                    settings.output_dir, current_path
                )
                print(f"Current checkpoint: {current_name}")
                for old_name in removed:
                    print(f"  removed old current: {old_name}")

            if runtime["is_main"] and writer is not None:
                writer.flush()

        if runtime["is_main"]:
            print("Training complete.")
            for dataset_name in experiment.selected_names:
                handle = registry.handles[dataset_name]
                if handle.val_loader is None:
                    continue
                metric_key, metric_label = selection_metric_for_spec(
                    handle.config.spec
                )
                best_value = dataset_best_values.get(
                    dataset_name, -float("inf")
                )
                best_epoch = dataset_best_epochs.get(dataset_name, 0)
                if best_epoch > 0:
                    print(
                        f"  {dataset_name}: ValBest "
                        f"{metric_label}={best_value:.4f} "
                        f"@ Ep{best_epoch:03d}"
                    )
                else:
                    print(
                        f"  {dataset_name}: no validation best recorded "
                        f"({metric_key})"
                    )

    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
