#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR multi-dataset training.

Model construction lives in models/pair.py. This file only owns:
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
"""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from datasets.config_loader import ExperimentConfig, load_experiment_config
from datasets.multi_dataset import DatasetRegistry, MultiDatasetScheduler
from loss import PAIRSemanticChangeLoss
from metrics import PAIRMetrics
from models.lora import lora_parameter_count, lora_state_dict, load_lora_state_dict
from models.pair import PAIRModel


# =============================================================================
# CLI / runtime
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/pair_train.json"))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def build_settings(experiment: ExperimentConfig, cli):
    o = experiment.optimizer
    t = experiment.training
    v = experiment.validation
    lg = experiment.logging

    output_dir = cli.output_dir
    if output_dir is None:
        output_dir = Path(lg.get("output_dir", f"outputs/{experiment.experiment['name']}"))

    return SimpleNamespace(
        lr=float(o.get("lr", 1e-4)),
        lora_lr=float(o.get("lora_lr", 2e-5)),
        weight_decay=float(o.get("weight_decay", 0.01)),
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
    return torch.cat([sample["target"][key].reshape(-1) for sample in samples], dim=0)


def _cat_valid_or_true(samples, value_key, valid_key):
    parts = []
    for sample in samples:
        target = sample["target"]
        value = target[value_key]

        if valid_key in target:
            valid = target[valid_key]
        else:
            valid = torch.ones_like(value, dtype=torch.bool)

        parts.append(valid.reshape(-1).bool())

    return torch.cat(parts, dim=0)


def merge_targets(samples, route):
    """
    Dataset files only carry valid masks when an explicit ignored_id exists.
    loss.py / metrics.py currently consume a mask unconditionally, so fully
    supervised samples receive an all-True mask here at the training boundary.
    No magic ignore label is introduced into the dataset.
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
        return {
            "semantic_t1": _cat_target(samples, "semantic_t1"),
            "semantic_t2": _cat_target(samples, "semantic_t2"),
            "change_t1": _cat_target(samples, "change_t1"),
            "change_t2": _cat_target(samples, "change_t2"),
            "semantic_valid_t1": _cat_valid_or_true(samples, "semantic_t1", "semantic_valid_t1"),
            "semantic_valid_t2": _cat_valid_or_true(samples, "semantic_t2", "semantic_valid_t2"),
            "change_valid_t1": _cat_valid_or_true(samples, "change_t1", "change_valid_t1"),
            "change_valid_t2": _cat_valid_or_true(samples, "change_t2", "change_valid_t2"),
        }

    raise NotImplementedError(
        "2D+3D target collation is deferred together with world-coordinate decoder wiring"
    )


# =============================================================================
# Forward / loss / validation
# =============================================================================

def forward_loss(model, criterion, samples, spec):
    prompts = [sample["prompt"] for sample in samples]

    if spec.route == "2d":
        output_sizes = [tuple(sample["target"]["change"].shape[-2:]) for sample in samples]
        prediction = model(
            task_mode="2d",
            images_t1=[sample["images_t1"] for sample in samples],
            images_t2=[sample["images_t2"] for sample in samples],
            prompts=prompts,
            class_names=spec.class_names,
            output_sizes=output_sizes,
        )
    elif spec.route == "3d":
        prediction = model(
            task_mode="3d",
            point_dicts_t1=[sample["point_dict_t1"] for sample in samples],
            point_dicts_t2=[sample["point_dict_t2"] for sample in samples],
            prompts=prompts,
            class_names=spec.class_names,
        )
    else:
        raise NotImplementedError(
            "PAIR 2D+3D training waits for real world-coordinate image/point correspondence"
        )

    target = merge_targets(samples, spec.route)
    loss_output = criterion(prediction=prediction, target=target, class_names=spec.class_names)
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

    evaluator = PAIRMetrics(
        spec.class_names,
        runtime["device"],
        settings.change_threshold,
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

def build_optimizer(model, settings):
    main, lora = [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if "lora_" in name:
            lora.append(parameter)
        else:
            main.append(parameter)

    groups = []
    if main:
        groups.append({"params": main, "lr": settings.lr, "name": "pair"})
    if lora:
        groups.append({"params": lora, "lr": settings.lora_lr, "name": "lora"})

    if not groups:
        raise RuntimeError("PAIR has no trainable parameters")

    optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
    return optimizer, main, lora


def build_scheduler(optimizer, total_updates, warmup_ratio, kind):
    warmup = int(round(total_updates * warmup_ratio))

    if kind == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_updates,
        )

    if kind == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)

    raise ValueError("optimizer.scheduler must be 'cosine' or 'constant'")


# =============================================================================
# Checkpoints
# =============================================================================

def non_qwen_trainable_state(model):
    base = unwrap(model)
    state = base.state_dict()

    names = {
        name
        for name, parameter in base.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.qwen_backbone.model.")
    }

    return {key: value.detach().cpu() for key, value in state.items() if key in names}


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
        "pair_trainable": non_qwen_trainable_state(model),
        "lora": lora_state_dict(base.qwen_backbone.model),
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

    # New unified checkpoint.
    trainable_state = ckpt.get("pair_trainable", {})
    if trainable_state:
        current = base.state_dict()
        current.update(trainable_state)
        base.load_state_dict(current, strict=False)

    # Compatibility with older checkpoints where these lived outside PAIRModel.
    if "decoder" in ckpt:
        base.decoder.load_state_dict(ckpt["decoder"])
    if "image_adapter" in ckpt:
        base.image_adapter.load_state_dict(ckpt["image_adapter"])

    load_lora_state_dict(base.qwen_backbone.model, ckpt.get("lora", {}))
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
        return "semantic/mIoU", "mIoU"

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
            raise KeyError(
                f"{name}: validation metric {metric_key!r} missing; available={sorted(scalars)}"
            )

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
        if item is None:
            continue
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
    if writer is None:
        return

    for key in (
        "loss",
        "loss_semantic_t1",
        "loss_semantic_t2",
        "loss_change_bce",
        "loss_change_dice",
    ):
        value = values.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(f"train/{dataset_name}/{key}", value, step)


def log_tensorboard_val(writer, result, epoch, dataset_name, spec):
    if writer is None:
        return

    total_loss = result["losses"].get("loss")
    if isinstance(total_loss, (int, float)):
        writer.add_scalar(f"val/{dataset_name}/loss", total_loss, epoch)

    scalars = result["scalars"]
    for display_name, key in validation_metric_layout(spec, scalars):
        value = scalars.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(f"val/{dataset_name}/{display_name}", value, epoch)


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

        registry = DatasetRegistry(experiment, runtime, num_workers=settings.num_workers)
        dataset_scheduler = MultiDatasetScheduler(experiment, registry, settings.grad_accum)

        updates_per_epoch = dataset_scheduler.updates_per_epoch
        total_updates = updates_per_epoch * settings.epochs

        # Complete model construction is owned by models/pair.py.
        model = PAIRModel.from_config(experiment.model, runtime["device"])

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
        scheduler = build_scheduler(
            optimizer,
            total_updates,
            settings.warmup_ratio,
            settings.scheduler,
        )

        start_epoch = 0
        start_update_in_epoch = 0
        optimizer_step = 0
        dataset_best_values = {name: -float("inf") for name in experiment.selected_names}
        dataset_best_epochs = {name: 0 for name in experiment.selected_names}

        if settings.resume is not None:
            (
                start_epoch,
                start_update_in_epoch,
                optimizer_step,
                loaded_best_values,
                loaded_best_epochs,
            ) = load_checkpoint(settings.resume, model, optimizer, scheduler)

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
                json.dumps(resolved_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            base = unwrap(model)
            lora_trainable, _ = lora_parameter_count(base.qwen_backbone.model)
            total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("=" * 96)
            print("PAIR MULTI-DATASET TRAINING")
            print("=" * 96)
            print("Experiment:", experiment.experiment["name"])
            print("Datasets:", ", ".join(experiment.selected_names))
            print("Qwen tuning:", base.qwen_tuning)
            print("Utonia frozen parameters:", f"{base.point_encoder.parameter_count() / 1e6:.2f} M")
            print("PointAdapter trainable:", f"{base.point_adapter.trainable_parameter_count() / 1e6:.2f} M")
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
                            _, loss_output, _ = forward_loss(model, criterion, samples, spec)
                            loss = loss_output.total / accumulation_steps

                        loss.backward()

                    for key, value in loss_output.as_dict().items():
                        update_sums[key] = update_sums.get(key, 0.0) + float(value.detach().cpu())

                trainable = [p for p in model.parameters() if p.requires_grad]

                if settings.max_grad_norm > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(trainable, settings.max_grad_norm)
                        .detach()
                        .cpu()
                    )
                else:
                    grad_norm = float("nan")

                optimizer.step()
                scheduler.step()
                optimizer_step += 1

                means = {key: value / accumulation_steps for key, value in update_sums.items()}
                lrs = {
                    group.get("name", str(i)): group["lr"]
                    for i, group in enumerate(optimizer.param_groups)
                }

                elapsed = time.time() - update_start
                samples_per_sec = update_samples * runtime["world_size"] / max(elapsed, 1e-6)

                if runtime["is_main"]:
                    write_log_only(
                        log_file,
                        (
                            f"ITER E{epoch+1:03d} U{optimizer_step:06d} "
                            f"dataset={dataset_name} route={handle.config.route} "
                            f"update_in_epoch={update_idx+1} accu={accumulation_steps} "
                            f"loss={means['loss']:.6f} "
                            f"sem1={means['loss_semantic_t1']:.6f} "
                            f"sem2={means['loss_semantic_t2']:.6f} "
                            f"bce={means['loss_change_bce']:.6f} "
                            f"dice={means['loss_change_dice']:.6f} "
                            f"grad={grad_norm:.6f} "
                            f"lr_pair={lrs.get('pair', 0.0):.8e} "
                            f"lr_lora={lrs.get('lora', 0.0):.8e} "
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

                        print(
                            f"E{epoch+1:03d} U{optimizer_step:06d} [{mix}] | "
                            f"avg_loss={window_means['loss']:.4f} "
                            f"avg_sem1={window_means['loss_semantic_t1']:.4f} "
                            f"avg_sem2={window_means['loss_semantic_t2']:.4f} "
                            f"avg_bce={window_means['loss_change_bce']:.4f} "
                            f"avg_dice={window_means['loss_change_dice']:.4f}"
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
