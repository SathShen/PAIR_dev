#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

PAIR unified training framework.

PAIR 2D training uses true vectorized physical batching inside each GPU.

Batch control is explicit:
    --per-gpu-batch-size   physical samples processed by each GPU per forward
    --grad-accum           number of forward/backward micro-steps per optimizer update

Effective global batch is reported as:
    per_gpu_batch_size * world_size * grad_accum

Training modes:
    --qwen-tuning frozen
    --qwen-tuning lora   (default)
    --qwen-tuning full

Validation reports:
    binary change metrics
    T1/T2 semantic metrics
    combined semantic metrics
    SECOND-style F_scd / SeK / mIoU / OA / Score when an unchanged class exists

Logs:
    stdout
    train_log.jsonl
    val_log.jsonl
    TensorBoard
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from datasets.pair_dataset import DatasetSpec, UnifiedPAIRDataset
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
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/pair"))

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--per-gpu-batch-size", type=int, default=4,
                   help="True physical batch size processed by each GPU per forward")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-4, help="Decoder/adapter LR")
    p.add_argument("--lora-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--decoder-dim", type=int, default=256)
    p.add_argument("--qwen-tuning", choices=("frozen", "lora", "full"), default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")

    p.add_argument("--change-threshold", type=float, default=0.5)
    p.add_argument("--unchanged-class-id", type=int, default=None)
    p.add_argument("--val-every-epochs", type=int, default=1)
    p.add_argument("--val-max-samples", type=int, default=0,
                   help="0 = full validation set")
    p.add_argument("--log-every", type=int, default=20,
                   help="Optimizer updates between train logs")
    p.add_argument("--save-every-epochs", type=int, default=1)
    p.add_argument("--best-metric", default="scd/F_scd")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
# Dataset
# =============================================================================

def load_spec(name):
    module = importlib.import_module("dataset.config")
    if not hasattr(module, name):
        available = [k for k, v in vars(module).items() if isinstance(v, DatasetSpec)]
        raise AttributeError(f"dataset.config.{name} not found. Available: {available}")

    spec = getattr(module, name)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(f"dataset.config.{name} is not DatasetSpec")
    if not isinstance(spec.class_names, dict) or not spec.class_names:
        raise TypeError(f"{name}.class_names must be a non-empty Dict[int, str]")
    return spec


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


def collate_batch(batch):
    if not batch:
        raise RuntimeError("Empty batch")
    return list(batch)


def make_train_loader(dataset, runtime, num_workers, per_gpu_batch_size):
    sampler = DistributedSampler(
        dataset, num_replicas=runtime["world_size"], rank=runtime["rank"],
        shuffle=True, seed=0, drop_last=False,
    )
    loader = DataLoader(
        dataset, batch_size=per_gpu_batch_size, sampler=sampler, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0, collate_fn=collate_batch,
    )
    return loader, sampler


def make_val_loader(dataset, runtime, num_workers, per_gpu_batch_size):
    sampler = DistributedSampler(
        dataset, num_replicas=runtime["world_size"], rank=runtime["rank"],
        shuffle=False, drop_last=False,
    ) if runtime["distributed"] else None

    loader = DataLoader(
        dataset, batch_size=per_gpu_batch_size, sampler=sampler, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0, collate_fn=collate_batch,
    )
    return loader


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


def save_checkpoint(path, model, optimizer, scheduler, epoch, batch_in_epoch,
                    optimizer_step, args, best_metric, best_value):
    base = unwrap(model)
    checkpoint = {
        "epoch": int(epoch), "batch_in_epoch": int(batch_in_epoch),
        "optimizer_step": int(optimizer_step),
        "decoder": base.decoder.state_dict(),
        "image_adapter": base.image_adapter.state_dict(),
        "lora": lora_state_dict(base.pair.qwen_backbone.model),
        "pair_trainable": non_qwen_pair_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric, "best_value": best_value,
        "args": vars(args),
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
        int(ckpt.get("epoch", 0)), int(ckpt.get("batch_in_epoch", 0)),
        int(ckpt.get("optimizer_step", 0)),
        ckpt.get("best_metric"), ckpt.get("best_value"),
    )


# =============================================================================
# Forward / validation / logging
# =============================================================================

def forward_loss(model, criterion, samples, spec):
    if spec.task_mode != "2d":
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
        unchanged_raw_id=args.unchanged_class_id,
    )
    sums, count = {}, 0
    start = time.time()

    for samples in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, loss_output, merged_target = forward_loss(model, criterion, samples, spec)

        evaluator.update(prediction, merged_target)
        batch_n = len(samples)
        for key, value in loss_output.as_dict().items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_n
        count += batch_n

        if args.val_max_samples > 0 and count >= args.val_max_samples:
            break

    evaluator.reduce_distributed()
    losses = all_reduce_loss_sums(sums, count, runtime["device"])
    result = evaluator.compute()
    result["losses"] = losses
    result["seconds"] = time.time() - start
    model.train()
    return result


def log_tensorboard_train(writer, values, step):
    if writer is None:
        return
    for key, value in values.items():
        writer.add_scalar(f"train/{key}", value, step)


def log_tensorboard_val(writer, result, step):
    if writer is None:
        return

    for key, value in result["losses"].items():
        writer.add_scalar(f"val/{key}", value, step)
    for key, value in result["scalars"].items():
        writer.add_scalar(f"val/{key}", value, step)

    for class_name, values in result["per_class"].items():
        safe = class_name.replace("/", "_")
        for metric in ("IoU", "F1", "Precision", "Recall"):
            writer.add_scalar(f"val/per_class/{safe}/{metric}", values[metric], step)

    for name, cm in result["confusion"].items():
        writer.add_image(
            f"val/confusion/{name}", normalized_confusion_image(cm),
            step, dataformats="CHW",
        )


def write_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_val(result):
    s = result["scalars"]
    loss = result["losses"].get("loss", float("nan"))
    print(
        f"VAL | loss={loss:.4f} | change F1={s['change/F1']:.4f} "
        f"IoU={s['change/IoU']:.4f} | sem mIoU={s['semantic/mIoU']:.4f} "
        f"mF1={s['semantic/mF1']:.4f}",
        end="",
    )
    if "scd/F_scd" in s:
        print(
            f" | F_scd={s['scd/F_scd']:.4f} SeK={s['scd/SeK']:.4f} "
            f"mIoU_scd={s['scd/mIoU']:.4f} OA={s['scd/OA']:.4f}"
        )
    else:
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")

    runtime = setup_distributed()
    writer = None

    try:
        set_seed(args.seed, runtime["rank"])
        spec = load_spec(args.spec)
        root = args.dataset_root.expanduser().resolve()
        train_ds = UnifiedPAIRDataset(root / "manifests" / f"{args.train_split}.jsonl", spec)

        val_path = root / "manifests" / f"{args.val_split}.jsonl"
        val_ds = UnifiedPAIRDataset(val_path, spec) if val_path.exists() else None

        train_loader, train_sampler = make_train_loader(
            train_ds, runtime, args.num_workers, args.per_gpu_batch_size
        )
        val_loader = make_val_loader(
            val_ds, runtime, args.num_workers, args.per_gpu_batch_size
        ) if val_ds is not None else None

        grad_accum = args.grad_accum
        physical_global = runtime["world_size"] * args.per_gpu_batch_size
        effective_global = physical_global * grad_accum

        model = build_model(args, runtime)
        if runtime["distributed"]:
            model = DDP(
                model, device_ids=[runtime["local_rank"]],
                output_device=runtime["local_rank"],
                broadcast_buffers=False, find_unused_parameters=False,
            )

        criterion = PAIRSemanticChangeLoss().to(runtime["device"])
        optimizer, main_params, lora_params = build_optimizer(model, args)
        updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
        total_updates = updates_per_epoch * args.epochs
        scheduler = build_scheduler(optimizer, total_updates, args.warmup_ratio, args.scheduler)

        start_epoch = start_batch = optimizer_step = 0
        best_metric, best_value = args.best_metric, -float("inf")
        if args.resume is not None:
            start_epoch, start_batch, optimizer_step, old_metric, old_value = load_checkpoint(
                args.resume, model, optimizer, scheduler
            )
            if old_metric is not None:
                best_metric = old_metric
            if old_value is not None:
                best_value = float(old_value)

        if runtime["is_main"]:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            writer = make_writer(args.output_dir, True)
            (args.output_dir / "config.json").write_text(
                json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            base = unwrap(model)
            lora_trainable, _ = lora_parameter_count(base.pair.qwen_backbone.model)
            total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("=" * 92)
            print("PAIR UNIFIED TRAINING")
            print("=" * 92)
            print("Spec:", args.spec)
            print("Classes:", spec.class_names)
            print("Train/Val:", len(train_ds), "/", 0 if val_ds is None else len(val_ds))
            print("Qwen tuning:", args.qwen_tuning)
            print("LoRA trainable:", f"{lora_trainable / 1e6:.2f} M")
            print("Total trainable:", f"{total_trainable / 1e6:.2f} M")
            print("GPUs:", runtime["world_size"])
            print("Per-GPU batch size:", args.per_gpu_batch_size)
            print("Gradient accumulation:", grad_accum)
            print("Effective global batch:", effective_global)
            print("Updates/epoch:", updates_per_epoch)
            print("Total optimizer updates:", total_updates)
            print("TensorBoard:", args.output_dir / "tensorboard")
            print()

        train_json = args.output_dir / "train_log.jsonl"
        val_json = args.output_dir / "val_log.jsonl"
        model.train()

        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            window, window_count, window_samples = {}, 0, 0
            window_start = time.time()

            for batch_idx, samples in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                accum_count += 1
                is_last = batch_idx + 1 == len(train_loader)
                update_now = accum_count == grad_accum or is_last

                sync_context = contextlib.nullcontext()
                if isinstance(model, DDP) and not update_now:
                    sync_context = model.no_sync()

                with sync_context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss_output, _ = forward_loss(model, criterion, samples, spec)
                        loss = loss_output.total / grad_accum
                    loss.backward()

                for key, value in loss_output.as_dict().items():
                    window[key] = window.get(key, 0.0) + float(value.detach().cpu())
                window_count += 1
                window_samples += len(samples)

                if not update_now:
                    continue

                if accum_count != grad_accum:
                    scale = grad_accum / accum_count
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)

                params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    params, args.max_grad_norm
                ).detach().cpu()) if args.max_grad_norm > 0 else float("nan")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accum_count = 0

                if optimizer_step % args.log_every == 0:
                    elapsed = time.time() - window_start
                    means = {k: v / max(window_count, 1) for k, v in window.items()}
                    lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)}

                    log = {
                        **means, "epoch": epoch + 1, "optimizer_step": optimizer_step,
                        "grad_norm": grad_norm,
                        "samples_per_sec": window_samples * runtime["world_size"] / max(elapsed, 1e-6),
                        "gpu_alloc_GiB": torch.cuda.memory_allocated() / 1024**3,
                        "gpu_reserved_GiB": torch.cuda.memory_reserved() / 1024**3,
                        "lr_pair": lrs.get("pair", 0.0), "lr_lora": lrs.get("lora", 0.0),
                    }

                    if runtime["is_main"]:
                        print(
                            f"E{epoch+1:03d} U{optimizer_step:06d} | "
                            f"loss={means['loss']:.4f} sem1={means['loss_semantic_t1']:.4f} "
                            f"sem2={means['loss_semantic_t2']:.4f} "
                            f"bce={means['loss_change_bce']:.4f} dice={means['loss_change_dice']:.4f} | "
                            f"grad={grad_norm:.3f} lr={lrs.get('pair', 0):.2e}/"
                            f"{lrs.get('lora', 0):.2e} | {log['samples_per_sec']:.2f} sample/s"
                        )
                        write_jsonl(train_json, log)
                        log_tensorboard_train(writer, log, optimizer_step)

                    window, window_count, window_samples = {}, 0, 0
                    window_start = time.time()

            start_batch = 0

            # Full validation at epoch interval.
            val_result = None
            if val_loader is not None and (epoch + 1) % args.val_every_epochs == 0:
                if runtime["distributed"]:
                    dist.barrier()
                val_result = validate(model, criterion, val_loader, spec, runtime, args)
                if runtime["is_main"]:
                    print_val(val_result)
                    record = {
                        "epoch": epoch + 1, "optimizer_step": optimizer_step,
                        "seconds": val_result["seconds"],
                        "losses": val_result["losses"],
                        "metrics": val_result["scalars"],
                        "per_class": val_result["per_class"],
                        "confusion": {
                            k: v.tolist() for k, v in val_result["confusion"].items()
                        },
                    }
                    write_jsonl(val_json, record)
                    log_tensorboard_val(writer, val_result, optimizer_step)

                    metric_value = val_result["scalars"].get(best_metric)
                    if metric_value is None:
                        metric_value = val_result["scalars"].get("change/F1", -float("inf"))
                    if metric_value > best_value:
                        best_value = float(metric_value)
                        save_checkpoint(
                            args.output_dir / "best.pt", model, optimizer, scheduler,
                            epoch + 1, 0, optimizer_step, args, best_metric, best_value,
                        )
                        print(f"Best checkpoint: {best_metric}={best_value:.6f}")

                if runtime["distributed"]:
                    dist.barrier()

            if runtime["is_main"] and (epoch + 1) % args.save_every_epochs == 0:
                save_checkpoint(
                    args.output_dir / f"epoch_{epoch+1:03d}.pt", model, optimizer, scheduler,
                    epoch + 1, 0, optimizer_step, args, best_metric, best_value,
                )

            if runtime["is_main"]:
                save_checkpoint(
                    args.output_dir / "last.pt", model, optimizer, scheduler,
                    epoch + 1, 0, optimizer_step, args, best_metric, best_value,
                )
                if writer is not None:
                    writer.flush()

        if runtime["is_main"]:
            print(f"Training complete. Best {best_metric}={best_value:.6f}")

    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
