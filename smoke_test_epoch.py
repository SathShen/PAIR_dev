#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR one-epoch real training test.

Pipeline:
    dataset/config.py -> real SECOND sample
    -> PAIR
       -> pre-LLM ViT dense features
       -> post-LLM reasoning features
       -> task hidden
    -> thin 2D adapter
    -> UnifiedChangeDecoder
    -> 2D topology restoration
    -> unified semantic-change loss
    -> backward
    -> optimizer.step

This is intentionally a FULL train epoch, not a one-step smoke test.

Expected current files:
    dataset/config.py
    dataset/pair_dataset.py
    models/pair.py
    models/qwen3vl_backbone.py
    models/change_decoder.py
    loss.py

Run:
    CUDA_VISIBLE_DEVICES=2 python test_train_epoch.py \
        --dataset-root /home/sht/Datasets/SECONDpair \
        --spec SECOND_SPEC \
        --output-dir outputs/test_epoch
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from datasets.pair_dataset import DatasetSpec, UnifiedPAIRDataset
from loss import PAIRSemanticChangeLoss
from models.change_decoder import (
    UnifiedChangeDecoder,
    UnifiedTokenSet,
    build_identity_temporal_links,
)
from models.pair import PAIRModel
from models.qwen3vl_backbone import Qwen3VLBackbone


DEFAULT_MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--spec", default="SECOND_SPEC")
    p.add_argument("--split", default="train")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/test_epoch"))
    p.add_argument("--decoder-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-shuffle", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_spec(name):
    module = importlib.import_module("datasets.configs")
    if not hasattr(module, name):
        specs = [k for k, v in vars(module).items() if isinstance(v, DatasetSpec)]
        raise AttributeError(f"datasets.configs.{name} not found. Available: {specs}")

    spec = getattr(module, name)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(f"datasets.configs.{name} is not DatasetSpec")
    if not isinstance(spec.class_names, dict) or not spec.class_names:
        raise TypeError(f"{name}.class_names must be non-empty Dict[int, str]")
    if spec.task_mode != "2d":
        raise ValueError("This first full-epoch test currently uses the real 2D SECOND dataset")
    return spec


def collate_one(batch):
    if len(batch) != 1:
        raise RuntimeError("PAIR V1 currently uses per-GPU batch size 1")
    return batch[0]


def tensor_to_pil(image):
    x = image.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("Image contains NaN/Inf")
    if float(x.min()) < -1e-4 or float(x.max()) > 1.0001:
        raise ValueError(f"Expected image in [0,1], got [{float(x.min())}, {float(x.max())}]")

    x = x.clamp(0, 1).mul(255).round().to(torch.uint8)
    arr = x.permute(1, 2, 0).contiguous().numpy()
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


class ImageDenseAdapter(nn.Module):
    """Only a thin modality adapter; this is NOT part of the unified decoder."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        return self.proj(x)


def make_grid_positions(shape, device):
    if shape is None:
        raise RuntimeError("PAIR did not return image_token_shape")
    t, h, w = shape
    if t != 1:
        raise NotImplementedError(f"Expected one image frame, got grid {shape}")

    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    zz = torch.zeros_like(xx)
    return torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)


def make_token_set(features, positions):
    n = features.shape[0]
    if positions.shape[0] != n:
        raise RuntimeError(f"Feature/position count mismatch: {n} vs {positions.shape[0]}")

    return UnifiedTokenSet(
        features=features,
        positions=positions,
        modality_ids=torch.zeros(n, dtype=torch.long, device=features.device),
        batch_ids=torch.zeros(n, dtype=torch.long, device=features.device),
    )


def restore_semantic(logits, shape, output_size):
    t, h, w = shape
    if t != 1 or logits.shape[0] != h * w:
        raise RuntimeError(f"Cannot restore semantic logits {tuple(logits.shape)} with grid {shape}")

    k = logits.shape[1]
    x = logits.T.reshape(1, k, h, w)
    x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
    return x[0].permute(1, 2, 0).reshape(-1, k)


def restore_change(logits, shape, output_size):
    t, h, w = shape
    if t != 1 or logits.numel() != h * w:
        raise RuntimeError(f"Cannot restore change logits {tuple(logits.shape)} with grid {shape}")

    x = logits.reshape(1, 1, h, w)
    x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
    return x[0, 0].reshape(-1)


class OneEpochPAIR(nn.Module):
    def __init__(self, pair, decoder, image_adapter):
        super().__init__()
        self.pair = pair
        self.decoder = decoder
        self.image_adapter = image_adapter

        # Full epoch test trains decoder + thin adapter only.
        self.pair.qwen_backbone.freeze()

    def train(self, mode=True):
        super().train(mode)
        self.pair.qwen_backbone.model.eval()
        return self

    def forward(self, image_t1, image_t2, prompt, class_names, output_size):
        # Foundation model remains frozen; decoder/adapter retain autograd.
        with torch.no_grad():
            out = self.pair(
                task_mode="2d",
                prompt=prompt,
                images_t1=image_t1,
                images_t2=image_t2,
                return_logits=False,
                return_hidden_states=True,
                return_dense_features=True,
                use_cache=False,
            )

        if out.image_dense_t1 is None or out.image_dense_t2 is None:
            raise RuntimeError(
                "image_dense_t1/t2 is None. Use the new pair.py that exposes pre-LLM vision features."
            )
        if out.image_hidden_t1 is None or out.image_hidden_t2 is None:
            raise RuntimeError("LLM image reasoning hidden is missing")
        if out.task_hidden is None:
            raise RuntimeError("task_hidden is missing")

        shape1 = out.aux["image_token_shape_t1"]
        shape2 = out.aux["image_token_shape_t2"]

        dense1 = self.image_adapter(out.image_dense_t1)
        dense2 = self.image_adapter(out.image_dense_t2)

        pos1 = make_grid_positions(shape1, dense1.device)
        pos2 = make_grid_positions(shape2, dense2.device)

        dense_t1 = make_token_set(dense1, pos1)
        dense_t2 = make_token_set(dense2, pos2)
        reasoning_t1 = make_token_set(out.image_hidden_t1, pos1)
        reasoning_t2 = make_token_set(out.image_hidden_t2, pos2)

        links12 = build_identity_temporal_links(dense1.shape[0], device=dense1.device)
        links21 = build_identity_temporal_links(dense2.shape[0], device=dense2.device)

        pred = self.decoder(
            dense_t1=dense_t1,
            dense_t2=dense_t2,
            reasoning_t1=reasoning_t1,
            reasoning_t2=reasoning_t2,
            task_hidden=out.task_hidden,
            links_t1_to_t2=links12,
            links_t2_to_t1=links21,
            class_names=class_names,
            qwen_backbone=self.pair.qwen_backbone,
            detach_qwen_class_encoder=True,
        )

        token_shapes = {
            "image_dense_t1": tuple(out.image_dense_t1.shape),
            "image_dense_t2": tuple(out.image_dense_t2.shape),
            "image_hidden_t1": tuple(out.image_hidden_t1.shape),
            "image_hidden_t2": tuple(out.image_hidden_t2.shape),
            "task_hidden": tuple(out.task_hidden.shape),
            "grid_t1": shape1,
            "grid_t2": shape2,
            "token_semantic_t1": tuple(pred.semantic_logits_t1.shape),
            "token_semantic_t2": tuple(pred.semantic_logits_t2.shape),
            "token_change_t1": tuple(pred.change_logits_t1.shape),
            "token_change_t2": tuple(pred.change_logits_t2.shape),
        }

        pred.semantic_logits_t1 = restore_semantic(pred.semantic_logits_t1, shape1, output_size)
        pred.semantic_logits_t2 = restore_semantic(pred.semantic_logits_t2, shape2, output_size)
        pred.change_logits_t1 = restore_change(pred.change_logits_t1, shape1, output_size)
        pred.change_logits_t2 = restore_change(pred.change_logits_t2, shape2, output_size)

        token_shapes.update({
            "pixel_semantic_t1": tuple(pred.semantic_logits_t1.shape),
            "pixel_semantic_t2": tuple(pred.semantic_logits_t2.shape),
            "pixel_change_t1": tuple(pred.change_logits_t1.shape),
            "pixel_change_t2": tuple(pred.change_logits_t2.shape),
        })

        return pred, token_shapes


def grad_summary(module):
    count, tensors, max_abs = 0, 0, 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        tensors += 1
        count += p.grad.numel()
        max_abs = max(max_abs, float(p.grad.detach().abs().max().cpu()))
    return tensors, count, max_abs


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_seed(args.seed)
    device = torch.device("cuda:0")
    root = args.dataset_root.expanduser().resolve()
    manifest = root / "manifests" / f"{args.split}.jsonl"
    spec = load_spec(args.spec)

    dataset = UnifiedPAIRDataset(manifest, spec)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_one,
    )

    print("=" * 96)
    print("PAIR FULL ONE-EPOCH TRAINING TEST")
    print("=" * 96)
    print("Dataset      :", root)
    print("Manifest     :", manifest)
    print("Spec         :", f"dataset.config.{args.spec}")
    print("Classes      :", spec.class_names)
    print("Samples      :", len(dataset))
    print("Epoch steps  :", len(loader))
    print("Device       :", torch.cuda.get_device_name(0))
    print()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("[1] Build Qwen + PAIR + Unified Decoder")
    qwen = Qwen3VLBackbone(
        model_dir=args.model_dir,
        dtype=torch.bfloat16,
        device="cuda:0",
        device_map="cuda:0",
        local_files_only=True,
    )
    pair = PAIRModel(qwen_backbone=qwen)
    adapter = ImageDenseAdapter(qwen.hidden_size, args.decoder_dim).to(device)
    decoder = UnifiedChangeDecoder(
        qwen_dim=qwen.hidden_size,
        decoder_dim=args.decoder_dim,
    ).to(device)

    model = OneEpochPAIR(pair, decoder, adapter).to(device)
    criterion = PAIRSemanticChangeLoss().to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    print("Trainable params:", f"{sum(p.numel() for p in trainable) / 1e6:.2f} M")
    print("Qwen frozen     :", not any(p.requires_grad for p in qwen.model.parameters()))
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "epoch_log.jsonl"

    model.train()
    epoch_start = time.time()
    running = {"loss": 0.0, "sem1": 0.0, "sem2": 0.0, "bce": 0.0, "dice": 0.0}

    first_weight = next(decoder.parameters()).detach().clone()

    for step, sample in enumerate(loader, start=1):
        step_start = time.time()
        target = sample["target"]
        output_size = tuple(target["change"].shape[-2:])

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, debug = model(
                image_t1=tensor_to_pil(sample["images_t1"]),
                image_t2=tensor_to_pil(sample["images_t2"]),
                prompt=sample["prompt"],
                class_names=spec.class_names,
                output_size=output_size,
            )
            losses = criterion(
                prediction=prediction,
                target=target,
                class_names=spec.class_names,
            )

        if not torch.isfinite(losses.total):
            raise RuntimeError(f"Non-finite total loss at step {step}: {losses.total}")

        losses.total.backward()

        adapter_grad = grad_summary(adapter)
        decoder_grad = grad_summary(decoder)

        if adapter_grad[0] == 0:
            raise RuntimeError(f"ImageDenseAdapter has no gradients at step {step}")
        if decoder_grad[0] == 0:
            raise RuntimeError(f"UnifiedChangeDecoder has no gradients at step {step}")

        if args.max_grad_norm > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm).detach().cpu()
            )
        else:
            grad_norm = float("nan")

        optimizer.step()

        values = {
            "loss": float(losses.total.detach().cpu()),
            "sem1": float(losses.semantic_t1.detach().cpu()),
            "sem2": float(losses.semantic_t2.detach().cpu()),
            "bce": float(losses.change_bce.detach().cpu()),
            "dice": float(losses.change_dice.detach().cpu()),
        }
        for key in running:
            running[key] += values[key]

        if step == 1:
            print("[2] First real sample")
            print("sample_id              :", sample["sample_id"])
            for key, value in debug.items():
                print(f"{key:23s}:", value)
            print("loss                    :", values)
            print("adapter grad tensors    :", adapter_grad[0], "max|grad| =", adapter_grad[2])
            print("decoder grad tensors    :", decoder_grad[0], "max|grad| =", decoder_grad[2])
            print("global grad norm        :", grad_norm)
            print("Qwen grad tensors       :", sum(p.grad is not None for p in qwen.model.parameters()))
            print()

        if step % args.log_every == 0 or step == len(loader):
            elapsed = time.time() - epoch_start
            mean = {k: v / step for k, v in running.items()}
            sec_per_step = elapsed / step
            eta = sec_per_step * (len(loader) - step)

            print(
                f"[{step:4d}/{len(loader)}] "
                f"loss={values['loss']:.4f} "
                f"avg={mean['loss']:.4f} "
                f"sem1={values['sem1']:.4f} "
                f"sem2={values['sem2']:.4f} "
                f"bce={values['bce']:.4f} "
                f"dice={values['dice']:.4f} "
                f"{sec_per_step:.2f}s/step "
                f"ETA={eta/60:.1f}min"
            )

            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step,
                    "sample_id": sample["sample_id"],
                    **values,
                    "avg_loss": mean["loss"],
                    "sec_per_step": sec_per_step,
                }) + "\n")

    epoch_time = time.time() - epoch_start
    mean = {k: v / len(loader) for k, v in running.items()}
    weight_delta = float((next(decoder.parameters()).detach() - first_weight).abs().mean().cpu())

    checkpoint = {
        "decoder": decoder.state_dict(),
        "image_adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "spec": args.spec,
        "epoch": 1,
    }
    torch.save(checkpoint, args.output_dir / "epoch_1.pt")

    print()
    print("=" * 96)
    print("ONE EPOCH PASS")
    print("=" * 96)
    print("steps                 :", len(loader))
    print("mean total loss       :", mean["loss"])
    print("mean semantic T1      :", mean["sem1"])
    print("mean semantic T2      :", mean["sem2"])
    print("mean change BCE       :", mean["bce"])
    print("mean change Dice      :", mean["dice"])
    print("decoder weight delta  :", weight_delta)
    print("epoch time            :", f"{epoch_time/60:.2f} min")
    print("peak allocated GPU    :", f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
    print("peak reserved GPU     :", f"{torch.cuda.max_memory_reserved()/1024**3:.2f} GiB")
    print("checkpoint            :", args.output_dir / "epoch_1.pt")
    print("log                   :", log_path)

    if weight_delta <= 0:
        raise RuntimeError("Epoch completed but decoder parameters did not update")


if __name__ == "__main__":
    main()
