#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR training framework V1.

Current first closed loop
-------------------------
dataset/config.py DatasetSpec
    -> UnifiedPAIRDataset
    -> PAIR temporal 2D backbone
    -> Temporal2DChangeDecoder
    -> semantic T1 CE
    -> semantic T2 CE
    -> binary change BCE + Dice
    -> backward / optimizer / scheduler / checkpoint / validation

Design inherited from the retained Qwen3D training framework:
- torch distributed / DDP
- BF16 autocast
- AdamW
- warmup schedule
- gradient clipping
- checkpoint / resume
- periodic validation

But this file deliberately does NOT bring back Detectron2, Mask2Former,
ScanNet mappers/evaluators, or Slurm.

PAIR V1 currently supports batch size 1. Use gradient accumulation for a
larger effective batch.

Single GPU smoke:
    CUDA_VISIBLE_DEVICES=2 python train_pair.py \
        --dataset-root /home/sht/Datasets/SECONDpair \
        --spec SECOND_SPEC \
        --max-steps 20 \
        --output-dir outputs/second_smoke

Later 4-GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc_per_node=4 train_pair.py \
        --dataset-root /home/sht/Datasets/SECONDpair \
        --spec SECOND_SPEC \
        --max-steps 10000 \
        --output-dir outputs/second
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_constant_schedule_with_warmup

from dataset.pair_dataset import (
    DatasetSpec,
    UnifiedPAIRDataset,
)
from models.pair import PAIRModel
from models.qwen3vl_backbone import Qwen3VLBackbone
from models.change_decoder import (
    Temporal2DChangeDecoder,
)
from loss import PAIRSemanticChangeLoss


DEFAULT_MODEL_DIR = (
    "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"
)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--spec",
        type=str,
        required=True,
        help="DatasetSpec variable in dataset/config.py",
    )

    p.add_argument(
        "--train-split",
        default="train",
    )

    p.add_argument(
        "--val-split",
        default="val",
    )

    p.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pair"),
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )

    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--decoder-dim",
        type=int,
        default=256,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    p.add_argument(
        "--val-every",
        type=int,
        default=500,
    )

    p.add_argument(
        "--val-steps",
        type=int,
        default=50,
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=500,
    )

    p.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--train-qwen",
        action="store_true",
        help=(
            "Unfreeze full Qwen. Not recommended for the first smoke test. "
            "The first V1 run should train the dense decoder only; LoRA will "
            "be added next for efficient Qwen adaptation."
        ),
    )

    return p.parse_args()


# ======================================================================
# Distributed
# ======================================================================

def setup_distributed():
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    distributed = world_size > 1

    if distributed:
        local_rank = int(
            os.environ[
                "LOCAL_RANK"
            ]
        )

        torch.cuda.set_device(
            local_rank
        )

        dist.init_process_group(
            backend="nccl"
        )

        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        torch.cuda.set_device(0)

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": torch.device(
            "cuda",
            local_rank,
        ),
        "is_main": rank == 0,
    }


def cleanup_distributed():
    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


# ======================================================================
# Reproducibility
# ======================================================================

def set_seed(seed, rank=0):
    seed = int(seed) + int(rank)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================
# Dataset config
# ======================================================================

def load_spec(name: str) -> DatasetSpec:
    module = importlib.import_module(
        "dataset.config"
    )

    if not hasattr(
        module,
        name,
    ):
        available = [
            key
            for key, value
            in vars(module).items()
            if isinstance(
                value,
                DatasetSpec,
            )
        ]

        raise AttributeError(
            f"dataset.config.{name} not found. "
            f"Available specs: {available}"
        )

    spec = getattr(
        module,
        name,
    )

    if not isinstance(
        spec,
        DatasetSpec,
    ):
        raise TypeError(
            f"dataset.config.{name} is not DatasetSpec"
        )

    if not isinstance(
        spec.class_names,
        dict,
    ):
        raise TypeError(
            f"{name}.class_names must be Dict[int, str]. "
            "PAIR does not infer classes from labels."
        )

    return spec


# ======================================================================
# Image bridge
# ======================================================================

def tensor_to_pil(
    image: torch.Tensor,
) -> Image.Image:
    x = (
        image.detach()
        .cpu()
        .float()
    )

    if (
        x.ndim != 3
        or x.shape[0] not in (1, 3, 4)
    ):
        raise ValueError(
            f"expected [C,H,W], got {tuple(x.shape)}"
        )

    if not torch.isfinite(
        x
    ).all():
        raise ValueError(
            "image contains NaN/Inf"
        )

    lo = float(
        x.min()
    )
    hi = float(
        x.max()
    )

    if (
        lo < -1e-4
        or hi > 1.0001
    ):
        raise ValueError(
            f"image expected [0,1], got [{lo},{hi}]"
        )

    x = (
        x.clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
    )

    arr = (
        x.permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    if arr.shape[-1] == 1:
        arr = arr[..., 0]

    return Image.fromarray(
        arr
    )


# ======================================================================
# Trainable wrapper
# ======================================================================

class PAIR2DTrainModel(torch.nn.Module):
    def __init__(
        self,
        *,
        pair_model: PAIRModel,
        decoder: Temporal2DChangeDecoder,
        freeze_qwen: bool,
    ):
        super().__init__()

        self.pair = pair_model
        self.decoder = decoder
        self.freeze_qwen = bool(
            freeze_qwen
        )

        if self.freeze_qwen:
            self.pair.qwen_backbone.model.requires_grad_(
                False
            )

    def train(self, mode: bool = True):
        super().train(mode)

        # Frozen foundation model behaves deterministically and stores no
        # dropout/training-state behavior. Decoder remains in train mode.
        if (
            mode
            and self.freeze_qwen
        ):
            self.pair.qwen_backbone.model.eval()

        return self

    def forward(
        self,
        *,
        image_t1,
        image_t2,
        prompt,
        class_names,
        output_size,
    ):
        pair_kwargs = dict(
            task_mode="2d",
            prompt=prompt,
            images_t1=image_t1,
            images_t2=image_t2,
            return_logits=False,
            return_hidden_states=True,
            use_cache=False,
        )

        if self.freeze_qwen:
            with torch.no_grad():
                pair_out = self.pair(
                    **pair_kwargs
                )
        else:
            pair_out = self.pair(
                **pair_kwargs
            )

        dense_t1 = pair_out.aux[
            "image_hidden_2d_t1"
        ]

        dense_t2 = pair_out.aux[
            "image_hidden_2d_t2"
        ]

        prediction = self.decoder(
            image_hidden_2d_t1=dense_t1,
            image_hidden_2d_t2=dense_t2,
            task_hidden=pair_out.task_hidden,
            class_names=class_names,
            qwen_backbone=self.pair.qwen_backbone,
            output_size=output_size,
        )

        return prediction


# ======================================================================
# Data
# ======================================================================

def trivial_collate(batch):
    # PAIR V1 batch size = 1.
    if len(batch) != 1:
        raise RuntimeError(
            "PAIR V1 currently requires per-GPU batch size 1."
        )

    return batch[0]


def make_loader(
    dataset,
    *,
    distributed,
    shuffle,
    num_workers,
):
    sampler = None

    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=(
            shuffle
            and sampler is None
        ),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
        collate_fn=trivial_collate,
    )

    return loader, sampler


def infinite_loader(
    loader,
    sampler=None,
):
    epoch = 0

    while True:
        if sampler is not None:
            sampler.set_epoch(
                epoch
            )

        for sample in loader:
            yield sample

        epoch += 1


# ======================================================================
# Build model
# ======================================================================

def build_model(
    args,
    runtime,
):
    device_string = (
        f"cuda:{runtime['local_rank']}"
    )

    qwen = Qwen3VLBackbone(
        model_dir=args.model_dir,
        dtype=torch.bfloat16,
        device=device_string,
        device_map=device_string,
        local_files_only=True,
    )

    pair = PAIRModel(
        qwen_backbone=qwen,
    )

    decoder = Temporal2DChangeDecoder(
        qwen_dim=qwen.hidden_size,
        decoder_dim=args.decoder_dim,
    ).to(
        runtime["device"]
    )

    model = PAIR2DTrainModel(
        pair_model=pair,
        decoder=decoder,
        freeze_qwen=(
            not args.train_qwen
        ),
    )

    model.to(
        runtime["device"]
    )

    return model


# ======================================================================
# Checkpoint
# ======================================================================

def model_for_state(model):
    return (
        model.module
        if isinstance(
            model,
            DDP,
        )
        else model
    )


def trainable_parameter_state(
    model,
):
    """
    Save trainable Qwen parameters only.

    For V1 with frozen Qwen this is empty, so checkpoints remain small.
    """
    base = model_for_state(
        model
    )

    trainable_names = {
        name
        for name, param
        in base.pair.named_parameters()
        if param.requires_grad
    }

    state = base.pair.state_dict()

    return {
        key: value.detach().cpu()
        for key, value
        in state.items()
        if key in trainable_names
    }


def save_checkpoint(
    *,
    path,
    model,
    optimizer,
    scheduler,
    step,
    args,
    spec_name,
):
    base = model_for_state(
        model
    )

    checkpoint = {
        "step": int(step),
        "decoder": (
            base.decoder
            .state_dict()
        ),
        "pair_trainable": (
            trainable_parameter_state(
                model
            )
        ),
        "optimizer": (
            optimizer.state_dict()
        ),
        "scheduler": (
            scheduler.state_dict()
        ),
        "args": vars(args),
        "spec_name": spec_name,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    *,
    path,
    model,
    optimizer,
    scheduler,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    base = model_for_state(
        model
    )

    base.decoder.load_state_dict(
        checkpoint[
            "decoder"
        ]
    )

    pair_state = checkpoint.get(
        "pair_trainable",
        {},
    )

    if pair_state:
        current = base.pair.state_dict()

        current.update(
            pair_state
        )

        base.pair.load_state_dict(
            current,
            strict=False,
        )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler"
        ]
    )

    return int(
        checkpoint[
            "step"
        ]
    )


# ======================================================================
# One forward/loss
# ======================================================================

def forward_loss(
    *,
    model,
    criterion,
    sample,
    spec,
):
    image_t1 = tensor_to_pil(
        sample[
            "images_t1"
        ]
    )

    image_t2 = tensor_to_pil(
        sample[
            "images_t2"
        ]
    )

    target = sample[
        "target"
    ]

    output_size = tuple(
        target[
            "change"
        ].shape[-2:]
    )

    prediction = model(
        image_t1=image_t1,
        image_t2=image_t2,
        prompt=sample[
            "prompt"
        ],
        class_names=spec.class_names,
        output_size=output_size,
    )

    loss_output = criterion(
        prediction=prediction,
        target=target,
        class_names=spec.class_names,
    )

    return prediction, loss_output


# ======================================================================
# Validation
# ======================================================================

@torch.no_grad()
def validate(
    *,
    model,
    criterion,
    loader,
    spec,
    max_steps,
    runtime,
):
    model.eval()

    sums = {}
    count = 0

    for sample in loader:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            _, loss_output = (
                forward_loss(
                    model=model,
                    criterion=criterion,
                    sample=sample,
                    spec=spec,
                )
            )

        values = (
            loss_output.as_dict()
        )

        for key, value in values.items():
            sums[key] = (
                sums.get(
                    key,
                    0.0,
                )
                + float(
                    value.detach()
                    .cpu()
                )
            )

        count += 1

        if count >= max_steps:
            break

    model.train()

    if count == 0:
        return {}

    result = {
        key: value / count
        for key, value
        in sums.items()
    }

    if (
        runtime["distributed"]
    ):
        # First V1 keeps validation logging local to rank 0.
        # Full distributed metric reduction comes with evaluator V2.
        pass

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    runtime = setup_distributed()

    try:
        set_seed(
            args.seed,
            runtime["rank"],
        )

        spec = load_spec(
            args.spec
        )

        if (
            spec.task_mode
            .lower()
            .strip()
            != "2d"
        ):
            raise ValueError(
                "Training V1 currently closes the 2D loop first."
            )

        root = (
            args.dataset_root
            .expanduser()
            .resolve()
        )

        train_manifest = (
            root
            / "manifests"
            / f"{args.train_split}.jsonl"
        )

        val_manifest = (
            root
            / "manifests"
            / f"{args.val_split}.jsonl"
        )

        train_dataset = (
            UnifiedPAIRDataset(
                train_manifest,
                spec,
            )
        )

        val_dataset = (
            UnifiedPAIRDataset(
                val_manifest,
                spec,
            )
            if val_manifest.exists()
            else None
        )

        train_loader, train_sampler = (
            make_loader(
                train_dataset,
                distributed=runtime[
                    "distributed"
                ],
                shuffle=True,
                num_workers=args.num_workers,
            )
        )

        if val_dataset is not None:
            val_loader, _ = (
                make_loader(
                    val_dataset,
                    distributed=False,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
            )
        else:
            val_loader = None

        model = build_model(
            args,
            runtime,
        )

        if runtime["distributed"]:
            model = DDP(
                model,
                device_ids=[
                    runtime[
                        "local_rank"
                    ]
                ],
                output_device=runtime[
                    "local_rank"
                ],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        criterion = (
            PAIRSemanticChangeLoss(
                semantic_weight=1.0,
                change_bce_weight=1.0,
                change_dice_weight=1.0,
            )
            .to(
                runtime["device"]
            )
        )

        params = [
            p
            for p in model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = (
            get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
            )
        )

        start_step = 0

        if args.resume is not None:
            start_step = load_checkpoint(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

        if runtime["is_main"]:
            args.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print("=" * 88)
            print("PAIR TRAINING V1")
            print("=" * 88)
            print(
                "Spec:",
                args.spec,
            )
            print(
                "Classes:",
                spec.class_names,
            )
            print(
                "Train samples:",
                len(train_dataset),
            )
            print(
                "Val samples:",
                (
                    len(val_dataset)
                    if val_dataset is not None
                    else 0
                ),
            )
            print(
                "World size:",
                runtime["world_size"],
            )
            print(
                "Train Qwen:",
                args.train_qwen,
            )
            print(
                "Gradient accumulation:",
                args.grad_accum,
            )
            print(
                "Effective global batch:",
                (
                    runtime[
                        "world_size"
                    ]
                    * args.grad_accum
                ),
            )
            print()

        model.train()

        train_iter = infinite_loader(
            train_loader,
            train_sampler,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        log_sums = {}
        log_count = 0
        wall_start = time.time()

        for step in range(
            start_step,
            args.max_steps,
        ):
            sample = next(
                train_iter
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                _, loss_output = (
                    forward_loss(
                        model=model,
                        criterion=criterion,
                        sample=sample,
                        spec=spec,
                    )
                )

                loss = (
                    loss_output.total
                    / args.grad_accum
                )

            loss.backward()

            values = (
                loss_output.as_dict()
            )

            for key, value in values.items():
                log_sums[key] = (
                    log_sums.get(
                        key,
                        0.0,
                    )
                    + float(
                        value.detach()
                        .cpu()
                    )
                )

            log_count += 1

            update = (
                (step + 1)
                % args.grad_accum
                == 0
            )

            if update:
                if (
                    args.max_grad_norm
                    > 0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        params,
                        args.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            global_step = step + 1

            if (
                runtime["is_main"]
                and global_step
                % args.log_every
                == 0
            ):
                elapsed = (
                    time.time()
                    - wall_start
                )

                means = {
                    key: value
                    / max(
                        log_count,
                        1,
                    )
                    for key, value
                    in log_sums.items()
                }

                lr = optimizer.param_groups[
                    0
                ][
                    "lr"
                ]

                print(
                    f"step {global_step:6d} | "
                    f"loss {means['loss']:.4f} | "
                    f"sem1 {means['loss_semantic_t1']:.4f} | "
                    f"sem2 {means['loss_semantic_t2']:.4f} | "
                    f"bce {means['loss_change_bce']:.4f} | "
                    f"dice {means['loss_change_dice']:.4f} | "
                    f"lr {lr:.3e} | "
                    f"{elapsed / args.log_every:.2f}s/step"
                )

                record = {
                    "step": global_step,
                    "lr": lr,
                    **means,
                }

                with (
                    args.output_dir
                    / "train_log.jsonl"
                ).open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        json.dumps(
                            record
                        )
                        + "\n"
                    )

                log_sums = {}
                log_count = 0
                wall_start = time.time()

            if (
                val_loader is not None
                and global_step
                % args.val_every
                == 0
            ):
                if runtime["distributed"]:
                    dist.barrier()

                if runtime["is_main"]:
                    val_result = validate(
                        model=model_for_state(
                            model
                        ),
                        criterion=criterion,
                        loader=val_loader,
                        spec=spec,
                        max_steps=args.val_steps,
                        runtime=runtime,
                    )

                    print(
                        "VAL",
                        global_step,
                        val_result,
                    )

                if runtime["distributed"]:
                    dist.barrier()

                model.train()

            if (
                runtime["is_main"]
                and global_step
                % args.save_every
                == 0
            ):
                save_checkpoint(
                    path=(
                        args.output_dir
                        / f"step_{global_step:07d}.pt"
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    args=args,
                    spec_name=args.spec,
                )

        if runtime["is_main"]:
            save_checkpoint(
                path=(
                    args.output_dir
                    / "last.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=args.max_steps,
                args=args,
                spec_name=args.spec,
            )

            print("Training complete.")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
