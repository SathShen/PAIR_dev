"""
PAIR multi-dataset runtime.

One optimizer update selects one dataset. All gradient-accumulation
microbatches in that update use the same dataset. Under DDP, rank 0 creates
the epoch dataset schedule and broadcasts it to every rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from datasets.config_loader import DatasetConfig, ExperimentConfig
from datasets.pair_dataset import UnifiedPAIRDataset


def collate_pair(batch):
    if not batch:
        raise RuntimeError("PAIR received an empty batch")
    return list(batch)


class DistributedEvalSampler(Sampler[int]):
    """Distributed validation without duplicate/padded examples."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(
            range(self.rank, len(self.dataset), self.num_replicas)
        )

    def __len__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.num_replicas + 1


class CyclingLoader:
    def __init__(self, loader, sampler=None):
        self.loader = loader
        self.sampler = sampler
        self.base_epoch = 0
        self.cycle = 0
        self.iterator = None

    def reset(self, epoch):
        self.base_epoch = int(epoch)
        self.cycle = 0
        if self.sampler is not None:
            self.sampler.set_epoch(self.base_epoch)
        self.iterator = iter(self.loader)

    def next(self):
        if self.iterator is None:
            self.reset(self.base_epoch)

        try:
            return next(self.iterator)
        except StopIteration:
            self.cycle += 1
            if self.sampler is not None:
                self.sampler.set_epoch(
                    self.base_epoch * 100000 + self.cycle
                )
            self.iterator = iter(self.loader)
            return next(self.iterator)


@dataclass
class DatasetHandle:
    config: DatasetConfig
    train_dataset: UnifiedPAIRDataset
    train_loader: DataLoader
    train_sampler: Optional[DistributedSampler]
    train_cycle: CyclingLoader
    val_dataset: Optional[UnifiedPAIRDataset]
    val_loader: Optional[DataLoader]

    @property
    def train_batches(self):
        return len(self.train_loader)


class DatasetRegistry:
    def __init__(
        self,
        experiment: ExperimentConfig,
        runtime: Dict,
        num_workers: int,
    ):
        self.experiment = experiment
        self.runtime = runtime
        self.handles: Dict[str, DatasetHandle] = {}

        for name in experiment.selected_names:
            cfg = experiment.datasets[name]

            train_ds = UnifiedPAIRDataset(
                cfg.train_manifest,
                cfg.spec,
            )

            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=runtime["world_size"],
                rank=runtime["rank"],
                shuffle=True,
                seed=int(
                    experiment.experiment.get("seed", 42)
                ),
                drop_last=False,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.per_gpu_batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                collate_fn=collate_pair,
            )

            val_ds = None
            val_loader = None

            if cfg.val_manifest is not None:
                val_ds = UnifiedPAIRDataset(
                    cfg.val_manifest,
                    cfg.spec,
                )

                val_sampler = (
                    DistributedEvalSampler(
                        val_ds,
                        runtime["world_size"],
                        runtime["rank"],
                    )
                    if runtime["distributed"]
                    else None
                )

                val_loader = DataLoader(
                    val_ds,
                    batch_size=cfg.per_gpu_batch_size,
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    collate_fn=collate_pair,
                )

            self.handles[name] = DatasetHandle(
                config=cfg,
                train_dataset=train_ds,
                train_loader=train_loader,
                train_sampler=train_sampler,
                train_cycle=CyclingLoader(
                    train_loader,
                    train_sampler,
                ),
                val_dataset=val_ds,
                val_loader=val_loader,
            )

    def reset_epoch(self, epoch):
        for handle in self.handles.values():
            handle.train_cycle.reset(epoch)

    def next_train_batch(self, name):
        return self.handles[name].train_cycle.next()

    def default_updates_per_epoch(self, grad_accum):
        return sum(
            math.ceil(
                handle.train_batches / grad_accum
            )
            for handle in self.handles.values()
        )

    def runtime_summary(self):
        result = {}
        for name, handle in self.handles.items():
            cfg = handle.config
            result[name] = {
                "route": cfg.route,
                "modalities": list(cfg.modalities),
                "label_mode": cfg.label_mode,
                "unchanged_raw_id": cfg.unchanged_raw_id,
                "train_samples": len(handle.train_dataset),
                "val_samples": (
                    0
                    if handle.val_dataset is None
                    else len(handle.val_dataset)
                ),
                "train_batches_per_loader_pass": len(
                    handle.train_loader
                ),
                "per_gpu_batch_size": cfg.per_gpu_batch_size,
                "sampling_weight": cfg.sampling_weight,
            }
        return result


class MultiDatasetScheduler:
    """
    Deterministic weighted optimizer-update scheduler.

    Rank 0 samples dataset IDs and broadcasts the full epoch schedule.
    """

    def __init__(self, experiment: ExperimentConfig):
        self.names = tuple(experiment.selected_names)

        weights = [
            float(
                experiment.datasets[name].sampling_weight
            )
            for name in self.names
        ]

        w = torch.tensor(
            weights,
            dtype=torch.float64,
        )
        self.probabilities = w / w.sum()
        self.seed = int(
            experiment.experiment.get("seed", 42)
        )

    def epoch_schedule(
        self,
        epoch: int,
        num_updates: int,
        runtime: Dict,
    ) -> List[str]:
        if num_updates <= 0:
            return []

        if runtime["is_main"]:
            generator = torch.Generator(
                device="cpu"
            )
            generator.manual_seed(
                self.seed + int(epoch)
            )

            indices = torch.multinomial(
                self.probabilities,
                num_samples=int(num_updates),
                replacement=True,
                generator=generator,
            ).to(
                dtype=torch.long,
                device=runtime["device"],
            )
        else:
            indices = torch.empty(
                int(num_updates),
                dtype=torch.long,
                device=runtime["device"],
            )

        if runtime["distributed"]:
            dist.broadcast(indices, src=0)

        return [
            self.names[int(i)]
            for i in indices.cpu().tolist()
        ]

    def summary(self):
        return {
            name: float(
                self.probabilities[i].item()
            )
            for i, name in enumerate(self.names)
        }
