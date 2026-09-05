"""
PAIR multi-dataset runtime.

Automatic dataset balancing
---------------------------
PAIR does not use user-authored sampling weights.

For every dataset, one experiment epoch consumes exactly one pass of that
dataset's per-rank DataLoader. With gradient accumulation G:

    full_updates = train_batches // G
    final_microbatches = train_batches % G

The final optimizer update for a dataset may therefore use fewer than G
micro-batches. This avoids wrapping/oversampling merely because a loader length
is not divisible by G.

The resulting per-dataset optimizer-update entries are concatenated and
deterministically shuffled each epoch. Under DDP, rank 0 creates the shuffled
plan and broadcasts it, so every rank executes the same modality branch at the
same optimizer update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
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
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.num_replicas + 1


class CyclingLoader:
    """
    Stateful loader wrapper.

    With the default exact epoch plan this should never wrap inside an epoch;
    wrapping remains as a defensive fallback for future custom schedules.
    """

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
                self.sampler.set_epoch(self.base_epoch * 100000 + self.cycle)
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


@dataclass(frozen=True)
class DatasetUpdate:
    dataset_name: str
    microbatches: int


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

            train_ds = UnifiedPAIRDataset(cfg.train_manifest, cfg.spec)

            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=runtime["world_size"],
                rank=runtime["rank"],
                shuffle=True,
                seed=int(experiment.experiment.get("seed", 42)),
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
                val_ds = UnifiedPAIRDataset(cfg.val_manifest, cfg.spec)

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
                train_cycle=CyclingLoader(train_loader, train_sampler),
                val_dataset=val_ds,
                val_loader=val_loader,
            )

    def reset_epoch(self, epoch):
        for handle in self.handles.values():
            handle.train_cycle.reset(epoch)

    def next_train_batch(self, name):
        return self.handles[name].train_cycle.next()

    def consume_updates(self, updates: Sequence[DatasetUpdate]):
        """
        Advance loader iterators when resuming in the middle of an epoch.
        """
        for update in updates:
            for _ in range(update.microbatches):
                self.next_train_batch(update.dataset_name)

    def runtime_summary(self, grad_accum):
        result = {}
        total_updates = sum(
            math.ceil(handle.train_batches / grad_accum)
            for handle in self.handles.values()
        )

        for name, handle in self.handles.items():
            cfg = handle.config
            train_batches = handle.train_batches
            optimizer_updates = math.ceil(train_batches / grad_accum)

            result[name] = {
                "route": cfg.route,
                "modalities": list(cfg.modalities),
                "label_mode": cfg.label_mode,
                "unchanged_raw_id": cfg.unchanged_raw_id,
                "train_samples": len(handle.train_dataset),
                "val_samples": (
                    0 if handle.val_dataset is None else len(handle.val_dataset)
                ),
                "train_batches": train_batches,
                "optimizer_updates_per_epoch": optimizer_updates,
                "update_fraction": (
                    optimizer_updates / total_updates if total_updates else 0.0
                ),
                "per_gpu_batch_size": cfg.per_gpu_batch_size,
            }

        return result


class MultiDatasetScheduler:
    """
    Build an exact, deterministic one-pass multi-dataset epoch plan.
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        registry: DatasetRegistry,
        grad_accum: int,
    ):
        if grad_accum <= 0:
            raise ValueError("grad_accum must be > 0")

        self.names = tuple(experiment.selected_names)
        self.registry = registry
        self.grad_accum = int(grad_accum)
        self.seed = int(experiment.experiment.get("seed", 42))

        self.base_plan: List[DatasetUpdate] = []
        self.per_dataset = {}

        for name in self.names:
            train_batches = registry.handles[name].train_batches
            full_updates, remainder = divmod(train_batches, self.grad_accum)

            updates = [
                DatasetUpdate(name, self.grad_accum)
                for _ in range(full_updates)
            ]
            if remainder:
                updates.append(DatasetUpdate(name, remainder))

            self.base_plan.extend(updates)
            self.per_dataset[name] = {
                "train_batches": train_batches,
                "full_updates": full_updates,
                "final_microbatches": remainder,
                "optimizer_updates": len(updates),
            }

        if not self.base_plan:
            raise ValueError("Multi-dataset epoch plan is empty")

    @property
    def updates_per_epoch(self) -> int:
        return len(self.base_plan)

    @property
    def microbatches_per_epoch(self) -> int:
        return sum(x.microbatches for x in self.base_plan)

    def epoch_schedule(
        self,
        epoch: int,
        runtime: Dict,
    ) -> List[DatasetUpdate]:
        """
        Shuffle exact update entries without changing per-dataset quotas.
        """
        n = len(self.base_plan)

        if runtime["is_main"]:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + int(epoch))
            permutation = torch.randperm(
                n,
                generator=generator,
                dtype=torch.long,
            ).to(runtime["device"])
        else:
            permutation = torch.empty(
                n,
                dtype=torch.long,
                device=runtime["device"],
            )

        if runtime["distributed"]:
            dist.broadcast(permutation, src=0)

        order = permutation.cpu().tolist()
        return [self.base_plan[i] for i in order]

    def summary(self):
        total = self.updates_per_epoch
        return {
            name: {
                **info,
                "update_fraction": (
                    info["optimizer_updates"] / total if total else 0.0
                ),
            }
            for name, info in self.per_dataset.items()
        }
