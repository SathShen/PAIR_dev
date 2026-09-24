"""
PAIR unified loss — V1.

Design goals
------------
1) Keep the loss aligned with PAIR's task protocol instead of coupling it to
   any specific optimizer.
2) Use the same small set of robust primitives across 2D and 3D:
      semantic: CE + Lovasz-Softmax
      binary change: BCE + Dice
      2D SCD auxiliary: PerASCD Soft Semantic Consistency Loss (SSCLoss)
      3D event: active-support CE + derived change Dice
3) Normalize across ACTIVE task groups so a batch with more annotated heads
   does not automatically contribute a proportionally larger total loss.
4) Keep the existing 3D event protocol exactly:
      0 unchanged
      1 added
      2 removed
      T1 support = {0, 2}
      T2 support = {0, 1}
   There is still NO separate 3D binary-change head. The event Dice term is
   derived directly from the shared event head.

Routes
------
2D SCD:
    semantic_logits_t1/t2 + binary change_logits_t1/t2
    semantic CE/Lovasz are applied on changed pixels only when
    semantic_changed_only=True (the train.py semantic_pair protocol).

2D BCD:
    semantic supervision may be fully masked out; binary change remains active.

3D:
    semantic_logits_t1/t2 + event_logits_t1/t2

Future 2D+3D:
    binary change logits and event logits may coexist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PAIR_EVENT_NUM_CLASSES = 3
PAIR_EVENT_ACTIVE_SUPPORT = {
    1: (0, 2),  # T1: unchanged / removed
    2: (0, 1),  # T2: unchanged / added
}


@dataclass
class ChangeLossOutput:
    total: torch.Tensor

    # Semantic CE kept under the old names for train.py compatibility.
    semantic_t1: torch.Tensor
    semantic_t2: torch.Tensor
    semantic_ce: torch.Tensor
    semantic_lovasz_t1: torch.Tensor
    semantic_lovasz_t2: torch.Tensor
    semantic_lovasz: torch.Tensor
    semantic: torch.Tensor

    # Binary change.
    change_bce: torch.Tensor
    change_dice: torch.Tensor
    change: torch.Tensor

    # PerASCD Soft Semantic Consistency auxiliary loss.
    ssc: torch.Tensor

    # 3D event. event_t1/t2 remain the active-support CE terms.
    event_t1: torch.Tensor
    event_t2: torch.Tensor
    event_ce: torch.Tensor
    event_dice_t1: torch.Tensor
    event_dice_t2: torch.Tensor
    event_dice: torch.Tensor
    event: torch.Tensor

    # Scalar tensor containing the denominator used by active-task
    # normalization. Useful for debugging/logging.
    active_weight_sum: torch.Tensor

    def as_dict(self):
        # Preserve all legacy keys while exposing the new components.
        return {
            "loss": self.total,
            "loss_semantic_t1": self.semantic_t1,
            "loss_semantic_t2": self.semantic_t2,
            "loss_semantic_ce": self.semantic_ce,
            "loss_semantic_lovasz_t1": self.semantic_lovasz_t1,
            "loss_semantic_lovasz_t2": self.semantic_lovasz_t2,
            "loss_semantic_lovasz": self.semantic_lovasz,
            "loss_semantic": self.semantic,
            "loss_change_bce": self.change_bce,
            "loss_change_dice": self.change_dice,
            "loss_change": self.change,
            "loss_ssc": self.ssc,
            "loss_event_t1": self.event_t1,
            "loss_event_t2": self.event_t2,
            "loss_event_ce": self.event_ce,
            "loss_event_dice_t1": self.event_dice_t1,
            "loss_event_dice_t2": self.event_dice_t2,
            "loss_event_dice": self.event_dice,
            "loss_event": self.event,
            "loss_active_weight_sum": self.active_weight_sum,
        }


class PAIRSemanticChangeLoss(nn.Module):
    def __init__(
        self,
        # Keep the original positional argument order intact.
        semantic_weight=1.0,
        change_bce_weight=1.0,
        change_dice_weight=1.0,
        event_weight=1.0,
        dice_eps=1.0,
        # V1 additions.
        semantic_lovasz_weight=0.5,
        change_weight=1.0,
        event_dice_weight=1.0,
        normalize_active_tasks=True,
        # PerASCD SSCLoss. Appended to preserve old positional arguments.
        ssc_weight=0.01,
        ssc_margin=0.1,
        ssc_tau=0.01,
        ssc_eps=1e-8,
    ):
        super().__init__()

        # Task-group weights.
        self.semantic_weight = float(semantic_weight)
        self.change_weight = float(change_weight)
        self.event_weight = float(event_weight)

        # Within-task weights.
        self.semantic_lovasz_weight = float(semantic_lovasz_weight)
        self.change_bce_weight = float(change_bce_weight)
        self.change_dice_weight = float(change_dice_weight)
        self.event_dice_weight = float(event_dice_weight)

        self.dice_eps = float(dice_eps)
        self.normalize_active_tasks = bool(normalize_active_tasks)

        # PerASCD SSCLoss:
        #   unchanged: 1 - cos(x1, x2)
        #   changed  : tau * softplus((cos(x1, x2) - margin) / tau)
        #
        # PerASCD computes it from the two semantic outputs after removing the
        # explicit unchanged channel. SECOND uses a small temperature; the
        # released PerASCD training entry defaults to tau=0.01.
        self.ssc_weight = float(ssc_weight)
        self.ssc_margin = float(ssc_margin)
        self.ssc_tau = float(ssc_tau)
        self.ssc_eps = float(ssc_eps)

        for name, value in (
            ("semantic_weight", self.semantic_weight),
            ("change_weight", self.change_weight),
            ("event_weight", self.event_weight),
            ("semantic_lovasz_weight", self.semantic_lovasz_weight),
            ("change_bce_weight", self.change_bce_weight),
            ("change_dice_weight", self.change_dice_weight),
            ("event_dice_weight", self.event_dice_weight),
            ("ssc_weight", self.ssc_weight),
            ("ssc_eps", self.ssc_eps),
            ("dice_eps", self.dice_eps),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if self.ssc_tau <= 0:
            raise ValueError(f"ssc_tau must be > 0, got {self.ssc_tau}")
        if not (-1.0 <= self.ssc_margin <= 1.0):
            raise ValueError(
                "ssc_margin must lie in [-1, 1] for cosine similarity, "
                f"got {self.ssc_margin}"
            )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_raw_to_local(class_names: Dict[int, str]) -> Dict[int, int]:
        if not isinstance(class_names, dict) or not class_names:
            raise TypeError("class_names must be a non-empty Dict[int, str]")
        raw_ids = sorted(int(k) for k in class_names)
        return {raw_id: local_id for local_id, raw_id in enumerate(raw_ids)}

    @staticmethod
    def _valid_or_true(target, key, value, device):
        valid = target.get(key)
        if valid is None:
            return torch.ones(value.numel(), dtype=torch.bool, device=device)

        valid = valid.to(device).reshape(-1).bool()
        if valid.numel() != value.numel():
            raise ValueError(
                f"{key} size {valid.numel()} does not match target size {value.numel()}"
            )
        return valid

    @staticmethod
    def _zero_from(tensor):
        return tensor.sum() * 0.0

    @staticmethod
    def _mean_over_active(losses: Sequence[torch.Tensor], active: Sequence[bool]):
        selected = [loss for loss, is_active in zip(losses, active) if is_active]
        if not selected:
            return losses[0].sum() * 0.0
        return torch.stack(selected).mean()

    @classmethod
    def remap_semantic_target(
        cls,
        raw_target,
        valid_mask,
        class_names,
        ignore_index=-100,
    ):
        raw_target = raw_target.reshape(-1).long()
        valid_mask = valid_mask.reshape(-1).bool()

        if raw_target.numel() != valid_mask.numel():
            raise ValueError("semantic target and valid mask sizes differ")

        local = torch.full_like(raw_target, int(ignore_index))
        matched = torch.zeros_like(valid_mask)

        for raw_id, local_id in cls.make_raw_to_local(class_names).items():
            mask = valid_mask & (raw_target == raw_id)
            local[mask] = local_id
            matched |= mask

        bad = valid_mask & ~matched
        if bad.any():
            values = torch.unique(raw_target[bad]).detach().cpu().tolist()
            raise ValueError(
                f"Semantic labels {values} are not declared in DatasetSpec.class_names"
            )

        return local

    # ------------------------------------------------------------------
    # Lovasz-Softmax
    # ------------------------------------------------------------------

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = gt_sorted.numel()
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union.clamp_min(1e-12)

        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[:-1]

        return jaccard

    @classmethod
    def _lovasz_softmax_flat(
        cls,
        probas: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if probas.numel() == 0:
            return probas.sum() * 0.0

        num_classes = probas.shape[1]
        losses = []

        # "present" classes only. This avoids penalizing a crop for semantic
        # classes that are absent from its valid supervision.
        for class_id in range(num_classes):
            fg = (labels == class_id).float()
            if fg.sum() == 0:
                continue

            errors = (fg - probas[:, class_id]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]

            losses.append(
                torch.dot(
                    errors_sorted,
                    cls._lovasz_grad(fg_sorted),
                )
            )

        if not losses:
            return probas.sum() * 0.0

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Semantic task: CE + Lovasz
    # ------------------------------------------------------------------

    @classmethod
    def _prepare_semantic(
        cls,
        logits,
        raw_target,
        valid_mask,
        class_names,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits is None or logits.ndim != 2:
            shape = None if logits is None else tuple(logits.shape)
            raise ValueError(f"semantic logits must be [N,K], got {shape}")

        raw_target = raw_target.to(logits.device).reshape(-1)
        valid_mask = valid_mask.to(logits.device).reshape(-1).bool()

        if logits.shape[0] != raw_target.numel():
            raise ValueError(
                f"semantic logits/target size mismatch: "
                f"{logits.shape[0]} vs {raw_target.numel()}"
            )

        if logits.shape[1] != len(class_names):
            raise ValueError(
                f"semantic logits K={logits.shape[1]} but "
                f"class_names has {len(class_names)} classes"
            )

        local_target = cls.remap_semantic_target(
            raw_target,
            valid_mask,
            class_names,
        )

        return logits.float(), local_target, valid_mask

    @classmethod
    def semantic_ce(cls, logits, raw_target, valid_mask, class_names):
        logits, local_target, valid_mask = cls._prepare_semantic(
            logits,
            raw_target,
            valid_mask,
            class_names,
        )

        if not valid_mask.any():
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits,
            local_target,
            ignore_index=-100,
        )

    @classmethod
    def semantic_lovasz(cls, logits, raw_target, valid_mask, class_names):
        logits, local_target, valid_mask = cls._prepare_semantic(
            logits,
            raw_target,
            valid_mask,
            class_names,
        )

        if not valid_mask.any():
            return logits.sum() * 0.0

        probas = F.softmax(logits[valid_mask], dim=1)
        labels = local_target[valid_mask]

        return cls._lovasz_softmax_flat(probas, labels)

    def _semantic_losses(
        self,
        prediction,
        target,
        class_names,
        *,
        changed_only: bool = False,
    ):
        raw1 = target["semantic_t1"]
        raw2 = target["semantic_t2"]

        valid1 = self._valid_or_true(
            target,
            "semantic_valid_t1",
            raw1,
            prediction.semantic_logits_t1.device,
        )
        valid2 = self._valid_or_true(
            target,
            "semantic_valid_t2",
            raw2,
            prediction.semantic_logits_t2.device,
        )

        # Keep track of whether semantic supervision exists BEFORE the
        # changed-only restriction. SSCLoss still needs the unchanged pixels
        # on 2D SCD batches, so its activation must not depend on whether the
        # current crop happens to contain a changed semantic pixel.
        semantic_supervision_active = bool(
            valid1.any().item() or valid2.any().item()
        )

        if changed_only:
            # 2D semantic change detection protocol:
            #   - binary branch learns changed vs unchanged over the full
            #     valid image;
            #   - semantic CE/Lovasz learn semantic discrimination only where
            #     the binary target says the pixel changed.
            #
            # Use the temporal-aware helper so future change_t1/change_t2
            # targets remain compatible without changing the present SECOND /
            # LandsatSCD shared-change convention.
            change1, change_valid1 = self._change_target(target, 1)
            change2, change_valid2 = self._change_target(target, 2)

            change1 = change1.to(valid1.device).reshape(-1)
            change2 = change2.to(valid2.device).reshape(-1)
            change_valid1 = change_valid1.to(valid1.device).reshape(-1).bool()
            change_valid2 = change_valid2.to(valid2.device).reshape(-1).bool()

            if (
                change1.numel() != valid1.numel()
                or change_valid1.numel() != valid1.numel()
                or change2.numel() != valid2.numel()
                or change_valid2.numel() != valid2.numel()
            ):
                raise ValueError(
                    "changed-only semantic mask must match semantic target size"
                )

            for change, change_valid, name in (
                (change1, change_valid1, "change_t1"),
                (change2, change_valid2, "change_t2"),
            ):
                if change_valid.any():
                    values = change[change_valid]
                    if not torch.all((values == 0) | (values == 1)):
                        bad = torch.unique(values).detach().cpu().tolist()
                        raise ValueError(
                            f"{name} valid target must contain only 0/1, got {bad}"
                        )

            valid1 = valid1 & change_valid1 & (change1 == 1)
            valid2 = valid2 & change_valid2 & (change2 == 1)

        active1 = bool(valid1.any().item())
        active2 = bool(valid2.any().item())

        sem1 = self.semantic_ce(
            prediction.semantic_logits_t1,
            raw1,
            valid1,
            class_names,
        )
        sem2 = self.semantic_ce(
            prediction.semantic_logits_t2,
            raw2,
            valid2,
            class_names,
        )

        lov1 = self.semantic_lovasz(
            prediction.semantic_logits_t1,
            raw1,
            valid1,
            class_names,
        )
        lov2 = self.semantic_lovasz(
            prediction.semantic_logits_t2,
            raw2,
            valid2,
            class_names,
        )

        sem_ce = self._mean_over_active(
            (sem1, sem2),
            (active1, active2),
        )
        sem_lovasz = self._mean_over_active(
            (lov1, lov2),
            (active1, active2),
        )

        semantic = sem_ce + self.semantic_lovasz_weight * sem_lovasz
        semantic_active = active1 or active2

        return (
            sem1,
            sem2,
            sem_ce,
            lov1,
            lov2,
            sem_lovasz,
            semantic,
            semantic_active,
            semantic_supervision_active,
        )

    # ------------------------------------------------------------------
    # PerASCD Soft Semantic Consistency Loss (SSCLoss)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        return " ".join(
            str(name)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .split()
        )

    @classmethod
    def _unchanged_local_index(
        cls,
        class_names: Dict[int, str],
    ) -> Optional[int]:
        """
        Find the model-local index of an explicit unchanged/no-change class.

        Do not reinterpret a generic "background" class as unchanged. PerASCD
        removes its explicit class-0 unchanged channel before SSCLoss.
        """
        aliases = {"unchanged", "no change", "non change"}
        matches = [
            int(raw_id)
            for raw_id, class_name in class_names.items()
            if cls._normalize_class_name(class_name) in aliases
        ]

        if not matches:
            return None

        if len(matches) != 1:
            raise ValueError(
                "SSCLoss requires at most one explicit unchanged/no-change "
                f"class, got raw IDs {matches}"
            )

        return cls.make_raw_to_local(class_names)[matches[0]]

    def soft_semantic_consistency(
        self,
        logits_t1: torch.Tensor,
        logits_t2: torch.Tensor,
        change_target: torch.Tensor,
        valid_mask: torch.Tensor,
        unchanged_local_index: int,
    ) -> torch.Tensor:
        """
        PerASCD SSCLoss adapted to PAIR's flattened semantic logits [N,K].

        PerASCD uses the semantic outputs after excluding the unchanged channel:

            unchanged:
                1 - cos(x1, x2)

            changed:
                tau * softplus((cos(x1, x2) - margin) / tau)

        PAIR's binary convention is 0=unchanged, 1=changed.
        """
        if logits_t1 is None or logits_t2 is None:
            raise ValueError("SSCLoss requires both T1/T2 semantic logits")

        if logits_t1.ndim != 2 or logits_t2.ndim != 2:
            raise ValueError(
                "SSCLoss semantic logits must be [N,K], got "
                f"{tuple(logits_t1.shape)} and {tuple(logits_t2.shape)}"
            )

        if logits_t1.shape != logits_t2.shape:
            raise ValueError(
                "SSCLoss requires aligned T1/T2 semantic logits, got "
                f"{tuple(logits_t1.shape)} vs {tuple(logits_t2.shape)}"
            )

        n, k = logits_t1.shape
        unchanged_local_index = int(unchanged_local_index)

        if not (0 <= unchanged_local_index < k):
            raise ValueError(
                f"SSCLoss unchanged index {unchanged_local_index} "
                f"is outside K={k}"
            )
        if k <= 1:
            raise ValueError(
                "SSCLoss needs at least one semantic class after removing "
                "the unchanged channel"
            )

        device = logits_t1.device
        change_target = change_target.to(device).reshape(-1)
        valid_mask = valid_mask.to(device).reshape(-1).bool()

        if change_target.numel() != n or valid_mask.numel() != n:
            raise ValueError(
                "SSCLoss logits/target/mask size mismatch: "
                f"N={n}, target={change_target.numel()}, "
                f"mask={valid_mask.numel()}"
            )

        if valid_mask.any():
            y = change_target[valid_mask]
            if not torch.all((y == 0) | (y == 1)):
                values = torch.unique(y).detach().cpu().tolist()
                raise ValueError(
                    "SSCLoss valid change target must contain only 0/1, "
                    f"got {values}"
                )

        if not valid_mask.any():
            return (logits_t1.sum() + logits_t2.sum()) * 0.0

        keep = torch.ones(k, dtype=torch.bool, device=device)
        keep[unchanged_local_index] = False

        # PerASCD constrains semantic outputs, not softmax probabilities.
        # Promote to FP32 so BF16/AMP does not weaken the softplus transition.
        x1 = logits_t1[valid_mask][:, keep].float()
        x2 = logits_t2[valid_mask][:, keep].float()

        cosine = F.cosine_similarity(
            x1,
            x2,
            dim=1,
            eps=self.ssc_eps,
        ).clamp(-1.0, 1.0)

        changed = change_target[valid_mask].float()

        unchanged_loss = 1.0 - cosine
        changed_loss = self.ssc_tau * F.softplus(
            (cosine - self.ssc_margin) / self.ssc_tau
        )

        per_pixel = torch.where(
            changed > 0.5,
            changed_loss,
            unchanged_loss,
        )
        return per_pixel.mean()

    def _ssc_loss(
        self,
        prediction,
        target,
        class_names: Dict[int, str],
        semantic_active: bool,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Apply SSCLoss only to aligned 2D semantic-change batches.

        Conditions:
        - shared 2D binary target `change` exists;
        - semantic supervision is active;
        - class_names has one explicit unchanged/no-change class.

        Therefore NYC-SCD's 3D event route and pure BCD batches are unchanged.
        """
        zero = self._zero_from(prediction.semantic_logits_t1)

        if self.ssc_weight <= 0 or not semantic_active:
            return zero, False
        if "change" not in target:
            return zero, False

        unchanged_local = self._unchanged_local_index(class_names)
        if unchanged_local is None:
            return zero, False

        raw1 = target["semantic_t1"]
        raw2 = target["semantic_t2"]

        valid_sem1 = self._valid_or_true(
            target,
            "semantic_valid_t1",
            raw1,
            prediction.semantic_logits_t1.device,
        )
        valid_sem2 = self._valid_or_true(
            target,
            "semantic_valid_t2",
            raw2,
            prediction.semantic_logits_t2.device,
        )

        change = target["change"]
        change_valid = target.get("change_valid")
        if change_valid is None:
            change_valid = torch.ones_like(change, dtype=torch.bool)

        device = prediction.semantic_logits_t1.device
        valid = (
            valid_sem1.to(device).reshape(-1).bool()
            & valid_sem2.to(device).reshape(-1).bool()
            & change_valid.to(device).reshape(-1).bool()
        )

        if not valid.any():
            return zero, False

        ssc = self.soft_semantic_consistency(
            prediction.semantic_logits_t1,
            prediction.semantic_logits_t2,
            change,
            valid,
            unchanged_local,
        )
        return ssc, True

    # ------------------------------------------------------------------
    # Binary change task: BCE + Dice
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_change(logits, target, valid_mask):
        if logits is None:
            raise ValueError("binary change logits are missing")

        logits = logits.reshape(-1)
        target = target.to(logits.device).reshape(-1)
        valid_mask = valid_mask.to(logits.device).reshape(-1).bool()

        if logits.numel() != target.numel() or target.numel() != valid_mask.numel():
            raise ValueError(
                f"change logits/target/mask size mismatch: "
                f"{logits.numel()}, {target.numel()}, {valid_mask.numel()}"
            )

        if valid_mask.any():
            y = target[valid_mask]
            if not torch.all((y == 0) | (y == 1)):
                values = torch.unique(y).detach().cpu().tolist()
                raise ValueError(
                    f"valid change target must contain only 0/1, got {values}"
                )

        return logits, target.float(), valid_mask

    def change_bce(self, logits, target, valid_mask):
        logits, target, valid_mask = self._prepare_change(
            logits,
            target,
            valid_mask,
        )

        if not valid_mask.any():
            return logits.sum() * 0.0

        return F.binary_cross_entropy_with_logits(
            logits[valid_mask].float(),
            target[valid_mask],
        )

    def change_dice(self, logits, target, valid_mask):
        logits, target, valid_mask = self._prepare_change(
            logits,
            target,
            valid_mask,
        )

        if not valid_mask.any():
            return logits.sum() * 0.0

        prob = torch.sigmoid(logits[valid_mask].float())
        target = target[valid_mask]

        intersection = (prob * target).sum()
        dice = (2.0 * intersection + self.dice_eps) / (
            prob.sum() + target.sum() + self.dice_eps
        )

        return 1.0 - dice

    @staticmethod
    def _change_target(target, time_id):
        key = f"change_t{time_id}"
        valid_key = f"change_valid_t{time_id}"

        if key in target:
            change = target[key]
            valid = target.get(valid_key)

            if valid is None:
                valid = torch.ones_like(change, dtype=torch.bool)

            return change, valid

        change = target["change"]
        valid = target.get("change_valid")

        if valid is None:
            valid = torch.ones_like(change, dtype=torch.bool)

        return change, valid

    def _binary_losses(self, prediction, target):
        if prediction.change_logits_t1 is None or prediction.change_logits_t2 is None:
            raise ValueError(
                "Binary branch must provide both change_logits_t1 and change_logits_t2"
            )

        change_t1, valid_t1 = self._change_target(target, 1)
        change_t2, valid_t2 = self._change_target(target, 2)

        valid_t1 = valid_t1.to(prediction.change_logits_t1.device).reshape(-1).bool()
        valid_t2 = valid_t2.to(prediction.change_logits_t2.device).reshape(-1).bool()

        active1 = bool(valid_t1.any().item())
        active2 = bool(valid_t2.any().item())

        bce1 = self.change_bce(
            prediction.change_logits_t1,
            change_t1,
            valid_t1,
        )
        bce2 = self.change_bce(
            prediction.change_logits_t2,
            change_t2,
            valid_t2,
        )

        dice1 = self.change_dice(
            prediction.change_logits_t1,
            change_t1,
            valid_t1,
        )
        dice2 = self.change_dice(
            prediction.change_logits_t2,
            change_t2,
            valid_t2,
        )

        change_bce = self._mean_over_active(
            (bce1, bce2),
            (active1, active2),
        )
        change_dice = self._mean_over_active(
            (dice1, dice2),
            (active1, active2),
        )

        change = (
            self.change_bce_weight * change_bce
            + self.change_dice_weight * change_dice
        )

        change_active = active1 or active2

        return change_bce, change_dice, change, change_active

    # ------------------------------------------------------------------
    # 3D event task: active-support CE + derived change Dice
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_event(logits, target, valid_mask, allowed_ids, name):
        if (
            logits is None
            or logits.ndim != 2
            or logits.shape[1] != PAIR_EVENT_NUM_CLASSES
        ):
            shape = None if logits is None else tuple(logits.shape)
            raise ValueError(
                f"event logits must be [N,{PAIR_EVENT_NUM_CLASSES}], got {shape}"
            )

        target = target.to(logits.device).reshape(-1).long()
        valid_mask = valid_mask.to(logits.device).reshape(-1).bool()

        if logits.shape[0] != target.numel() or target.numel() != valid_mask.numel():
            raise ValueError(
                f"event logits/target/mask size mismatch: "
                f"{logits.shape[0]}, {target.numel()}, {valid_mask.numel()}"
            )

        if valid_mask.any():
            y = target[valid_mask]

            bad = (y < 0) | (y >= PAIR_EVENT_NUM_CLASSES)
            if bad.any():
                values = torch.unique(y[bad]).detach().cpu().tolist()
                raise ValueError(
                    f"{name} contains invalid event IDs {values}; "
                    f"expected 0..{PAIR_EVENT_NUM_CLASSES - 1}"
                )

            support_ok = torch.zeros_like(y, dtype=torch.bool)
            for event_id in allowed_ids:
                support_ok |= y == event_id

            if (~support_ok).any():
                values = torch.unique(y[~support_ok]).detach().cpu().tolist()
                raise ValueError(
                    f"{name} contains event IDs {values} outside "
                    f"its active support {tuple(allowed_ids)}"
                )

        allowed = torch.tensor(
            allowed_ids,
            dtype=torch.long,
            device=logits.device,
        )

        return logits.float(), target, valid_mask, allowed

    @classmethod
    def event_ce(cls, logits, target, valid_mask, allowed_ids, name):
        logits, target, valid_mask, allowed = cls._prepare_event(
            logits,
            target,
            valid_mask,
            allowed_ids,
            name,
        )

        if not valid_mask.any():
            return logits.sum() * 0.0

        # Restrict the class space BEFORE CE.
        # T1 global [0,2] -> local [0,1]
        # T2 global [0,1] -> local [0,1]
        active_logits = logits[valid_mask].index_select(1, allowed)
        global_target = target[valid_mask]
        local_target = torch.empty_like(global_target)

        for local_id, global_id in enumerate(allowed_ids):
            local_target[global_target == global_id] = local_id

        return F.cross_entropy(
            active_logits,
            local_target,
        )

    def event_change_dice(self, logits, target, valid_mask, allowed_ids, name):
        logits, target, valid_mask, allowed = self._prepare_event(
            logits,
            target,
            valid_mask,
            allowed_ids,
            name,
        )

        if not valid_mask.any():
            return logits.sum() * 0.0

        active_logits = logits[valid_mask].index_select(1, allowed)
        active_prob = F.softmax(active_logits, dim=1)

        # In both active supports the second local class is the changed class:
        # T1 [unchanged, removed], T2 [unchanged, added].
        change_prob = active_prob[:, 1]
        change_target = (target[valid_mask] != 0).float()

        intersection = (change_prob * change_target).sum()
        dice = (2.0 * intersection + self.dice_eps) / (
            change_prob.sum() + change_target.sum() + self.dice_eps
        )

        return 1.0 - dice

    def _event_losses(self, prediction, target):
        if prediction.event_logits_t1 is None or prediction.event_logits_t2 is None:
            raise ValueError(
                "Event branch must provide both event_logits_t1 and event_logits_t2"
            )

        if "event_t1" not in target or "event_t2" not in target:
            raise KeyError(
                "Event loss requires target['event_t1'] and target['event_t2']"
            )

        event1 = target["event_t1"]
        event2 = target["event_t2"]

        valid1 = self._valid_or_true(
            target,
            "event_valid_t1",
            event1,
            prediction.event_logits_t1.device,
        )
        valid2 = self._valid_or_true(
            target,
            "event_valid_t2",
            event2,
            prediction.event_logits_t2.device,
        )

        active1 = bool(valid1.any().item())
        active2 = bool(valid2.any().item())

        event_t1 = self.event_ce(
            prediction.event_logits_t1,
            event1,
            valid1,
            allowed_ids=PAIR_EVENT_ACTIVE_SUPPORT[1],
            name="event_t1",
        )

        event_t2 = self.event_ce(
            prediction.event_logits_t2,
            event2,
            valid2,
            allowed_ids=PAIR_EVENT_ACTIVE_SUPPORT[2],
            name="event_t2",
        )

        dice_t1 = self.event_change_dice(
            prediction.event_logits_t1,
            event1,
            valid1,
            allowed_ids=PAIR_EVENT_ACTIVE_SUPPORT[1],
            name="event_t1",
        )

        dice_t2 = self.event_change_dice(
            prediction.event_logits_t2,
            event2,
            valid2,
            allowed_ids=PAIR_EVENT_ACTIVE_SUPPORT[2],
            name="event_t2",
        )

        event_ce = self._mean_over_active(
            (event_t1, event_t2),
            (active1, active2),
        )
        event_dice = self._mean_over_active(
            (dice_t1, dice_t2),
            (active1, active2),
        )

        event = event_ce + self.event_dice_weight * event_dice
        event_active = active1 or active2

        return (
            event_t1,
            event_t2,
            event_ce,
            dice_t1,
            dice_t2,
            event_dice,
            event,
            event_active,
        )

    # ------------------------------------------------------------------
    # Forward / active-task normalization
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        prediction,
        target,
        class_names: Dict[int, str],
        dataset_name: Optional[str] = None,
        semantic_changed_only: bool = False,
    ):
        # Kept for API compatibility. The supervision actually present in the
        # batch determines which task groups are active.
        del dataset_name

        (
            sem1,
            sem2,
            sem_ce,
            sem_lov1,
            sem_lov2,
            sem_lovasz,
            semantic,
            semantic_active,
            semantic_supervision_active,
        ) = self._semantic_losses(
            prediction,
            target,
            class_names,
            changed_only=bool(semantic_changed_only),
        )

        has_binary_t1 = prediction.change_logits_t1 is not None
        has_binary_t2 = prediction.change_logits_t2 is not None
        has_event_t1 = prediction.event_logits_t1 is not None
        has_event_t2 = prediction.event_logits_t2 is not None

        if has_binary_t1 != has_binary_t2:
            raise ValueError(
                "Binary branch is incomplete: change_logits_t1/t2 must be present together"
            )

        if has_event_t1 != has_event_t2:
            raise ValueError(
                "Event branch is incomplete: event_logits_t1/t2 must be present together"
            )

        has_binary = has_binary_t1 and has_binary_t2
        has_event = has_event_t1 and has_event_t2

        if not has_binary and not has_event:
            raise ValueError(
                "PAIR loss received neither binary-change logits nor event logits"
            )

        zero = self._zero_from(prediction.semantic_logits_t1)

        change_bce = zero
        change_dice = zero
        change = zero
        change_active = False

        event_t1 = zero
        event_t2 = zero
        event_ce = zero
        event_dice_t1 = zero
        event_dice_t2 = zero
        event_dice = zero
        event = zero
        event_active = False

        ssc = zero
        ssc_active = False

        if has_binary:
            (
                change_bce,
                change_dice,
                change,
                change_active,
            ) = self._binary_losses(
                prediction,
                target,
            )

        # SSCLoss couples the paired 2D semantic outputs with binary change
        # supervision. Keep it outside the active-task denominator so adding
        # it does not silently turn PAIR's existing 1/2 semantic + 1/2 change
        # balance into three 1/3 task groups.
        if has_binary:
            ssc, ssc_active = self._ssc_loss(
                prediction,
                target,
                class_names,
                semantic_supervision_active,
            )

        if has_event:
            (
                event_t1,
                event_t2,
                event_ce,
                event_dice_t1,
                event_dice_t2,
                event_dice,
                event,
                event_active,
            ) = self._event_losses(
                prediction,
                target,
            )

        weighted_terms = []
        active_weights = []

        if semantic_active and self.semantic_weight > 0:
            weighted_terms.append(
                self.semantic_weight * semantic
            )
            active_weights.append(
                self.semantic_weight
            )

        if change_active and self.change_weight > 0:
            weighted_terms.append(
                self.change_weight * change
            )
            active_weights.append(
                self.change_weight
            )

        if event_active and self.event_weight > 0:
            weighted_terms.append(
                self.event_weight * event
            )
            active_weights.append(
                self.event_weight
            )

        if not weighted_terms:
            raise ValueError(
                "PAIR loss has no active supervised task after valid masks/weights are applied"
            )

        numerator = torch.stack(weighted_terms).sum()
        weight_sum_value = float(sum(active_weights))

        if self.normalize_active_tasks:
            total = numerator / weight_sum_value
        else:
            total = numerator

        # PerASCD adds SSCLoss as an auxiliary term with unit coefficient.
        # `ssc_weight` exposes that coefficient without changing PAIR's task
        # normalization. Current PAIR default=0.01 keeps SSC auxiliary at the measured gradient scale.
        if ssc_active and self.ssc_weight > 0:
            total = total + self.ssc_weight * ssc

        active_weight_sum = numerator.new_tensor(
            weight_sum_value
        )

        return ChangeLossOutput(
            total=total,
            semantic_t1=sem1,
            semantic_t2=sem2,
            semantic_ce=sem_ce,
            semantic_lovasz_t1=sem_lov1,
            semantic_lovasz_t2=sem_lov2,
            semantic_lovasz=sem_lovasz,
            semantic=semantic,
            change_bce=change_bce,
            change_dice=change_dice,
            change=change,
            ssc=ssc,
            event_t1=event_t1,
            event_t2=event_t2,
            event_ce=event_ce,
            event_dice_t1=event_dice_t1,
            event_dice_t2=event_dice_t2,
            event_dice=event_dice,
            event=event,
            active_weight_sum=active_weight_sum,
        )

def _self_test():
    from types import SimpleNamespace

    torch.manual_seed(0)
    criterion = PAIRSemanticChangeLoss()

    # ------------------------------------------------------------------
    # Legacy-like semantic pair without an explicit "unchanged" class:
    # SSCLoss must remain inactive and baseline arithmetic must not change.
    # ------------------------------------------------------------------
    class_names_plain = {
        0: "ground",
        1: "building",
        2: "vegetation",
        3: "clutter",
    }
    sem1_plain = torch.tensor([0, 1, 2, 3, 1, 2])
    sem2_plain = torch.tensor([0, 1, 2, 3, 1, 2])

    pred_plain = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=torch.randn(6, requires_grad=True),
        change_logits_t2=torch.randn(6, requires_grad=True),
        event_logits_t1=None,
        event_logits_t2=None,
    )
    target_plain = {
        "semantic_t1": sem1_plain,
        "semantic_t2": sem2_plain,
        "change": torch.tensor([0, 1, 0, 1, 1, 0]),
        "change_valid": torch.ones(6, dtype=torch.bool),
    }

    out_plain = criterion(
        prediction=pred_plain,
        target=target_plain,
        class_names=class_names_plain,
    )
    expected_plain = 0.5 * (out_plain.semantic + out_plain.change)
    assert torch.allclose(out_plain.ssc, torch.tensor(0.0))
    assert torch.allclose(
        out_plain.total, expected_plain, atol=1e-6, rtol=1e-6
    )

    # ------------------------------------------------------------------
    # SECOND-like 2D SCD: explicit unchanged class activates SSCLoss.
    # ------------------------------------------------------------------
    class_names_scd = {
        0: "unchanged",
        1: "water",
        2: "building",
        3: "tree",
    }

    pred_scd = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=torch.randn(6, requires_grad=True),
        change_logits_t2=torch.randn(6, requires_grad=True),
        event_logits_t1=None,
        event_logits_t2=None,
    )
    target_scd = {
        "semantic_t1": torch.tensor([0, 1, 0, 2, 3, 0]),
        "semantic_t2": torch.tensor([0, 2, 0, 3, 1, 0]),
        "semantic_valid_t1": torch.ones(6, dtype=torch.bool),
        "semantic_valid_t2": torch.ones(6, dtype=torch.bool),
        "change": torch.tensor([0, 1, 0, 1, 1, 0]),
        "change_valid": torch.ones(6, dtype=torch.bool),
    }

    out_scd = criterion(
        prediction=pred_scd,
        target=target_scd,
        class_names=class_names_scd,
    )
    expected_scd = (
        0.5 * (out_scd.semantic + out_scd.change)
        + criterion.ssc_weight * out_scd.ssc
    )
    assert torch.isfinite(out_scd.ssc)
    assert out_scd.ssc.item() >= 0.0
    assert torch.allclose(
        out_scd.total, expected_scd, atol=1e-6, rtol=1e-6
    )

    # Changed-only semantic supervision must ignore unchanged pixels for BOTH
    # CE and Lovasz while keeping the same binary and SSC targets.
    out_scd_changed_only = criterion(
        prediction=pred_scd,
        target=target_scd,
        class_names=class_names_scd,
        semantic_changed_only=True,
    )
    changed = target_scd["change"].bool()
    expected_ce_t1 = F.cross_entropy(
        pred_scd.semantic_logits_t1[changed].float(),
        target_scd["semantic_t1"][changed],
    )
    expected_ce_t2 = F.cross_entropy(
        pred_scd.semantic_logits_t2[changed].float(),
        target_scd["semantic_t2"][changed],
    )
    assert torch.allclose(
        out_scd_changed_only.semantic_t1, expected_ce_t1, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        out_scd_changed_only.semantic_t2, expected_ce_t2, atol=1e-6, rtol=1e-6
    )
    # Binary change and SSC definitions themselves are unchanged.
    assert torch.allclose(
        out_scd_changed_only.change, out_scd.change, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        out_scd_changed_only.ssc, out_scd.ssc, atol=1e-6, rtol=1e-6
    )

    # A crop containing only unchanged pixels has no active semantic CE/Lovasz
    # task under the changed-only protocol, but SSC must still remain active on
    # the unchanged pair instead of being disabled accidentally.
    pred_unchanged = SimpleNamespace(
        semantic_logits_t1=torch.randn(4, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(4, 4, requires_grad=True),
        change_logits_t1=torch.randn(4, requires_grad=True),
        change_logits_t2=torch.randn(4, requires_grad=True),
        event_logits_t1=None,
        event_logits_t2=None,
    )
    target_unchanged = {
        "semantic_t1": torch.zeros(4, dtype=torch.long),
        "semantic_t2": torch.zeros(4, dtype=torch.long),
        "semantic_valid_t1": torch.ones(4, dtype=torch.bool),
        "semantic_valid_t2": torch.ones(4, dtype=torch.bool),
        "change": torch.zeros(4, dtype=torch.long),
        "change_valid": torch.ones(4, dtype=torch.bool),
    }
    out_unchanged = criterion(
        prediction=pred_unchanged,
        target=target_unchanged,
        class_names=class_names_scd,
        semantic_changed_only=True,
    )
    assert torch.allclose(out_unchanged.semantic, torch.tensor(0.0))
    assert torch.isfinite(out_unchanged.ssc)
    assert out_unchanged.ssc.item() >= 0.0
    expected_unchanged_total = (
        out_unchanged.change + criterion.ssc_weight * out_unchanged.ssc
    )
    assert torch.allclose(
        out_unchanged.total, expected_unchanged_total, atol=1e-6, rtol=1e-6
    )

    out_scd.total.backward()
    assert pred_scd.semantic_logits_t1.grad is not None
    assert pred_scd.semantic_logits_t2.grad is not None
    assert pred_scd.change_logits_t1.grad is not None

    # Formula sanity check:
    # identical semantic vectors at an unchanged pixel -> zero SSC.
    identical_t1 = torch.tensor([[9.0, 1.0, 2.0, 3.0]])
    identical_t2 = identical_t1.clone()
    direct = criterion.soft_semantic_consistency(
        identical_t1,
        identical_t2,
        torch.tensor([0]),
        torch.tensor([True]),
        unchanged_local_index=0,
    )
    assert torch.allclose(direct, torch.tensor(0.0), atol=1e-6)

    # ------------------------------------------------------------------
    # 2D BCD: semantic supervision fully masked -> SSCLoss inactive.
    # ------------------------------------------------------------------
    pred_bcd = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=torch.randn(6, requires_grad=True),
        change_logits_t2=torch.randn(6, requires_grad=True),
        event_logits_t1=None,
        event_logits_t2=None,
    )
    target_bcd = {
        "semantic_t1": torch.zeros(6, dtype=torch.long),
        "semantic_t2": torch.zeros(6, dtype=torch.long),
        "semantic_valid_t1": torch.zeros(6, dtype=torch.bool),
        "semantic_valid_t2": torch.zeros(6, dtype=torch.bool),
        "change": torch.tensor([0, 1, 0, 1, 1, 0]),
        "change_valid": torch.ones(6, dtype=torch.bool),
    }

    out_bcd = criterion(
        prediction=pred_bcd,
        target=target_bcd,
        class_names=class_names_scd,
    )
    assert torch.allclose(out_bcd.ssc, torch.tensor(0.0))
    assert torch.allclose(
        out_bcd.total, out_bcd.change, atol=1e-6, rtol=1e-6
    )

    # ------------------------------------------------------------------
    # 3D: no shared binary `change` target -> SSCLoss stays inactive.
    # Event active-support behavior must remain unchanged.
    # ------------------------------------------------------------------
    class_names_3d = {
        0: "ground",
        1: "building",
        2: "vegetation",
        3: "clutter",
    }
    pred_3d = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=None,
        change_logits_t2=None,
        event_logits_t1=torch.randn(6, 3, requires_grad=True),
        event_logits_t2=torch.randn(6, 3, requires_grad=True),
    )
    target_3d = {
        "semantic_t1": torch.tensor([0, 1, 2, 3, 1, 2]),
        "semantic_t2": torch.tensor([0, 1, 2, 3, 1, 2]),
        "event_t1": torch.tensor([0, 2, 0, 2, 0, 2]),
        "event_t2": torch.tensor([0, 1, 0, 1, 0, 1]),
        "event_valid_t1": torch.ones(6, dtype=torch.bool),
        "event_valid_t2": torch.ones(6, dtype=torch.bool),
    }

    out_3d = criterion(
        prediction=pred_3d,
        target=target_3d,
        class_names=class_names_3d,
        dataset_name="NYC-SCD",
    )
    assert torch.allclose(out_3d.ssc, torch.tensor(0.0))
    expected_3d = 0.5 * (out_3d.semantic + out_3d.event)
    assert torch.allclose(
        out_3d.total, expected_3d, atol=1e-6, rtol=1e-6
    )

    out_3d.total.backward()

    # T1 illegal class Added(1); T2 illegal class Removed(2).
    assert torch.allclose(
        pred_3d.event_logits_t1.grad[:, 1],
        torch.zeros_like(pred_3d.event_logits_t1.grad[:, 1]),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        pred_3d.event_logits_t2.grad[:, 2],
        torch.zeros_like(pred_3d.event_logits_t2.grad[:, 2]),
        atol=0.0,
        rtol=0.0,
    )

    print("loss.py V2 + PerASCD SSCLoss self-test: PASS")
    print(
        "2D SCD total=%.6f | semantic=%.6f | change=%.6f | ssc=%.6f"
        % (
            float(out_scd.total.detach()),
            float(out_scd.semantic.detach()),
            float(out_scd.change.detach()),
            float(out_scd.ssc.detach()),
        )
    )
    print(
        "2D BCD total=%.6f | ssc=%.6f"
        % (
            float(out_bcd.total.detach()),
            float(out_bcd.ssc.detach()),
        )
    )
    print(
        "3D total=%.6f | semantic=%.6f | event=%.6f | ssc=%.6f"
        % (
            float(out_3d.total.detach()),
            float(out_3d.semantic.detach()),
            float(out_3d.event.detach()),
            float(out_3d.ssc.detach()),
        )
    )


if __name__ == "__main__":
    _self_test()
