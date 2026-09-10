"""
PAIR unified loss.

2D:
    semantic_logits_t1/t2 + binary change_logits_t1/t2
    Existing SCD / BCD loss behavior is preserved.

3D:
    semantic_logits_t1/t2 + event_logits_t1/t2
    Event head always has six global PAIR classes:
        0 unchanged
        1 added
        2 removed
        3 class_change
        4 height_up
        5 height_down

Important:
A dataset is trained only over the event classes it actually supervises.
Inactive event classes are removed from the CE softmax denominator rather than
being treated as negatives.

Current NYC-SCD support:
    T1 active classes: [0, 2]  unchanged / removed
    T2 active classes: [0, 1]  unchanged / added

event_valid masks point-level temporal support. Active-class selection is a
separate dataset-level rule and therefore lives in the loss rather than in the
prepared target files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PAIR_EVENT_NUM_CLASSES = 6
PAIR_EVENT_NAMES = (
    "unchanged",
    "added",
    "removed",
    "class_change",
    "height_up",
    "height_down",
)

# Dataset-level supervision support. Keep this out of user config and out of
# prepared point targets. Add future 3D datasets here only when their true event
# supervision protocol is known.
_EVENT_ACTIVE_CLASSES = {
    "nyc-scd": ((0, 2), (0, 1)),
}


@dataclass
class ChangeLossOutput:
    total: torch.Tensor
    semantic_t1: torch.Tensor
    semantic_t2: torch.Tensor
    change_bce: torch.Tensor
    change_dice: torch.Tensor
    event_t1: torch.Tensor
    event_t2: torch.Tensor
    event: torch.Tensor

    def as_dict(self):
        return {
            "loss": self.total,
            "loss_semantic_t1": self.semantic_t1,
            "loss_semantic_t2": self.semantic_t2,
            "loss_change_bce": self.change_bce,
            "loss_change_dice": self.change_dice,
            "loss_event_t1": self.event_t1,
            "loss_event_t2": self.event_t2,
            "loss_event": self.event,
        }


class PAIRSemanticChangeLoss(nn.Module):
    def __init__(self, semantic_weight=1.0, change_bce_weight=1.0, change_dice_weight=1.0,
                 event_weight=1.0, dice_eps=1.0):
        super().__init__()
        self.semantic_weight = float(semantic_weight)
        self.change_bce_weight = float(change_bce_weight)
        self.change_dice_weight = float(change_dice_weight)
        self.event_weight = float(event_weight)
        self.dice_eps = float(dice_eps)

    # =========================================================================
    # Shared helpers
    # =========================================================================

    @staticmethod
    def _zero_from(*values):
        for value in values:
            if torch.is_tensor(value):
                return value.sum() * 0.0
        raise ValueError("Cannot construct zero loss without a tensor")

    @staticmethod
    def _valid_or_true(target: Dict[str, torch.Tensor], valid_key: str, value: torch.Tensor, device):
        valid = target.get(valid_key)
        if valid is None:
            return torch.ones(value.numel(), dtype=torch.bool, device=device)
        valid = valid.to(device).reshape(-1).bool()
        if valid.numel() != value.numel():
            raise ValueError(
                f"{valid_key} size {valid.numel()} does not match target size {value.numel()}"
            )
        return valid

    @staticmethod
    def make_raw_to_local(class_names: Dict[int, str]) -> Dict[int, int]:
        if not isinstance(class_names, dict) or not class_names:
            raise TypeError("class_names must be a non-empty Dict[int, str]")
        raw_ids = sorted(int(k) for k in class_names.keys())
        return {raw_id: local_id for local_id, raw_id in enumerate(raw_ids)}

    # =========================================================================
    # Semantic CE
    # =========================================================================

    @classmethod
    def remap_semantic_target(cls, raw_target, valid_mask, class_names, ignore_index=-100):
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
            raise ValueError(f"Semantic labels {values} are not declared in DatasetSpec.class_names")
        return local

    @classmethod
    def semantic_ce(cls, logits, raw_target, valid_mask, class_names):
        if logits is None or logits.ndim != 2:
            shape = None if logits is None else tuple(logits.shape)
            raise ValueError(f"semantic logits must be [N,K], got {shape}")

        raw_target = raw_target.to(logits.device).reshape(-1)
        valid_mask = valid_mask.to(logits.device).reshape(-1).bool()
        if logits.shape[0] != raw_target.numel():
            raise ValueError(
                f"semantic logits/target size mismatch: {logits.shape[0]} vs {raw_target.numel()}"
            )
        if logits.shape[1] != len(class_names):
            raise ValueError(
                f"semantic logits K={logits.shape[1]} but class_names has {len(class_names)} classes"
            )
        if not valid_mask.any():
            return logits.sum() * 0.0

        local_target = cls.remap_semantic_target(raw_target, valid_mask, class_names)
        return F.cross_entropy(logits.float(), local_target, ignore_index=-100)

    def _semantic_losses(self, prediction, target, class_names):
        raw1 = target["semantic_t1"]
        raw2 = target["semantic_t2"]
        valid1 = self._valid_or_true(
            target, "semantic_valid_t1", raw1, prediction.semantic_logits_t1.device
        )
        valid2 = self._valid_or_true(
            target, "semantic_valid_t2", raw2, prediction.semantic_logits_t2.device
        )
        sem1 = self.semantic_ce(prediction.semantic_logits_t1, raw1, valid1, class_names)
        sem2 = self.semantic_ce(prediction.semantic_logits_t2, raw2, valid2, class_names)
        return sem1, sem2

    # =========================================================================
    # Existing 2D binary change loss
    # =========================================================================

    @staticmethod
    def _prepare_change(logits, target, valid_mask):
        if logits is None:
            raise ValueError("Binary change loss requested but prediction.change_logits is None")

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
                raise ValueError(f"valid change target must contain only 0/1, got {values}")
        return logits, target.float(), valid_mask

    def change_bce(self, logits, target, valid_mask):
        logits, target, valid_mask = self._prepare_change(logits, target, valid_mask)
        if not valid_mask.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits[valid_mask].float(), target[valid_mask]
        )

    def change_dice(self, logits, target, valid_mask):
        logits, target, valid_mask = self._prepare_change(logits, target, valid_mask)
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
            value = target[key]
            valid = target.get(valid_key)
        else:
            if "change" not in target:
                raise KeyError("2D binary change loss requires target['change']")
            value = target["change"]
            valid = target.get("change_valid")

        if valid is None:
            valid = torch.ones_like(value, dtype=torch.bool)
        return value, valid

    def _forward_binary(self, prediction, target, sem1, sem2):
        if prediction.change_logits_t1 is None or prediction.change_logits_t2 is None:
            raise ValueError("2D prediction must provide change_logits_t1 and change_logits_t2")
        if prediction.event_logits_t1 is not None or prediction.event_logits_t2 is not None:
            raise ValueError("2D prediction must not provide event logits")

        change_t1, valid_t1 = self._change_target(target, 1)
        change_t2, valid_t2 = self._change_target(target, 2)

        bce1 = self.change_bce(prediction.change_logits_t1, change_t1, valid_t1)
        bce2 = self.change_bce(prediction.change_logits_t2, change_t2, valid_t2)
        dice1 = self.change_dice(prediction.change_logits_t1, change_t1, valid_t1)
        dice2 = self.change_dice(prediction.change_logits_t2, change_t2, valid_t2)
        change_bce = 0.5 * (bce1 + bce2)
        change_dice = 0.5 * (dice1 + dice2)

        zero = self._zero_from(prediction.change_logits_t1)
        total = (
            self.semantic_weight * (sem1 + sem2)
            + self.change_bce_weight * change_bce
            + self.change_dice_weight * change_dice
        )
        return ChangeLossOutput(
            total=total,
            semantic_t1=sem1,
            semantic_t2=sem2,
            change_bce=change_bce,
            change_dice=change_dice,
            event_t1=zero,
            event_t2=zero,
            event=zero,
        )

    # =========================================================================
    # 3D active-class event CE
    # =========================================================================

    @staticmethod
    def _normalize_dataset_name(dataset_name: Optional[str]) -> Optional[str]:
        if dataset_name is None:
            return None
        return str(dataset_name).strip().lower().replace("_", "-")

    @classmethod
    def resolve_event_active_classes(
        cls,
        *,
        dataset_name: Optional[str],
        active_t1: Optional[Sequence[int]] = None,
        active_t2: Optional[Sequence[int]] = None,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Resolve dataset-level event supervision support.

        Explicit active_t1/active_t2 are useful for isolated tests or a future
        runtime registry. Normal training should pass dataset_name and keep this
        information out of the user-authored config.
        """
        if (active_t1 is None) != (active_t2 is None):
            raise ValueError("active_t1 and active_t2 must be provided together")

        if active_t1 is not None:
            t1, t2 = tuple(int(x) for x in active_t1), tuple(int(x) for x in active_t2)
        else:
            key = cls._normalize_dataset_name(dataset_name)
            if key not in _EVENT_ACTIVE_CLASSES:
                raise ValueError(
                    f"3D event loss has no declared active-class protocol for dataset "
                    f"{dataset_name!r}. Add its true event supervision support to loss.py; "
                    "do not silently train all six event classes."
                )
            t1, t2 = _EVENT_ACTIVE_CLASSES[key]

        for name, classes in (("T1", t1), ("T2", t2)):
            if not classes:
                raise ValueError(f"{name} active event classes cannot be empty")
            if len(set(classes)) != len(classes):
                raise ValueError(f"{name} active event classes contain duplicates: {classes}")
            bad = [x for x in classes if x < 0 or x >= PAIR_EVENT_NUM_CLASSES]
            if bad:
                raise ValueError(f"{name} active event classes contain invalid IDs: {bad}")
            if 0 not in classes:
                raise ValueError(f"{name} active event classes must include 0=unchanged")
        return t1, t2

    @staticmethod
    def event_ce(logits, target, valid_mask, active_classes: Sequence[int]):
        """
        Active-class CE.

        Example NYC T1:
            global logits [N,6] -> logits[:, [0,2]]
            target 0 -> local 0
            target 2 -> local 1

        Inactive columns never enter the softmax denominator, so a dataset that
        cannot supervise event classes 3/4/5 does not push those logits down.
        """
        if logits is None or logits.ndim != 2 or logits.shape[1] != PAIR_EVENT_NUM_CLASSES:
            shape = None if logits is None else tuple(logits.shape)
            raise ValueError(f"event logits must be [N,{PAIR_EVENT_NUM_CLASSES}], got {shape}")

        target = target.to(logits.device).reshape(-1).long()
        valid_mask = valid_mask.to(logits.device).reshape(-1).bool()
        if logits.shape[0] != target.numel() or target.numel() != valid_mask.numel():
            raise ValueError(
                f"event logits/target/mask size mismatch: "
                f"{logits.shape[0]}, {target.numel()}, {valid_mask.numel()}"
            )

        bad_global = (target < 0) | (target >= PAIR_EVENT_NUM_CLASSES)
        if bad_global.any():
            values = torch.unique(target[bad_global]).detach().cpu().tolist()
            raise ValueError(f"event target contains invalid global IDs {values}; expected 0..5")

        active = tuple(int(x) for x in active_classes)
        active_tensor = torch.tensor(active, dtype=torch.long, device=logits.device)
        local_target = torch.full_like(target, -100)
        matched = torch.zeros_like(valid_mask)
        for local_id, global_id in enumerate(active):
            mask = valid_mask & (target == global_id)
            local_target[mask] = local_id
            matched |= mask

        bad_valid = valid_mask & ~matched
        if bad_valid.any():
            values = torch.unique(target[bad_valid]).detach().cpu().tolist()
            raise ValueError(
                f"event_valid=True contains targets {values} outside active classes {list(active)}"
            )
        if not valid_mask.any():
            return logits.sum() * 0.0

        active_logits = logits.index_select(1, active_tensor).float()
        return F.cross_entropy(active_logits, local_target, ignore_index=-100)

    def _forward_event(
        self,
        prediction,
        target,
        sem1,
        sem2,
        *,
        dataset_name,
        event_active_classes_t1,
        event_active_classes_t2,
    ):
        if prediction.event_logits_t1 is None or prediction.event_logits_t2 is None:
            raise ValueError("3D prediction must provide event_logits_t1 and event_logits_t2")
        if prediction.change_logits_t1 is not None or prediction.change_logits_t2 is not None:
            raise ValueError("3D prediction must not provide binary change logits")
        if "event_t1" not in target or "event_t2" not in target:
            raise KeyError("3D event loss requires target['event_t1'] and target['event_t2']")

        active_t1, active_t2 = self.resolve_event_active_classes(
            dataset_name=dataset_name,
            active_t1=event_active_classes_t1,
            active_t2=event_active_classes_t2,
        )

        event1 = target["event_t1"]
        event2 = target["event_t2"]
        valid1 = self._valid_or_true(
            target, "event_valid_t1", event1, prediction.event_logits_t1.device
        )
        valid2 = self._valid_or_true(
            target, "event_valid_t2", event2, prediction.event_logits_t2.device
        )
        loss_event_t1 = self.event_ce(
            prediction.event_logits_t1, event1, valid1, active_t1
        )
        loss_event_t2 = self.event_ce(
            prediction.event_logits_t2, event2, valid2, active_t2
        )
        loss_event = 0.5 * (loss_event_t1 + loss_event_t2)

        zero = self._zero_from(prediction.event_logits_t1)
        total = self.semantic_weight * (sem1 + sem2) + self.event_weight * loss_event
        return ChangeLossOutput(
            total=total,
            semantic_t1=sem1,
            semantic_t2=sem2,
            change_bce=zero,
            change_dice=zero,
            event_t1=loss_event_t1,
            event_t2=loss_event_t2,
            event=loss_event,
        )

    # =========================================================================
    # Route by decoder output, not by a user task_mode config
    # =========================================================================

    def forward(
        self,
        *,
        prediction,
        target,
        class_names: Dict[int, str],
        dataset_name: Optional[str] = None,
        event_active_classes_t1: Optional[Sequence[int]] = None,
        event_active_classes_t2: Optional[Sequence[int]] = None,
    ):
        sem1, sem2 = self._semantic_losses(prediction, target, class_names)

        has_binary = prediction.change_logits_t1 is not None or prediction.change_logits_t2 is not None
        has_event = prediction.event_logits_t1 is not None or prediction.event_logits_t2 is not None
        if has_binary == has_event:
            raise ValueError(
                "PAIR loss expects exactly one prediction branch: "
                "binary change logits for 2D or event logits for 3D"
            )

        if has_binary:
            return self._forward_binary(prediction, target, sem1, sem2)

        return self._forward_event(
            prediction,
            target,
            sem1,
            sem2,
            dataset_name=dataset_name,
            event_active_classes_t1=event_active_classes_t1,
            event_active_classes_t2=event_active_classes_t2,
        )
