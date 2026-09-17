"""
PAIR unified loss.

2D keeps the existing semantic + binary-change objective.
3D uses semantic + full six-class event CE:
    0 unchanged, 1 added, 2 removed,
    3 class_change, 4 height_up, 5 height_down.

The current 3D protocol supervises all six event classes on both epochs.
Point-level event_valid masks decide which points participate. There is no
NYC-specific active-class registry and no 3D binary-change loss/head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PAIR_EVENT_NUM_CLASSES = 6
PAIR_EVENT_NAMES = (
    "unchanged", "added", "removed", "class_change", "height_up", "height_down"
)


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
    def __init__(self, semantic_weight=1.0, change_bce_weight=1.0,
                 change_dice_weight=1.0, event_weight=1.0, dice_eps=1.0):
        super().__init__()
        self.semantic_weight = float(semantic_weight)
        self.change_bce_weight = float(change_bce_weight)
        self.change_dice_weight = float(change_dice_weight)
        self.event_weight = float(event_weight)
        self.dice_eps = float(dice_eps)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _zero_from(*values):
        for value in values:
            if torch.is_tensor(value):
                return value.sum() * 0.0
        raise ValueError("Cannot construct zero loss without a tensor")

    @staticmethod
    def _valid_or_true(target: Dict[str, torch.Tensor], valid_key: str,
                       value: torch.Tensor, device):
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
        raw_ids = sorted(int(k) for k in class_names)
        return {raw_id: local_id for local_id, raw_id in enumerate(raw_ids)}

    # -------------------------------------------------------------------------
    # Semantic CE
    # -------------------------------------------------------------------------

    @classmethod
    def remap_semantic_target(cls, raw_target, valid_mask, class_names,
                              ignore_index=-100):
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

        local_target = cls.remap_semantic_target(
            raw_target, valid_mask, class_names
        )
        return F.cross_entropy(logits.float(), local_target, ignore_index=-100)

    def _semantic_losses(self, prediction, target, class_names):
        raw1, raw2 = target["semantic_t1"], target["semantic_t2"]
        valid1 = self._valid_or_true(
            target, "semantic_valid_t1", raw1, prediction.semantic_logits_t1.device
        )
        valid2 = self._valid_or_true(
            target, "semantic_valid_t2", raw2, prediction.semantic_logits_t2.device
        )
        sem1 = self.semantic_ce(
            prediction.semantic_logits_t1, raw1, valid1, class_names
        )
        sem2 = self.semantic_ce(
            prediction.semantic_logits_t2, raw2, valid2, class_names
        )
        return sem1, sem2

    # -------------------------------------------------------------------------
    # Existing 2D binary-change loss
    # -------------------------------------------------------------------------

    @staticmethod
    def _prepare_change(logits, target, valid_mask):
        if logits is None:
            raise ValueError("Binary change loss requested but change logits are missing")
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
            logits, target, valid_mask
        )
        if not valid_mask.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits[valid_mask].float(), target[valid_mask]
        )

    def change_dice(self, logits, target, valid_mask):
        logits, target, valid_mask = self._prepare_change(
            logits, target, valid_mask
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
        key, valid_key = f"change_t{time_id}", f"change_valid_t{time_id}"
        if key in target:
            value, valid = target[key], target.get(valid_key)
        else:
            if "change" not in target:
                raise KeyError("2D binary change loss requires target['change']")
            value, valid = target["change"], target.get("change_valid")
        if valid is None:
            valid = torch.ones_like(value, dtype=torch.bool)
        return value, valid

    def _binary_losses(self, prediction, target):
        if prediction.change_logits_t1 is None or prediction.change_logits_t2 is None:
            raise ValueError(
                "Binary branch must provide both change_logits_t1 and change_logits_t2"
            )

        change_t1, valid_t1 = self._change_target(target, 1)
        change_t2, valid_t2 = self._change_target(target, 2)

        bce1 = self.change_bce(
            prediction.change_logits_t1, change_t1, valid_t1
        )
        bce2 = self.change_bce(
            prediction.change_logits_t2, change_t2, valid_t2
        )
        dice1 = self.change_dice(
            prediction.change_logits_t1, change_t1, valid_t1
        )
        dice2 = self.change_dice(
            prediction.change_logits_t2, change_t2, valid_t2
        )

        change_bce = 0.5 * (bce1 + bce2)
        change_dice = 0.5 * (dice1 + dice2)
        return change_bce, change_dice

    # -------------------------------------------------------------------------
    # 3D full six-class event CE
    # -------------------------------------------------------------------------

    @staticmethod
    def event_ce(logits, target, valid_mask):
        if logits is None or logits.ndim != 2 or logits.shape[1] != PAIR_EVENT_NUM_CLASSES:
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
                    f"event_valid=True contains invalid event IDs {values}; expected 0..5"
                )
        if not valid_mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid_mask].float(), target[valid_mask])

    def _event_losses(self, prediction, target):
        if prediction.event_logits_t1 is None or prediction.event_logits_t2 is None:
            raise ValueError(
                "Event branch must provide both event_logits_t1 and event_logits_t2"
            )
        if "event_t1" not in target or "event_t2" not in target:
            raise KeyError(
                "Event loss requires target['event_t1'] and target['event_t2']"
            )

        event1, event2 = target["event_t1"], target["event_t2"]
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

        loss_event_t1 = self.event_ce(
            prediction.event_logits_t1, event1, valid1
        )
        loss_event_t2 = self.event_ce(
            prediction.event_logits_t2, event2, valid2
        )
        loss_event = 0.5 * (loss_event_t1 + loss_event_t2)
        return loss_event_t1, loss_event_t2, loss_event

    # -------------------------------------------------------------------------
    # Route by decoder output, not by a task-mode config
    # -------------------------------------------------------------------------

    def forward(self, *, prediction, target, class_names: Dict[int, str],
                dataset_name: Optional[str] = None, **kwargs):
        # Kept temporarily for train.py compatibility. Event supervision is no
        # longer selected by dataset name or by active-class arguments.
        del dataset_name
        if kwargs:
            stale = sorted(kwargs)
            raise TypeError(
                f"Unsupported loss arguments {stale}. Active-class event loss has been removed."
            )

        sem1, sem2 = self._semantic_losses(
            prediction, target, class_names
        )

        has_binary_t1 = prediction.change_logits_t1 is not None
        has_binary_t2 = prediction.change_logits_t2 is not None
        has_event_t1 = prediction.event_logits_t1 is not None
        has_event_t2 = prediction.event_logits_t2 is not None

        if has_binary_t1 != has_binary_t2:
            raise ValueError(
                "Binary branch is incomplete: change_logits_t1/t2 must be "
                "present together"
            )
        if has_event_t1 != has_event_t2:
            raise ValueError(
                "Event branch is incomplete: event_logits_t1/t2 must be "
                "present together"
            )

        has_binary = has_binary_t1 and has_binary_t2
        has_event = has_event_t1 and has_event_t2

        if not has_binary and not has_event:
            raise ValueError(
                "PAIR loss received neither binary-change logits nor event logits"
            )

        # One shared semantic term, plus whichever change-supervision branches
        # are actually present. This supports:
        #
        #   2D    -> semantic + binary
        #   3D    -> semantic + event
        #   2D3D  -> semantic + binary + event
        #
        # The loss layer must not encode the current implementation status of
        # forward_2d3d as a permanent architectural restriction.
        total = self.semantic_weight * (sem1 + sem2)

        reference = prediction.semantic_logits_t1
        zero = self._zero_from(reference)

        change_bce = zero
        change_dice = zero
        event_t1 = zero
        event_t2 = zero
        event = zero

        if has_binary:
            change_bce, change_dice = self._binary_losses(
                prediction, target
            )
            total = (
                total
                + self.change_bce_weight * change_bce
                + self.change_dice_weight * change_dice
            )

        if has_event:
            event_t1, event_t2, event = self._event_losses(
                prediction, target
            )
            total = total + self.event_weight * event

        return ChangeLossOutput(
            total=total,
            semantic_t1=sem1,
            semantic_t2=sem2,
            change_bce=change_bce,
            change_dice=change_dice,
            event_t1=event_t1,
            event_t2=event_t2,
            event=event,
        )


def _self_test():
    from types import SimpleNamespace

    torch.manual_seed(0)
    loss_fn = PAIRSemanticChangeLoss()
    class_names = {
        0: "ground",
        1: "building",
        2: "vegetation",
        3: "clutter",
    }

    semantic_t1 = torch.tensor([0, 1, 2, 3, 1, 2])
    semantic_t2 = torch.tensor([0, 1, 2, 3, 1, 2])

    # ------------------------------------------------------------------
    # 2D: semantic + binary
    # ------------------------------------------------------------------
    pred_2d = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=torch.randn(6, requires_grad=True),
        change_logits_t2=torch.randn(6, requires_grad=True),
        event_logits_t1=None,
        event_logits_t2=None,
    )
    target_2d = {
        "semantic_t1": semantic_t1,
        "semantic_t2": semantic_t2,
        "change_t1": torch.tensor([0, 1, 0, 1, 1, 0]),
        "change_t2": torch.tensor([0, 1, 0, 1, 1, 0]),
    }
    out_2d = loss_fn(
        prediction=pred_2d,
        target=target_2d,
        class_names=class_names,
        dataset_name="SECOND",
    )
    out_2d.total.backward()
    assert pred_2d.change_logits_t1.grad is not None
    assert pred_2d.change_logits_t2.grad is not None

    # ------------------------------------------------------------------
    # 3D: semantic + full six-class event
    # ------------------------------------------------------------------
    pred_3d = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=None,
        change_logits_t2=None,
        event_logits_t1=torch.randn(6, 6, requires_grad=True),
        event_logits_t2=torch.randn(6, 6, requires_grad=True),
    )
    target_3d = {
        "semantic_t1": semantic_t1,
        "semantic_t2": semantic_t2,
        "event_t1": torch.tensor([0, 1, 2, 3, 4, 5]),
        "event_t2": torch.tensor([5, 4, 3, 2, 1, 0]),
        "event_valid_t1": torch.ones(6, dtype=torch.bool),
        "event_valid_t2": torch.ones(6, dtype=torch.bool),
    }
    out_3d = loss_fn(
        prediction=pred_3d,
        target=target_3d,
        class_names=class_names,
        dataset_name="NYC-SCD",
    )
    out_3d.total.backward()
    assert pred_3d.event_logits_t1.grad is not None
    assert pred_3d.event_logits_t2.grad is not None

    # ------------------------------------------------------------------
    # 2D+3D: semantic + binary + event simultaneously.
    # This is the important regression for unified multimodal PAIR.
    # ------------------------------------------------------------------
    pred_2d3d = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=torch.randn(6, requires_grad=True),
        change_logits_t2=torch.randn(6, requires_grad=True),
        event_logits_t1=torch.randn(6, 6, requires_grad=True),
        event_logits_t2=torch.randn(6, 6, requires_grad=True),
    )
    target_2d3d = {
        "semantic_t1": semantic_t1,
        "semantic_t2": semantic_t2,
        "change_t1": torch.tensor([0, 1, 0, 1, 1, 0]),
        "change_t2": torch.tensor([0, 1, 0, 1, 1, 0]),
        "event_t1": torch.tensor([0, 1, 2, 3, 4, 5]),
        "event_t2": torch.tensor([5, 4, 3, 2, 1, 0]),
        "event_valid_t1": torch.ones(6, dtype=torch.bool),
        "event_valid_t2": torch.ones(6, dtype=torch.bool),
    }
    out_2d3d = loss_fn(
        prediction=pred_2d3d,
        target=target_2d3d,
        class_names=class_names,
    )
    out_2d3d.total.backward()

    assert pred_2d3d.change_logits_t1.grad is not None
    assert pred_2d3d.change_logits_t2.grad is not None
    assert pred_2d3d.event_logits_t1.grad is not None
    assert pred_2d3d.event_logits_t2.grad is not None
    assert torch.isfinite(out_2d3d.total)

    # Invalid event labels behind event_valid=False are ignored.
    logits = torch.zeros(2, 6, requires_grad=True)
    masked = loss_fn.event_ce(
        logits,
        torch.tensor([5, 99]),
        torch.tensor([True, False]),
    )
    masked.backward()
    assert logits.grad is not None

    print("loss.py self-test: PASS (2D / 3D / 2D3D)")


if __name__ == "__main__":
    _self_test()
