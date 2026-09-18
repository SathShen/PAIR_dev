"""
PAIR unified loss.

2D:
    semantic_logits_t1/t2 + binary change_logits_t1/t2

3D:
    semantic_logits_t1/t2 + event_logits_t1/t2

2D+3D:
    binary change logits and event logits may coexist.

3D event protocol:
    0 unchanged
    1 added
    2 removed

Active event support:
    T1: unchanged / removed = {0, 2}
    T2: unchanged / added   = {0, 1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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
    def __init__(
        self,
        semantic_weight=1.0,
        change_bce_weight=1.0,
        change_dice_weight=1.0,
        event_weight=1.0,
        dice_eps=1.0,
    ):
        super().__init__()
        self.semantic_weight = float(semantic_weight)
        self.change_bce_weight = float(change_bce_weight)
        self.change_dice_weight = float(change_dice_weight)
        self.event_weight = float(event_weight)
        self.dice_eps = float(dice_eps)

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
            raise ValueError(f"{key} size {valid.numel()} does not match target size {value.numel()}")
        return valid

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

        return F.cross_entropy(
            logits.float(),
            local_target,
            ignore_index=-100,
        )

    def _semantic_losses(self, prediction, target, class_names):
        raw1 = target["semantic_t1"]
        raw2 = target["semantic_t2"]

        valid1 = self._valid_or_true(
            target, "semantic_valid_t1", raw1, prediction.semantic_logits_t1.device
        )
        valid2 = self._valid_or_true(
            target, "semantic_valid_t2", raw2, prediction.semantic_logits_t2.device
        )

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

        return sem1, sem2

    @staticmethod
    def _zero_from(tensor):
        return tensor.sum() * 0.0

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
                raise ValueError(f"valid change target must contain only 0/1, got {values}")

        return logits, target.float(), valid_mask

    def change_bce(self, logits, target, valid_mask):
        logits, target, valid_mask = self._prepare_change(logits, target, valid_mask)

        if not valid_mask.any():
            return logits.sum() * 0.0

        return F.binary_cross_entropy_with_logits(
            logits[valid_mask].float(),
            target[valid_mask],
        )

    def change_dice(self, logits, target, valid_mask):
        logits, target, valid_mask = self._prepare_change(logits, target, valid_mask)

        if not valid_mask.any():
            return logits.sum() * 0.0

        prob = torch.sigmoid(logits[valid_mask].float())
        target = target[valid_mask]

        intersection = (prob * target).sum()
        dice = (2 * intersection + self.dice_eps) / (
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

        change_bce = 0.5 * (bce1 + bce2)
        change_dice = 0.5 * (dice1 + dice2)

        return change_bce, change_dice

    @staticmethod
    def event_ce(logits, target, valid_mask, allowed_ids, name):
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

        if not valid_mask.any():
            return logits.sum() * 0.0

        allowed = torch.tensor(
            allowed_ids,
            dtype=torch.long,
            device=logits.device,
        )

        # IMPORTANT:
        # Restrict the class space BEFORE softmax / cross entropy.
        #
        # T1:
        #   global event IDs [0, 2] -> local CE IDs [0, 1]
        #
        # T2:
        #   global event IDs [0, 1] -> local CE IDs [0, 1]
        active_logits = logits[valid_mask].float().index_select(1, allowed)
        global_target = target[valid_mask]

        local_target = torch.empty_like(global_target)

        for local_id, global_id in enumerate(allowed_ids):
            local_target[global_target == global_id] = local_id

        return F.cross_entropy(
            active_logits,
            local_target,
        )

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

        event = 0.5 * (event_t1 + event_t2)

        return event_t1, event_t2, event

    def forward(
        self,
        *,
        prediction,
        target,
        class_names: Dict[int, str],
        dataset_name: Optional[str] = None,
    ):
        # train.py may still pass dataset_name.
        # Loss behavior is determined by available outputs, not dataset name.
        del dataset_name

        sem1, sem2 = self._semantic_losses(
            prediction,
            target,
            class_names,
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

        total = self.semantic_weight * (sem1 + sem2)

        zero = self._zero_from(prediction.semantic_logits_t1)

        change_bce = zero
        change_dice = zero
        event_t1 = zero
        event_t2 = zero
        event = zero

        if has_binary:
            change_bce, change_dice = self._binary_losses(
                prediction,
                target,
            )

            total = (
                total
                + self.change_bce_weight * change_bce
                + self.change_dice_weight * change_dice
            )

        if has_event:
            event_t1, event_t2, event = self._event_losses(
                prediction,
                target,
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

    criterion = PAIRSemanticChangeLoss()

    class_names = {
        0: "ground",
        1: "building",
        2: "vegetation",
        3: "clutter",
    }

    semantic_t1 = torch.tensor([0, 1, 2, 3, 1, 2])
    semantic_t2 = torch.tensor([0, 1, 2, 3, 1, 2])

    # ------------------------------------------------------------------
    # 2D
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

    out_2d = criterion(
        prediction=pred_2d,
        target=target_2d,
        class_names=class_names,
    )

    out_2d.total.backward()

    # ------------------------------------------------------------------
    # 3D
    # ------------------------------------------------------------------
    pred_3d = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=None,
        change_logits_t2=None,
        event_logits_t1=torch.randn(6, 3, requires_grad=True),
        event_logits_t2=torch.randn(6, 3, requires_grad=True),
    )

    target_3d = {
        "semantic_t1": semantic_t1,
        "semantic_t2": semantic_t2,
        "event_t1": torch.tensor([0, 2, 0, 2, 0, 2]),
        "event_t2": torch.tensor([0, 1, 0, 1, 0, 1]),
        "event_valid_t1": torch.ones(6, dtype=torch.bool),
        "event_valid_t2": torch.ones(6, dtype=torch.bool),
    }

    out_3d = criterion(
        prediction=pred_3d,
        target=target_3d,
        class_names=class_names,
        dataset_name="NYC-SCD",
    )

    out_3d.total.backward()

    # Illegal event classes must receive zero gradient:
    # T1 illegal = Added (1)
    # T2 illegal = Removed (2)
    assert torch.allclose(
        pred_3d.event_logits_t1.grad[:, 1],
        torch.zeros_like(pred_3d.event_logits_t1.grad[:, 1]),
    )

    assert torch.allclose(
        pred_3d.event_logits_t2.grad[:, 2],
        torch.zeros_like(pred_3d.event_logits_t2.grad[:, 2]),
    )

    # ------------------------------------------------------------------
    # Future 2D+3D route compatibility
    # ------------------------------------------------------------------
    pred_2d3d = SimpleNamespace(
        semantic_logits_t1=torch.randn(6, 4, requires_grad=True),
        semantic_logits_t2=torch.randn(6, 4, requires_grad=True),
        change_logits_t1=torch.randn(6, requires_grad=True),
        change_logits_t2=torch.randn(6, requires_grad=True),
        event_logits_t1=torch.randn(6, 3, requires_grad=True),
        event_logits_t2=torch.randn(6, 3, requires_grad=True),
    )

    target_2d3d = {
        "semantic_t1": semantic_t1,
        "semantic_t2": semantic_t2,
        "change_t1": torch.tensor([0, 1, 0, 1, 1, 0]),
        "change_t2": torch.tensor([0, 1, 0, 1, 1, 0]),
        "event_t1": torch.tensor([0, 2, 0, 2, 0, 2]),
        "event_t2": torch.tensor([0, 1, 0, 1, 0, 1]),
        "event_valid_t1": torch.ones(6, dtype=torch.bool),
        "event_valid_t2": torch.ones(6, dtype=torch.bool),
    }

    out_2d3d = criterion(
        prediction=pred_2d3d,
        target=target_2d3d,
        class_names=class_names,
    )

    out_2d3d.total.backward()

    assert torch.isfinite(out_2d.total)
    assert torch.isfinite(out_3d.total)
    assert torch.isfinite(out_2d3d.total)

    assert pred_2d.change_logits_t1.grad is not None
    assert pred_3d.event_logits_t1.grad is not None
    assert pred_2d3d.change_logits_t1.grad is not None
    assert pred_2d3d.event_logits_t1.grad is not None

    print("loss.py self-test: PASS")


if __name__ == "__main__":
    _self_test()