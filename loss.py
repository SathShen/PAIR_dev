"""
PAIR unified semantic-change loss.

Decoder-side logits:
    semantic_logits_t1 : [N1, K]
    semantic_logits_t2 : [N2, K]
    change_logits_t1   : [N1]
    change_logits_t2   : [N2]

Targets may be 2D maps [H,W] or 3D point labels [N]; they are flattened here.

Dataset class IDs do NOT need to be continuous. Example:
    class_names = {0: "unchanged", 10: "water", 50: "building"}

The model-local classifier order is the sorted raw IDs:
    0 -> local 0
    10 -> local 1
    50 -> local 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChangeLossOutput:
    total: torch.Tensor
    semantic_t1: torch.Tensor
    semantic_t2: torch.Tensor
    change_bce: torch.Tensor
    change_dice: torch.Tensor

    def as_dict(self):
        return {
            "loss": self.total,
            "loss_semantic_t1": self.semantic_t1,
            "loss_semantic_t2": self.semantic_t2,
            "loss_change_bce": self.change_bce,
            "loss_change_dice": self.change_dice,
        }


class PAIRSemanticChangeLoss(nn.Module):
    def __init__(self, semantic_weight=1.0, change_bce_weight=1.0,
                 change_dice_weight=1.0, dice_eps=1.0):
        super().__init__()
        self.semantic_weight = float(semantic_weight)
        self.change_bce_weight = float(change_bce_weight)
        self.change_dice_weight = float(change_dice_weight)
        self.dice_eps = float(dice_eps)

    @staticmethod
    def make_raw_to_local(class_names: Dict[int, str]) -> Dict[int, int]:
        if not isinstance(class_names, dict) or not class_names:
            raise TypeError("class_names must be a non-empty Dict[int, str]")
        raw_ids = sorted(int(k) for k in class_names.keys())
        return {raw_id: local_id for local_id, raw_id in enumerate(raw_ids)}

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
            raise ValueError(
                f"Semantic labels {values} are not declared in DatasetSpec.class_names"
            )
        return local

    @classmethod
    def semantic_ce(cls, logits, raw_target, valid_mask, class_names):
        if logits.ndim != 2:
            raise ValueError(f"semantic logits must be [N,K], got {tuple(logits.shape)}")

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

    @staticmethod
    def _prepare_change(logits, target, valid_mask):
        logits = logits.reshape(-1)
        target = target.to(logits.device).reshape(-1)
        valid_mask = valid_mask.to(logits.device).reshape(-1).bool()

        if logits.numel() != target.numel() or target.numel() != valid_mask.numel():
            raise ValueError(
                f"change logits/target/mask size mismatch: "
                f"{logits.numel()}, {target.numel()}, {valid_mask.numel()}"
            )
        if not valid_mask.any():
            return logits, target.float(), valid_mask

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
        dice = (2 * intersection + self.dice_eps) / (
            prob.sum() + target.sum() + self.dice_eps
        )
        return 1.0 - dice

    @staticmethod
    def _change_target(target, time_id):
        """
        Current 2D datasets use one shared change map:
            change / change_valid

        Future 3D datasets may have different T1/T2 point topologies and can provide:
            change_t1 / change_valid_t1
            change_t2 / change_valid_t2
        """
        key = f"change_t{time_id}"
        valid_key = f"change_valid_t{time_id}"
        if key in target:
            if valid_key not in target:
                raise KeyError(f"{key} exists but {valid_key} is missing")
            return target[key], target[valid_key]
        return target["change"], target["change_valid"]

    def forward(self, *, prediction, target, class_names: Dict[int, str]):
        sem1 = self.semantic_ce(
            prediction.semantic_logits_t1, target["semantic_t1"],
            target["semantic_valid_t1"], class_names
        )
        sem2 = self.semantic_ce(
            prediction.semantic_logits_t2, target["semantic_t2"],
            target["semantic_valid_t2"], class_names
        )

        change_t1, valid_t1 = self._change_target(target, 1)
        change_t2, valid_t2 = self._change_target(target, 2)

        bce1 = self.change_bce(prediction.change_logits_t1, change_t1, valid_t1)
        bce2 = self.change_bce(prediction.change_logits_t2, change_t2, valid_t2)
        dice1 = self.change_dice(prediction.change_logits_t1, change_t1, valid_t1)
        dice2 = self.change_dice(prediction.change_logits_t2, change_t2, valid_t2)

        change_bce = 0.5 * (bce1 + bce2)
        change_dice = 0.5 * (dice1 + dice2)

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
        )
