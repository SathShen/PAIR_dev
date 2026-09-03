"""
PAIR semantic-change loss.

Supports arbitrary raw dataset class IDs declared manually in DatasetSpec:

    class_names = {
        0: "unchanged",
        10: "water",
        50: "building",
    }

Raw semantic labels are remapped only according to this dictionary:

    raw 0  -> local 0
    raw 10 -> local 1
    raw 50 -> local 2

No label-file scanning or automatic class discovery is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

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
    def __init__(
        self,
        *,
        semantic_weight: float = 1.0,
        change_bce_weight: float = 1.0,
        change_dice_weight: float = 1.0,
        dice_eps: float = 1.0,
    ):
        super().__init__()

        self.semantic_weight = float(
            semantic_weight
        )
        self.change_bce_weight = float(
            change_bce_weight
        )
        self.change_dice_weight = float(
            change_dice_weight
        )
        self.dice_eps = float(
            dice_eps
        )

    # ------------------------------------------------------------------
    # Raw -> local class mapping
    # ------------------------------------------------------------------

    @staticmethod
    def make_raw_to_local(
        class_names: Dict[int, str],
    ) -> Dict[int, int]:
        if not isinstance(class_names, dict):
            raise TypeError(
                "class_names must be Dict[int, str]"
            )

        raw_ids = sorted(
            int(k)
            for k in class_names.keys()
        )

        return {
            raw_id: local_id
            for local_id, raw_id
            in enumerate(raw_ids)
        }

    @classmethod
    def remap_semantic_target(
        cls,
        *,
        raw_target: torch.Tensor,
        valid_mask: torch.Tensor,
        class_names: Dict[int, str],
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Convert arbitrary source class IDs into CE-compatible 0..K-1 indices.

        Only positions where valid_mask=True are checked.
        """
        if raw_target.shape != valid_mask.shape:
            raise ValueError(
                "semantic target and valid mask shapes differ"
            )

        mapping = cls.make_raw_to_local(
            class_names
        )

        local = torch.full_like(
            raw_target,
            fill_value=int(ignore_index),
            dtype=torch.long,
        )

        matched = torch.zeros_like(
            valid_mask,
            dtype=torch.bool,
        )

        for raw_id, local_id in mapping.items():
            mask = (
                valid_mask
                & (raw_target == raw_id)
            )

            local[mask] = int(
                local_id
            )

            matched |= mask

        bad = (
            valid_mask
            & ~matched
        )

        if bad.any():
            bad_values = torch.unique(
                raw_target[bad]
            ).detach().cpu().tolist()

            raise ValueError(
                "Valid semantic labels contain raw class IDs not declared "
                f"in DatasetSpec.class_names: {bad_values}"
            )

        return local

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    @staticmethod
    def masked_semantic_ce(
        *,
        logits: torch.Tensor,
        raw_target: torch.Tensor,
        valid_mask: torch.Tensor,
        class_names: Dict[int, str],
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(
                f"semantic logits must be [B,K,H,W], got {tuple(logits.shape)}"
            )

        if raw_target.ndim == 2:
            raw_target = raw_target.unsqueeze(
                0
            )

        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(
                0
            )

        local_target = (
            PAIRSemanticChangeLoss
            .remap_semantic_target(
                raw_target=raw_target,
                valid_mask=valid_mask,
                class_names=class_names,
            )
        )

        if not valid_mask.any():
            # Differentiable zero.
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits.float(),
            local_target,
            ignore_index=-100,
        )

    @staticmethod
    def masked_change_bce(
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                "change_logits must be [B,1,H,W]"
            )

        logits = logits[:, 0]

        if target.ndim == 2:
            target = target.unsqueeze(
                0
            )

        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(
                0
            )

        if not valid_mask.any():
            return logits.sum() * 0.0

        y = target[
            valid_mask
        ].float()

        if not torch.all(
            (y == 0) | (y == 1)
        ):
            values = torch.unique(
                y
            ).detach().cpu().tolist()

            raise ValueError(
                f"valid change target must contain only 0/1, got {values}"
            )

        return F.binary_cross_entropy_with_logits(
            logits[valid_mask].float(),
            y,
        )

    def masked_change_dice(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits[:, 0]

        if target.ndim == 2:
            target = target.unsqueeze(
                0
            )

        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(
                0
            )

        if not valid_mask.any():
            return logits.sum() * 0.0

        prob = torch.sigmoid(
            logits.float()
        )

        target = target.float()

        prob = prob[
            valid_mask
        ]

        target = target[
            valid_mask
        ]

        intersection = (
            prob * target
        ).sum()

        denominator = (
            prob.sum()
            + target.sum()
        )

        dice = (
            2.0 * intersection
            + self.dice_eps
        ) / (
            denominator
            + self.dice_eps
        )

        return 1.0 - dice

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        prediction,
        target,
        class_names: Dict[int, str],
    ) -> ChangeLossOutput:
        sem_t1 = target[
            "semantic_t1"
        ].to(
            prediction.semantic_logits_t1.device
        )

        sem_t2 = target[
            "semantic_t2"
        ].to(
            prediction.semantic_logits_t2.device
        )

        sem_valid_t1 = target[
            "semantic_valid_t1"
        ].to(
            prediction.semantic_logits_t1.device
        )

        sem_valid_t2 = target[
            "semantic_valid_t2"
        ].to(
            prediction.semantic_logits_t2.device
        )

        change = target[
            "change"
        ].to(
            prediction.change_logits.device
        )

        change_valid = target[
            "change_valid"
        ].to(
            prediction.change_logits.device
        )

        loss_sem1 = self.masked_semantic_ce(
            logits=prediction.semantic_logits_t1,
            raw_target=sem_t1,
            valid_mask=sem_valid_t1,
            class_names=class_names,
        )

        loss_sem2 = self.masked_semantic_ce(
            logits=prediction.semantic_logits_t2,
            raw_target=sem_t2,
            valid_mask=sem_valid_t2,
            class_names=class_names,
        )

        loss_bce = self.masked_change_bce(
            logits=prediction.change_logits,
            target=change,
            valid_mask=change_valid,
        )

        loss_dice = self.masked_change_dice(
            logits=prediction.change_logits,
            target=change,
            valid_mask=change_valid,
        )

        total = (
            self.semantic_weight
            * (loss_sem1 + loss_sem2)
            + self.change_bce_weight
            * loss_bce
            + self.change_dice_weight
            * loss_dice
        )

        return ChangeLossOutput(
            total=total,
            semantic_t1=loss_sem1,
            semantic_t2=loss_sem2,
            change_bce=loss_bce,
            change_dice=loss_dice,
        )
