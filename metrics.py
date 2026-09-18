"""
Unified streaming metrics for PAIR semantic change detection.

2D:
    semantic_logits_t1/t2 + binary change_logits_t1/t2

3D:
    semantic_logits_t1/t2 + 3-class event_logits_t1/t2
    event protocol: 0 unchanged, 1 added, 2 removed
    binary change metrics are derived from event != 0 when no binary head exists.
"""

from __future__ import annotations

from typing import Dict

import torch

from datasets.pair_dataset import infer_unchanged_raw_id


EPS = 1e-12
PAIR_EVENT_NAMES = {
    0: "unchanged",
    1: "added",
    2: "removed",
}
PAIR_EVENT_NUM_CLASSES = len(PAIR_EVENT_NAMES)
PAIR_EVENT_ACTIVE_SUPPORT = {
    1: (0, 2),  # T1: unchanged / removed
    2: (0, 1),  # T2: unchanged / added
}

def _safe_div(a, b):
    return a / b.clamp_min(EPS)


def _kappa(cm):
    cm = cm.double()
    total = cm.sum()
    if total <= 0:
        return torch.tensor(0.0, device=cm.device)

    po = torch.diag(cm).sum() / total
    pe = (cm.sum(1) * cm.sum(0)).sum() / (total * total)
    return (po - pe) / (1.0 - pe).clamp_min(EPS)


def _classification_metrics(cm):
    cm = cm.double()
    tp = torch.diag(cm)
    true_count = cm.sum(1)
    pred_count = cm.sum(0)
    union = true_count + pred_count - tp

    precision = _safe_div(tp, pred_count)
    recall = _safe_div(tp, true_count)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, union)

    valid_iou = union > 0
    valid_f1 = true_count > 0
    total = cm.sum()

    return {
        "OA": float((tp.sum() / total.clamp_min(1)).item()),
        "mIoU": float(iou[valid_iou].mean().item()) if valid_iou.any() else 0.0,
        "mF1": float(f1[valid_f1].mean().item()) if valid_f1.any() else 0.0,
        "Kappa": float(_kappa(cm).item()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "support": true_count,
    }


def _binary_metrics(cm):
    cm = cm.double()
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    precision = tp / (tp + fp).clamp_min(EPS)
    recall = tp / (tp + fn).clamp_min(EPS)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(EPS)
    iou = tp / (tp + fp + fn).clamp_min(EPS)
    oa = (tp + tn) / cm.sum().clamp_min(1)

    return {
        "OA": float(oa.item()),
        "Precision": float(precision.item()),
        "Recall": float(recall.item()),
        "F1": float(f1.item()),
        "IoU": float(iou.item()),
        "Kappa": float(_kappa(cm).item()),
        "TP": int(tp.item()),
        "TN": int(tn.item()),
        "FP": int(fp.item()),
        "FN": int(fn.item()),
    }


def _scd_metrics(cm, unchanged_index):
    order = [unchanged_index] + [
        i for i in range(cm.shape[0]) if i != unchanged_index
    ]

    q = cm[order][:, order].double()
    total = q.sum()

    if total <= 0:
        return {
            "OA": 0.0,
            "IoU_nc": 0.0,
            "IoU_c": 0.0,
            "mIoU": 0.0,
            "P_scd": 0.0,
            "R_scd": 0.0,
            "F_scd": 0.0,
            "SeK": 0.0,
            "Score": 0.0,
        }

    q00 = q[0, 0]
    iou_nc = q00 / (q[0, :].sum() + q[:, 0].sum() - q00).clamp_min(EPS)
    iou_c = q[1:, 1:].sum() / (total - q00).clamp_min(EPS)
    miou = 0.5 * (iou_nc + iou_c)

    correct_changed = torch.diag(q)[1:].sum()
    p_scd = correct_changed / q[:, 1:].sum().clamp_min(EPS)
    r_scd = correct_changed / q[1:, :].sum().clamp_min(EPS)
    f_scd = 2 * p_scd * r_scd / (p_scd + r_scd).clamp_min(EPS)

    qhat = q.clone()
    qhat[0, 0] = 0
    kappa_sep = _kappa(qhat)
    sek = kappa_sep * torch.exp(iou_c - 1.0)
    score = 0.3 * miou + 0.7 * sek
    oa = torch.diag(q).sum() / total

    return {
        "OA": float(oa.item()),
        "IoU_nc": float(iou_nc.item()),
        "IoU_c": float(iou_c.item()),
        "mIoU": float(miou.item()),
        "P_scd": float(p_scd.item()),
        "R_scd": float(r_scd.item()),
        "F_scd": float(f_scd.item()),
        "SeK": float(sek.item()),
        "Score": float(score.item()),
    }


class PAIRMetrics:
    def __init__(
        self,
        class_names: Dict[int, str],
        device,
        change_threshold=0.5,
        unchanged_raw_id=None,
    ):
        if not isinstance(class_names, dict) or not class_names:
            raise TypeError("class_names must be a non-empty Dict[int, str]")

        self.class_names = {int(k): str(v) for k, v in class_names.items()}
        self.raw_ids = tuple(sorted(self.class_names))
        self.names = tuple(self.class_names[k] for k in self.raw_ids)
        self.raw_to_local = {
            raw: i for i, raw in enumerate(self.raw_ids)
        }

        self.k = len(self.raw_ids)
        self.device = torch.device(device)
        self.threshold = float(change_threshold)

        if unchanged_raw_id is None:
            unchanged_raw_id = infer_unchanged_raw_id(self.class_names)

        self.unchanged_raw_id = unchanged_raw_id
        self.unchanged_local = (
            None
            if unchanged_raw_id is None
            else self.raw_to_local.get(int(unchanged_raw_id))
        )

        self.semantic_t1 = torch.zeros(
            self.k,
            self.k,
            dtype=torch.long,
            device=self.device,
        )
        self.semantic_t2 = torch.zeros_like(self.semantic_t1)
        self.scd = torch.zeros_like(self.semantic_t1)

        self.change = torch.zeros(
            2,
            2,
            dtype=torch.long,
            device=self.device,
        )

        self.event_t1 = torch.zeros(
            PAIR_EVENT_NUM_CLASSES,
            PAIR_EVENT_NUM_CLASSES,
            dtype=torch.long,
            device=self.device,
        )
        self.event_t2 = torch.zeros_like(self.event_t1)

        self.jse_tp = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.jse_pred_change = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.jse_gt_change = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _mask_or_true(target, key, value, device):
        valid = target.get(key)
        if valid is None:
            return torch.ones(
                value.numel(),
                dtype=torch.bool,
                device=device,
            )
        valid = valid.to(device).reshape(-1).bool()
        if valid.numel() != value.numel():
            raise ValueError(
                f"{key} size {valid.numel()} does not match "
                f"target size {value.numel()}"
            )
        return valid

    def _raw_to_local_target(self, raw, valid):
        raw = raw.to(self.device).reshape(-1).long()
        valid = valid.to(self.device).reshape(-1).bool()

        local = torch.full_like(raw, -1)
        matched = torch.zeros_like(valid)

        for raw_id, local_id in self.raw_to_local.items():
            mask = valid & (raw == raw_id)
            local[mask] = local_id
            matched |= mask

        bad = valid & ~matched
        if bad.any():
            values = torch.unique(raw[bad]).detach().cpu().tolist()
            raise ValueError(
                "Metric target contains undeclared raw class IDs: "
                f"{values}"
            )

        return local, valid

    @staticmethod
    def _update_confusion(cm, target, pred, valid, k):
        target = target[valid]
        pred = pred[valid]

        if target.numel() == 0:
            return

        bins = torch.bincount(
            target * k + pred,
            minlength=k * k,
        )
        cm += bins.reshape(k, k)

    @staticmethod
    def _validate_binary_target(target, valid, name):
        if not valid.any():
            return
        y = target[valid]
        if not torch.all((y == 0) | (y == 1)):
            values = torch.unique(y).detach().cpu().tolist()
            raise ValueError(
                f"{name} must contain only 0/1 on valid entries, got {values}"
            )

    @staticmethod
    def _validate_event_target(
        target,
        valid,
        name,
        allowed_ids=None,
    ):
        if not valid.any():
            return

        y = target[valid]

        # First check the global event ID range.
        bad = (y < 0) | (y >= PAIR_EVENT_NUM_CLASSES)
        if bad.any():
            values = torch.unique(
                y[bad]
            ).detach().cpu().tolist()

            raise ValueError(
                f"{name} contains invalid event IDs {values}; "
                f"expected 0..{PAIR_EVENT_NUM_CLASSES - 1}"
            )

        # Then check the time-specific active support.
        if allowed_ids is not None:
            support_ok = torch.zeros_like(
                y,
                dtype=torch.bool,
            )

            for event_id in allowed_ids:
                support_ok |= (y == event_id)

            if (~support_ok).any():
                values = torch.unique(
                    y[~support_ok]
                ).detach().cpu().tolist()

                raise ValueError(
                    f"{name} contains event IDs {values} outside "
                    f"its active support {tuple(allowed_ids)}"
                )
            
    def _active_event_argmax(
        self,
        logits,
        time_id,
    ):
        """
        Predict event classes only within the active support
        of the corresponding time point.

        T1: {0 unchanged, 2 removed}
        T2: {0 unchanged, 1 added}
        """
        if time_id not in PAIR_EVENT_ACTIVE_SUPPORT:
            raise ValueError(
                f"Unsupported time_id={time_id}; expected 1 or 2"
            )

        scores = logits.detach().to(self.device)

        allowed = torch.tensor(
            PAIR_EVENT_ACTIVE_SUPPORT[time_id],
            dtype=torch.long,
            device=self.device,
        )

        # Restrict competition to valid event classes BEFORE argmax.
        local_pred = (
            scores
            .index_select(1, allowed)
            .argmax(-1)
        )

        # Map the local index back to the global event ID.
        pred = allowed[local_pred]

        return pred.reshape(-1)

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

    def _update_semantic(self, prediction, target):
        raw1 = target["semantic_t1"]
        raw2 = target["semantic_t2"]

        valid1 = self._mask_or_true(
            target,
            "semantic_valid_t1",
            raw1,
            self.device,
        )
        valid2 = self._mask_or_true(
            target,
            "semantic_valid_t2",
            raw2,
            self.device,
        )

        gt1, valid1 = self._raw_to_local_target(raw1, valid1)
        gt2, valid2 = self._raw_to_local_target(raw2, valid2)

        pred1 = (
            prediction.semantic_logits_t1
            .detach()
            .argmax(-1)
            .to(self.device)
            .reshape(-1)
        )
        pred2 = (
            prediction.semantic_logits_t2
            .detach()
            .argmax(-1)
            .to(self.device)
            .reshape(-1)
        )

        if pred1.numel() != gt1.numel() or pred2.numel() != gt2.numel():
            raise ValueError("Semantic prediction/target size mismatch")

        self._update_confusion(
            self.semantic_t1,
            gt1,
            pred1,
            valid1,
            self.k,
        )
        self._update_confusion(
            self.semantic_t2,
            gt2,
            pred2,
            valid2,
            self.k,
        )

        return pred1, pred2, gt1, gt2, valid1, valid2

    def _update_binary_change(
        self,
        prediction,
        target,
        pred1,
        pred2,
        gt1,
        gt2,
        valid1,
        valid2,
    ):
        logits1 = getattr(prediction, "change_logits_t1", None)
        logits2 = getattr(prediction, "change_logits_t2", None)

        if (logits1 is None) != (logits2 is None):
            raise ValueError(
                "change_logits_t1/t2 must be present together"
            )
        if logits1 is None:
            return False

        cgt1, cv1 = self._change_target(target, 1)
        cgt2, cv2 = self._change_target(target, 2)

        cgt1 = cgt1.to(self.device).reshape(-1).long()
        cgt2 = cgt2.to(self.device).reshape(-1).long()
        cv1 = cv1.to(self.device).reshape(-1).bool()
        cv2 = cv2.to(self.device).reshape(-1).bool()

        self._validate_binary_target(cgt1, cv1, "change_t1")
        self._validate_binary_target(cgt2, cv2, "change_t2")

        prob1 = torch.sigmoid(
            logits1.detach().float()
        ).to(self.device).reshape(-1)
        prob2 = torch.sigmoid(
            logits2.detach().float()
        ).to(self.device).reshape(-1)

        if prob1.numel() != cgt1.numel() or prob2.numel() != cgt2.numel():
            raise ValueError("Binary change prediction/target size mismatch")

        shared_change = (
            "change_t1" not in target
            and "change_t2" not in target
            and prob1.numel() == prob2.numel() == cgt1.numel()
        )

        if shared_change:
            prob = 0.5 * (prob1 + prob2)
            cpred = (prob >= self.threshold).long()

            self._update_confusion(
                self.change,
                cgt1,
                cpred,
                cv1,
                2,
            )

            if self.unchanged_local is not None:
                gated1 = pred1.clone()
                gated2 = pred2.clone()
                unchanged = ~cpred.bool()

                gated1[unchanged] = self.unchanged_local
                gated2[unchanged] = self.unchanged_local

                self._update_confusion(
                    self.scd,
                    gt1,
                    gated1,
                    valid1,
                    self.k,
                )
                self._update_confusion(
                    self.scd,
                    gt2,
                    gated2,
                    valid2,
                    self.k,
                )
        else:
            cpred1 = (prob1 >= self.threshold).long()
            cpred2 = (prob2 >= self.threshold).long()

            self._update_confusion(
                self.change,
                cgt1,
                cpred1,
                cv1,
                2,
            )
            self._update_confusion(
                self.change,
                cgt2,
                cpred2,
                cv2,
                2,
            )

            if self.unchanged_local is not None:
                gated1 = pred1.clone()
                gated2 = pred2.clone()

                gated1[~cpred1.bool()] = self.unchanged_local
                gated2[~cpred2.bool()] = self.unchanged_local

                self._update_confusion(
                    self.scd,
                    gt1,
                    gated1,
                    valid1,
                    self.k,
                )
                self._update_confusion(
                    self.scd,
                    gt2,
                    gated2,
                    valid2,
                    self.k,
                )

        return True

    def _update_event(
        self,
        prediction,
        target,
        pred1,
        pred2,
        gt1,
        gt2,
        valid1,
        valid2,
        update_change,
    ):
        logits1 = getattr(prediction, "event_logits_t1", None)
        logits2 = getattr(prediction, "event_logits_t2", None)

        if (logits1 is None) != (logits2 is None):
            raise ValueError(
                "event_logits_t1/t2 must be present together"
            )
        if logits1 is None:
            return False

        if (
            logits1.ndim != 2
            or logits2.ndim != 2
            or logits1.shape[1] != PAIR_EVENT_NUM_CLASSES
            or logits2.shape[1] != PAIR_EVENT_NUM_CLASSES
        ):
            raise ValueError(
                f"event logits must be [N,{PAIR_EVENT_NUM_CLASSES}]"
            )

        if "event_t1" not in target or "event_t2" not in target:
            raise KeyError(
                "Event metrics require event_t1 and event_t2 targets"
            )

        egt1 = target["event_t1"].to(self.device).reshape(-1).long()
        egt2 = target["event_t2"].to(self.device).reshape(-1).long()

        ev1 = self._mask_or_true(
            target,
            "event_valid_t1",
            target["event_t1"],
            self.device,
        )
        ev2 = self._mask_or_true(
            target,
            "event_valid_t2",
            target["event_t2"],
            self.device,
        )

        self._validate_event_target(
            egt1,
            ev1,
            "event_t1",
            allowed_ids=PAIR_EVENT_ACTIVE_SUPPORT[1],
        )
        self._validate_event_target(
            egt2,
            ev2,
            "event_t2",
            allowed_ids=PAIR_EVENT_ACTIVE_SUPPORT[2],
        )

        epred1 = self._active_event_argmax(
            logits1,
            time_id=1,
        )

        epred2 = self._active_event_argmax(
            logits2,
            time_id=2,
        )

        if epred1.numel() != egt1.numel() or epred2.numel() != egt2.numel():
            raise ValueError("Event prediction/target size mismatch")

        self._update_confusion(
            self.event_t1,
            egt1,
            epred1,
            ev1,
            PAIR_EVENT_NUM_CLASSES,
        )
        self._update_confusion(
            self.event_t2,
            egt2,
            epred2,
            ev2,
            PAIR_EVENT_NUM_CLASSES,
        )

        # 3D has no separate binary head. In that case derive:
        # unchanged -> 0, added/removed -> 1.
        if update_change:
            cgt1 = (egt1 != 0).long()
            cgt2 = (egt2 != 0).long()
            cpred1 = (epred1 != 0).long()
            cpred2 = (epred2 != 0).long()

            self._update_confusion(
                self.change,
                cgt1,
                cpred1,
                ev1,
                2,
            )
            self._update_confusion(
                self.change,
                cgt2,
                cpred2,
                ev2,
                2,
            )

            # Joint Semantic-Event F1 (JSE-F1).
            #
            # A changed point is a JSE true positive only when:
            #   1) it is truly changed,
            #   2) its event type is correct,
            #   3) its semantic class is correct.
            #
            # For changed GT points, semantic GT must be valid.
            # For unchanged GT points, semantic validity is not required because
            # a predicted change is still a valid false positive.
            jvalid1 = ev1 & ((egt1 == 0) | valid1)
            jvalid2 = ev2 & ((egt2 == 0) | valid2)

            gt_changed1 = egt1 != 0
            gt_changed2 = egt2 != 0

            pred_changed1 = epred1 != 0
            pred_changed2 = epred2 != 0

            joint_correct1 = (
                gt_changed1
                & (epred1 == egt1)
                & (pred1 == gt1)
            )
            joint_correct2 = (
                gt_changed2
                & (epred2 == egt2)
                & (pred2 == gt2)
            )

            self.jse_tp += (
                jvalid1 & joint_correct1
            ).sum()
            self.jse_tp += (
                jvalid2 & joint_correct2
            ).sum()

            self.jse_pred_change += (
                jvalid1 & pred_changed1
            ).sum()
            self.jse_pred_change += (
                jvalid2 & pred_changed2
            ).sum()

            self.jse_gt_change += (
                jvalid1 & gt_changed1
            ).sum()
            self.jse_gt_change += (
                jvalid2 & gt_changed2
            ).sum()

            if self.unchanged_local is not None:
                gated1 = pred1.clone()
                gated2 = pred2.clone()

                gated1[~cpred1.bool()] = self.unchanged_local
                gated2[~cpred2.bool()] = self.unchanged_local

                self._update_confusion(
                    self.scd,
                    gt1,
                    gated1,
                    valid1,
                    self.k,
                )
                self._update_confusion(
                    self.scd,
                    gt2,
                    gated2,
                    valid2,
                    self.k,
                )

        return True

    def update(self, prediction, target):
        if tuple(prediction.raw_class_ids) != self.raw_ids:
            raise ValueError(
                f"Decoder class order {prediction.raw_class_ids} != "
                f"metric class order {self.raw_ids}"
            )

        pred1, pred2, gt1, gt2, valid1, valid2 = self._update_semantic(
            prediction,
            target,
        )

        has_binary = self._update_binary_change(
            prediction,
            target,
            pred1,
            pred2,
            gt1,
            gt2,
            valid1,
            valid2,
        )

        has_event = self._update_event(
            prediction,
            target,
            pred1,
            pred2,
            gt1,
            gt2,
            valid1,
            valid2,
            update_change=not has_binary,
        )

        if not has_binary and not has_event:
            raise ValueError(
                "PAIRMetrics received neither binary change logits nor event logits"
            )

    def reduce_distributed(self):
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            for tensor in (
                self.semantic_t1,
                self.semantic_t2,
                self.scd,
                self.change,
                self.event_t1,
                self.event_t2,
                self.jse_tp,
                self.jse_pred_change,
                self.jse_gt_change,
            ):
                torch.distributed.all_reduce(
                    tensor,
                    op=torch.distributed.ReduceOp.SUM,
                )

    def compute(self):
        t1 = _classification_metrics(self.semantic_t1)
        t2 = _classification_metrics(self.semantic_t2)

        combined_cm = self.semantic_t1 + self.semantic_t2
        combined = _classification_metrics(combined_cm)
        change = _binary_metrics(self.change)

        result = {
            "change/OA": change["OA"],
            "change/Precision": change["Precision"],
            "change/Recall": change["Recall"],
            "change/F1": change["F1"],
            "change/IoU": change["IoU"],
            "change/Kappa": change["Kappa"],

            "semantic_t1/OA": t1["OA"],
            "semantic_t1/mIoU": t1["mIoU"],
            "semantic_t1/mF1": t1["mF1"],
            "semantic_t1/Kappa": t1["Kappa"],

            "semantic_t2/OA": t2["OA"],
            "semantic_t2/mIoU": t2["mIoU"],
            "semantic_t2/mF1": t2["mF1"],
            "semantic_t2/Kappa": t2["Kappa"],

            "semantic/OA": combined["OA"],
            "semantic/mIoU": combined["mIoU"],
            "semantic/mF1": combined["mF1"],
            "semantic/Kappa": combined["Kappa"],
        }

        per_class = {}
        for i, (raw_id, name) in enumerate(
            zip(self.raw_ids, self.names)
        ):
            per_class[name] = {
                "raw_id": raw_id,
                "IoU": float(combined["iou"][i].item()),
                "F1": float(combined["f1"][i].item()),
                "Precision": float(combined["precision"][i].item()),
                "Recall": float(combined["recall"][i].item()),
                "Support": int(combined["support"][i].item()),
            }

        event_per_class = {}
        event_combined_cm = self.event_t1 + self.event_t2

        if event_combined_cm.sum() > 0:
            event1 = _classification_metrics(self.event_t1)
            event2 = _classification_metrics(self.event_t2)
            event = _classification_metrics(event_combined_cm)

            result.update({
                "event_t1/OA": event1["OA"],
                "event_t1/mIoU": event1["mIoU"],
                "event_t1/mF1": event1["mF1"],
                "event_t1/Kappa": event1["Kappa"],

                "event_t2/OA": event2["OA"],
                "event_t2/mIoU": event2["mIoU"],
                "event_t2/mF1": event2["mF1"],
                "event_t2/Kappa": event2["Kappa"],

                "event/OA": event["OA"],
                "event/mIoU": event["mIoU"],
                "event/mF1": event["mF1"],
                "event/Kappa": event["Kappa"],
            })

            jse_tp = self.jse_tp.double()
            jse_pred_change = self.jse_pred_change.double()
            jse_gt_change = self.jse_gt_change.double()

            jse_precision = (
                jse_tp
                / jse_pred_change.clamp_min(EPS)
            )
            jse_recall = (
                jse_tp
                / jse_gt_change.clamp_min(EPS)
            )
            jse_f1 = (
                2.0 * jse_precision * jse_recall
                / (jse_precision + jse_recall).clamp_min(EPS)
            )

            result.update({
                "jse/Precision": float(jse_precision.item()),
                "jse/Recall": float(jse_recall.item()),
                "jse/F1": float(jse_f1.item()),
            })

            for i in range(PAIR_EVENT_NUM_CLASSES):
                name = PAIR_EVENT_NAMES[i]
                event_per_class[name] = {
                    "event_id": i,
                    "IoU": float(event["iou"][i].item()),
                    "F1": float(event["f1"][i].item()),
                    "Precision": float(event["precision"][i].item()),
                    "Recall": float(event["recall"][i].item()),
                    "Support": int(event["support"][i].item()),
                }

        if self.unchanged_local is not None:
            for key, value in _scd_metrics(
                self.scd,
                self.unchanged_local,
            ).items():
                result[f"scd/{key}"] = value

        return {
            "scalars": result,
            "per_class": per_class,
            "event_per_class": event_per_class,
            "confusion": {
                "semantic_t1": self.semantic_t1.detach().cpu(),
                "semantic_t2": self.semantic_t2.detach().cpu(),
                "semantic_combined": combined_cm.detach().cpu(),
                "scd_gated": self.scd.detach().cpu(),
                "change": self.change.detach().cpu(),
                "event_t1": self.event_t1.detach().cpu(),
                "event_t2": self.event_t2.detach().cpu(),
                "event_combined": event_combined_cm.detach().cpu(),
            },
        }


def normalized_confusion_image(cm):
    cm = cm.float()
    denom = cm.sum(1, keepdim=True).clamp_min(1)
    return (cm / denom).unsqueeze(0)


def _self_test():
    from types import SimpleNamespace

    device = "cpu"

    # 2D binary-change path.
    m2d = PAIRMetrics(
        {
            0: "unchanged",
            1: "building",
            2: "vegetation",
        },
        device,
    )

    p2d = SimpleNamespace(
        raw_class_ids=(0, 1, 2),
        semantic_logits_t1=torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]),
        semantic_logits_t2=torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]),
        change_logits_t1=torch.tensor([-5.0, 5.0, 5.0, 5.0]),
        change_logits_t2=torch.tensor([-5.0, 5.0, 5.0, 5.0]),
        event_logits_t1=None,
        event_logits_t2=None,
    )

    t2d = {
        "semantic_t1": torch.tensor([0, 1, 2, 1]),
        "semantic_t2": torch.tensor([0, 1, 2, 1]),
        "semantic_valid_t1": torch.ones(4, dtype=torch.bool),
        "semantic_valid_t2": torch.ones(4, dtype=torch.bool),
        "change": torch.tensor([0, 1, 1, 1]),
        "change_valid": torch.ones(4, dtype=torch.bool),
    }

    m2d.update(p2d, t2d)
    r2d = m2d.compute()
    assert r2d["scalars"]["change/F1"] == 1.0
    assert "event/mIoU" not in r2d["scalars"]

    # 3D event path.
    m3d = PAIRMetrics(
        {
            0: "ground",
            1: "building",
            2: "vegetation",
            3: "clutter",
        },
        device,
    )

    p3d = SimpleNamespace(
        raw_class_ids=(0, 1, 2, 3),
        semantic_logits_t1=torch.eye(4),
        semantic_logits_t2=torch.eye(4),
        change_logits_t1=None,
        change_logits_t2=None,
        event_logits_t1=torch.tensor([
            [5.0, 0.0, 0.0],  # unchanged
            [0.0, 0.0, 5.0],  # removed
            [5.0, 0.0, 0.0],  # unchanged
            [0.0, 0.0, 5.0],  # removed
        ]),
        event_logits_t2=torch.tensor([
            [5.0, 0.0, 0.0],  # unchanged
            [0.0, 5.0, 0.0],  # added
            [5.0, 0.0, 0.0],  # unchanged
            [0.0, 5.0, 0.0],  # added
        ]),
    )

    t3d = {
        "semantic_t1": torch.tensor([0, 1, 2, 3]),
        "semantic_t2": torch.tensor([0, 1, 2, 3]),
        "event_t1": torch.tensor([0, 2, 0, 2]),
        "event_t2": torch.tensor([0, 1, 0, 1]),
        "event_valid_t1": torch.ones(4, dtype=torch.bool),
        "event_valid_t2": torch.ones(4, dtype=torch.bool),
    }

    m3d.update(p3d, t3d)
    r3d = m3d.compute()
    assert r3d["scalars"]["change/F1"] == 1.0
    assert r3d["scalars"]["event/OA"] == 1.0
    assert r3d["scalars"]["jse/F1"] == 1.0
    assert r3d["confusion"]["event_combined"].shape == (3, 3)


    print("metrics.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
