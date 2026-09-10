"""
Unified streaming metrics for PAIR semantic change detection.

2D:
    - semantic T1/T2 metrics
    - binary change metrics
    - existing SCD gated metrics when class_names contains a true "unchanged" class

3D:
    - semantic T1/T2 metrics
    - active-class event metrics
    - binary change metrics derived from event != unchanged

PAIR 3D event taxonomy:
    0 unchanged
    1 added
    2 removed
    3 class_change
    4 height_up
    5 height_down

Current NYC-SCD event supervision:
    T1 active: [0, 2]  unchanged / removed
    T2 active: [0, 1]  unchanged / added

Inactive event classes do not participate in prediction argmax or metrics.
For NYC-SCD, event classes 3/4/5 are reported as N/A rather than zero.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from datasets.pair_dataset import infer_unchanged_raw_id


EPS = 1e-12
PAIR_EVENT_NUM_CLASSES = 6
PAIR_EVENT_NAMES = (
    "unchanged",
    "added",
    "removed",
    "class_change",
    "height_up",
    "height_down",
)

# Runtime supervision metadata, not user config.
_EVENT_ACTIVE_CLASSES = {
    "nyc-scd": ((0, 2), (0, 1)),
}


# =============================================================================
# Generic metrics
# =============================================================================

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
    order = [unchanged_index] + [i for i in range(cm.shape[0]) if i != unchanged_index]
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


# =============================================================================
# Streaming PAIR metrics
# =============================================================================

class PAIRMetrics:
    def __init__(
        self,
        class_names: Dict[int, str],
        device,
        change_threshold=0.5,
        unchanged_raw_id=None,
        dataset_name: Optional[str] = None,
        event_active_classes_t1: Optional[Sequence[int]] = None,
        event_active_classes_t2: Optional[Sequence[int]] = None,
    ):
        if not isinstance(class_names, dict) or not class_names:
            raise TypeError("class_names must be a non-empty Dict[int, str]")

        self.class_names = {int(k): str(v) for k, v in class_names.items()}
        self.raw_ids = tuple(sorted(self.class_names))
        self.names = tuple(self.class_names[k] for k in self.raw_ids)
        self.raw_to_local = {raw: i for i, raw in enumerate(self.raw_ids)}
        self.k = len(self.raw_ids)
        self.device = torch.device(device)
        self.threshold = float(change_threshold)
        self.dataset_name = dataset_name

        if unchanged_raw_id is None:
            unchanged_raw_id = infer_unchanged_raw_id(self.class_names)
        self.unchanged_raw_id = unchanged_raw_id
        self.unchanged_local = (
            None if unchanged_raw_id is None else self.raw_to_local.get(int(unchanged_raw_id))
        )

        # Semantic / existing 2D metrics.
        self.semantic_t1 = torch.zeros(self.k, self.k, dtype=torch.long, device=self.device)
        self.semantic_t2 = torch.zeros_like(self.semantic_t1)
        self.scd = torch.zeros_like(self.semantic_t1)
        self.change = torch.zeros(2, 2, dtype=torch.long, device=self.device)

        # 3D event metrics are resolved lazily so 2D datasets need no event metadata.
        self.event_active_t1 = None
        self.event_active_t2 = None
        self.event_union = None
        self.event_union_to_local = None
        self.event_t1 = None
        self.event_t2 = None
        self.event = None

        if (event_active_classes_t1 is None) != (event_active_classes_t2 is None):
            raise ValueError("event_active_classes_t1 and event_active_classes_t2 must be provided together")

        if event_active_classes_t1 is not None:
            self._initialize_event_metrics(
                tuple(int(x) for x in event_active_classes_t1),
                tuple(int(x) for x in event_active_classes_t2),
            )
        else:
            key = self._normalize_dataset_name(dataset_name)
            if key in _EVENT_ACTIVE_CLASSES:
                self._initialize_event_metrics(*_EVENT_ACTIVE_CLASSES[key])

    # =========================================================================
    # Shared target/confusion helpers
    # =========================================================================

    @staticmethod
    def _normalize_dataset_name(dataset_name):
        if dataset_name is None:
            return None
        return str(dataset_name).strip().lower().replace("_", "-")

    @staticmethod
    def _valid_or_true(target, key, value, device):
        valid = target.get(key)
        if valid is None:
            return torch.ones(value.numel(), dtype=torch.bool, device=device)
        valid = valid.to(device).reshape(-1).bool()
        if valid.numel() != value.numel():
            raise ValueError(f"{key} size {valid.numel()} does not match target size {value.numel()}")
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
            raise ValueError(f"Metric target contains undeclared raw class IDs: {values}")
        return local, valid

    @staticmethod
    def _update_confusion(cm, target, pred, valid, k):
        target = target[valid]
        pred = pred[valid]
        if target.numel() == 0:
            return
        bins = torch.bincount(target * k + pred, minlength=k * k)
        cm += bins.reshape(k, k)

    # =========================================================================
    # Semantic
    # =========================================================================

    def _update_semantic(self, prediction, target):
        if tuple(prediction.raw_class_ids) != self.raw_ids:
            raise ValueError(
                f"Decoder class order {prediction.raw_class_ids} != metric class order {self.raw_ids}"
            )

        pred1 = prediction.semantic_logits_t1.detach().argmax(-1).to(self.device)
        pred2 = prediction.semantic_logits_t2.detach().argmax(-1).to(self.device)

        raw1 = target["semantic_t1"]
        raw2 = target["semantic_t2"]
        valid1 = self._valid_or_true(target, "semantic_valid_t1", raw1, self.device)
        valid2 = self._valid_or_true(target, "semantic_valid_t2", raw2, self.device)
        gt1, valid1 = self._raw_to_local_target(raw1, valid1)
        gt2, valid2 = self._raw_to_local_target(raw2, valid2)

        if pred1.numel() != gt1.numel() or pred2.numel() != gt2.numel():
            raise ValueError("Semantic prediction/target size mismatch")

        self._update_confusion(self.semantic_t1, gt1, pred1, valid1, self.k)
        self._update_confusion(self.semantic_t2, gt2, pred2, valid2, self.k)
        return pred1, pred2, gt1, gt2, valid1, valid2

    # =========================================================================
    # Existing 2D binary change
    # =========================================================================

    @staticmethod
    def _change_target(target, time_id):
        key = f"change_t{time_id}"
        valid_key = f"change_valid_t{time_id}"
        if key in target:
            value = target[key]
            valid = target.get(valid_key)
        else:
            if "change" not in target:
                raise KeyError("2D binary metrics require target['change']")
            value = target["change"]
            valid = target.get("change_valid")

        if valid is None:
            valid = torch.ones_like(value, dtype=torch.bool)
        return value, valid

    def _update_binary_2d(self, prediction, target, pred1, pred2, gt1, gt2, valid1, valid2):
        if prediction.change_logits_t1 is None or prediction.change_logits_t2 is None:
            raise ValueError("2D prediction must provide change_logits_t1 and change_logits_t2")
        if prediction.event_logits_t1 is not None or prediction.event_logits_t2 is not None:
            raise ValueError("2D prediction must not provide event logits")

        cgt1, cv1 = self._change_target(target, 1)
        cgt2, cv2 = self._change_target(target, 2)
        cgt1 = cgt1.to(self.device).reshape(-1).long()
        cgt2 = cgt2.to(self.device).reshape(-1).long()
        cv1 = cv1.to(self.device).reshape(-1).bool()
        cv2 = cv2.to(self.device).reshape(-1).bool()

        prob1 = torch.sigmoid(prediction.change_logits_t1.detach().float()).to(self.device).reshape(-1)
        prob2 = torch.sigmoid(prediction.change_logits_t2.detach().float()).to(self.device).reshape(-1)

        shared_change = (
            "change_t1" not in target
            and "change_t2" not in target
            and prob1.numel() == prob2.numel() == cgt1.numel()
        )

        if shared_change:
            prob = 0.5 * (prob1 + prob2)
            cpred = (prob >= self.threshold).long()
            self._update_confusion(self.change, cgt1, cpred, cv1, 2)

            if self.unchanged_local is not None:
                gated1 = pred1.clone()
                gated2 = pred2.clone()
                unchanged = ~cpred.bool()
                gated1[unchanged] = self.unchanged_local
                gated2[unchanged] = self.unchanged_local
                self._update_confusion(self.scd, gt1, gated1, valid1, self.k)
                self._update_confusion(self.scd, gt2, gated2, valid2, self.k)
            return

        for cgt, cv, prob in ((cgt1, cv1, prob1), (cgt2, cv2, prob2)):
            if cgt.numel() != prob.numel() or cv.numel() != prob.numel():
                raise ValueError("Binary change prediction/target size mismatch")
            cpred = (prob >= self.threshold).long()
            self._update_confusion(self.change, cgt, cpred, cv, 2)

        if self.unchanged_local is not None:
            gated1 = pred1.clone()
            gated2 = pred2.clone()
            gated1[~(prob1 >= self.threshold)] = self.unchanged_local
            gated2[~(prob2 >= self.threshold)] = self.unchanged_local
            self._update_confusion(self.scd, gt1, gated1, valid1, self.k)
            self._update_confusion(self.scd, gt2, gated2, valid2, self.k)

    # =========================================================================
    # 3D event metrics
    # =========================================================================

    @staticmethod
    def _validate_active_classes(active, name):
        active = tuple(int(x) for x in active)
        if not active:
            raise ValueError(f"{name} active event classes cannot be empty")
        if len(set(active)) != len(active):
            raise ValueError(f"{name} active event classes contain duplicates: {active}")
        bad = [x for x in active if x < 0 or x >= PAIR_EVENT_NUM_CLASSES]
        if bad:
            raise ValueError(f"{name} active event classes contain invalid IDs: {bad}")
        if 0 not in active:
            raise ValueError(f"{name} active event classes must include 0=unchanged")
        return active

    def _initialize_event_metrics(self, active_t1, active_t2):
        active_t1 = self._validate_active_classes(active_t1, "T1")
        active_t2 = self._validate_active_classes(active_t2, "T2")
        union = tuple(sorted(set(active_t1) | set(active_t2)))

        self.event_active_t1 = active_t1
        self.event_active_t2 = active_t2
        self.event_union = union
        self.event_union_to_local = {global_id: local_id for local_id, global_id in enumerate(union)}
        self.event_t1 = torch.zeros(len(active_t1), len(active_t1), dtype=torch.long, device=self.device)
        self.event_t2 = torch.zeros(len(active_t2), len(active_t2), dtype=torch.long, device=self.device)
        self.event = torch.zeros(len(union), len(union), dtype=torch.long, device=self.device)

    def _ensure_event_metrics(self):
        if self.event_active_t1 is not None:
            return
        key = self._normalize_dataset_name(self.dataset_name)
        if key not in _EVENT_ACTIVE_CLASSES:
            raise ValueError(
                f"3D event metrics have no declared active-class protocol for dataset "
                f"{self.dataset_name!r}. Add its true event supervision support to metrics.py; "
                "do not silently evaluate all six classes."
            )
        self._initialize_event_metrics(*_EVENT_ACTIVE_CLASSES[key])

    @staticmethod
    def _active_event_prediction(logits, active_classes):
        if logits is None or logits.ndim != 2 or logits.shape[1] != PAIR_EVENT_NUM_CLASSES:
            shape = None if logits is None else tuple(logits.shape)
            raise ValueError(f"event logits must be [N,{PAIR_EVENT_NUM_CLASSES}], got {shape}")

        active = torch.tensor(active_classes, dtype=torch.long, device=logits.device)
        local_pred = logits.detach().index_select(1, active).argmax(-1)
        global_pred = active[local_pred]
        return local_pred, global_pred

    @staticmethod
    def _active_event_target(target, valid, active_classes, device):
        target = target.to(device).reshape(-1).long()
        valid = valid.to(device).reshape(-1).bool()
        if target.numel() != valid.numel():
            raise ValueError("event target and valid mask sizes differ")

        bad_global = valid & ((target < 0) | (target >= PAIR_EVENT_NUM_CLASSES))
        if bad_global.any():
            values = torch.unique(target[bad_global]).detach().cpu().tolist()
            raise ValueError(f"event target contains invalid global IDs {values}")

        local = torch.full_like(target, -1)
        matched = torch.zeros_like(valid)
        for local_id, global_id in enumerate(active_classes):
            mask = valid & (target == int(global_id))
            local[mask] = local_id
            matched |= mask

        bad_valid = valid & ~matched
        if bad_valid.any():
            values = torch.unique(target[bad_valid]).detach().cpu().tolist()
            raise ValueError(
                f"event_valid=True contains targets {values} outside active classes {list(active_classes)}"
            )
        return target, local, valid

    def _global_event_to_union_local(self, values):
        local = torch.full_like(values, -1)
        for global_id, local_id in self.event_union_to_local.items():
            local[values == global_id] = local_id
        return local

    def _update_event_one_time(self, logits, target, valid, active_classes, cm):
        local_pred, global_pred = self._active_event_prediction(logits, active_classes)
        global_gt, local_gt, valid = self._active_event_target(
            target, valid, active_classes, self.device
        )
        local_pred = local_pred.to(self.device)
        global_pred = global_pred.to(self.device)

        if local_pred.numel() != local_gt.numel():
            raise ValueError("Event prediction/target size mismatch")

        self._update_confusion(cm, local_gt, local_pred, valid, len(active_classes))

        union_gt = self._global_event_to_union_local(global_gt)
        union_pred = self._global_event_to_union_local(global_pred)
        self._update_confusion(self.event, union_gt, union_pred, valid, len(self.event_union))

        # Traditional binary change metric derived directly from event IDs.
        binary_gt = (global_gt != 0).long()
        binary_pred = (global_pred != 0).long()
        self._update_confusion(self.change, binary_gt, binary_pred, valid, 2)

    def _update_event_3d(self, prediction, target):
        self._ensure_event_metrics()

        if prediction.event_logits_t1 is None or prediction.event_logits_t2 is None:
            raise ValueError("3D prediction must provide event_logits_t1 and event_logits_t2")
        if prediction.change_logits_t1 is not None or prediction.change_logits_t2 is not None:
            raise ValueError("3D prediction must not provide binary change logits")
        if "event_t1" not in target or "event_t2" not in target:
            raise KeyError("3D event metrics require target['event_t1'] and target['event_t2']")

        event1 = target["event_t1"]
        event2 = target["event_t2"]
        valid1 = self._valid_or_true(target, "event_valid_t1", event1, self.device)
        valid2 = self._valid_or_true(target, "event_valid_t2", event2, self.device)

        self._update_event_one_time(
            prediction.event_logits_t1,
            event1,
            valid1,
            self.event_active_t1,
            self.event_t1,
        )
        self._update_event_one_time(
            prediction.event_logits_t2,
            event2,
            valid2,
            self.event_active_t2,
            self.event_t2,
        )

    # =========================================================================
    # Public update/reduce/compute
    # =========================================================================

    def update(self, prediction, target):
        pred1, pred2, gt1, gt2, valid1, valid2 = self._update_semantic(prediction, target)

        has_binary = prediction.change_logits_t1 is not None or prediction.change_logits_t2 is not None
        has_event = prediction.event_logits_t1 is not None or prediction.event_logits_t2 is not None
        if has_binary == has_event:
            raise ValueError(
                "PAIR metrics expect exactly one prediction branch: "
                "binary change logits for 2D or event logits for 3D"
            )

        if has_binary:
            self._update_binary_2d(prediction, target, pred1, pred2, gt1, gt2, valid1, valid2)
        else:
            self._update_event_3d(prediction, target)

    def reduce_distributed(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        tensors = [self.semantic_t1, self.semantic_t2, self.scd, self.change]
        if self.event_t1 is not None:
            tensors.extend([self.event_t1, self.event_t2, self.event])
        for tensor in tensors:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    @staticmethod
    def _put_classification_scalars(result, prefix, metrics):
        result[f"{prefix}/OA"] = metrics["OA"]
        result[f"{prefix}/mIoU"] = metrics["mIoU"]
        result[f"{prefix}/mF1"] = metrics["mF1"]
        result[f"{prefix}/Kappa"] = metrics["Kappa"]

    def _compute_event(self, result):
        if self.event_t1 is None:
            return None, None

        t1 = _classification_metrics(self.event_t1)
        t2 = _classification_metrics(self.event_t2)
        combined = _classification_metrics(self.event)

        self._put_classification_scalars(result, "event_t1", t1)
        self._put_classification_scalars(result, "event_t2", t2)
        self._put_classification_scalars(result, "event", combined)

        event_per_class = {}
        for global_id, name in enumerate(PAIR_EVENT_NAMES):
            if global_id not in self.event_union_to_local:
                event_per_class[name] = {
                    "event_id": global_id,
                    "IoU": None,
                    "F1": None,
                    "Precision": None,
                    "Recall": None,
                    "Support": None,
                    "status": "N/A",
                }
                continue

            i = self.event_union_to_local[global_id]
            event_per_class[name] = {
                "event_id": global_id,
                "IoU": float(combined["iou"][i].item()),
                "F1": float(combined["f1"][i].item()),
                "Precision": float(combined["precision"][i].item()),
                "Recall": float(combined["recall"][i].item()),
                "Support": int(combined["support"][i].item()),
                "status": "active",
            }

        return event_per_class, {
            "event_t1": self.event_t1.detach().cpu(),
            "event_t2": self.event_t2.detach().cpu(),
            "event_combined": self.event.detach().cpu(),
            "event_active_classes_t1": self.event_active_t1,
            "event_active_classes_t2": self.event_active_t2,
            "event_combined_classes": self.event_union,
        }

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
        }
        self._put_classification_scalars(result, "semantic_t1", t1)
        self._put_classification_scalars(result, "semantic_t2", t2)
        self._put_classification_scalars(result, "semantic", combined)

        semantic_per_class = {}
        for i, (raw_id, name) in enumerate(zip(self.raw_ids, self.names)):
            semantic_per_class[name] = {
                "raw_id": raw_id,
                "IoU": float(combined["iou"][i].item()),
                "F1": float(combined["f1"][i].item()),
                "Precision": float(combined["precision"][i].item()),
                "Recall": float(combined["recall"][i].item()),
                "Support": int(combined["support"][i].item()),
            }

        if self.unchanged_local is not None and self.event_t1 is None:
            for key, value in _scd_metrics(self.scd, self.unchanged_local).items():
                result[f"scd/{key}"] = value

        event_per_class, event_confusion = self._compute_event(result)

        confusion = {
            "semantic_t1": self.semantic_t1.detach().cpu(),
            "semantic_t2": self.semantic_t2.detach().cpu(),
            "semantic_combined": combined_cm.detach().cpu(),
            "scd_gated": self.scd.detach().cpu(),
            "change": self.change.detach().cpu(),
        }
        if event_confusion is not None:
            confusion.update(event_confusion)

        output = {
            "scalars": result,
            "per_class": semantic_per_class,
            "confusion": confusion,
        }
        if event_per_class is not None:
            output["event_per_class"] = event_per_class
        return output


def normalized_confusion_image(cm):
    cm = cm.float()
    denom = cm.sum(1, keepdim=True).clamp_min(1)
    return (cm / denom).unsqueeze(0)
