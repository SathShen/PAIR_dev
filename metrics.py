"""
Unified streaming metrics for PAIR semantic change detection.
"""

from __future__ import annotations

from typing import Dict, Optional
import torch

from datasets.pair_dataset import infer_unchanged_raw_id


EPS = 1e-12


def _safe_div(a, b):
    return a / b.clamp_min(EPS)


def _kappa(cm):
    cm = cm.double()
    total = cm.sum()
    if total <= 0:
        return torch.tensor(0.0, device=cm.device)

    po = torch.diag(cm).sum() / total
    pe = (
        cm.sum(1) * cm.sum(0)
    ).sum() / (total * total)

    return (
        (po - pe)
        / (1.0 - pe).clamp_min(EPS)
    )


def _classification_metrics(cm):
    cm = cm.double()
    tp = torch.diag(cm)
    true_count = cm.sum(1)
    pred_count = cm.sum(0)
    union = true_count + pred_count - tp

    precision = _safe_div(tp, pred_count)
    recall = _safe_div(tp, true_count)
    f1 = _safe_div(
        2 * precision * recall,
        precision + recall,
    )
    iou = _safe_div(tp, union)

    valid_iou = union > 0
    valid_f1 = true_count > 0
    total = cm.sum()

    return {
        "OA": float(
            (tp.sum() / total.clamp_min(1)).item()
        ),
        "mIoU": (
            float(iou[valid_iou].mean().item())
            if valid_iou.any()
            else 0.0
        ),
        "mF1": (
            float(f1[valid_f1].mean().item())
            if valid_f1.any()
            else 0.0
        ),
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
    f1 = (
        2 * precision * recall
        / (precision + recall).clamp_min(EPS)
    )
    iou = tp / (
        tp + fp + fn
    ).clamp_min(EPS)
    oa = (
        (tp + tn)
        / cm.sum().clamp_min(1)
    )

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
    order = (
        [unchanged_index]
        + [
            i for i in range(cm.shape[0])
            if i != unchanged_index
        ]
    )

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

    iou_nc = q00 / (
        q[0, :].sum()
        + q[:, 0].sum()
        - q00
    ).clamp_min(EPS)

    iou_c = q[1:, 1:].sum() / (
        total - q00
    ).clamp_min(EPS)

    miou = 0.5 * (
        iou_nc + iou_c
    )

    correct_changed = torch.diag(q)[1:].sum()
    p_scd = correct_changed / (
        q[:, 1:].sum()
    ).clamp_min(EPS)
    r_scd = correct_changed / (
        q[1:, :].sum()
    ).clamp_min(EPS)

    f_scd = (
        2 * p_scd * r_scd
        / (p_scd + r_scd).clamp_min(EPS)
    )

    qhat = q.clone()
    qhat[0, 0] = 0
    kappa_sep = _kappa(qhat)
    sek = kappa_sep * torch.exp(
        iou_c - 1.0
    )
    score = (
        0.3 * miou
        + 0.7 * sek
    )
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
        if not isinstance(
            class_names, dict
        ) or not class_names:
            raise TypeError(
                "class_names must be a non-empty Dict[int, str]"
            )

        self.class_names = {
            int(k): str(v)
            for k, v in class_names.items()
        }

        self.raw_ids = tuple(
            sorted(self.class_names)
        )
        self.names = tuple(
            self.class_names[k]
            for k in self.raw_ids
        )
        self.raw_to_local = {
            raw: i
            for i, raw in enumerate(self.raw_ids)
        }

        self.k = len(self.raw_ids)
        self.device = torch.device(device)
        self.threshold = float(
            change_threshold
        )

        if unchanged_raw_id is None:
            unchanged_raw_id = (
                infer_unchanged_raw_id(
                    self.class_names
                )
            )

        self.unchanged_raw_id = (
            unchanged_raw_id
        )
        self.unchanged_local = (
            None
            if unchanged_raw_id is None
            else self.raw_to_local.get(
                int(unchanged_raw_id)
            )
        )

        self.semantic_t1 = torch.zeros(
            self.k,
            self.k,
            dtype=torch.long,
            device=self.device,
        )
        self.semantic_t2 = torch.zeros_like(
            self.semantic_t1
        )
        self.scd = torch.zeros_like(
            self.semantic_t1
        )
        self.change = torch.zeros(
            2,
            2,
            dtype=torch.long,
            device=self.device,
        )

    def _raw_to_local_target(
        self,
        raw,
        valid,
    ):
        raw = raw.to(
            self.device
        ).reshape(-1).long()

        valid = valid.to(
            self.device
        ).reshape(-1).bool()

        local = torch.full_like(
            raw,
            -1,
        )

        matched = torch.zeros_like(
            valid
        )

        for (
            raw_id,
            local_id,
        ) in self.raw_to_local.items():
            mask = (
                valid
                & (raw == raw_id)
            )
            local[mask] = local_id
            matched |= mask

        bad = valid & ~matched
        if bad.any():
            values = torch.unique(
                raw[bad]
            ).detach().cpu().tolist()
            raise ValueError(
                "Metric target contains undeclared "
                f"raw class IDs: {values}"
            )

        return local, valid

    @staticmethod
    def _update_confusion(
        cm,
        target,
        pred,
        valid,
        k,
    ):
        target = target[valid]
        pred = pred[valid]

        if target.numel() == 0:
            return

        bins = torch.bincount(
            target * k + pred,
            minlength=k * k,
        )

        cm += bins.reshape(k, k)

    def _change_target(
        self,
        target,
        time_id,
    ):
        key = f"change_t{time_id}"
        valid_key = (
            f"change_valid_t{time_id}"
        )

        if key in target:
            return (
                target[key],
                target[valid_key],
            )

        return (
            target["change"],
            target["change_valid"],
        )

    def update(self, prediction, target):
        if tuple(
            prediction.raw_class_ids
        ) != self.raw_ids:
            raise ValueError(
                "Decoder class order "
                f"{prediction.raw_class_ids} != "
                f"metric class order {self.raw_ids}"
            )

        pred1 = (
            prediction.semantic_logits_t1
            .detach()
            .argmax(-1)
            .to(self.device)
        )
        pred2 = (
            prediction.semantic_logits_t2
            .detach()
            .argmax(-1)
            .to(self.device)
        )

        gt1, valid1 = (
            self._raw_to_local_target(
                target["semantic_t1"],
                target["semantic_valid_t1"],
            )
        )
        gt2, valid2 = (
            self._raw_to_local_target(
                target["semantic_t2"],
                target["semantic_valid_t2"],
            )
        )

        if (
            pred1.numel() != gt1.numel()
            or pred2.numel() != gt2.numel()
        ):
            raise ValueError(
                "Semantic prediction/target size mismatch"
            )

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

        cgt1, cv1 = self._change_target(
            target,
            1,
        )
        cgt2, cv2 = self._change_target(
            target,
            2,
        )

        cgt1 = cgt1.to(
            self.device
        ).reshape(-1).long()
        cgt2 = cgt2.to(
            self.device
        ).reshape(-1).long()

        cv1 = cv1.to(
            self.device
        ).reshape(-1).bool()
        cv2 = cv2.to(
            self.device
        ).reshape(-1).bool()

        prob1 = torch.sigmoid(
            prediction.change_logits_t1
            .detach()
            .float()
        ).to(
            self.device
        ).reshape(-1)

        prob2 = torch.sigmoid(
            prediction.change_logits_t2
            .detach()
            .float()
        ).to(
            self.device
        ).reshape(-1)

        shared_change = (
            "change_t1" not in target
            and "change_t2" not in target
            and prob1.numel()
            == prob2.numel()
            == cgt1.numel()
        )

        if shared_change:
            prob = 0.5 * (
                prob1 + prob2
            )
            cpred = (
                prob >= self.threshold
            ).long()

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

                gated1[unchanged] = (
                    self.unchanged_local
                )
                gated2[unchanged] = (
                    self.unchanged_local
                )

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
            for cgt, cv, prob in (
                (cgt1, cv1, prob1),
                (cgt2, cv2, prob2),
            ):
                cpred = (
                    prob >= self.threshold
                ).long()

                self._update_confusion(
                    self.change,
                    cgt,
                    cpred,
                    cv,
                    2,
                )

            if self.unchanged_local is not None:
                gated1 = pred1.clone()
                gated2 = pred2.clone()

                gated1[
                    ~(prob1 >= self.threshold)
                ] = self.unchanged_local

                gated2[
                    ~(prob2 >= self.threshold)
                ] = self.unchanged_local

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
            ):
                torch.distributed.all_reduce(
                    tensor,
                    op=torch.distributed.ReduceOp.SUM,
                )

    def compute(self):
        t1 = _classification_metrics(
            self.semantic_t1
        )
        t2 = _classification_metrics(
            self.semantic_t2
        )

        combined_cm = (
            self.semantic_t1
            + self.semantic_t2
        )
        combined = _classification_metrics(
            combined_cm
        )
        change = _binary_metrics(
            self.change
        )

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
        for i, (
            raw_id,
            name,
        ) in enumerate(
            zip(
                self.raw_ids,
                self.names,
            )
        ):
            per_class[name] = {
                "raw_id": raw_id,
                "IoU": float(
                    combined["iou"][i].item()
                ),
                "F1": float(
                    combined["f1"][i].item()
                ),
                "Precision": float(
                    combined["precision"][i].item()
                ),
                "Recall": float(
                    combined["recall"][i].item()
                ),
                "Support": int(
                    combined["support"][i].item()
                ),
            }

        if self.unchanged_local is not None:
            for key, value in _scd_metrics(
                self.scd,
                self.unchanged_local,
            ).items():
                result[
                    f"scd/{key}"
                ] = value

        return {
            "scalars": result,
            "per_class": per_class,
            "confusion": {
                "semantic_t1": (
                    self.semantic_t1
                    .detach()
                    .cpu()
                ),
                "semantic_t2": (
                    self.semantic_t2
                    .detach()
                    .cpu()
                ),
                "semantic_combined": (
                    combined_cm
                    .detach()
                    .cpu()
                ),
                "scd_gated": (
                    self.scd
                    .detach()
                    .cpu()
                ),
                "change": (
                    self.change
                    .detach()
                    .cpu()
                ),
            },
        }


def normalized_confusion_image(cm):
    cm = cm.float()
    denom = cm.sum(
        1,
        keepdim=True,
    ).clamp_min(1)
    return (
        cm / denom
    ).unsqueeze(0)
