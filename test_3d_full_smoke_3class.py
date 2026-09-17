#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAIR full 3D smoke test for the current 3-class event protocol.

Path under test:
    config_loader
    -> DatasetRegistry / NYC-SCD
    -> PAIRModel.from_config
    -> frozen Utonia
    -> PointAdapter
    -> Qwen reasoning
    -> batched KNN temporal links
    -> UnifiedChangeDecoder
    -> semantic logits + 3-class event logits
    -> loss
    -> backward
    -> PAIRMetrics

Event protocol:
    0 unchanged
    1 added
    2 removed

Temporal GT support:
    T1: unchanged / removed
    T2: unchanged / added

This is a smoke test, not a benchmark. By default each epoch is trimmed to
at most 4096 points after the real DatasetRegistry loader returns a sample.

Run from the PAIR repository root:
    python test_3d_full_smoke.py

Useful:
    python test_3d_full_smoke.py --max-points 8192
    python test_3d_full_smoke.py --max-points 0
    CUDA_VISIBLE_DEVICES=1 python test_3d_full_smoke.py --device cuda:0
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


EVENT_NUM_CLASSES = 3
T1_ALLOWED = {0, 2}
T2_ALLOWED = {0, 1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--config", type=Path, default=Path("configs/pair_train.json"))
    p.add_argument("--dataset", default="NYC-SCD")
    p.add_argument("--scan-batches", type=int, default=8)
    p.add_argument("--max-points", type=int, default=4096, help="Per epoch. 0 keeps all points.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def banner(text):
    print("\n" + "=" * 96)
    print(text)
    print("=" * 96)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preflight_sources(root: Path):
    """Catch partial/stale 6-class copies before allocating the model."""
    checks = {
        "models/change_decoder.py": {
            "must": [
                "event_head",
                'prediction_type == "binary"',
            ],
            "must_not": [
                "nn.Linear(self.decoder_dim, 6)",
            ],
        },
        "models/pair.py": {
            "must": [
                'prediction_type="event"',
                "event_logits_t1",
                "event_logits_t2",
            ],
            "must_not": [
                "(dense_t1.features.shape[0], 6)",
                "(dense_t2.features.shape[0], 6)",
            ],
        },
        "datasets/pair_dataset.py": {
            "must": [
                '"unchanged"',
                '"added"',
                '"removed"',
            ],
            "must_not": [
                '3: "class_change"',
                '4: "height_up"',
                '5: "height_down"',
                "PAIR_EVENT_NUM_CLASSES == 6",
            ],
        },
        "loss.py": {
            "must": [
                "PAIR_EVENT_NUM_CLASSES = 3",
                "event_logits_t1",
                "event_logits_t2",
            ],
            "must_not": [
                "PAIR_EVENT_NUM_CLASSES = 6",
            ],
        },
        "metrics.py": {
            "must": [
                "PAIR_EVENT_NAMES",
                '"unchanged"',
                '"added"',
                '"removed"',
                "event_per_class",
            ],
            "must_not": [
                '"class_change"',
                '"height_up"',
                '"height_down"',
            ],
        },
        "train.py": {
            "must": [
                '"event_t1"',
                '"event_t2"',
                '"event_valid_t1"',
                '"event_valid_t2"',
                "PAIRModel.from_config",
            ],
            "must_not": [
                "dataset_name=spec.name",
            ],
        },
    }

    problems = []
    for rel, rules in checks.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing file: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in rules["must"]:
            if token not in text:
                problems.append(f"{rel}: missing required token: {token!r}")
        for token in rules["must_not"]:
            if token in text:
                problems.append(f"{rel}: stale token still present: {token!r}")

    if problems:
        print("Local source preflight FAILED:")
        for item in problems:
            print("  -", item)
        raise RuntimeError("Local PAIR source is not fully on the 3-class event protocol.")

    print("[PASS] local source protocol preflight")


def distribution(x, valid=None):
    x = x.reshape(-1).long()
    if valid is not None:
        x = x[valid.reshape(-1).bool()]
    if x.numel() == 0:
        return {}
    values, counts = torch.unique(x, return_counts=True)
    return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}


def choose_indices(target, time_id, n, max_points, seed):
    if max_points <= 0 or n <= max_points:
        return torch.arange(n, dtype=torch.long)

    selected = []
    seen = set()

    def add_first(mask):
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel():
            value = int(idx[0].item())
            if value not in seen:
                seen.add(value)
                selected.append(value)

    semantic = target[f"semantic_t{time_id}"].reshape(-1)
    event = target[f"event_t{time_id}"].reshape(-1)
    valid = target[f"event_valid_t{time_id}"].reshape(-1).bool()

    for value in torch.unique(semantic).tolist():
        add_first(semantic == int(value))
    if valid.any():
        for value in torch.unique(event[valid]).tolist():
            add_first(valid & (event == int(value)))

    generator = torch.Generator().manual_seed(seed + time_id)
    for value in torch.randperm(n, generator=generator).tolist():
        if len(selected) >= max_points:
            break
        if value not in seen:
            seen.add(value)
            selected.append(value)

    return torch.tensor(selected[:max_points], dtype=torch.long)


def trim_sample(sample, max_points, seed):
    if max_points <= 0:
        return sample

    out = dict(sample)
    out["target"] = dict(sample["target"])

    for time_id in (1, 2):
        point_key = f"point_dict_t{time_id}"
        point = dict(sample[point_key])
        n = int(point["coord"].shape[0])
        idx = choose_indices(out["target"], time_id, n, max_points, seed)

        for key, value in list(point.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
                point[key] = value[idx]
        out[point_key] = point

        for name in ("semantic", "semantic_valid", "event", "event_valid"):
            key = f"{name}_t{time_id}"
            value = out["target"].get(key)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
                out["target"][key] = value[idx]

    return out


def validate_sample(sample):
    check("point_dict_t1" in sample and "point_dict_t2" in sample, "3D point dictionaries are missing")
    check("target" in sample, "sample target is missing")

    target = sample["target"]
    required = (
        "semantic_t1",
        "semantic_t2",
        "event_t1",
        "event_t2",
        "event_valid_t1",
        "event_valid_t2",
    )
    for key in required:
        check(key in target, f"required target key {key!r} is missing")

    for time_id, allowed in ((1, T1_ALLOWED), (2, T2_ALLOWED)):
        point = sample[f"point_dict_t{time_id}"]
        coord = point["coord"]
        n = int(coord.shape[0])

        check(coord.ndim == 2 and coord.shape[1] == 3, f"T{time_id} coord must be [N,3]")
        check(target[f"semantic_t{time_id}"].numel() == n, f"T{time_id} semantic length mismatch")
        check(target[f"event_t{time_id}"].numel() == n, f"T{time_id} event length mismatch")
        check(target[f"event_valid_t{time_id}"].numel() == n, f"T{time_id} event_valid length mismatch")

        valid = target[f"event_valid_t{time_id}"].reshape(-1).bool()
        check(valid.any(), f"T{time_id} has no valid event supervision")

        event = target[f"event_t{time_id}"].reshape(-1).long()
        valid_ids = set(torch.unique(event[valid]).tolist())
        bad = valid_ids - allowed
        check(not bad, f"T{time_id} valid event IDs violate protocol: {sorted(bad)}")

        if "rgb" in point:
            check(point["rgb"].shape[0] == n, f"T{time_id} rgb length mismatch")
        if "intensity" in point:
            check(point["intensity"].shape[0] == n, f"T{time_id} intensity length mismatch")


def print_sample(sample):
    print("sample_id:", sample.get("sample_id", "<unknown>"))
    for time_id in (1, 2):
        point = sample[f"point_dict_t{time_id}"]
        target = sample["target"]
        n = int(point["coord"].shape[0])
        optional = [key for key in ("rgb", "intensity") if key in point]
        valid = target[f"event_valid_t{time_id}"]
        print(f"T{time_id}: N={n:,} point keys={sorted(point)} optional={optional or 'none'}")
        print(f"    semantic: {distribution(target[f'semantic_t{time_id}'])}")
        print(f"    event all: {distribution(target[f'event_t{time_id}'])}")
        print(
            f"    event valid: {distribution(target[f'event_t{time_id}'], valid)} "
            f"({int(valid.bool().sum())}/{n})"
        )


def next_usable_sample(registry, dataset_name, scan_batches, max_points, seed):
    errors = []
    for batch_idx in range(max(int(scan_batches), 1)):
        try:
            samples = registry.next_train_batch(dataset_name)
            if not samples:
                raise RuntimeError("DatasetRegistry returned an empty batch")

            for sample_idx, sample in enumerate(samples):
                try:
                    sample = trim_sample(sample, max_points, seed + batch_idx * 100 + sample_idx)
                    validate_sample(sample)
                    return sample
                except Exception as exc:
                    errors.append(
                        f"batch {batch_idx} sample {sample_idx}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        except Exception as exc:
            errors.append(f"batch {batch_idx}: {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not obtain a usable NYC-SCD sample:\n  " + "\n  ".join(errors))


def grad_norm(parameters):
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        total += float(parameter.grad.detach().float().pow(2).sum().item())
    return total ** 0.5 if found else 0.0


def main():
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    banner("PAIR 3D FULL SMOKE TEST — 3-CLASS EVENT")
    print("project:", root)
    print("config:", args.config)
    print("dataset:", args.dataset)
    print("device:", args.device)
    print("max_points/epoch:", "ALL" if args.max_points <= 0 else f"{args.max_points:,}")

    preflight_sources(root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full PAIR 3D smoke test")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_seed(args.seed)

    from datasets.config_loader import load_experiment_config
    from datasets.multi_dataset import DatasetRegistry
    from loss import PAIRSemanticChangeLoss
    from metrics import PAIRMetrics
    from models.pair import PAIRModel
    from train import forward_loss

    check(
        "dataset_name" not in inspect.signature(PAIRMetrics.__init__).parameters,
        "metrics.py still exposes stale dataset_name routing",
    )

    banner("1. CONFIG + DATASET")
    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment = load_experiment_config(config_path, [args.dataset])

    runtime = {
        "distributed": False,
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "device": device,
        "is_main": True,
    }

    registry = DatasetRegistry(experiment, runtime, num_workers=0)
    check(args.dataset in registry.handles, f"DatasetRegistry has no dataset {args.dataset!r}")
    handle = registry.handles[args.dataset]
    spec = handle.config.spec

    print("route:", spec.route)
    print("label_mode:", spec.label_mode)
    print("class_names:", spec.class_names)
    print("ignored_id:", spec.ignored_id)
    print("root:", handle.config.root)

    check(spec.route == "3d", f"{args.dataset} route must be 3d, got {spec.route}")
    check(spec.label_mode == "semantic_pair", f"{args.dataset} label_mode must be semantic_pair")

    if hasattr(registry, "reset_epoch"):
        registry.reset_epoch(0)

    sample = next_usable_sample(
        registry,
        args.dataset,
        args.scan_batches,
        args.max_points,
        args.seed,
    )
    print_sample(sample)
    print("[PASS] DatasetRegistry / target protocol")

    banner("2. MODEL BUILD")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    build_start = time.time()
    model = PAIRModel.from_config(experiment.model, device)
    model.train()
    criterion = PAIRSemanticChangeLoss().to(device)

    print(f"model build: {time.time() - build_start:.2f}s")
    print("qwen_tuning:", model.qwen_tuning)
    print("Utonia output_dim:", model.point_encoder.output_dim)
    print("decoder_dim:", model.decoder.decoder_dim)
    print("event classes:", model.decoder.event_head.out_features)
    print("reasoning token budget:", model.point_adapter.num_tokens)

    check(model.decoder.event_head.out_features == EVENT_NUM_CLASSES, "event_head must have 3 outputs")
    frozen_bad = [name for name, p in model.point_encoder.named_parameters() if p.requires_grad]
    check(not frozen_bad, f"Utonia is not fully frozen: {frozen_bad[:5]}")
    print("[PASS] model construction / 3-class event head / Utonia frozen")

    reasoning_projection_outputs = []

    def capture_token_proj(module, inputs, output):
        if torch.is_tensor(output) and output.requires_grad:
            output.retain_grad()
            reasoning_projection_outputs.append(output)

    hook = model.point_adapter.token_proj.register_forward_hook(capture_token_proj)

    banner("3. REAL FORWARD THROUGH train.forward_loss")
    model.zero_grad(set_to_none=True)
    forward_start = time.time()

    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, loss_output, merged_target = forward_loss(
                model,
                criterion,
                [sample],
                spec,
            )
    finally:
        hook.remove()

    n1 = int(sample["point_dict_t1"]["coord"].shape[0])
    n2 = int(sample["point_dict_t2"]["coord"].shape[0])
    k = len(spec.class_names)

    print(f"forward: {time.time() - forward_start:.2f}s")
    print("semantic_logits_t1:", tuple(prediction.semantic_logits_t1.shape))
    print("semantic_logits_t2:", tuple(prediction.semantic_logits_t2.shape))
    print("event_logits_t1:", tuple(prediction.event_logits_t1.shape))
    print("event_logits_t2:", tuple(prediction.event_logits_t2.shape))
    print("change_logits_t1:", prediction.change_logits_t1)
    print("change_logits_t2:", prediction.change_logits_t2)

    check(tuple(prediction.semantic_logits_t1.shape) == (n1, k), "bad semantic_logits_t1 shape")
    check(tuple(prediction.semantic_logits_t2.shape) == (n2, k), "bad semantic_logits_t2 shape")
    check(tuple(prediction.event_logits_t1.shape) == (n1, 3), "bad event_logits_t1 shape")
    check(tuple(prediction.event_logits_t2.shape) == (n2, 3), "bad event_logits_t2 shape")
    check(
        prediction.change_logits_t1 is None and prediction.change_logits_t2 is None,
        "3D path must not return binary change logits",
    )

    for name in ("semantic_logits_t1", "semantic_logits_t2", "event_logits_t1", "event_logits_t2"):
        check(torch.isfinite(getattr(prediction, name)).all(), f"{name} contains NaN/Inf")

    valid1 = merged_target["event_valid_t1"].bool()
    valid2 = merged_target["event_valid_t2"].bool()
    ids1 = set(torch.unique(merged_target["event_t1"][valid1]).tolist())
    ids2 = set(torch.unique(merged_target["event_t2"][valid2]).tolist())
    check(ids1 <= T1_ALLOWED, f"merged T1 event protocol violation: {sorted(ids1)}")
    check(ids2 <= T2_ALLOWED, f"merged T2 event protocol violation: {sorted(ids2)}")

    print("[PASS] Dataset -> Utonia -> PointAdapter -> Qwen -> KNN -> Decoder")

    banner("4. LOSS")
    values = loss_output.as_dict()
    for key, value in values.items():
        print(f"{key}: {float(value.detach().cpu()):.6f}")
        check(torch.isfinite(value).all(), f"{key} is NaN/Inf")

    for key in ("loss_event_t1", "loss_event_t2", "loss_event"):
        check(key in values, f"{key} missing from loss output")

    check(float(values["loss_event"].detach().cpu()) >= 0.0, "event loss is negative")
    print("[PASS] semantic + 3-class event loss")

    banner("5. BACKWARD")
    backward_start = time.time()
    loss_output.total.backward()
    torch.cuda.synchronize(device)
    print(f"backward: {time.time() - backward_start:.2f}s")

    event_grad = model.decoder.event_head.weight.grad
    check(event_grad is not None, "event_head received no gradient")
    check(tuple(event_grad.shape)[0] == 3, f"event_head gradient has wrong first dim: {tuple(event_grad.shape)}")
    check(torch.isfinite(event_grad).all(), "event_head gradient contains NaN/Inf")
    check(float(event_grad.abs().max()) > 0, "event_head gradient is all zero")
    print(f"event_head grad max: {float(event_grad.abs().max()):.6e}")

    reasoning_grads = [
        float(x.grad.detach().float().norm().item())
        for x in reasoning_projection_outputs
        if x.grad is not None
    ]
    print("PointAdapter token_proj output grad norms:", reasoning_grads)
    check(
        reasoning_grads and max(reasoning_grads) > 0,
        "No gradient returned through Qwen into PointAdapter reasoning tokens",
    )

    point_adapter_grad = grad_norm(model.point_adapter.parameters())
    decoder_grad = grad_norm(model.decoder.parameters())
    print(f"PointAdapter parameter grad norm: {point_adapter_grad:.6e}")
    print(f"Decoder parameter grad norm: {decoder_grad:.6e}")
    check(point_adapter_grad > 0, "PointAdapter parameter gradient is zero")
    check(decoder_grad > 0, "Decoder parameter gradient is zero")

    utonia_grads = [
        name for name, p in model.point_encoder.named_parameters()
        if p.grad is not None
    ]
    check(not utonia_grads, f"Frozen Utonia unexpectedly has gradients: {utonia_grads[:5]}")

    if model.qwen_tuning == "lora":
        lora_params = [p for name, p in model.named_parameters() if "lora_" in name]
        lora_grad = grad_norm(lora_params)
        print(f"Qwen LoRA grad norm: {lora_grad:.6e}")
        check(lora_params, "qwen_tuning=lora but no LoRA parameters were found")
        check(lora_grad > 0, "Qwen LoRA parameters received no gradient")

    print("[PASS] loss.backward")
    print("[PASS] gradient crosses Qwen reasoning path into PointAdapter")
    print("[PASS] Utonia remains frozen")

    banner("6. METRICS")
    evaluator = PAIRMetrics(
        spec.class_names,
        device,
        change_threshold=float(experiment.validation.get("change_threshold", 0.5)),
    )
    evaluator.update(prediction, merged_target)
    result = evaluator.compute()

    required_scalars = (
        "semantic/OA",
        "semantic/mIoU",
        "event/OA",
        "event/mIoU",
        "event/mF1",
        "change/OA",
        "change/F1",
        "change/IoU",
    )
    for key in required_scalars:
        check(key in result["scalars"], f"metric {key!r} is missing")
        value = result["scalars"][key]
        check(np.isfinite(value), f"metric {key!r} is NaN/Inf")
        print(f"{key}: {value:.6f}")

    event_per_class = result.get("event_per_class", {})
    check(
        set(event_per_class) == {"unchanged", "added", "removed"},
        f"unexpected event_per_class keys: {sorted(event_per_class)}",
    )

    confusion = result["confusion"]
    check(tuple(confusion["event_t1"].shape) == (3, 3), "event_t1 confusion must be 3x3")
    check(tuple(confusion["event_t2"].shape) == (3, 3), "event_t2 confusion must be 3x3")
    check(tuple(confusion["event_combined"].shape) == (3, 3), "event_combined confusion must be 3x3")

    print("[PASS] semantic / 3-class event / derived-binary metrics")

    banner("RESULT")
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    print("FULL 3D SMOKE TEST: PASS")
    print(f"peak allocated: {peak:.2f} GiB")
    print(f"peak reserved:  {reserved:.2f} GiB")
    print("Tested: real loader -> model -> loss -> backward -> metrics.")


if __name__ == "__main__":
    main()
