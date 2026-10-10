#!/usr/bin/env python3
"""PAIR combined memory + Box-generation diagnostic.

Purpose
-------
One script, one PAIR process, both SECOND and NYC-SCD.
It answers two separate questions:

1) Why training reaches very high CUDA memory?
   - exact model parameter/buffer bytes
   - raw batch tensor bytes
   - CUDA input tensor footprint observed at Qwen/Utonia boundaries
   - unique CUDA storages saved by autograd for backward (excluding model params/buffers)
   - forward live/peak allocation
   - backward live/peak allocation
   - exact gradient bytes
   - Train -> Val transition WITHOUT empty_cache() in between

2) Why generated Box count is zero?
   - prints the exact raw Qwen text returned by generate_box_proposals()
   - prints parser failure count and parsed valid-box count
   - runs a synthetic parser sanity check before the real model test

Notes
-----
- No optimizer.step() is performed; model weights are not updated.
- Training memory uses the real configured train batch sizes and box_mode='joint'.
- Validation uses batch=1 and box_mode='generated'.
- BF16 autocast follows the real train/val path.
- Rows/peaks must NOT be added together.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

DATASETS = ("SECOND", "NYC-SCD")
GIB = 1024 ** 3
MIB = 1024 ** 2


def gib(x: int | float) -> float:
    return float(x) / GIB


def mib(x: int | float) -> float:
    return float(x) / MIB


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_snapshot(device: torch.device) -> Dict[str, int]:
    cuda_sync()
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def storage_key(t: torch.Tensor):
    """Key one physical tensor storage; robust to views/sharing."""
    if not torch.is_tensor(t):
        return None
    try:
        s = t.untyped_storage()
        return (t.device.type, t.device.index, int(s.data_ptr()), int(s.nbytes()))
    except Exception:
        # Fallback: enough for reporting, although views cannot be deduplicated.
        return (t.device.type, t.device.index, int(t.data_ptr()), int(t.numel() * t.element_size()))


def unique_storage_bytes(obj: Any, *, device_type: str | None = None) -> int:
    seen = set()

    def walk(x: Any):
        if torch.is_tensor(x):
            if device_type is not None and x.device.type != device_type:
                return
            k = storage_key(x)
            if k is not None:
                seen.add(k)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
            return
        # Do NOT recursively inspect arbitrary nn.Module / cache objects here;
        # this function is for explicit data structures crossing our boundaries.

    walk(obj)
    return sum(k[-1] for k in seen)


def model_storage_keys(model: torch.nn.Module):
    keys = set()
    for p in model.parameters():
        if p.is_cuda:
            k = storage_key(p)
            if k is not None:
                keys.add(k)
    for b in model.buffers():
        if b.is_cuda:
            k = storage_key(b)
            if k is not None:
                keys.add(k)
    return keys


def parameter_report(model: torch.nn.Module, train_module: Any) -> Dict[str, Any]:
    """Exact tensor bytes, grouped with train.py's architecture bucket when available."""
    rows: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "params": 0, "trainable": 0, "frozen": 0, "buffers": 0,
    })
    dtype_bytes = defaultdict(int)
    seen_param_ids = set()

    bucket_fn = getattr(train_module, "_architecture_module_bucket", None)

    for name, p in model.named_parameters():
        if id(p) in seen_param_ids:
            continue
        seen_param_ids.add(id(p))
        nbytes = int(p.numel() * p.element_size())
        bucket = bucket_fn(name) if callable(bucket_fn) else name.split(".", 1)[0]
        rows[bucket]["params"] += nbytes
        rows[bucket]["trainable" if p.requires_grad else "frozen"] += nbytes
        dtype_bytes[str(p.dtype)] += nbytes

    seen_buffer_ids = set()
    for name, b in model.named_buffers():
        if id(b) in seen_buffer_ids:
            continue
        seen_buffer_ids.add(id(b))
        nbytes = int(b.numel() * b.element_size())
        bucket = bucket_fn(name) if callable(bucket_fn) else name.split(".", 1)[0]
        rows[bucket]["buffers"] += nbytes
        dtype_bytes[str(b.dtype)] += nbytes

    total_params = sum(r["params"] for r in rows.values())
    total_buffers = sum(r["buffers"] for r in rows.values())

    print("\n" + "=" * 118)
    print("A. STATIC MODEL MEMORY (exact tensor bytes, not allocator cache)")
    print("=" * 118)
    print(f"{'Module':<28} {'Params':>12} {'Trainable':>12} {'Frozen':>12} {'Buffers':>12}")
    print("-" * 82)
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["params"] + kv[1]["buffers"], reverse=True):
        print(f"{name:<28} {gib(r['params']):10.3f}GiB {gib(r['trainable']):10.3f}GiB "
              f"{gib(r['frozen']):10.3f}GiB {gib(r['buffers']):10.3f}GiB")
    print("-" * 82)
    print(f"TOTAL parameter tensors: {gib(total_params):.3f} GiB")
    print(f"TOTAL buffers:           {gib(total_buffers):.3f} GiB")
    print(f"TOTAL static tensors:    {gib(total_params + total_buffers):.3f} GiB")
    print("Dtype breakdown:")
    for dt, n in sorted(dtype_bytes.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {dt:<18} {gib(n):.3f} GiB")

    return {
        "modules": rows,
        "dtype_bytes": dict(dtype_bytes),
        "parameter_bytes": total_params,
        "buffer_bytes": total_buffers,
        "static_tensor_bytes": total_params + total_buffers,
    }


class BoundaryInputTracker:
    """Observe explicit CUDA tensor structures entering expensive modules."""
    def __init__(self):
        self.phase = "idle"
        self.max_bytes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.handles = []

    def _record(self, name: str, obj: Any):
        b = unique_storage_bytes(obj, device_type="cuda")
        if b > self.max_bytes[self.phase][name]:
            self.max_bytes[self.phase][name] = b

    def install(self, model):
        # Actual Qwen model kwargs after PAIR has moved processor tensors to CUDA.
        def qwen_pre(_module, args, kwargs):
            self._record("Qwen model CUDA inputs", (args, kwargs))
        try:
            h = model.qwen_backbone.model.register_forward_pre_hook(qwen_pre, with_kwargs=True)
            self.handles.append(h)
        except TypeError:
            # Older torch fallback; kwargs unavailable.
            def qwen_pre_old(_module, args):
                self._record("Qwen model CUDA inputs", args)
            self.handles.append(model.qwen_backbone.model.register_forward_pre_hook(qwen_pre_old))

        # Raw point data already moved to CUDA before Utonia.
        if model.point_encoder is not None:
            def point_pre(_module, args, kwargs):
                self._record("Utonia raw CUDA point batch", (args, kwargs))
            try:
                self.handles.append(model.point_encoder.register_forward_pre_hook(point_pre, with_kwargs=True))
            except TypeError:
                self.handles.append(model.point_encoder.register_forward_pre_hook(
                    lambda _m, args: self._record("Utonia raw CUDA point batch", args)
                ))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def print_phase(self, phase: str):
        rows = self.max_bytes.get(phase, {})
        if not rows:
            print("  observed CUDA input tensors: none captured")
            return
        print("  observed CUDA input tensor footprint at module boundaries:")
        for name, b in sorted(rows.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {name:<34} {mib(b):9.1f} MiB")


class BoxCapture:
    """Capture exact Qwen generated text before PAIR discards box_texts."""
    def __init__(self, model):
        self.model = model
        self.original = model.qwen_backbone.generate_box_proposals
        self.phase = "idle"
        self.records = []

    def install(self):
        original = self.original
        capture = self

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            texts = list(out.get("box_texts", []))
            valid = out.get("box_valid")
            valid_count = int(valid.sum().item()) if torch.is_tensor(valid) else -1
            rec = {
                "phase": capture.phase,
                "kind": str(kwargs.get("kind", "?")),
                "valid_boxes": valid_count,
                "parse_failures": int(out.get("box_parse_failures", -1)),
                "texts": texts,
            }
            capture.records.append(rec)
            print(f"\n  [BOX RAW] phase={rec['phase']} kind={rec['kind']} "
                  f"valid={rec['valid_boxes']} parse_failures={rec['parse_failures']}")
            for i, text in enumerate(texts):
                # repr makes empty strings / whitespace obvious.
                shown = text if len(text) <= 3000 else text[:3000] + " ...<truncated>"
                print(f"    sample[{i}] = {shown!r}")
            return out

        self.model.qwen_backbone.generate_box_proposals = wrapped

    def remove(self):
        self.model.qwen_backbone.generate_box_proposals = self.original


def parser_sanity(model) -> Dict[str, Any]:
    parse = model.qwen_backbone._parse_box_json
    tests = {}
    two = '{"boxes":[{"box":[100,200,700,800],"score":0.5}]}'
    three = '{"boxes":[{"box":[100,200,300,700,800,900],"score":0.5}]}'
    tests["2d"] = parse(two, kind="2d", max_boxes=16)
    tests["3d"] = parse(three, kind="3d", max_boxes=16,
                        scene_bounds=[[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    print("\n" + "=" * 118)
    print("B. BOX PARSER SANITY (synthetic known-valid JSON)")
    print("=" * 118)
    print("2D parsed:", tests["2d"])
    print("3D parsed:", tests["3d"])
    if len(tests["2d"]) != 1 or len(tests["3d"]) != 1:
        raise RuntimeError("Synthetic Box parser sanity failed")
    print("[PASS] parser accepts known-valid non-empty 2D and 3D boxes")
    return tests


def grad_bytes(model: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        k = storage_key(g)
        if k is not None and k not in seen:
            seen.add(k)
            total += k[-1]
    return total


def raw_batch_bytes(samples: list[Any]) -> Dict[str, int]:
    return {
        "cpu": unique_storage_bytes(samples, device_type="cpu"),
        "cuda": unique_storage_bytes(samples, device_type="cuda"),
    }


def train_memory_probe(model, criterion, train_module, samples, spec, device,
                       dataset_name: str, tracker: BoundaryInputTracker,
                       static_keys: set) -> Dict[str, Any]:
    """One real forward+backward, no optimizer.step."""
    model.train()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()  # isolate this route; transition test below does NOT clear after backward.
    cuda_sync()

    rb = raw_batch_bytes(samples)
    baseline = cuda_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    saved_storages: Dict[Any, int] = {}

    def pack_hook(t: torch.Tensor):
        if torch.is_tensor(t) and t.is_cuda:
            k = storage_key(t)
            if k is not None and k not in static_keys:
                saved_storages[k] = k[-1]
        return t

    def unpack_hook(t: torch.Tensor):
        return t

    phase = f"{dataset_name}/train"
    tracker.phase = phase
    t0 = time.perf_counter()
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, loss_out, target = train_module.forward_loss(
                    model, criterion, samples, spec, box_mode="joint"
                )
        cuda_sync()
        after_forward = cuda_snapshot(device)
        forward_peak = int(torch.cuda.max_memory_allocated(device))
        forward_reserved_peak = int(torch.cuda.max_memory_reserved(device))

        # Reset peak to isolate backward's additional transient requirement.
        torch.cuda.reset_peak_memory_stats(device)
        loss_out.total.backward()
        cuda_sync()
        after_backward = cuda_snapshot(device)
        backward_peak = int(torch.cuda.max_memory_allocated(device))
        backward_reserved_peak = int(torch.cuda.max_memory_reserved(device))
        gbytes = grad_bytes(model)
        elapsed = time.perf_counter() - t0

        result = {
            "dataset": dataset_name,
            "batch_size": len(samples),
            "raw_cpu_batch_bytes": rb["cpu"],
            "raw_cuda_batch_bytes": rb["cuda"],
            "baseline_allocated": baseline["allocated"],
            "baseline_reserved": baseline["reserved"],
            "after_forward_allocated": after_forward["allocated"],
            "after_forward_reserved": after_forward["reserved"],
            "forward_peak_allocated": forward_peak,
            "forward_peak_reserved": forward_reserved_peak,
            "saved_for_backward_unique_cuda_bytes": int(sum(saved_storages.values())),
            "after_backward_allocated": after_backward["allocated"],
            "after_backward_reserved": after_backward["reserved"],
            "backward_peak_allocated": backward_peak,
            "backward_peak_reserved": backward_reserved_peak,
            "gradient_bytes": gbytes,
            "loss": float(loss_out.total.detach().float().cpu()),
            "seconds": elapsed,
        }

        print("\n" + "=" * 118)
        print(f"C. TRAIN MEMORY — {dataset_name}  batch={len(samples)}  box_mode=joint")
        print("=" * 118)
        print(f"Raw dataset tensors:         CPU={mib(rb['cpu']):.1f} MiB  CUDA(before model)={mib(rb['cuda']):.1f} MiB")
        print(f"Baseline live allocated:     {gib(baseline['allocated']):.3f} GiB")
        print(f"Baseline reserved cache:     {gib(baseline['reserved']):.3f} GiB")
        print(f"Forward end live:            {gib(after_forward['allocated']):.3f} GiB  "
              f"(Δ {gib(after_forward['allocated']-baseline['allocated']):+.3f} GiB)")
        print(f"Forward peak live:           {gib(forward_peak):.3f} GiB  "
              f"(extra {gib(max(0,forward_peak-baseline['allocated'])):.3f} GiB)")
        print(f"Autograd saved storages*:    {gib(sum(saved_storages.values())):.3f} GiB")
        print(f"After backward live:         {gib(after_backward['allocated']):.3f} GiB")
        print(f"Backward peak live:          {gib(backward_peak):.3f} GiB  "
              f"(extra over fwd-end {gib(max(0,backward_peak-after_forward['allocated'])):.3f} GiB)")
        print(f"Gradient tensors exact:      {gib(gbytes):.3f} GiB")
        print(f"Peak reserved (forward/bwd): {gib(forward_reserved_peak):.3f} / {gib(backward_reserved_peak):.3f} GiB")
        print(f"Loss={result['loss']:.6f}  elapsed={elapsed:.2f}s")
        print("* unique CUDA storages saved by autograd, excluding model parameter/buffer storages; "
              "it is a diagnostic of backward activation pressure, not an additive allocator accounting row.")
        tracker.print_phase(phase)

        # Drop graph and gradients, but intentionally DO NOT empty allocator cache.
        del prediction, loss_out, target
        model.zero_grad(set_to_none=True)
        gc.collect()
        cuda_sync()
        cleaned = cuda_snapshot(device)
        result["after_cleanup_allocated"] = cleaned["allocated"]
        result["after_cleanup_reserved"] = cleaned["reserved"]
        print(f"After graph+grad cleanup:     allocated={gib(cleaned['allocated']):.3f} GiB  "
              f"reserved={gib(cleaned['reserved']):.3f} GiB  (NO empty_cache)")
        return result

    except torch.cuda.OutOfMemoryError as exc:
        cuda_sync()
        snap = cuda_snapshot(device)
        print(f"\n[OOM during {dataset_name} training probe] {exc}")
        print(f"At failure: allocated={gib(snap['allocated']):.3f} GiB reserved={gib(snap['reserved']):.3f} GiB")
        raise


def val_transition_probe(model, criterion, train_module, sample, spec, device,
                         dataset_name: str, tracker: BoundaryInputTracker,
                         box_capture: BoxCapture) -> Dict[str, Any]:
    """Run generated-box val immediately after train cleanup, without empty_cache."""
    model.eval()
    phase = f"{dataset_name}/val_generated"
    tracker.phase = phase
    box_capture.phase = phase
    before = cuda_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction, loss_out, target = train_module.forward_loss(
            model, criterion, [sample], spec, box_mode="generated"
        )
    cuda_sync()
    after = cuda_snapshot(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    valid = getattr(prediction, "box_prediction_valid", None)
    valid_count = int(valid.sum().item()) if torch.is_tensor(valid) else -1
    parse_failures = int(getattr(prediction, "box_parse_failures", -1))
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 118)
    print(f"D. TRAIN -> VAL TRANSITION — {dataset_name}  val batch=1 generated Box")
    print("=" * 118)
    print("No torch.cuda.empty_cache() was called after the training probe.")
    print(f"Val entry:            allocated={gib(before['allocated']):.3f} GiB  reserved={gib(before['reserved']):.3f} GiB")
    print(f"Val peak live:        {gib(peak):.3f} GiB  extra={mib(max(0,peak-before['allocated'])):.1f} MiB")
    print(f"Val end live:         {gib(after['allocated']):.3f} GiB")
    print(f"Val peak reserved:    {gib(peak_reserved):.3f} GiB")
    print(f"Generated valid Box:  {valid_count}")
    print(f"Parse failures:       {parse_failures}")
    print(f"Val loss:             {float(loss_out.total.detach().float().cpu()):.6f}")
    tracker.print_phase(phase)

    result = {
        "dataset": dataset_name,
        "entry_allocated": before["allocated"],
        "entry_reserved": before["reserved"],
        "peak_allocated": peak,
        "peak_reserved": peak_reserved,
        "end_allocated": after["allocated"],
        "end_reserved": after["reserved"],
        "valid_boxes": valid_count,
        "parse_failures": parse_failures,
        "loss": float(loss_out.total.detach().float().cpu()),
        "seconds": elapsed,
    }

    del prediction, loss_out, target
    gc.collect()
    return result


def load_samples(dataset_cls, manifest, spec, count: int):
    ds = dataset_cls(manifest, spec)
    if len(ds) < count:
        raise RuntimeError(f"Dataset has only {len(ds)} samples but batch requires {count}")
    samples = [ds[i] for i in range(count)]
    return ds, samples


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Use the project in current working directory, not bundled model code.
    from datasets.config_loader import load_experiment_config
    from datasets.pair_dataset import UnifiedPAIRDataset
    from loss import PAIRSemanticChangeLoss
    from models.pair import PAIRModel
    import train

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    exp = load_experiment_config(args.config, DATASETS)
    flags = train.active_model_flags(exp)
    if not (flags["enable_2d"] and flags["enable_3d"]):
        raise RuntimeError("This diagnostic expects SECOND + NYC-SCD routes enabled")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before_model = cuda_snapshot(device)

    print("Building one joint PAIR model...", flush=True)
    model = PAIRModel.from_config(exp.model, device, **flags)
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        train.load_model_state_from_checkpoint(ckpt, model)
        del ckpt
        gc.collect()

    criterion = PAIRSemanticChangeLoss().to(device)
    after_model = cuda_snapshot(device)
    print(f"Allocator after model: allocated={gib(after_model['allocated']):.3f} GiB "
          f"reserved={gib(after_model['reserved']):.3f} GiB "
          f"(allocated delta={gib(after_model['allocated']-before_model['allocated']):.3f} GiB)")

    static_report = parameter_report(model, train)
    static_keys = model_storage_keys(model)
    parser_report = parser_sanity(model)

    # Generation config matters: this is text generation, not a randomly initialized Box head.
    gen_cfg = getattr(model.qwen_backbone.model, "generation_config", None)
    if gen_cfg is not None:
        print("\nQwen generation defaults:")
        print(f"  do_sample={getattr(gen_cfg, 'do_sample', None)}  "
              f"num_beams={getattr(gen_cfg, 'num_beams', None)}  "
              f"temperature={getattr(gen_cfg, 'temperature', None)}")
    print(f"PAIR box generation: max_boxes={model.box_generation_max_boxes}, "
          f"max_new_tokens={model.box_max_new_tokens}")

    tracker = BoundaryInputTracker()
    tracker.install(model)
    box_capture = BoxCapture(model)
    box_capture.install()

    reports = []
    datasets_to_delete = []
    try:
        for dataset_name in DATASETS:
            cfg = exp.datasets[dataset_name]
            train_batch = int(cfg.per_gpu_batch_size)
            if train_batch < 1:
                raise ValueError(f"{dataset_name}: invalid train batch {train_batch}")

            train_ds, train_samples = load_samples(
                UnifiedPAIRDataset, cfg.train_manifest, cfg.spec, train_batch
            )
            datasets_to_delete.append(train_ds)

            # Use first validation sample for immediate Train -> Val test.
            if cfg.val_manifest is None:
                raise RuntimeError(f"{dataset_name} has no val manifest")
            val_ds, val_samples = load_samples(
                UnifiedPAIRDataset, cfg.val_manifest, cfg.spec, 1
            )
            datasets_to_delete.append(val_ds)

            train_report = train_memory_probe(
                model, criterion, train, train_samples, cfg.spec, device,
                dataset_name, tracker, static_keys,
            )
            try:
                val_report = val_transition_probe(
                    model, criterion, train, val_samples[0], cfg.spec, device,
                    dataset_name, tracker, box_capture,
                )
            except torch.cuda.OutOfMemoryError as exc:
                snap = cuda_snapshot(device)
                print(f"\n[OOM during {dataset_name} Train->Val probe] {exc}")
                print(f"At failure: allocated={gib(snap['allocated']):.3f} GiB "
                      f"reserved={gib(snap['reserved']):.3f} GiB")
                val_report = {"dataset": dataset_name, "oom": True, "error": str(exc)}

            reports.append({"dataset": dataset_name, "train": train_report, "val": val_report})

            del train_samples, val_samples
            model.zero_grad(set_to_none=True)
            gc.collect()
            # Now isolate the next dataset; this does not affect the just-recorded Train->Val result.
            torch.cuda.empty_cache()

    finally:
        box_capture.remove()
        tracker.remove()
        for ds in datasets_to_delete:
            del ds
        gc.collect()

    print("\n" + "=" * 118)
    print("E. BOX GENERATION SUMMARY")
    print("=" * 118)
    if not box_capture.records:
        print("No generate_box_proposals() call was captured — generated Box path did not execute.")
    else:
        for rec in box_capture.records:
            print(f"{rec['phase']:<30} kind={rec['kind']:<3} valid={rec['valid_boxes']:<3} "
                  f"parse_failures={rec['parse_failures']}")
            for text in rec["texts"]:
                normalized = text.strip().replace("\n", " ")
                if normalized == '{"boxes":[]}' or normalized == '{"boxes": []}':
                    print("  -> Qwen itself emitted an EMPTY but valid Box list; parser did not delete boxes.")

    print("\n" + "=" * 118)
    print("F. HOW TO READ THE MEMORY NUMBERS")
    print("=" * 118)
    print("1) Static model tensor bytes are exact and persistent.")
    print("2) Raw batch bytes are exact tensor-storage bytes in the dataset sample before the model.")
    print("3) 'Autograd saved storages' directly measures unique non-parameter CUDA storages saved for backward;")
    print("   it is the most useful activation-pressure diagnostic here, but do not add it to allocator peaks.")
    print("4) Forward/backward peak allocated are the authoritative live CUDA peaks.")
    print("5) Reserved is allocator cache. A 45 GiB reserved value after training does not mean 45 GiB live tensors.")
    print("6) This test intentionally does NOT optimizer.step(); AdamW state is therefore not included.")

    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "cuda_device": torch.cuda.get_device_name(device),
        "static_model": {
            "parameter_bytes": static_report["parameter_bytes"],
            "buffer_bytes": static_report["buffer_bytes"],
            "static_tensor_bytes": static_report["static_tensor_bytes"],
            "dtype_bytes": static_report["dtype_bytes"],
            "modules": {k: dict(v) for k, v in static_report["modules"].items()},
        },
        "parser_sanity": parser_report,
        "boundary_cuda_input_max_bytes": {
            phase: dict(rows) for phase, rows in tracker.max_bytes.items()
        },
        "box_generation": box_capture.records,
        "dataset_reports": reports,
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/pair_train_qwen4b.json"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("pair_memory_box_debug_v2.json"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
