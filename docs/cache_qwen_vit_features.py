#!/usr/bin/env python3
"""Offline Qwen2.5-VL ViT feature cache for ScanNet frames.

Reads RGB frames from ``frames_square_highres`` and writes per-frame ViT
features as bfloat16 ``.pt`` tensors, matching
``VideoMaskFormer._cache_vit_features`` layout:

    FEATURE_DIR/<scene_id>/<frame>.pt   e.g. .../scene0051_03/140.pt

Examples
--------
# 3B features -> /data/user_data/lucylin/scannet_qwen_feat_3b
python scripts/cache_qwen_vit_features.py --model 3b

# 7B features -> /data/user_data/lucylin/scannet_qwen_7b
python scripts/cache_qwen_vit_features.py --model 7b

# Shard across 4 jobs (e.g. 4 GPUs / SLURM array tasks)
python scripts/cache_qwen_vit_features.py --model 3b --shard-id 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


FRAMES_ROOT = Path("/data/user_data/lucylin/SEMSEG_100k/frames_square_highres")
USER_DATA = Path("/data/user_data/lucylin")

MODEL_PRESETS = {
    "3b": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "out_dir": USER_DATA / "scannet_qwen_feat_3b",
    },
    "7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "out_dir": USER_DATA / "scannet_qwen_7b",
    },
}

# Match univlg/config.py defaults used by the dataset mapper / model.
DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 390000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True, help="Which Qwen ViT to run.")
    p.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    p.add_argument("--out-dir", type=Path, default=None, help="Override default FEATURE_DIR for the model.")
    p.add_argument("--hf-id", type=str, default=None, help="Override HuggingFace model id.")
    p.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    p.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--batch-size", type=int, default=8, help="Frames per ViT forward.")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if .pt already exists.")
    p.add_argument("--scenes", nargs="*", default=None, help="Optional scene id whitelist.")
    return p.parse_args()


def list_scene_dirs(frames_root: Path, scenes: list[str] | None):
    if scenes:
        dirs = [frames_root / s for s in scenes]
        missing = [str(d) for d in dirs if not d.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing scene dirs: {missing[:5]}")
        return sorted(dirs)

    return sorted(p for p in frames_root.iterdir() if p.is_dir() and (p / "color").is_dir())


def list_frame_jobs(scene_dirs: list[Path], out_dir: Path, overwrite: bool):
    """Return (rgb_path, out_path) jobs, skipping existing caches unless overwrite."""
    jobs = []
    for scene_dir in scene_dirs:
        color_dir = scene_dir / "color"
        scene_out = out_dir / scene_dir.name
        for rgb_path in sorted(color_dir.iterdir()):
            if rgb_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            frame_name = rgb_path.stem  # "140" from "140.png"
            out_path = scene_out / f"{frame_name}.pt"
            if not overwrite and out_path.is_file():
                continue
            jobs.append((rgb_path, out_path))
    return jobs


def split_vit_features_by_frame(featurecloud: torch.Tensor, image_grid_thw: torch.Tensor, merge: int):
    """Same logic as VideoMaskFormer._split_vit_features_by_frame."""
    tokens_per_frame = (
        image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] // (merge * merge)
    ).tolist()
    assert sum(tokens_per_frame) == featurecloud.shape[0], (
        f"ViT token count {featurecloud.shape[0]} != sum(tokens_per_frame) {sum(tokens_per_frame)}"
    )
    return list(torch.split(featurecloud, tokens_per_frame, dim=0))


@torch.inference_mode()
def encode_batch(visual, processor, rgb_paths: list[Path], device: torch.device):
    images = [Image.open(p).convert("RGB") for p in rgb_paths]
    out = processor.image_processor(images=images, do_rescale=True, return_tensors="pt")
    pixel_values = out["pixel_values"].to(device=device, dtype=torch.bfloat16)
    image_grid_thw = out["image_grid_thw"].to(device=device)
    featurecloud = visual(pixel_values, grid_thw=image_grid_thw)
    return featurecloud, image_grid_thw


def save_features(frame_features, out_paths: list[Path]):
    for fc, path in zip(frame_features, out_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write so a killed job doesn't leave a truncated .pt
        tmp = path.with_suffix(".pt.tmp")
        torch.save(fc.detach().to(dtype=torch.bfloat16).cpu().contiguous(), tmp)
        os.replace(tmp, path)


def main():
    args = parse_args()
    preset = MODEL_PRESETS[args.model]
    hf_id = args.hf_id or preset["hf_id"]
    out_dir = args.out_dir or preset["out_dir"]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard-id must be in [0, {args.num_shards})")

    scene_dirs = list_scene_dirs(args.frames_root, args.scenes)
    # Shard by scene so each job owns whole scenes (cleaner resume / fewer races).
    scene_dirs = scene_dirs[args.shard_id :: args.num_shards]
    jobs = list_frame_jobs(scene_dirs, out_dir, args.overwrite)

    print(f"model={hf_id}")
    print(f"frames_root={args.frames_root}")
    print(f"out_dir={out_dir}")
    print(f"shard={args.shard_id}/{args.num_shards}  scenes={len(scene_dirs)}  frames_to_do={len(jobs)}")
    if not jobs:
        print("Nothing to do.")
        return

    processor = AutoProcessor.from_pretrained(
        hf_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    # Stock HF ViT only (same as video_maskformer_model.py self.visual).
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    )
    visual = model.visual.to(device)
    visual.eval()
    visual.requires_grad_(False)
    del model
    merge = getattr(visual, "spatial_merge_size", 2)

    for start in tqdm(range(0, len(jobs), args.batch_size), desc="batches"):
        batch = jobs[start : start + args.batch_size]
        rgb_paths = [j[0] for j in batch]
        out_paths = [j[1] for j in batch]
        try:
            featurecloud, image_grid_thw = encode_batch(visual, processor, rgb_paths, device)
            frame_features = split_vit_features_by_frame(featurecloud, image_grid_thw, merge)
            assert len(frame_features) == len(out_paths)
            save_features(frame_features, out_paths)
        except Exception as e:
            # Fall back to per-frame so one bad image doesn't kill the batch.
            print(f"\nBatch failed ({e}); retrying frame-by-frame starting at {rgb_paths[0]}")
            for rgb_path, out_path in batch:
                try:
                    featurecloud, image_grid_thw = encode_batch(visual, processor, [rgb_path], device)
                    frame_features = split_vit_features_by_frame(featurecloud, image_grid_thw, merge)
                    save_features(frame_features, [out_path])
                except Exception as e2:
                    print(f"  SKIP {rgb_path}: {e2}")

    print("Done.")


if __name__ == "__main__":
    main()
