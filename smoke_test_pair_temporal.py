#!/usr/bin/env python3
"""
Temporal PAIR full-model integration smoke test.

Tests
-----
1. Shared Qwen3-VL + shared PTv3 + shared PointAdapter construction.
2. 2D temporal mode:
       Image T1 + Image T2 + prompt
3. 3D temporal mode:
       Point T1 + Point T2 + prompt
4. 2D+3D temporal mode:
       Image T1 + Image T2 + Point T1 + Point T2 + prompt
5. Correct T1/T2 hidden-state separation.
6. Point-token injection.
7. Temporal representations actually differ when T1/T2 inputs differ.
8. Native Qwen text generation remains functional.

Run
---
CUDA_VISIBLE_DEVICES=2 python smoke_test_pair_temporal.py
"""

import os
import time

import torch
from PIL import Image, ImageDraw

from models.pair import PAIRModel
from models.qwen3vl_backbone import Qwen3VLBackbone
from models.point_encoder import PTv3Backbone, PTv3Config
from models.point_adapter import PointAdapter, PointAdapterConfig


MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"

IMAGE_SIZE = 448
NUM_SIDE = 16
NUM_POINTS = NUM_SIDE ** 3
POINT_IN_CHANNELS = 6
NUM_POINT_TOKENS = 32


def gib(x: int) -> float:
    return x / (1024 ** 3)


def tensor_summary(name, tensor):
    if tensor is None:
        print(f"  {name}: None")
        return

    finite = bool(torch.isfinite(tensor).all().item())

    print(
        f"  {name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"finite={finite}"
    )


def assert_finite(name, tensor):
    assert tensor is not None, f"{name} is None."
    assert torch.isfinite(tensor).all(), (
        f"{name} contains NaN/Inf."
    )


# ======================================================================
# Synthetic temporal images
# ======================================================================

def make_image_t1(size=IMAGE_SIZE):
    image = Image.new(
        "RGB",
        (size, size),
        (238, 238, 238),
    )

    draw = ImageDraw.Draw(image)

    # Building-like rectangle.
    draw.rectangle(
        (55, 85, 205, 245),
        outline=(210, 50, 50),
        width=6,
    )

    # Tree-like circle.
    draw.ellipse(
        (270, 120, 365, 215),
        outline=(50, 160, 70),
        width=6,
    )

    # Road-like line.
    draw.line(
        (0, 370, size, 370),
        fill=(75, 75, 75),
        width=8,
    )

    draw.text(
        (25, 25),
        "Time 1",
        fill=(0, 0, 0),
    )

    return image


def make_image_t2(size=IMAGE_SIZE):
    image = Image.new(
        "RGB",
        (size, size),
        (238, 238, 238),
    )

    draw = ImageDraw.Draw(image)

    # Existing building.
    draw.rectangle(
        (55, 85, 205, 245),
        outline=(210, 50, 50),
        width=6,
    )

    # NEW building/object.
    draw.rectangle(
        (220, 245, 335, 340),
        outline=(40, 90, 210),
        width=6,
    )

    # Tree moved/changed.
    draw.ellipse(
        (290, 105, 390, 205),
        outline=(50, 160, 70),
        width=6,
    )

    # Same road.
    draw.line(
        (0, 370, size, 370),
        fill=(75, 75, 75),
        width=8,
    )

    draw.text(
        (25, 25),
        "Time 2",
        fill=(0, 0, 0),
    )

    return image


# ======================================================================
# Synthetic temporal point clouds
# ======================================================================

def make_point_pair(device):
    """
    Build two point clouds with identical topology but a controlled temporal
    change in one spatial region.

    T2 changes:
        - raises Z in the x/y upper-right region
        - modifies corresponding feature channels

    Keeping grid_coord valid and deterministic makes this a safe PTv3
    integration test while still ensuring T1/T2 features are different.
    """

    axis = torch.arange(
        NUM_SIDE,
        dtype=torch.int32,
    )

    gx, gy, gz = torch.meshgrid(
        axis,
        axis,
        axis,
        indexing="ij",
    )

    grid_coord = torch.stack(
        [
            gx.reshape(-1),
            gy.reshape(-1),
            gz.reshape(-1),
        ],
        dim=1,
    ).contiguous()

    coord_t1 = grid_coord.float() * 0.05

    scale = max(
        float(coord_t1.max().item()),
        1e-6,
    )

    xyz_t1 = coord_t1 / scale
    x1, y1, z1 = xyz_t1.unbind(dim=1)

    feat_t1 = torch.stack(
        [
            x1,
            y1,
            z1,
            torch.sin(x1 * torch.pi),
            torch.cos(y1 * torch.pi),
            z1 * 0.5,
        ],
        dim=1,
    ).float()

    # --------------------------------------------------------------
    # T2: create a local synthetic 3D change.
    # --------------------------------------------------------------
    coord_t2 = coord_t1.clone()
    feat_t2 = feat_t1.clone()

    changed = (
        (grid_coord[:, 0] >= NUM_SIDE // 2)
        & (grid_coord[:, 1] >= NUM_SIDE // 2)
        & (grid_coord[:, 2] >= NUM_SIDE // 3)
    )

    coord_t2[changed, 2] += 0.25

    feat_t2[changed, 2] += 0.40
    feat_t2[changed, 3] *= -1.0
    feat_t2[changed, 5] += 0.75

    batch = torch.zeros(
        NUM_POINTS,
        dtype=torch.long,
    )

    common = {
        "grid_coord": grid_coord.to(device),
        "batch": batch.to(device),
    }

    point_t1 = {
        "coord": coord_t1.to(device),
        "grid_coord": common["grid_coord"].clone(),
        "feat": feat_t1.to(device),
        "batch": common["batch"].clone(),
    }

    point_t2 = {
        "coord": coord_t2.to(device),
        "grid_coord": common["grid_coord"].clone(),
        "feat": feat_t2.to(device),
        "batch": common["batch"].clone(),
    }

    return point_t1, point_t2, int(changed.sum().item())


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 88)
    print("PAIR TEMPORAL FULL MODEL INTEGRATION SMOKE TEST")
    print("=" * 88)

    assert os.path.isdir(MODEL_DIR), (
        f"Qwen checkpoint not found: {MODEL_DIR}"
    )

    assert torch.cuda.is_available(), (
        "CUDA is required."
    )

    device = torch.device("cuda")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))
    print("Qwen checkpoint:", MODEL_DIR)
    print()

    # ==================================================================
    # 1. Architecture ownership
    # ==================================================================
    print("[0] Architecture check")

    assert hasattr(
        PAIRModel,
        "_make_point_injection_hook",
    ), (
        "PAIRModel does not own point injection."
    )

    assert not hasattr(
        Qwen3VLBackbone,
        "_make_point_injection_hook",
    ), (
        "Qwen3VLBackbone should no longer own point injection."
    )

    print("  PAIR owns point injection: YES")
    print("  Qwen wrapper owns point injection: NO")
    print("  architecture: TEMPORAL REFACTORED")
    print()

    # ==================================================================
    # 2. Build modules
    # ==================================================================
    print("[1] Building Qwen3VLBackbone...")

    t0 = time.perf_counter()

    qwen = Qwen3VLBackbone(
        model_dir=MODEL_DIR,
        dtype=torch.bfloat16,
        device="cuda",
        device_map="cuda",
        local_files_only=True,
    )

    torch.cuda.synchronize()

    print(
        f"  hidden size: {qwen.hidden_size}"
    )
    print(
        f"  image token id: {qwen.image_token_id}"
    )
    print(
        f"  point token id: {qwen.point_token_id}"
    )
    print(
        f"  task token id: {qwen.task_token_id}"
    )
    print(
        f"  load time: "
        f"{time.perf_counter() - t0:.2f} s"
    )
    print()

    print("[2] Building shared PTv3...")

    ptv3_cfg = PTv3Config(
        in_channels=POINT_IN_CHANNELS,
    )

    point_encoder = PTv3Backbone(
        ptv3_cfg
    ).to(device)

    assert point_encoder.output_dim == 64

    print(
        "  output dim:",
        point_encoder.output_dim,
    )
    print(
        "  params:",
        f"{point_encoder.parameter_count() / 1e6:.3f} M",
    )
    print()

    print("[3] Building shared PointAdapter...")

    adapter_cfg = PointAdapterConfig(
        in_dim=point_encoder.output_dim,
        out_dim=qwen.hidden_size,
        num_tokens=NUM_POINT_TOKENS,
        sampling="uniform",
        use_layer_norm=True,
    )

    point_adapter = PointAdapter(
        adapter_cfg
    ).to(device)

    print(
        f"  adapter: "
        f"{adapter_cfg.in_dim} -> "
        f"{adapter_cfg.out_dim}"
    )
    print(
        f"  temporal tokens per point cloud: "
        f"{adapter_cfg.num_tokens}"
    )
    print()

    print("[4] Building PAIRModel...")

    model = PAIRModel(
        qwen_backbone=qwen,
        point_encoder=point_encoder,
        point_adapter=point_adapter,
    )

    model.eval()

    print("  PAIR temporal model built.")
    print()

    # ==================================================================
    # 3. Build synthetic temporal pair
    # ==================================================================
    print("[5] Building T1/T2 synthetic data...")

    image_t1 = make_image_t1()
    image_t2 = make_image_t2()

    point_t1, point_t2, changed_points = (
        make_point_pair(device)
    )

    print(
        "  image T1:",
        image_t1.size,
    )
    print(
        "  image T2:",
        image_t2.size,
    )
    print(
        "  point T1:",
        tuple(point_t1["coord"].shape),
        tuple(point_t1["feat"].shape),
    )
    print(
        "  point T2:",
        tuple(point_t2["coord"].shape),
        tuple(point_t2["feat"].shape),
    )
    print(
        "  synthetically changed 3D points:",
        changed_points,
    )
    print(
        "  TEMPORAL INPUT PAIR OK"
    )
    print()

    prompt = (
        "Compare Time 1 and Time 2. "
        "Use the available remote-sensing modalities to identify "
        "and summarize changes between the two observations."
    )

    # ==================================================================
    # 4. Full 2D + 3D temporal forward
    # ==================================================================
    print("[6] Running temporal 2D+3D forward...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        joint = model(
            task_mode="2d3d",
            prompt=prompt,

            images_t1=image_t1,
            images_t2=image_t2,

            point_dict_t1=point_t1,
            point_dict_t2=point_t2,

            return_logits=False,
            return_hidden_states=True,
        )

    torch.cuda.synchronize()

    print(
        f"  forward time: "
        f"{time.perf_counter() - t0:.3f} s"
    )

    tensor_summary(
        "image_hidden_t1",
        joint.image_hidden_t1,
    )
    tensor_summary(
        "image_hidden_t2",
        joint.image_hidden_t2,
    )
    tensor_summary(
        "point_hidden_t1",
        joint.point_hidden_t1,
    )
    tensor_summary(
        "point_hidden_t2",
        joint.point_hidden_t2,
    )
    tensor_summary(
        "task_hidden",
        joint.task_hidden,
    )

    assert_finite(
        "image_hidden_t1",
        joint.image_hidden_t1,
    )
    assert_finite(
        "image_hidden_t2",
        joint.image_hidden_t2,
    )
    assert_finite(
        "point_hidden_t1",
        joint.point_hidden_t1,
    )
    assert_finite(
        "point_hidden_t2",
        joint.point_hidden_t2,
    )
    assert_finite(
        "task_hidden",
        joint.task_hidden,
    )

    assert joint.image_hidden_t1.shape == (
        196,
        qwen.hidden_size,
    )

    assert joint.image_hidden_t2.shape == (
        196,
        qwen.hidden_size,
    )

    assert joint.point_hidden_t1.shape == (
        NUM_POINT_TOKENS,
        qwen.hidden_size,
    )

    assert joint.point_hidden_t2.shape == (
        NUM_POINT_TOKENS,
        qwen.hidden_size,
    )

    assert joint.task_hidden.shape == (
        1,
        qwen.hidden_size,
    )

    assert (
        joint.aux["point_injection_replaced"]
        is True
    )

    print(
        "  point injection calls:",
        joint.aux["point_injection_calls"],
    )
    print(
        "  point injection replaced:",
        joint.aux["point_injection_replaced"],
    )

    print(
        "  image token counts T1/T2:",
        joint.aux["image_count_t1"],
        joint.aux["image_count_t2"],
    )

    print(
        "  point token counts T1/T2:",
        joint.aux["point_count_t1"],
        joint.aux["point_count_t2"],
    )

    # Dense PTv3 outputs
    dense_t1 = joint.aux[
        "point_encoded_t1"
    ].features

    dense_t2 = joint.aux[
        "point_encoded_t2"
    ].features

    tensor_summary(
        "PTv3 dense_t1",
        dense_t1,
    )
    tensor_summary(
        "PTv3 dense_t2",
        dense_t2,
    )

    assert dense_t1.shape == (
        NUM_POINTS,
        64,
    )

    assert dense_t2.shape == (
        NUM_POINTS,
        64,
    )

    # Spatial Qwen image hidden
    spatial_t1 = joint.aux[
        "image_hidden_2d_t1"
    ]
    spatial_t2 = joint.aux[
        "image_hidden_2d_t2"
    ]

    tensor_summary(
        "image_hidden_2d_t1",
        spatial_t1,
    )
    tensor_summary(
        "image_hidden_2d_t2",
        spatial_t2,
    )

    assert spatial_t1.shape == (
        1,
        14,
        14,
        qwen.hidden_size,
    )

    assert spatial_t2.shape == (
        1,
        14,
        14,
        qwen.hidden_size,
    )

    print("  TEMPORAL 2D+3D FORWARD OK")
    print()

    # ==================================================================
    # 5. T1/T2 representation difference
    # ==================================================================
    print("[7] Checking T1/T2 representation differences...")

    image_delta = (
        joint.image_hidden_t1.float()
        - joint.image_hidden_t2.float()
    ).abs().mean().item()

    point_delta = (
        joint.point_hidden_t1.float()
        - joint.point_hidden_t2.float()
    ).abs().mean().item()

    dense_point_delta = (
        dense_t1.float()
        - dense_t2.float()
    ).abs().mean().item()

    print(
        "  mean |image T1 - T2|:",
        image_delta,
    )
    print(
        "  mean |point contextual T1 - T2|:",
        point_delta,
    )
    print(
        "  mean |PTv3 dense T1 - T2|:",
        dense_point_delta,
    )

    assert image_delta > 0.0, (
        "T1/T2 image representations are identical."
    )

    assert point_delta > 0.0, (
        "T1/T2 contextual point representations are identical."
    )

    assert dense_point_delta > 0.0, (
        "T1/T2 PTv3 dense representations are identical."
    )

    print("  TEMPORAL DIFFERENCE PROPAGATION OK")
    print()

    # ==================================================================
    # 6. 2D-only temporal routing
    # ==================================================================
    print("[8] Running temporal 2D-only routing...")

    with torch.inference_mode():
        out_2d = model(
            task_mode="2d",
            prompt=prompt,

            images_t1=image_t1,
            images_t2=image_t2,

            return_logits=False,
            return_hidden_states=True,
        )

    assert_finite(
        "2d.image_hidden_t1",
        out_2d.image_hidden_t1,
    )
    assert_finite(
        "2d.image_hidden_t2",
        out_2d.image_hidden_t2,
    )
    assert_finite(
        "2d.task_hidden",
        out_2d.task_hidden,
    )

    assert out_2d.point_hidden_t1 is None
    assert out_2d.point_hidden_t2 is None

    assert out_2d.image_hidden_t1.shape == (
        196,
        qwen.hidden_size,
    )
    assert out_2d.image_hidden_t2.shape == (
        196,
        qwen.hidden_size,
    )

    print("  2D TEMPORAL ROUTING OK")
    print()

    # ==================================================================
    # 7. 3D-only temporal routing
    # ==================================================================
    print("[9] Running temporal 3D-only routing...")

    with torch.inference_mode():
        out_3d = model(
            task_mode="3d",
            prompt=prompt,

            point_dict_t1=point_t1,
            point_dict_t2=point_t2,

            return_logits=False,
            return_hidden_states=True,
        )

    assert out_3d.image_hidden_t1 is None
    assert out_3d.image_hidden_t2 is None

    assert_finite(
        "3d.point_hidden_t1",
        out_3d.point_hidden_t1,
    )
    assert_finite(
        "3d.point_hidden_t2",
        out_3d.point_hidden_t2,
    )
    assert_finite(
        "3d.task_hidden",
        out_3d.task_hidden,
    )

    assert out_3d.point_hidden_t1.shape == (
        NUM_POINT_TOKENS,
        qwen.hidden_size,
    )

    assert out_3d.point_hidden_t2.shape == (
        NUM_POINT_TOKENS,
        qwen.hidden_size,
    )

    assert (
        out_3d.aux["point_injection_replaced"]
        is True
    )

    print("  3D TEMPORAL ROUTING OK")
    print()

    # ==================================================================
    # 8. Check prompt actually contains temporal structure
    # ==================================================================
    print("[10] Checking temporal prompt layout...")

    prompt_text = joint.aux["prompt_text"]

    required_phrases = [
        "Time 1 image:",
        "Time 2 image:",
        "Time 1 point cloud:",
        "Time 2 point cloud:",
    ]

    for phrase in required_phrases:
        assert phrase in prompt_text, (
            f"Temporal prompt is missing: {phrase}"
        )

    assert prompt_text.count("<TASK>") == 1

    print("  T1/T2 labels present in Qwen prompt")
    print("  exactly one <TASK> present")
    print("  TEMPORAL PROMPT LAYOUT OK")
    print()

    # ==================================================================
    # 9. Native generation
    # ==================================================================
    print("[11] Running temporal 2D+3D generation...")

    with torch.inference_mode():
        generated = model.generate(
            task_mode="2d3d",
            prompt=prompt,

            images_t1=image_t1,
            images_t2=image_t2,

            point_dict_t1=point_t1,
            point_dict_t2=point_t2,

            max_new_tokens=40,
            do_sample=False,
        )

    print("  generated text:")
    print(" ", generated.generated_text)

    assert generated.generated_ids is not None
    assert isinstance(
        generated.generated_text,
        str,
    )

    assert (
        generated.aux["point_injection_replaced"]
        is True
    )

    print(
        "  generation point injection replaced:",
        generated.aux["point_injection_replaced"],
    )

    print("  TEMPORAL GENERATION OK")
    print()

    # ==================================================================
    # Final
    # ==================================================================
    torch.cuda.synchronize()

    peak_allocated = gib(
        torch.cuda.max_memory_allocated()
    )

    peak_reserved = gib(
        torch.cuda.max_memory_reserved()
    )

    print("=" * 88)
    print("SUCCESS")
    print("=" * 88)
    print()
    print("PAIR temporal model chain is connected:")
    print()
    print("  Image T1 ----\\")
    print("                \\")
    print("  Image T2 ------> Qwen3-VL -----------------------\\")
    print("                                                   \\")
    print("  Point T1 -> shared PTv3 -> shared PointAdapter ----> Qwen LLM")
    print("                                                   /")
    print("  Point T2 -> shared PTv3 -> shared PointAdapter --/")
    print("  Prompt + <TASK> --------------------------------/")
    print()
    print(
        "  image_hidden_t1:",
        tuple(joint.image_hidden_t1.shape),
    )
    print(
        "  image_hidden_t2:",
        tuple(joint.image_hidden_t2.shape),
    )
    print(
        "  point_hidden_t1:",
        tuple(joint.point_hidden_t1.shape),
    )
    print(
        "  point_hidden_t2:",
        tuple(joint.point_hidden_t2.shape),
    )
    print(
        "  task_hidden:",
        tuple(joint.task_hidden.shape),
    )
    print()
    print(
        f"Peak allocated GPU memory: "
        f"{peak_allocated:.2f} GiB"
    )
    print(
        f"Peak reserved GPU memory:  "
        f"{peak_reserved:.2f} GiB"
    )
    print()
    print("Temporal 2D routing:       PASS")
    print("Temporal 3D routing:       PASS")
    print("Temporal 2D+3D routing:    PASS")
    print("T1/T2 image split:         PASS")
    print("T1/T2 point split:         PASS")
    print("Point injection:           PASS")
    print("Temporal difference flow:  PASS")
    print("Text generation:           PASS")


if __name__ == "__main__":
    main()
