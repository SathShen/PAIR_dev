#!/usr/bin/env python3
"""
PAIR full-model integration smoke test.

This test intentionally uses only the public top-level PAIR API so it can
validate both:

1) the current repository layout where Qwen3VLBackbone still owns injection;
2) the refactored layout where PAIRModel owns injection and Qwen3VLBackbone
   only prepares Qwen inputs.

Validated chain
---------------
Image ---------------------------> Qwen3-VL Vision ----\
                                                        \
Point cloud -> PTv3 -> PointAdapter -> <POINT> tokens ---> Qwen LLM
                                                          |
Prompt + <TASK> ------------------------------------------/
                                                          |
                      image_hidden / point_hidden / task_hidden
                                                          |
                                                     text generate

It also tests the three routing modes:
    2d
    3d
    2d3d
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

NUM_SIDE = 16
NUM_POINTS = NUM_SIDE ** 3       # 4096
POINT_IN_CHANNELS = 6
NUM_POINT_TOKENS = 32
IMAGE_SIZE = 448


def gib(x):
    return x / (1024 ** 3)


def make_test_image(size=IMAGE_SIZE):
    """Small deterministic synthetic image."""
    image = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(image)

    draw.rectangle((40, 55, 185, 220), outline=(210, 50, 50), width=5)
    draw.rectangle((245, 120, 395, 300), outline=(50, 110, 210), width=5)
    draw.ellipse((125, 265, 225, 365), outline=(60, 170, 70), width=5)
    draw.line(
        (0, size - 42, size, size - 42),
        fill=(70, 70, 70),
        width=6,
    )
    draw.text((48, 22), "PAIR integration smoke test", fill=(0, 0, 0))

    return image


def make_synthetic_point_cloud(device):
    """
    16 x 16 x 16 regular 3D grid.

    Features:
        normalized XYZ
        + sin(X)
        + cos(Y)
        + Z
      = 6 channels
    """
    axis = torch.arange(NUM_SIDE, dtype=torch.int32)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")

    grid_coord = torch.stack(
        [
            gx.reshape(-1),
            gy.reshape(-1),
            gz.reshape(-1),
        ],
        dim=1,
    ).contiguous()

    coord = grid_coord.float() * 0.05

    denom = max(float(coord.max().item()), 1e-6)
    xyz = coord / denom

    x, y, z = xyz.unbind(dim=1)

    extra = torch.stack(
        [
            torch.sin(x * torch.pi),
            torch.cos(y * torch.pi),
            z,
        ],
        dim=1,
    )

    feat = torch.cat([xyz, extra], dim=1).float()

    batch = torch.zeros(NUM_POINTS, dtype=torch.long)

    return {
        "coord": coord.to(device),
        "grid_coord": grid_coord.to(device),
        "feat": feat.to(device),
        "batch": batch.to(device),
    }


def get_aux(output, key, default=None):
    """
    Compatibility helper.

    Old layout:
        output.aux["qwen"][key]

    Refactored layout:
        output.aux[key]
    """
    if output.aux is None:
        return default

    if key in output.aux:
        return output.aux[key]

    qwen_aux = output.aux.get("qwen")

    if isinstance(qwen_aux, dict) and key in qwen_aux:
        return qwen_aux[key]

    return default


def tensor_summary(name, tensor):
    if tensor is None:
        print(f"  {name}: None")
        return

    finite = bool(torch.isfinite(tensor).all().item())

    print(
        f"  {name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, device={tensor.device}, finite={finite}"
    )


def assert_finite(name, tensor):
    assert tensor is not None, f"{name} is None."
    assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf."


def print_architecture_ownership():
    """
    This is informational, not a failure condition.

    It tells us whether the GitHub checkout is still using the older layout
    or the refactored layout agreed for PAIR.
    """
    pair_owns_hook = hasattr(PAIRModel, "_make_point_injection_hook")
    qwen_owns_hook = hasattr(
        Qwen3VLBackbone,
        "_make_point_injection_hook",
    )

    print("[0] Injection ownership check")
    print("  PAIRModel owns point-injection hook:", pair_owns_hook)
    print("  Qwen3VLBackbone owns point-injection hook:", qwen_owns_hook)

    if pair_owns_hook and not qwen_owns_hook:
        print("  layout: REFACTORED / intended current layout")
    elif (not pair_owns_hook) and qwen_owns_hook:
        print("  layout: LEGACY but functionally compatible")
        print(
            "  note: update pair.py + qwen3vl_backbone.py when ready; "
            "this smoke test can still validate the functional chain."
        )
    elif pair_owns_hook and qwen_owns_hook:
        print("  WARNING: injection logic exists in BOTH modules.")
    else:
        print("  WARNING: no point-injection hook was found.")

    print()


def main():
    print("=" * 82)
    print("PAIR FULL MODEL INTEGRATION SMOKE TEST")
    print("=" * 82)

    assert os.path.isdir(MODEL_DIR), (
        f"Qwen checkpoint not found: {MODEL_DIR}"
    )

    assert torch.cuda.is_available(), "CUDA is required."

    device = torch.device("cuda")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))
    print("Qwen checkpoint:", MODEL_DIR)
    print()

    print_architecture_ownership()

    # ==================================================================
    # 1. Build the actual PAIR modules
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

    print(f"  Qwen hidden size: {qwen.hidden_size}")
    print(f"  image token id: {qwen.image_token_id}")
    print(f"  point token id: {qwen.point_token_id}")
    print(f"  task token id: {qwen.task_token_id}")
    print(f"  load time: {time.perf_counter() - t0:.2f} s")
    print()

    print("[2] Building PTv3...")

    ptv3_cfg = PTv3Config(
        in_channels=POINT_IN_CHANNELS,
    )

    point_encoder = PTv3Backbone(
        ptv3_cfg
    ).to(device)

    assert point_encoder.output_dim == 64

    print("  PTv3 output dim:", point_encoder.output_dim)
    print(
        "  PTv3 params:",
        f"{point_encoder.parameter_count() / 1e6:.3f} M",
    )
    print()

    print("[3] Building PointAdapter...")

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
        f"  adapter: {adapter_cfg.in_dim} -> "
        f"{adapter_cfg.out_dim}"
    )
    print("  point tokens:", adapter_cfg.num_tokens)
    print(
        "  adapter params:",
        f"{point_adapter.parameter_count() / 1e6:.3f} M",
    )
    print()

    print("[4] Building top-level PAIRModel...")

    model = PAIRModel(
        qwen_backbone=qwen,
        point_encoder=point_encoder,
        point_adapter=point_adapter,
    )

    model.eval()

    print("  PAIRModel built.")
    print()

    # ==================================================================
    # 2. Inputs
    # ==================================================================
    print("[5] Building synthetic inputs...")

    image = make_test_image()
    point_dict = make_synthetic_point_cloud(device)

    print("  image size:", image.size)
    print("  point coord:", tuple(point_dict["coord"].shape))
    print("  point feat:", tuple(point_dict["feat"].shape))

    assert point_dict["coord"].shape == (NUM_POINTS, 3)
    assert point_dict["feat"].shape == (
        NUM_POINTS,
        POINT_IN_CHANNELS,
    )

    print("  SYNTHETIC INPUTS OK")
    print()

    # ==================================================================
    # 3. 2D + 3D main joint forward
    # ==================================================================
    print("[6] Running PAIR 2D+3D forward...")

    joint_prompt = (
        "Analyze the image and the 3D point-cloud information together. "
        "Briefly summarize the multimodal scene."
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        joint = model(
            task_mode="2d3d",
            prompt=joint_prompt,
            images=image,
            point_dict=point_dict,
            return_logits=False,
            return_hidden_states=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  joint forward time: {elapsed:.3f} s")

    tensor_summary("image_hidden", joint.image_hidden)
    tensor_summary("point_hidden", joint.point_hidden)
    tensor_summary("task_hidden", joint.task_hidden)

    assert_finite("joint.image_hidden", joint.image_hidden)
    assert_finite("joint.point_hidden", joint.point_hidden)
    assert_finite("joint.task_hidden", joint.task_hidden)

    assert joint.image_hidden.ndim == 2
    assert joint.image_hidden.shape[-1] == qwen.hidden_size

    assert joint.point_hidden.shape == (
        NUM_POINT_TOKENS,
        qwen.hidden_size,
    )

    assert joint.task_hidden.shape == (
        1,
        qwen.hidden_size,
    )

    injection_replaced = get_aux(
        joint,
        "point_injection_replaced",
        None,
    )

    injection_calls = get_aux(
        joint,
        "point_injection_calls",
        None,
    )

    print("  point injection calls:", injection_calls)
    print("  point injection replaced:", injection_replaced)

    assert injection_replaced is True, (
        "PTv3-derived point tokens were not injected into Qwen."
    )

    image_hidden_2d = get_aux(
        joint,
        "image_hidden_2d",
        None,
    )

    if image_hidden_2d is not None:
        tensor_summary("image_hidden_2d", image_hidden_2d)

        assert image_hidden_2d.shape[-1] == qwen.hidden_size

        flattened = (
            image_hidden_2d.shape[0]
            * image_hidden_2d.shape[1]
            * image_hidden_2d.shape[2]
        )

        assert flattened == joint.image_hidden.shape[0]

    point_encoded = joint.aux.get("point_encoded")
    point_tokens_aux = joint.aux.get("point_tokens")

    assert point_encoded is not None
    assert point_tokens_aux is not None

    dense_features = getattr(
        point_encoded,
        "features",
        None,
    )

    assert dense_features is not None
    assert dense_features.shape == (NUM_POINTS, 64)

    print("  PTv3 dense features:", tuple(dense_features.shape))
    print("  adapter/Qwen point tokens:", tuple(point_tokens_aux.shape))
    print("  2D+3D FORWARD OK")
    print()

    # ==================================================================
    # 4. 2D-only routing
    # ==================================================================
    print("[7] Running PAIR 2D-only routing...")

    with torch.inference_mode():
        out_2d = model(
            task_mode="2d",
            prompt="Describe the image briefly.",
            images=image,
            return_logits=False,
            return_hidden_states=True,
        )

    assert_finite("2d.image_hidden", out_2d.image_hidden)
    assert_finite("2d.task_hidden", out_2d.task_hidden)

    assert out_2d.point_hidden is None
    assert out_2d.image_hidden.shape[-1] == qwen.hidden_size
    assert out_2d.task_hidden.shape == (1, qwen.hidden_size)

    tensor_summary("2d image_hidden", out_2d.image_hidden)
    tensor_summary("2d task_hidden", out_2d.task_hidden)

    print("  2D ROUTING OK")
    print()

    # ==================================================================
    # 5. 3D-only routing
    # ==================================================================
    print("[8] Running PAIR 3D-only routing...")

    with torch.inference_mode():
        out_3d = model(
            task_mode="3d",
            prompt="Analyze the provided 3D point-cloud information.",
            point_dict=point_dict,
            return_logits=False,
            return_hidden_states=True,
        )

    assert out_3d.image_hidden is None

    assert_finite("3d.point_hidden", out_3d.point_hidden)
    assert_finite("3d.task_hidden", out_3d.task_hidden)

    assert out_3d.point_hidden.shape == (
        NUM_POINT_TOKENS,
        qwen.hidden_size,
    )

    assert out_3d.task_hidden.shape == (
        1,
        qwen.hidden_size,
    )

    assert get_aux(
        out_3d,
        "point_injection_replaced",
        None,
    ) is True

    tensor_summary("3d point_hidden", out_3d.point_hidden)
    tensor_summary("3d task_hidden", out_3d.task_hidden)

    print("  3D ROUTING OK")
    print()

    # ==================================================================
    # 6. Sanity: adding 3D context changes the task representation
    # ==================================================================
    print("[9] Multimodal task-hidden sanity check...")

    task_delta_2d_vs_joint = (
        out_2d.task_hidden.float()
        - joint.task_hidden.float()
    ).abs().mean().item()

    print(
        "  mean |TASK(2D) - TASK(2D+3D)|:",
        task_delta_2d_vs_joint,
    )

    assert task_delta_2d_vs_joint > 0.0

    print("  MULTIMODAL TASK REPRESENTATION CHANGED OK")
    print()

    # ==================================================================
    # 7. Native language generation through the top-level PAIR model
    # ==================================================================
    print("[10] Running PAIR 2D+3D generate()...")

    with torch.inference_mode():
        generated = model.generate(
            task_mode="2d3d",
            prompt=joint_prompt,
            images=image,
            point_dict=point_dict,
            max_new_tokens=32,
            do_sample=False,
        )

    print("  generated text:")
    print(" ", generated.generated_text)

    assert generated.generated_ids is not None
    assert isinstance(generated.generated_text, str)

    generation_injected = get_aux(
        generated,
        "point_injection_replaced",
        None,
    )

    print(
        "  generation point injection replaced:",
        generation_injected,
    )

    assert generation_injected is True

    print("  PAIR GENERATION OK")
    print()

    # ==================================================================
    # Final report
    # ==================================================================
    torch.cuda.synchronize()

    peak_allocated = gib(
        torch.cuda.max_memory_allocated()
    )
    peak_reserved = gib(
        torch.cuda.max_memory_reserved()
    )

    print("=" * 82)
    print("SUCCESS")
    print("=" * 82)
    print("PAIR model chain is connected:")
    print()
    print("  Image -> Qwen3-VL Vision ------------------------\\")
    print("                                                    -> Qwen LLM")
    print("  Point -> PTv3 -> PointAdapter -> <POINT> --------/")
    print("  Prompt + <TASK> ---------------------------------/")
    print()
    print(
        "  PTv3 dense features:       ",
        tuple(dense_features.shape),
    )
    print(
        "  contextual image hidden:  ",
        tuple(joint.image_hidden.shape),
    )
    print(
        "  contextual point hidden:  ",
        tuple(joint.point_hidden.shape),
    )
    print(
        "  contextual task hidden:   ",
        tuple(joint.task_hidden.shape),
    )
    print()
    print(f"Peak allocated GPU memory: {peak_allocated:.2f} GiB")
    print(f"Peak reserved GPU memory:  {peak_reserved:.2f} GiB")
    print()
    print("2D routing:       PASS")
    print("3D routing:       PASS")
    print("2D+3D routing:    PASS")
    print("Point injection:  PASS")
    print("Text generation:  PASS")


if __name__ == "__main__":
    main()
