import os
import time
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from ptv3.model import PointTransformerV3


MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"

NUM_SIDE = 16                 # 16^3 = 4096 synthetic points
NUM_POINTS = NUM_SIDE ** 3
PTV3_IN_CHANNELS = 6
PTV3_OUT_CHANNELS = 64
NUM_POINT_TOKENS = 32
PATCH_SIZE = 128


class PointAdapter(nn.Module):
    """
    Minimal V1 adapter for the smoke test only:
        PTv3 feature [N, 64]
            -> Linear(64, 2560)
            -> LayerNorm
            -> Qwen point token [N, 2560]
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


def make_test_image(size=448):
    img = Image.new("RGB", (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    draw.rectangle((40, 60, 180, 220), outline=(220, 50, 50), width=5)
    draw.rectangle((240, 120, 390, 300), outline=(50, 120, 220), width=5)
    draw.ellipse((130, 260, 220, 350), outline=(60, 180, 60), width=5)
    draw.line((0, size - 40, size, size - 40), fill=(80, 80, 80), width=6)
    draw.text((55, 25), "PAIR PTv3-Qwen test", fill=(0, 0, 0))

    return img


def make_synthetic_point_cloud(device):
    """
    Deterministic 16x16x16 grid = 4096 points.

    Input feature:
      normalized XYZ (3)
      + 3 synthetic attributes (3)
      = 6 channels
    """
    axis = torch.arange(NUM_SIDE, dtype=torch.int32)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")

    grid_coord = torch.stack(
        [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)],
        dim=1,
    ).contiguous()

    coord = grid_coord.float() * 0.05

    denom = max(float(coord.max().item()), 1e-6)
    xyz_norm = coord / denom

    x, y, z = xyz_norm.unbind(dim=1)
    extra = torch.stack(
        [
            torch.sin(x * 3.14159265),
            torch.cos(y * 3.14159265),
            z,
        ],
        dim=1,
    )

    feat = torch.cat([xyz_norm, extra], dim=1).float()
    batch = torch.zeros(NUM_POINTS, dtype=torch.long)

    return {
        "coord": coord.to(device),
        "grid_coord": grid_coord.to(device),
        "feat": feat.to(device),
        "batch": batch.to(device),
    }


def build_ptv3(device):
    model = PointTransformerV3(
        in_channels=PTV3_IN_CHANNELS,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(PATCH_SIZE,) * 5,
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(PATCH_SIZE,) * 4,
        drop_path=0.0,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
    ).to(device)

    model.eval()
    return model


def uniform_resample(feat, num_tokens):
    """
    Very simple smoke-test resampler.
    No FPS / learned resampler yet.

    feat: [N, C]
    return:
        sampled_feat: [num_tokens, C]
        sampled_idx:  [num_tokens]
    """
    assert feat.ndim == 2
    assert feat.shape[0] >= num_tokens

    idx = torch.linspace(
        0,
        feat.shape[0] - 1,
        num_tokens,
        device=feat.device,
    ).long()

    return feat[idx], idx


def mean_abs_delta(a, b):
    return (a.float() - b.float()).abs().mean().item()


def main():
    print("=" * 78)
    print("PTv3 -> PointAdapter -> Qwen3-VL JOINT SMOKE TEST")
    print("=" * 78)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    assert torch.cuda.is_available(), "CUDA is required for this smoke test."
    assert os.path.isdir(MODEL_DIR), f"Model directory not found: {MODEL_DIR}"

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(device))
    print("Qwen model:", MODEL_DIR)
    print("Synthetic PTv3 points:", NUM_POINTS)
    print("Qwen point tokens:", NUM_POINT_TOKENS)
    print()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()

    # ==================================================================
    # 1. Load processor + register PAIR special tokens
    # ==================================================================
    print("[1] Loading Qwen3-VL processor / tokenizer...")

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer

    num_added = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<POINT>",
                "<TASK>",
            ]
        }
    )

    point_token_id = tokenizer.convert_tokens_to_ids("<POINT>")
    task_token_id = tokenizer.convert_tokens_to_ids("<TASK>")

    print("Processor loaded.")
    print("  added special tokens:", num_added)
    print("  <POINT> token id:", point_token_id)
    print("  <TASK> token id:", task_token_id)
    print("  tokenizer size:", len(tokenizer))
    print()

    # ==================================================================
    # 2. Load Qwen3-VL
    # ==================================================================
    print("[2] Loading Qwen3-VL...")

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    qwen.resize_token_embeddings(len(tokenizer))
    qwen.eval()

    qwen_hidden = qwen.config.text_config.hidden_size
    image_token_id = qwen.config.image_token_id
    qwen_dtype = next(qwen.parameters()).dtype

    print("Qwen loaded.")
    print("  text hidden size:", qwen_hidden)
    print("  Qwen dtype:", qwen_dtype)
    print("  image token id:", image_token_id)
    print()

    # ==================================================================
    # 3. Build PTv3 + minimal PointAdapter
    # ==================================================================
    print("[3] Building PTv3 + PointAdapter...")

    ptv3 = build_ptv3(device)

    point_adapter = PointAdapter(
        in_dim=PTV3_OUT_CHANNELS,
        out_dim=qwen_hidden,
    ).to(device=device, dtype=torch.float32)
    point_adapter.eval()

    ptv3_params = sum(p.numel() for p in ptv3.parameters())
    adapter_params = sum(p.numel() for p in point_adapter.parameters())

    print("PTv3 built.")
    print(f"  PTv3 parameters: {ptv3_params / 1e6:.3f} M")
    print(f"  PointAdapter parameters: {adapter_params / 1e6:.3f} M")
    print(f"  Adapter mapping: {PTV3_OUT_CHANNELS} -> {qwen_hidden}")
    print()

    # ==================================================================
    # 4. Synthetic point cloud -> PTv3
    # ==================================================================
    print("[4] Running synthetic point cloud through PTv3...")

    point_dict = make_synthetic_point_cloud(device)

    print("  input coord:", tuple(point_dict["coord"].shape))
    print("  input feat:", tuple(point_dict["feat"].shape))

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        point_output = ptv3(point_dict)

    torch.cuda.synchronize()
    ptv3_time = time.perf_counter() - t0

    dense_point_feat = point_output.feat

    print("  PTv3 output.feat:", tuple(dense_point_feat.shape))
    print("  PTv3 output dtype:", dense_point_feat.dtype)
    print(f"  PTv3 forward time: {ptv3_time:.3f} s")

    assert dense_point_feat.shape == (NUM_POINTS, PTV3_OUT_CHANNELS)
    assert torch.isfinite(dense_point_feat).all()

    print("  PTV3 FORWARD OK")
    print()

    # ==================================================================
    # 5. Resample 4096 dense features -> 32 point tokens
    # ==================================================================
    print("[5] Resampling PTv3 features...")

    sampled_feat, sampled_idx = uniform_resample(
        dense_point_feat,
        NUM_POINT_TOKENS,
    )

    sampled_coord = point_output.coord[sampled_idx]

    print("  sampled feature:", tuple(sampled_feat.shape))
    print("  sampled coord:", tuple(sampled_coord.shape))
    print("  first 10 sampled indices:", sampled_idx[:10].tolist())

    assert sampled_feat.shape == (NUM_POINT_TOKENS, PTV3_OUT_CHANNELS)
    assert sampled_coord.shape == (NUM_POINT_TOKENS, 3)

    print("  POINT RESAMPLING OK")
    print()

    # ==================================================================
    # 6. PointAdapter: 64 -> 2560
    # ==================================================================
    print("[6] Projecting PTv3 features into Qwen hidden space...")

    with torch.inference_mode():
        point_tokens_fp32 = point_adapter(sampled_feat.float())

    point_tokens = point_tokens_fp32.to(dtype=qwen_dtype)

    print("  adapter FP32 output:", tuple(point_tokens_fp32.shape), point_tokens_fp32.dtype)
    print("  Qwen point tokens:", tuple(point_tokens.shape), point_tokens.dtype)

    assert point_tokens.shape == (NUM_POINT_TOKENS, qwen_hidden)
    assert torch.isfinite(point_tokens).all()

    print("  POINT ADAPTER OK")
    print()

    # ==================================================================
    # 7. Build native Qwen image + point placeholders + task sequence
    # ==================================================================
    print("[7] Building Image + PTv3 Points + Prompt + <TASK> sequence...")

    image = make_test_image(448)
    point_placeholders = "".join(["<POINT>"] * NUM_POINT_TOKENS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Analyze the image together with the provided 3D point features.\n"
                        f"{point_placeholders}\n"
                        "Describe the scene briefly. <TASK>"
                    ),
                },
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Prompt preview:")
    print(prompt_text[:500])
    print()

    qwen_inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    input_ids_cpu = qwen_inputs["input_ids"]

    image_mask_cpu = input_ids_cpu == image_token_id
    point_mask_cpu = input_ids_cpu == point_token_id
    task_mask_cpu = input_ids_cpu == task_token_id

    image_count = int(image_mask_cpu.sum().item())
    point_count = int(point_mask_cpu.sum().item())
    task_count = int(task_mask_cpu.sum().item())

    point_positions = torch.nonzero(
        point_mask_cpu[0], as_tuple=False
    ).flatten().tolist()

    print("  sequence length:", input_ids_cpu.shape[1])
    print("  image token count:", image_count)
    print("  <POINT> count:", point_count)
    print("  <TASK> count:", task_count)
    print("  <POINT> positions:", point_positions)

    assert point_count == NUM_POINT_TOKENS
    assert task_count == 1

    expected_point_positions = list(
        range(point_positions[0], point_positions[0] + NUM_POINT_TOKENS)
    )
    assert point_positions == expected_point_positions

    image_grid_thw = qwen_inputs["image_grid_thw"]
    t, h_patch, w_patch = image_grid_thw[0].tolist()
    merge = qwen.config.vision_config.spatial_merge_size
    expected_image_count = t * (h_patch // merge) * (w_patch // merge)

    assert image_count == expected_image_count

    print("  TOKEN LAYOUT OK")
    print()

    # ==================================================================
    # 8. Move Qwen inputs to GPU
    # ==================================================================
    gpu_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in qwen_inputs.items()
    }

    image_mask = image_mask_cpu.to(device)
    point_mask = point_mask_cpu.to(device)
    task_mask = task_mask_cpu.to(device)

    # ==================================================================
    # 9. Native baseline:
    #    Real image, but ordinary <POINT> token embeddings
    # ==================================================================
    print("[8] Running native baseline Qwen forward...")

    with torch.inference_mode():
        baseline_outputs = qwen(
            **gpu_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    baseline_last = baseline_outputs.hidden_states[-1]
    baseline_point_hidden = baseline_last[0][point_mask[0]]
    baseline_task_hidden = baseline_last[0][task_mask[0]]

    print("  baseline last hidden:", tuple(baseline_last.shape))
    print("  baseline point hidden:", tuple(baseline_point_hidden.shape))
    print("  baseline task hidden:", tuple(baseline_task_hidden.shape))
    print("  BASELINE QWEN FORWARD OK")
    print()

    # ==================================================================
    # 10. Temporary embedding hook:
    #     replace <POINT> embeddings with REAL PTv3-derived point tokens
    # ==================================================================
    print("[9] Installing PTv3 -> Qwen point-token injection hook...")

    full_seq_len = gpu_inputs["input_ids"].shape[1]
    embedding_module = qwen.get_input_embeddings()

    hook_stats = {
        "calls": 0,
        "replaced": False,
    }

    def inject_ptv3_hook(module, args, output):
        hook_stats["calls"] += 1

        if (
            torch.is_tensor(output)
            and output.dim() == 3
            and output.shape[0] == 1
            and output.shape[1] == full_seq_len
            and output.shape[2] == qwen_hidden
        ):
            out = output.clone()

            out[0, point_mask[0]] = point_tokens.to(
                device=out.device,
                dtype=out.dtype,
            )

            hook_stats["replaced"] = True
            return out

        return output

    hook_handle = embedding_module.register_forward_hook(inject_ptv3_hook)

    try:
        print("[10] Running Qwen with PTv3-derived point tokens...")

        with torch.inference_mode():
            joint_outputs = qwen(
                **gpu_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

    finally:
        hook_handle.remove()

    assert hook_stats["replaced"], (
        "PTv3 point-token injection hook never replaced <POINT> embeddings."
    )

    joint_last = joint_outputs.hidden_states[-1]

    image_hidden = joint_last[0][image_mask[0]]
    point_hidden = joint_last[0][point_mask[0]]
    task_hidden = joint_last[0][task_mask[0]]

    print("  hook calls:", hook_stats["calls"])
    print("  PTv3 replacement happened:", hook_stats["replaced"])
    print("  image hidden:", tuple(image_hidden.shape))
    print("  point hidden:", tuple(point_hidden.shape))
    print("  task hidden:", tuple(task_hidden.shape))

    h_token = h_patch // merge
    w_token = w_patch // merge

    image_hidden_2d = image_hidden.reshape(
        t,
        h_token,
        w_token,
        qwen_hidden,
    )

    print("  spatial image hidden:", tuple(image_hidden_2d.shape))

    assert image_hidden.shape == (image_count, qwen_hidden)
    assert point_hidden.shape == (NUM_POINT_TOKENS, qwen_hidden)
    assert task_hidden.shape == (1, qwen_hidden)

    assert torch.isfinite(image_hidden).all()
    assert torch.isfinite(point_hidden).all()
    assert torch.isfinite(task_hidden).all()

    print("  JOINT HIDDEN EXTRACTION OK")
    print()

    # ==================================================================
    # 11. Sanity check:
    #     PTv3-derived tokens must change point + task representations
    # ==================================================================
    print("[11] Sanity check: do PTv3 features affect Qwen / <TASK>?")

    point_delta = mean_abs_delta(
        point_hidden,
        baseline_point_hidden,
    )

    task_delta = mean_abs_delta(
        task_hidden,
        baseline_task_hidden,
    )

    print("  mean |POINT hidden delta|:", point_delta)
    print("  mean |TASK hidden delta|:", task_delta)

    assert point_delta > 0.0
    assert task_delta > 0.0

    print("  PTV3 FEATURES AFFECT QWEN REPRESENTATION OK")
    print()

    # ==================================================================
    # 12. Joint generation with the same PTv3 point-token injection
    # ==================================================================
    print("[12] Joint generate() smoke test...")

    gen_stats = {"replaced": False}

    def generation_ptv3_hook(module, args, output):
        if (
            torch.is_tensor(output)
            and output.dim() == 3
            and output.shape[0] == 1
            and output.shape[1] == full_seq_len
            and output.shape[2] == qwen_hidden
        ):
            out = output.clone()

            out[0, point_mask[0]] = point_tokens.to(
                device=out.device,
                dtype=out.dtype,
            )

            gen_stats["replaced"] = True
            return out

        return output

    gen_handle = embedding_module.register_forward_hook(
        generation_ptv3_hook
    )

    try:
        with torch.inference_mode():
            generated_ids = qwen.generate(
                **gpu_inputs,
                max_new_tokens=32,
                do_sample=False,
            )
    finally:
        gen_handle.remove()

    new_tokens = generated_ids[
        :,
        gpu_inputs["input_ids"].shape[1]:
    ]

    generated_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    print("  generation PTv3 replacement happened:", gen_stats["replaced"])
    print("  Generated text:")
    print(" ", generated_text)

    assert gen_stats["replaced"]

    print("  JOINT GENERATION OK")
    print()

    # ==================================================================
    # 13. Final report
    # ==================================================================
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Peak allocated GPU memory: {peak_mem:.2f} GB")
    print()

    print("=" * 78)
    print("SUCCESS")
    print("=" * 78)
    print(
        "Synthetic point cloud -> PTv3 -> resampler -> PointAdapter -> "
        "Qwen3-VL <POINT> injection -> image/point/task hidden + text generation "
        "completed successfully."
    )
    print()
    print("Core tensor path:")
    print(f"  PTv3 dense feature:     {tuple(dense_point_feat.shape)}")
    print(f"  sampled point feature:  {tuple(sampled_feat.shape)}")
    print(f"  Qwen point tokens:      {tuple(point_tokens.shape)}")
    print(f"  contextual point hidden:{tuple(point_hidden.shape)}")
    print(f"  task hidden:            {tuple(task_hidden.shape)}")
    print(f"  image hidden:           {tuple(image_hidden.shape)}")


if __name__ == "__main__":
    main()
