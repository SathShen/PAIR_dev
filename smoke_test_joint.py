import os
import torch
from PIL import Image, ImageDraw

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"
NUM_FAKE_POINTS = 32


def make_test_image(size=448):
    img = Image.new("RGB", (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 60, 180, 220), outline=(220, 50, 50), width=5)
    draw.rectangle((240, 120, 390, 300), outline=(50, 120, 220), width=5)
    draw.ellipse((130, 260, 220, 350), outline=(60, 180, 60), width=5)
    draw.line((0, size - 40, size, size - 40), fill=(80, 80, 80), width=6)
    draw.text((55, 25), "PAIR smoke test", fill=(0, 0, 0))
    return img


def mean_abs_delta(a, b):
    return (a.float() - b.float()).abs().mean().item()


def main():
    print("=" * 72)
    print("QWEN3-VL FINAL JOINT INTERFACE SMOKE TEST")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model dir:", MODEL_DIR)
    print("Fake point tokens:", NUM_FAKE_POINTS)
    print()

    assert os.path.isdir(MODEL_DIR), f"Model dir not found: {MODEL_DIR}"

    # ------------------------------------------------------------------
    # 1. Processor / tokenizer
    # ------------------------------------------------------------------
    print("[1] Loading processor...")
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
    print("Added special tokens:", num_added)
    print("<POINT> token id:", point_token_id)
    print("<TASK> token id:", task_token_id)
    print("Tokenizer vocab size:", len(tokenizer))
    print()

    # ------------------------------------------------------------------
    # 2. Load native Qwen3-VL
    # ------------------------------------------------------------------
    print("[2] Loading Qwen3-VL model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    hidden_size = model.config.text_config.hidden_size
    image_token_id = model.config.image_token_id

    print("Model loaded.")
    print("Text hidden size:", hidden_size)
    print("Model dtype:", model_dtype)
    print("Image token id:", image_token_id)
    print("Embedding size:", model.get_input_embeddings().weight.shape[0])
    print()

    # ------------------------------------------------------------------
    # 3. Build ONE native multimodal sequence:
    #    IMAGE + contiguous POINT placeholders + TASK
    # ------------------------------------------------------------------
    print("[3] Building joint image + point + task input...")

    image = make_test_image(448)

    # No spaces: point placeholder positions should be contiguous.
    point_placeholders = "".join(["<POINT>"] * NUM_FAKE_POINTS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Analyze the image and the following 3D point features.\n"
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

    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    input_ids_cpu = inputs["input_ids"]
    image_grid_thw = inputs["image_grid_thw"]

    print("  image_grid_thw:", image_grid_thw.tolist())
    print()

    # ------------------------------------------------------------------
    # 4. Verify token layout
    # ------------------------------------------------------------------
    print("[4] Checking token layout...")

    image_mask_cpu = input_ids_cpu == image_token_id
    point_mask_cpu = input_ids_cpu == point_token_id
    task_mask_cpu = input_ids_cpu == task_token_id

    image_count = int(image_mask_cpu.sum().item())
    point_count = int(point_mask_cpu.sum().item())
    task_count = int(task_mask_cpu.sum().item())

    image_positions = torch.nonzero(
        image_mask_cpu[0], as_tuple=False
    ).flatten().tolist()

    point_positions = torch.nonzero(
        point_mask_cpu[0], as_tuple=False
    ).flatten().tolist()

    task_positions = torch.nonzero(
        task_mask_cpu[0], as_tuple=False
    ).flatten().tolist()

    print("  image token count:", image_count)
    print("  <POINT> count:", point_count)
    print("  <TASK> count:", task_count)
    print("  first 10 image positions:", image_positions[:10], "...")
    print("  <POINT> positions:", point_positions)
    print("  <TASK> position:", task_positions)

    assert image_count > 0
    assert point_count == NUM_FAKE_POINTS
    assert task_count == 1

    expected_point_positions = list(
        range(point_positions[0], point_positions[0] + NUM_FAKE_POINTS)
    )
    assert point_positions == expected_point_positions, (
        "Point placeholder block is not contiguous."
    )

    # Check image token count from grid + spatial merge.
    t, h_patch, w_patch = image_grid_thw[0].tolist()
    merge = model.config.vision_config.spatial_merge_size
    expected_image_count = t * (h_patch // merge) * (w_patch // merge)

    print("  expected image tokens from grid:", expected_image_count)

    assert image_count == expected_image_count, (
        f"Image token mismatch: {image_count} vs {expected_image_count}"
    )

    print("  TOKEN LAYOUT OK")
    print()

    # ------------------------------------------------------------------
    # 5. Move native processor outputs to GPU.
    #
    # We KEEP pixel_values/image_grid_thw and let Qwen3-VL itself execute
    # its normal image path, including its native visual processing.
    # ------------------------------------------------------------------
    gpu_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    point_mask = point_mask_cpu.to(device)
    image_mask = image_mask_cpu.to(device)
    task_mask = task_mask_cpu.to(device)

    # ------------------------------------------------------------------
    # 6. Native baseline:
    #    image is real, <POINT> still uses its ordinary token embedding.
    # ------------------------------------------------------------------
    print("[5] Running native baseline forward...")

    with torch.inference_mode():
        baseline_outputs = model(
            **gpu_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    baseline_last = baseline_outputs.hidden_states[-1]

    baseline_image_hidden = baseline_last[0][image_mask[0]]
    baseline_point_hidden = baseline_last[0][point_mask[0]]
    baseline_task_hidden = baseline_last[0][task_mask[0]]

    print("  baseline last hidden:", tuple(baseline_last.shape))
    print("  baseline image hidden:", tuple(baseline_image_hidden.shape))
    print("  baseline point hidden:", tuple(baseline_point_hidden.shape))
    print("  baseline task hidden:", tuple(baseline_task_hidden.shape))

    assert baseline_image_hidden.shape == (image_count, hidden_size)
    assert baseline_point_hidden.shape == (NUM_FAKE_POINTS, hidden_size)
    assert baseline_task_hidden.shape == (1, hidden_size)

    print("  NATIVE IMAGE PATH BASELINE OK")
    print()

    # ------------------------------------------------------------------
    # 7. Create external fake point embeddings.
    #
    # Center them around the <POINT> embedding only to keep their numerical
    # scale reasonable. Later this tensor will be replaced by:
    #
    #     PTv3 -> PointAdapter -> [N, 2560]
    # ------------------------------------------------------------------
    print("[6] Creating fake external point embeddings...")

    with torch.inference_mode():
        point_base = (
            model.get_input_embeddings()
            .weight[point_token_id]
            .detach()
            .clone()
        )

        fake_point_embeds = (
            point_base.unsqueeze(0)
            .repeat(NUM_FAKE_POINTS, 1)
        )

        fake_point_embeds = (
            fake_point_embeds
            + 0.01 * torch.randn_like(fake_point_embeds)
        )

    fake_point_embeds = fake_point_embeds.to(
        device=device,
        dtype=model_dtype,
    )

    print("  fake point embeds:", tuple(fake_point_embeds.shape))
    print("  fake point dtype:", fake_point_embeds.dtype)

    assert fake_point_embeds.shape == (NUM_FAKE_POINTS, hidden_size)
    assert torch.isfinite(fake_point_embeds).all()

    print("  FAKE POINT EMBEDDINGS READY")
    print()

    # ------------------------------------------------------------------
    # 8. Install a TEMPORARY embedding hook.
    #
    # Important:
    # We do NOT manually reconstruct the image embeddings.
    # We let Qwen3-VL perform its native image pathway.
    #
    # The hook modifies only the initial full-sequence embedding output:
    #     <POINT> positions -> external fake point embeddings
    #
    # All scalar/small internal embedding lookups are left untouched.
    # ------------------------------------------------------------------
    print("[7] Installing temporary <POINT> embedding injection hook...")

    full_seq_len = gpu_inputs["input_ids"].shape[1]

    hook_stats = {
        "replaced": False,
        "calls": 0,
    }

    def inject_point_hook(module, args, output):
        hook_stats["calls"] += 1

        # Only modify the normal initial [B, L, D] embedding tensor.
        if (
            torch.is_tensor(output)
            and output.dim() == 3
            and output.shape[0] == 1
            and output.shape[1] == full_seq_len
            and output.shape[2] == hidden_size
        ):
            out = output.clone()
            out[0, point_mask[0]] = fake_point_embeds.to(
                device=out.device,
                dtype=out.dtype,
            )
            hook_stats["replaced"] = True
            return out

        return output

    embedding_module = model.get_input_embeddings()
    hook_handle = embedding_module.register_forward_hook(inject_point_hook)

    try:
        # --------------------------------------------------------------
        # 9. Joint forward:
        #    Native IMAGE path + external POINT embeddings + TASK
        # --------------------------------------------------------------
        print("[8] Running JOINT forward...")

        with torch.inference_mode():
            joint_outputs = model(
                **gpu_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

    finally:
        # Never leave the hook installed after the test.
        hook_handle.remove()

    assert hook_stats["replaced"], (
        "The embedding hook never replaced the <POINT> positions."
    )

    joint_last = joint_outputs.hidden_states[-1]

    image_hidden = joint_last[0][image_mask[0]]
    point_hidden = joint_last[0][point_mask[0]]
    task_hidden = joint_last[0][task_mask[0]]

    print("  embedding hook calls:", hook_stats["calls"])
    print("  point replacement happened:", hook_stats["replaced"])
    print("  joint last hidden:", tuple(joint_last.shape))
    print("  image hidden:", tuple(image_hidden.shape))
    print("  point hidden:", tuple(point_hidden.shape))
    print("  task hidden:", tuple(task_hidden.shape))

    assert image_hidden.shape == (image_count, hidden_size)
    assert point_hidden.shape == (NUM_FAKE_POINTS, hidden_size)
    assert task_hidden.shape == (1, hidden_size)

    assert torch.isfinite(image_hidden).all()
    assert torch.isfinite(point_hidden).all()
    assert torch.isfinite(task_hidden).all()

    # Restore image grid representation.
    h_token = h_patch // merge
    w_token = w_patch // merge

    image_hidden_2d = image_hidden.reshape(
        t,
        h_token,
        w_token,
        hidden_size,
    )

    print("  spatial image hidden:", tuple(image_hidden_2d.shape))
    print("  JOINT HIDDEN EXTRACTION OK")
    print()

    # ------------------------------------------------------------------
    # 10. Sanity check:
    #     external point features must affect point and TASK states.
    #
    # Under causal attention, image tokens occur BEFORE point tokens.
    # Therefore later point tokens are NOT expected to modify earlier
    # image-token states. That is normal.
    # ------------------------------------------------------------------
    print("[9] Sanity check: do external points affect Qwen / <TASK>?")

    point_delta = mean_abs_delta(
        point_hidden,
        baseline_point_hidden,
    )

    task_delta = mean_abs_delta(
        task_hidden,
        baseline_task_hidden,
    )

    image_delta = mean_abs_delta(
        image_hidden,
        baseline_image_hidden,
    )

    print("  mean |POINT hidden delta|:", point_delta)
    print("  mean |TASK hidden delta|:", task_delta)
    print("  mean |IMAGE hidden delta|:", image_delta)
    print("  (IMAGE delta may be ~0 because image tokens precede point tokens.)")

    assert point_delta > 0.0, (
        "External point embeddings did not change point hidden states."
    )

    assert task_delta > 0.0, (
        "External point embeddings did not change <TASK> hidden state."
    )

    print("  EXTERNAL POINT FEATURES AFFECT JOINT REPRESENTATION OK")
    print()

    # ------------------------------------------------------------------
    # 11. Optional generation smoke test with the same injection.
    #
    # The hook only replaces embeddings on the initial full sequence.
    # During autoregressive decoding, one-token embedding calls are ignored.
    # ------------------------------------------------------------------
    print("[10] Joint generate() smoke test...")

    gen_hook_stats = {"replaced": False}

    def generation_point_hook(module, args, output):
        if (
            torch.is_tensor(output)
            and output.dim() == 3
            and output.shape[0] == 1
            and output.shape[1] == full_seq_len
            and output.shape[2] == hidden_size
        ):
            out = output.clone()
            out[0, point_mask[0]] = fake_point_embeds.to(
                device=out.device,
                dtype=out.dtype,
            )
            gen_hook_stats["replaced"] = True
            return out
        return output

    gen_handle = embedding_module.register_forward_hook(generation_point_hook)

    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **gpu_inputs,
                max_new_tokens=32,
                do_sample=False,
            )
    finally:
        gen_handle.remove()

    new_tokens = generated_ids[:, gpu_inputs["input_ids"].shape[1]:]

    generated_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    print("  generation point replacement happened:", gen_hook_stats["replaced"])
    print("  Generated text:")
    print(" ", generated_text)

    assert gen_hook_stats["replaced"], (
        "Point embeddings were not injected during the generation prefill."
    )

    print("  JOINT GENERATION OK")
    print()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak allocated GPU memory: {peak_mem:.2f} GB")
        print()

    print("=" * 72)
    print("SUCCESS")
    print("=" * 72)
    print(
        "Native Qwen3-VL image pathway + external fake 3D embeddings + "
        "<TASK> + text generation completed successfully."
    )


if __name__ == "__main__":
    main()
