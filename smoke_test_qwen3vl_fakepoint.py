import os
import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"
NUM_FAKE_POINTS = 32


def main():
    print("=" * 72)
    print("QWEN3-VL FAKE POINT EMBEDDING SMOKE TEST")
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
    # 2. Qwen3-VL
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

    hidden_size = model.config.text_config.hidden_size
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    print("Model loaded.")
    print("Text hidden size:", hidden_size)
    print("Model dtype:", model_dtype)
    print("Embedding size:", model.get_input_embeddings().weight.shape[0])
    print()

    # ------------------------------------------------------------------
    # 3. Build a PURE TEXT + POINT-PLACEHOLDER sequence.
    #
    # No image is used in this test.  We only test whether external
    # embeddings can replace <POINT> token embeddings and travel through
    # the Qwen3-VL language model.
    # ------------------------------------------------------------------
    print("[3] Building point-placeholder prompt...")

    point_placeholders = " ".join(["<POINT>"] * NUM_FAKE_POINTS)

    prompt_text = (
        "<|im_start|>user\n"
        "Analyze the following 3D point features.\n"
        f"{point_placeholders}\n"
        "Describe the scene briefly. <TASK>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print("Prompt preview:")
    print(prompt_text[:500])
    print()

    inputs = processor(
        text=[prompt_text],
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("input_ids:", tuple(input_ids.shape), input_ids.dtype)
    print("attention_mask:", tuple(attention_mask.shape), attention_mask.dtype)
    print()

    # ------------------------------------------------------------------
    # 4. Verify placeholders
    # ------------------------------------------------------------------
    print("[4] Checking <POINT> / <TASK> tokenization...")

    point_mask_cpu = input_ids == point_token_id
    task_mask_cpu = input_ids == task_token_id

    point_count = int(point_mask_cpu.sum().item())
    task_count = int(task_mask_cpu.sum().item())

    point_positions = torch.nonzero(
        point_mask_cpu[0], as_tuple=False
    ).flatten().tolist()

    task_positions = torch.nonzero(
        task_mask_cpu[0], as_tuple=False
    ).flatten().tolist()

    print("  <POINT> count:", point_count)
    print("  <TASK> count:", task_count)
    print("  <POINT> positions:", point_positions)
    print("  <TASK> position:", task_positions)

    assert point_count == NUM_FAKE_POINTS, (
        f"Expected {NUM_FAKE_POINTS} <POINT> tokens, got {point_count}"
    )
    assert task_count == 1, (
        f"Expected exactly one <TASK> token, got {task_count}"
    )

    print("  PLACEHOLDER TOKENIZATION OK")
    print()

    # ------------------------------------------------------------------
    # 5. Convert normal tokens to Qwen embeddings
    # ------------------------------------------------------------------
    print("[5] Creating external fake point embeddings...")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    point_mask = point_mask_cpu.to(device)
    task_mask = task_mask_cpu.to(device)

    with torch.inference_mode():
        baseline_embeds = model.get_input_embeddings()(input_ids)

    # Keep an untouched baseline for the sanity check later.
    baseline_embeds = baseline_embeds.clone()

    # Use the learned/random <POINT> embedding as the center, then perturb
    # every fake point slightly so that all 32 points are different.
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

    print("  baseline inputs_embeds:", tuple(baseline_embeds.shape))
    print("  fake point embeds:", tuple(fake_point_embeds.shape))
    print("  fake point dtype:", fake_point_embeds.dtype)

    assert fake_point_embeds.shape == (NUM_FAKE_POINTS, hidden_size)
    assert torch.isfinite(fake_point_embeds).all()

    # Replace ONLY the <POINT> token embeddings.
    injected_embeds = baseline_embeds.clone()
    injected_embeds[0, point_mask[0]] = fake_point_embeds

    assert torch.isfinite(injected_embeds).all()

    replacement_delta = (
        injected_embeds[0, point_mask[0]].float()
        - baseline_embeds[0, point_mask[0]].float()
    ).abs().mean()

    print(
        "  mean embedding replacement delta:",
        float(replacement_delta.item()),
    )
    print("  EXTERNAL POINT EMBEDDINGS INSERTED")
    print()

    # ------------------------------------------------------------------
    # 6. Forward with injected inputs_embeds
    #
    # IMPORTANT:
    # Qwen3-VL accepts either input_ids OR inputs_embeds, not both.
    # attention_mask is kept so Qwen can construct sequential position ids.
    # ------------------------------------------------------------------
    print("[6] Forward with injected fake point embeddings...")

    with torch.inference_mode():
        outputs = model(
            inputs_embeds=injected_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    last_hidden = outputs.hidden_states[-1]

    print("Forward OK.")
    print("  logits:", tuple(outputs.logits.shape))
    print("  hidden-state layers:", len(outputs.hidden_states))
    print("  last hidden:", tuple(last_hidden.shape))
    print()

    # ------------------------------------------------------------------
    # 7. Extract contextualized point / task hidden states
    # ------------------------------------------------------------------
    print("[7] Extracting contextualized point and task hidden states...")

    point_hidden = last_hidden[0][point_mask[0]]
    task_hidden = last_hidden[0][task_mask[0]]

    print("  contextualized point hidden:", tuple(point_hidden.shape))
    print("  task hidden:", tuple(task_hidden.shape))

    assert point_hidden.shape == (NUM_FAKE_POINTS, hidden_size)
    assert task_hidden.shape == (1, hidden_size)
    assert torch.isfinite(point_hidden).all()
    assert torch.isfinite(task_hidden).all()

    print("  POINT HIDDEN EXTRACTION OK")
    print("  TASK HIDDEN EXTRACTION OK")
    print()

    # ------------------------------------------------------------------
    # 8. Sanity check:
    #    run exactly the same sequence WITHOUT external point replacement.
    #    The <TASK> representation should change when fake point features
    #    are injected.
    # ------------------------------------------------------------------
    print("[8] Sanity check: do fake point features affect <TASK>?")

    with torch.inference_mode():
        baseline_outputs = model(
            inputs_embeds=baseline_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    baseline_last_hidden = baseline_outputs.hidden_states[-1]
    baseline_task_hidden = baseline_last_hidden[0][task_mask[0]]
    baseline_point_hidden = baseline_last_hidden[0][point_mask[0]]

    task_delta = (
        task_hidden.float() - baseline_task_hidden.float()
    ).abs().mean()

    point_delta = (
        point_hidden.float() - baseline_point_hidden.float()
    ).abs().mean()

    print("  mean |POINT hidden delta|:", float(point_delta.item()))
    print("  mean |TASK hidden delta|:", float(task_delta.item()))

    assert point_delta.item() > 0.0, (
        "Point hidden states did not change after external embedding injection."
    )
    assert task_delta.item() > 0.0, (
        "<TASK> hidden did not change after external point embedding injection."
    )

    print("  EXTERNAL POINT FEATURES AFFECT QWEN REPRESENTATION OK")
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
        "Fake external 3D embeddings -> Qwen3-VL -> "
        "contextualized point hidden + <TASK> hidden completed successfully."
    )


if __name__ == "__main__":
    main()
