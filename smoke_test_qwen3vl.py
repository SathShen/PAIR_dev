import os
import torch
from PIL import Image, ImageDraw

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_DIR = "/data2/sht/checkpoints/Qwen/Qwen3-VL-4B-Instruct"


def make_test_image(size=448):
    img = Image.new("RGB", (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    draw.rectangle((40, 60, 180, 220), outline=(220, 50, 50), width=5)
    draw.rectangle((240, 120, 390, 300), outline=(50, 120, 220), width=5)
    draw.ellipse((130, 260, 220, 350), outline=(60, 180, 60), width=5)
    draw.line((0, size - 40, size, size - 40), fill=(80, 80, 80), width=6)

    draw.text((55, 25), "PAIR smoke test", fill=(0, 0, 0))
    return img


def main():
    print("=" * 72)
    print("QWEN3-VL SMOKE TEST")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model dir:", MODEL_DIR)
    print()

    assert os.path.isdir(MODEL_DIR), f"Model dir not found: {MODEL_DIR}"

    print("[1] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    print("Processor loaded.")
    print()

    print("[2] Loading Qwen3-VL model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    print("Model loaded.")
    print()

    # Try to print useful config info
    hidden_size = None
    vision_hidden_size = None
    vision_out_hidden_size = None

    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size

    if hasattr(model.config, "vision_config"):
        if hasattr(model.config.vision_config, "hidden_size"):
            vision_hidden_size = model.config.vision_config.hidden_size
        if hasattr(model.config.vision_config, "out_hidden_size"):
            vision_out_hidden_size = model.config.vision_config.out_hidden_size

    print("Config summary:")
    print("  text hidden size:", hidden_size)
    print("  vision hidden size:", vision_hidden_size)
    print("  vision out hidden size:", vision_out_hidden_size)
    print("  image token id:", getattr(model.config, "image_token_id", None))
    print()

    print("[3] Building synthetic image...")
    image = make_test_image(448)
    print("Synthetic image size:", image.size)
    print()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]

    print("[4] Applying chat template...")
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Prompt preview:")
    print(prompt_text[:300])
    print()

    print("[5] Building model inputs...")
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

    image_grid_thw = inputs.get("image_grid_thw", None)
    if image_grid_thw is not None:
        print("  image_grid_thw:", image_grid_thw.tolist())

        # Estimate visual token count after spatial merge
        spatial_merge = 2
        if hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "spatial_merge_size"):
            spatial_merge = model.config.vision_config.spatial_merge_size

        t, h, w = image_grid_thw[0].tolist()
        estimated_visual_tokens = t * (h // spatial_merge) * (w // spatial_merge)
        print("  estimated visual token count:", estimated_visual_tokens)
    else:
        estimated_visual_tokens = None

    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    print()

    print("[6] Forward pass with hidden states...")
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    print()
    print("[6.1] Extract contextualized image tokens...")

    print()
    print("[6.2] Inspect model structure / DeepStack...")

    print("  vision config deepstack indexes:",
        getattr(model.config.vision_config, "deepstack_visual_indexes", None))

    print("  model.model type:",
        type(model.model).__name__ if hasattr(model, "model") else None)

    if hasattr(model, "model") and hasattr(model.model, "visual"):
        print("  vision module:",
            type(model.model.visual).__name__)

    print("  output keys:",
        outputs.keys())

    image_token_id = model.config.image_token_id

    # input_ids 中哪些位置是 image token
    image_mask = inputs["input_ids"] == image_token_id

    num_image_tokens = image_mask.sum(dim=1)
    print("  image token id:", image_token_id)
    print("  image token count:", num_image_tokens.tolist())

    # Qwen 最后一层 hidden
    last_hidden = outputs.hidden_states[-1]
    print("  last hidden:", tuple(last_hidden.shape))

    # 当前 smoke test batch_size = 1
    image_hidden = last_hidden[0][image_mask[0]]

    print("  image hidden:", tuple(image_hidden.shape))

    # 检查数量
    assert image_hidden.shape[0] == estimated_visual_tokens, (
        f"Mismatch: hidden={image_hidden.shape[0]}, "
        f"expected={estimated_visual_tokens}"
    )

    # 恢复二维 spatial grid
    t, h_patch, w_patch = image_grid_thw[0].tolist()

    merge_size = model.config.vision_config.spatial_merge_size

    h_token = h_patch // merge_size
    w_token = w_patch // merge_size

    image_hidden_2d = image_hidden.reshape(
        t,
        h_token,
        w_token,
        image_hidden.shape[-1],
    )

    print("  spatial image feature:", tuple(image_hidden_2d.shape))

    assert torch.isfinite(image_hidden).all()
    assert torch.isfinite(image_hidden_2d).all()

    print("  IMAGE TOKEN EXTRACTION OK")

    print("Forward OK.")
    print("Output type:", type(outputs).__name__)

    if hasattr(outputs, "logits") and outputs.logits is not None:
        print("  logits shape:", tuple(outputs.logits.shape))

    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        print("  hidden_states layers:", len(outputs.hidden_states))
        print("  last hidden state shape:", tuple(outputs.hidden_states[-1].shape))

    input_len = inputs["input_ids"].shape[1]
    print("  input_ids seq len:", input_len)

    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        out_len = outputs.hidden_states[-1].shape[1]
        print("  output seq len:", out_len)

    print()

    print("[7] Generate text...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
    generated_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    print("Generated text:")
    print(generated_text)
    print()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak allocated GPU memory: {peak_mem:.2f} GB")

    print()
    print("=" * 72)
    print("QWEN3-VL SMOKE TEST SUCCESS")
    print("=" * 72)


if __name__ == "__main__":
    main()