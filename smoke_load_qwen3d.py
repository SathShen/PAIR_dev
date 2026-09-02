import os
import torch

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from qwen3d import add_maskformer2_config, add_maskformer2_video_config


def main():
    print("=" * 70)
    print("Qwen-3D smoke load test")
    print("=" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # 1. Build official Qwen-3D config
    # --------------------------------------------------
    cfg = get_cfg()

    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)

    cfg.merge_from_file("qwen3d/configs/qwen_3d.yaml")

    cfg.defrost()

    # Important: use 3B backbone to match qwen3d_3b.pth
    cfg.QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    cfg.MODEL.WEIGHTS = "ckpts/qwen3d_3b.pth"

    # Don't connect to author's wandb project
    cfg.USE_WANDB = False

    # Local output folder only
    cfg.OUTPUT_DIR = "./output/smoke_load"

    # Single GPU
    cfg.MODEL.DEVICE = "cuda"

    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\nConfig:")
    print("  META_ARCHITECTURE:", cfg.MODEL.META_ARCHITECTURE)
    print("  QWEN_MODEL:", cfg.QWEN_MODEL)
    print("  WEIGHTS:", cfg.MODEL.WEIGHTS)
    print("  USE_LORA:", cfg.USE_LORA)
    print("  LORA_RANK:", cfg.LORA_RANK)
    print("  ROPE_TYPE:", cfg.ROPE_TYPE)

    # --------------------------------------------------
    # 2. Construct complete Qwen-3D architecture
    # --------------------------------------------------
    print("\n[1/2] Building Qwen-3D model...")

    model = build_model(cfg)

    print("Model built successfully.")
    print("Model class:", type(model).__name__)

    # --------------------------------------------------
    # 3. Load official fine-tuned Qwen-3D checkpoint
    # --------------------------------------------------
    print("\n[2/2] Loading Qwen-3D checkpoint...")

    checkpointer = DetectionCheckpointer(
        model,
        save_dir=cfg.OUTPUT_DIR,
    )

    checkpoint_info = checkpointer.load(cfg.MODEL.WEIGHTS)

    print("\nCheckpoint loaded successfully.")

    # --------------------------------------------------
    # Model information
    # --------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\nModel statistics:")
    print(f"  Total params:     {total_params / 1e9:.3f} B")
    print(f"  Trainable params: {trainable_params / 1e6:.3f} M")
    print(
        f"  Trainable ratio:  "
        f"{100 * trainable_params / total_params:.4f}%"
    )

    print("\nImportant modules:")
    print("  qwen_model:", type(model.qwen_model).__name__)
    print("  visual:", type(model.visual).__name__)
    print("  connector:", model.connector)
    print("  text_connector:", model.text_connector)
    print("  mask_decoder:", type(model.mask_decoder).__name__)

    model.eval()

    print("\n" + "=" * 70)
    print("QWEN-3D MODEL + CHECKPOINT LOAD OK")
    print("=" * 70)


if __name__ == "__main__":
    main()