"""
LoRA helpers for Qwen3-VL inside PAIR.

Default target modules follow the official Qwen3-VL fine-tuning example:
    q_proj, k_proj, v_proj, o_proj

Base Qwen/ViT weights stay frozen; only LoRA parameters are trainable.
"""

from __future__ import annotations

from typing import Sequence


def apply_qwen_lora(qwen_backbone, r=16, alpha=32, dropout=0.05,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA requires PEFT. Install it with: pip install peft") from exc

    qwen_backbone.freeze()
    config = LoraConfig(
        r=int(r), lora_alpha=int(alpha), lora_dropout=float(dropout),
        target_modules=list(target_modules), bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    qwen_backbone.model = get_peft_model(qwen_backbone.model, config)
    return qwen_backbone.model


def is_peft_model(model):
    return hasattr(model, "peft_config")


def lora_state_dict(model):
    if not is_peft_model(model):
        return {}
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}


def load_lora_state_dict(model, state):
    if not state:
        return
    if not is_peft_model(model):
        raise RuntimeError("Checkpoint contains LoRA weights but current Qwen is not a PEFT model")
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state)


def lora_parameter_count(model):
    total = trainable = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return trainable, total
