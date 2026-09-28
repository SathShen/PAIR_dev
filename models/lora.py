"""
LoRA helpers for PAIR.

There are three independent adapter families:

1) Qwen LLM LoRA
   Implemented with PEFT and applied to text self-attention projections
   (q_proj/k_proj/v_proj/o_proj by default).

2) Qwen Vision LoRA
   Implemented with PAIRLoRALinear and applied only to each vision
   Transformer block's attention qkv/proj linears.

3) Utonia / PointTransformerV3 LoRA
   Implemented with PAIRLoRALinear and applied only to each
   SerializedAttention qkv/proj linear.

All pretrained base weights remain frozen.  The custom PAIR LoRA wrappers use
standard low-rank updates

    y = W x + (alpha / r) B A dropout(x)

with A Kaiming-initialized and B initialized to zero, so enabling an adapter
preserves the pretrained forward exactly at initialization.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PAIR_LORA_A = "pair_lora_A"
PAIR_LORA_B = "pair_lora_B"


class PAIRLoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank residual branch."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int,
        alpha: float,
        dropout: float,
        kind: str,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"PAIRLoRALinear expects nn.Linear, got {type(base).__name__}")
        if int(r) <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError(f"LoRA dropout must be in [0,1), got {dropout}")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout_p = float(dropout)
        self.kind = str(kind)

        # Base foundation-model weights are never updated.
        self.base.requires_grad_(False)

        # Keep custom adapters in FP32 for stable optimization.  The branch
        # output is cast back to the frozen base output dtype before addition.
        self.pair_lora_A = nn.Parameter(
            torch.empty(self.r, self.base.in_features, dtype=torch.float32)
        )
        self.pair_lora_B = nn.Parameter(
            torch.zeros(self.base.out_features, self.r, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.pair_lora_A, a=math.sqrt(5))

        # Base backbones are intentionally kept in eval mode.  This explicit
        # flag lets LoRA dropout still follow PAIR train/eval state without
        # turning foundation-model dropout / stochastic depth back on.
        self._pair_lora_training = True

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def set_lora_training(self, mode: bool) -> None:
        self._pair_lora_training = bool(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        lora_x = F.dropout(
            x,
            p=self.dropout_p,
            training=self._pair_lora_training and self.dropout_p > 0.0,
        )
        # Under CUDA autocast, keep FP32 master adapter parameters while
        # allowing GEMMs to use the active autocast dtype (BF16 in PAIR).
        # Without autocast, cast the input explicitly for dtype compatibility.
        if not torch.is_autocast_enabled():
            lora_x = lora_x.to(dtype=self.pair_lora_A.dtype)
        delta = F.linear(F.linear(lora_x, self.pair_lora_A), self.pair_lora_B)
        delta = delta.mul(self.scaling).to(dtype=base_out.dtype)
        return base_out + delta


def _resolve_attr_bfs(root: nn.Module, attr: str):
    queue = [root]
    seen = set()
    while queue:
        module = queue.pop(0)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        if hasattr(module, attr):
            return getattr(module, attr)
        for child_name in ("base_model", "model"):
            child = getattr(module, child_name, None)
            if isinstance(child, nn.Module) and child is not module:
                queue.append(child)
    raise AttributeError(f"Could not resolve submodule attribute {attr!r}")


def apply_qwen_lora(
    qwen_backbone,
    r=16,
    alpha=32,
    dropout=0.05,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
):
    """Apply PEFT LoRA to Qwen LLM attention modules."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA requires PEFT. Install it with: pip install peft") from exc

    qwen_backbone.freeze()
    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    qwen_backbone.model = get_peft_model(qwen_backbone.model, config)
    return qwen_backbone.model


def apply_qwen_vision_lora(
    qwen_backbone,
    *,
    r: int = 16,
    alpha: float = 32,
    dropout: float = 0.05,
) -> Dict[str, int]:
    """
    Add LoRA to Qwen3-VL Vision Transformer attention qkv/proj linears.

    The implementation deliberately walks vision.blocks[*].attn rather than
    matching generic names such as "proj", which would otherwise risk touching
    patch mergers or unrelated projections.
    """
    vision = _resolve_attr_bfs(qwen_backbone.model, "visual")
    blocks = getattr(vision, "blocks", None)
    if blocks is None:
        raise RuntimeError("Qwen vision module does not expose .blocks")

    wrapped = 0
    params = 0
    for block_idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            raise RuntimeError(f"Qwen vision block {block_idx} has no .attn module")

        for attr in ("qkv", "proj"):
            linear = getattr(attn, attr, None)
            if isinstance(linear, PAIRLoRALinear):
                if linear.kind != "vision":
                    raise RuntimeError(
                        f"Qwen vision block {block_idx}.{attr} already has incompatible LoRA kind"
                    )
                continue
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(
                    f"Qwen vision block {block_idx}.attn.{attr} expected nn.Linear, "
                    f"got {type(linear).__name__}"
                )
            wrapped_linear = PAIRLoRALinear(
                linear,
                r=r,
                alpha=alpha,
                dropout=dropout,
                kind="vision",
            ).to(device=linear.weight.device)
            setattr(attn, attr, wrapped_linear)
            wrapped += 1
            params += wrapped_linear.pair_lora_A.numel() + wrapped_linear.pair_lora_B.numel()

    if wrapped == 0:
        raise RuntimeError("Qwen Vision LoRA matched zero attention linears")
    return {"wrapped_linears": wrapped, "trainable_parameters": params}


def apply_utonia_lora(
    model: nn.Module,
    *,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
) -> Dict[str, int]:
    """
    Add LoRA only to Utonia/PTv3 SerializedAttention qkv/proj linears.

    Utonia's official attention modules expose both `qkv` and `proj` Linear
    layers.  Requiring both attributes avoids accidentally adapting GridPooling
    or other projection layers that are not attention.
    """
    wrapped = 0
    attention_modules = 0
    params = 0

    for module_name, module in list(model.named_modules()):
        qkv = getattr(module, "qkv", None)
        proj = getattr(module, "proj", None)
        if qkv is None or proj is None:
            continue
        if not isinstance(qkv, (nn.Linear, PAIRLoRALinear)):
            continue
        if not isinstance(proj, (nn.Linear, PAIRLoRALinear)):
            continue

        # Utonia attention is uniquely characterized by the fused qkv shape.
        qkv_base = qkv.base if isinstance(qkv, PAIRLoRALinear) else qkv
        proj_base = proj.base if isinstance(proj, PAIRLoRALinear) else proj
        if qkv_base.out_features != 3 * qkv_base.in_features:
            continue
        if proj_base.in_features != proj_base.out_features:
            continue
        if proj_base.in_features != qkv_base.in_features:
            continue

        attention_modules += 1
        for attr in ("qkv", "proj"):
            linear = getattr(module, attr)
            if isinstance(linear, PAIRLoRALinear):
                if linear.kind != "utonia":
                    raise RuntimeError(
                        f"Utonia attention {module_name}.{attr} already has incompatible LoRA kind"
                    )
                continue
            wrapped_linear = PAIRLoRALinear(
                linear,
                r=r,
                alpha=alpha,
                dropout=dropout,
                kind="utonia",
            ).to(device=linear.weight.device)
            setattr(module, attr, wrapped_linear)
            wrapped += 1
            params += wrapped_linear.pair_lora_A.numel() + wrapped_linear.pair_lora_B.numel()

    if attention_modules == 0 or wrapped == 0:
        raise RuntimeError("Utonia LoRA matched zero SerializedAttention qkv/proj linears")
    return {
        "attention_modules": attention_modules,
        "wrapped_linears": wrapped,
        "trainable_parameters": params,
    }


def set_custom_lora_training(module: nn.Module, mode: bool) -> None:
    for child in module.modules():
        if isinstance(child, PAIRLoRALinear):
            child.set_lora_training(mode)


def custom_lora_parameter_count(module: nn.Module, *, kind: str | None = None) -> Tuple[int, int]:
    trainable = total = 0
    for child in module.modules():
        if not isinstance(child, PAIRLoRALinear):
            continue
        if kind is not None and child.kind != kind:
            continue
        for parameter in (child.pair_lora_A, child.pair_lora_B):
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
    return trainable, total


def custom_lora_state_dict(module: nn.Module, *, kind: str | None = None):
    allowed = set()
    for module_name, child in module.named_modules():
        if not isinstance(child, PAIRLoRALinear):
            continue
        if kind is not None and child.kind != kind:
            continue
        prefix = f"{module_name}." if module_name else ""
        allowed.add(prefix + PAIR_LORA_A)
        allowed.add(prefix + PAIR_LORA_B)

    state = module.state_dict()
    return {
        key: value.detach().cpu()
        for key, value in state.items()
        if key in allowed
    }


def load_custom_lora_state_dict(module: nn.Module, state, *, kind: str | None = None) -> None:
    if not state:
        return

    current = dict(module.named_parameters())
    expected = set()
    for module_name, child in module.named_modules():
        if not isinstance(child, PAIRLoRALinear):
            continue
        if kind is not None and child.kind != kind:
            continue
        prefix = f"{module_name}." if module_name else ""
        expected.add(prefix + PAIR_LORA_A)
        expected.add(prefix + PAIR_LORA_B)

    unknown = sorted(set(state) - expected)
    missing = sorted(expected - set(state))
    if unknown:
        raise RuntimeError(f"Unexpected custom LoRA checkpoint keys: {unknown[:8]}")
    if missing:
        raise RuntimeError(f"Missing custom LoRA checkpoint keys: {missing[:8]}")

    with torch.no_grad():
        for key, value in state.items():
            if key not in current:
                raise RuntimeError(f"Custom LoRA parameter {key!r} not found in current model")
            current[key].copy_(value.to(device=current[key].device, dtype=current[key].dtype))


def is_peft_model(model):
    return hasattr(model, "peft_config")


def lora_state_dict(model):
    """PEFT/Qwen-LLM LoRA state only (custom vision LoRA saved separately)."""
    if not is_peft_model(model):
        return {}
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}


def load_lora_state_dict(model, state):
    if not state:
        return
    if not is_peft_model(model):
        raise RuntimeError("Checkpoint contains Qwen LLM LoRA weights but current Qwen is not a PEFT model")
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state)


def lora_parameter_count(model):
    """PEFT/Qwen-LLM LoRA count only; excludes PAIR custom LoRA wrappers."""
    total = trainable = 0
    for name, param in model.named_parameters():
        if "lora_" in name and "pair_lora_" not in name:
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return trainable, total
