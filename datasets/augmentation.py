"""
PAIR paired 2D training augmentation.

This module intentionally contains ONLY the first augmentation stage requested
for PAIR:

- paired geometric augmentation matching the PerASCD recipe:
    * 50% no rotation / 50% 90-degree counter-clockwise rotation
    * then uniformly choose one of:
        no flip / vertical flip / horizontal flip / 180-degree rotation
- shared ColorJitter for T1/T2:
    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1

Important protocol rules
------------------------
1. The SAME geometric transform is applied to T1, T2 and every raster target /
   validity mask.
2. The SAME photometric transform (same factors AND same operation order) is
   applied to T1 and T2.
3. Input images are expected to be float tensors in [0, 1].
4. No dataset-specific normalization is performed here. Qwen's processor keeps
   ownership of its own image preprocessing.
5. No random crop and no temporal swap are implemented in this stage.
6. This module is intended only for TRAIN samples on route == "2d".
   3D and future 2D+3D geometry must not use it unless their coordinates are
   transformed consistently too.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Optional

import torch

try:
    from torchvision.transforms import functional as TVF
except ImportError as exc:
    raise ImportError(
        "PAIR 2D augmentation requires torchvision. "
        "Install torchvision matching the current PyTorch build."
    ) from exc


# Raster targets that must stay pixel-aligned with the image pair.
_RASTER_TARGET_KEYS = (
    "change",
    "semantic_t1",
    "semantic_t2",
    "change_valid",
    "semantic_valid_t1",
    "semantic_valid_t2",
)


@dataclass(frozen=True)
class Pair2DAugmentConfig:
    """Fixed first-stage PAIR augmentation settings."""

    rotate90_prob: float = 0.5

    # PerASCD ColorJitter strengths.
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    hue: float = 0.1

    def validate(self) -> None:
        if not 0.0 <= self.rotate90_prob <= 1.0:
            raise ValueError(
                f"rotate90_prob must be in [0,1], got {self.rotate90_prob}"
            )

        for name in ("brightness", "contrast", "saturation"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if not 0.0 <= float(self.hue) <= 0.5:
            raise ValueError(
                f"hue must be in [0,0.5], got {self.hue}"
            )


DEFAULT_PAIR_2D_AUGMENT = Pair2DAugmentConfig()


def _validate_image_pair(
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
) -> None:
    if not torch.is_tensor(image_t1) or not torch.is_tensor(image_t2):
        raise TypeError("PAIR 2D augmentation expects torch.Tensor images")

    if image_t1.ndim != 3 or image_t2.ndim != 3:
        raise ValueError(
            "PAIR 2D augmentation expects CHW images, got "
            f"{tuple(image_t1.shape)} and {tuple(image_t2.shape)}"
        )

    if image_t1.shape != image_t2.shape:
        raise ValueError(
            "PAIR 2D augmentation requires aligned T1/T2 images, got "
            f"{tuple(image_t1.shape)} vs {tuple(image_t2.shape)}"
        )

    if not image_t1.is_floating_point() or not image_t2.is_floating_point():
        raise TypeError("PAIR 2D augmentation expects floating-point images")

    if not torch.isfinite(image_t1).all() or not torch.isfinite(image_t2).all():
        raise ValueError("PAIR 2D image contains NaN/Inf")

    # read_image() already converts image data to [0,1]. Fail loudly rather
    # than accidentally applying torchvision hue/brightness to normalized Qwen
    # tensors or arbitrary-range rasters.
    eps = 1e-5
    if (
        float(image_t1.min()) < -eps
        or float(image_t1.max()) > 1.0 + eps
        or float(image_t2.min()) < -eps
        or float(image_t2.max()) > 1.0 + eps
    ):
        raise ValueError(
            "PAIR 2D augmentation expects images in [0,1] BEFORE the "
            "Qwen processor"
        )


def _apply_geometry(
    tensor: torch.Tensor,
    *,
    rotate90: bool,
    flip_mode: str,
) -> torch.Tensor:
    """
    Apply one PerASCD-style spatial transform to the last two dimensions.

    rotate90=True matches PIL.Image.ROTATE_90 (counter-clockwise).
    """
    out = tensor

    if rotate90:
        out = torch.rot90(out, k=1, dims=(-2, -1))

    if flip_mode == "none":
        return out
    if flip_mode == "vertical":
        return torch.flip(out, dims=(-2,))
    if flip_mode == "horizontal":
        return torch.flip(out, dims=(-1,))
    if flip_mode == "both":
        # Equivalent to PIL.Image.ROTATE_180 used by PerASCD rand_flip_SCD.
        return torch.flip(out, dims=(-2, -1))

    raise ValueError(f"Unknown flip_mode={flip_mode!r}")


def _sample_geometry(
    rng: random.Random,
    config: Pair2DAugmentConfig,
):
    # PerASCD:
    #   rand_rot90_SCD: 50% no rotation / 50% ROTATE_90
    # followed by rand_flip_SCD:
    #   25% none / 25% vertical / 25% horizontal / 25% both.
    rotate90 = rng.random() >= (1.0 - config.rotate90_prob)

    r = rng.random()
    if r < 0.25:
        flip_mode = "none"
    elif r < 0.50:
        flip_mode = "vertical"
    elif r < 0.75:
        flip_mode = "horizontal"
    else:
        flip_mode = "both"

    return rotate90, flip_mode


def _sample_color_ops(
    rng: random.Random,
    config: Pair2DAugmentConfig,
    *,
    rgb: bool,
):
    """
    Reproduce torchvision/PerASCD ColorJitter semantics while making the
    sampled parameters explicit so the exact same transform is reused on both
    temporal images.
    """
    ops = []

    if config.brightness > 0:
        ops.append(
            (
                "brightness",
                rng.uniform(
                    max(0.0, 1.0 - config.brightness),
                    1.0 + config.brightness,
                ),
            )
        )

    if config.contrast > 0:
        ops.append(
            (
                "contrast",
                rng.uniform(
                    max(0.0, 1.0 - config.contrast),
                    1.0 + config.contrast,
                ),
            )
        )

    # Saturation and hue have RGB semantics. PAIR's Qwen route is normally RGB,
    # but skipping them for a non-RGB raster is safer than silently inventing
    # channels.
    if rgb and config.saturation > 0:
        ops.append(
            (
                "saturation",
                rng.uniform(
                    max(0.0, 1.0 - config.saturation),
                    1.0 + config.saturation,
                ),
            )
        )

    if rgb and config.hue > 0:
        ops.append(
            (
                "hue",
                rng.uniform(-config.hue, config.hue),
            )
        )

    rng.shuffle(ops)
    return tuple(ops)


def _apply_color_ops(
    image: torch.Tensor,
    ops,
) -> torch.Tensor:
    out = image

    for name, factor in ops:
        if name == "brightness":
            out = TVF.adjust_brightness(out, factor)
        elif name == "contrast":
            out = TVF.adjust_contrast(out, factor)
        elif name == "saturation":
            out = TVF.adjust_saturation(out, factor)
        elif name == "hue":
            out = TVF.adjust_hue(out, factor)
        else:
            raise RuntimeError(f"Unknown color jitter op {name!r}")

    # Torchvision already keeps ordinary float images in range, but clamp once
    # at the augmentation boundary to preserve the explicit [0,1] contract for
    # the Qwen processor.
    return out.clamp_(0.0, 1.0)


def augment_pair_2d(
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    target: Dict[str, Any],
    *,
    config: Pair2DAugmentConfig = DEFAULT_PAIR_2D_AUGMENT,
    rng: Optional[random.Random] = None,
):
    """
    Apply synchronized PerASCD-style augmentation to one PAIR 2D sample.

    Returns
    -------
    image_t1_aug, image_t2_aug, target_aug

    The input target dictionary is not mutated.
    """
    config.validate()
    _validate_image_pair(image_t1, image_t2)

    # By default use Python's process/worker RNG. PyTorch DataLoader workers
    # seed Python's random module, so this remains worker-safe and reproducible
    # under the normal PAIR seed setup.
    rng = random if rng is None else rng

    rotate90, flip_mode = _sample_geometry(rng, config)

    image_t1 = _apply_geometry(
        image_t1,
        rotate90=rotate90,
        flip_mode=flip_mode,
    )
    image_t2 = _apply_geometry(
        image_t2,
        rotate90=rotate90,
        flip_mode=flip_mode,
    )

    target_aug = dict(target)
    original_hw = tuple(target.get("change", image_t1).shape[-2:])

    for key in _RASTER_TARGET_KEYS:
        value = target.get(key)
        if not torch.is_tensor(value) or value.ndim < 2:
            continue

        # Only transform raster-shaped supervision. This prevents a future
        # non-raster auxiliary field from being rotated accidentally.
        if tuple(value.shape[-2:]) != original_hw:
            continue

        target_aug[key] = _apply_geometry(
            value,
            rotate90=rotate90,
            flip_mode=flip_mode,
        )

    rgb = image_t1.shape[0] == 3
    color_ops = _sample_color_ops(
        rng,
        config,
        rgb=rgb,
    )

    # Exactly the same factors and op order are applied to both dates.
    image_t1 = _apply_color_ops(image_t1, color_ops)
    image_t2 = _apply_color_ops(image_t2, color_ops)

    return image_t1, image_t2, target_aug


def _self_test():
    # Two identical dates must remain photometrically identical because the
    # jitter parameters are shared.
    base = torch.linspace(
        0.0,
        1.0,
        steps=3 * 8 * 8,
        dtype=torch.float32,
    ).reshape(3, 8, 8)

    sem1 = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [3, 3, 2, 2, 1, 1, 0, 0],
            [3, 3, 2, 2, 1, 1, 0, 0],
            [3, 3, 2, 2, 1, 1, 0, 0],
            [3, 3, 2, 2, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    sem2 = sem1.clone()
    sem2[2:6, 2:6] = (sem2[2:6, 2:6] + 1) % 4

    change = (sem1 != sem2).long()
    target = {
        "semantic_t1": sem1,
        "semantic_t2": sem2,
        "change": change,
        "semantic_valid_t1": torch.ones_like(sem1, dtype=torch.bool),
        "semantic_valid_t2": torch.ones_like(sem2, dtype=torch.bool),
        "change_valid": torch.ones_like(change, dtype=torch.bool),
    }

    rng = random.Random(12345)
    a1, a2, ta = augment_pair_2d(
        base.clone(),
        base.clone(),
        target,
        rng=rng,
    )

    assert a1.shape == base.shape
    assert a2.shape == base.shape
    assert torch.allclose(a1, a2, atol=0.0, rtol=0.0)
    assert float(a1.min()) >= 0.0
    assert float(a1.max()) <= 1.0

    # Geometry must preserve semantic/change alignment exactly.
    derived_change = (
        ta["semantic_t1"] != ta["semantic_t2"]
    ).long()
    assert torch.equal(ta["change"], derived_change)

    for key in (
        "semantic_valid_t1",
        "semantic_valid_t2",
        "change_valid",
    ):
        assert ta[key].dtype == torch.bool
        assert bool(ta[key].all())

    # Input target was not mutated.
    assert torch.equal(target["change"], change)

    print("pair_augmentation.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
