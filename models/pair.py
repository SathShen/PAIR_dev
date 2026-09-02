"""
PAIR: Prompt-Aware Image-Point Reasoning

Top-level model scaffold.

This file intentionally contains only the orchestration layer.
The actual implementations will live in:

    qwen3vl_backbone.py
    point_encoder.py
    point_adapter.py
    multimodal_injector.py

Current goal:
    - define a clean PAIRModel interface
    - support 2D / 3D / 2D+3D routing
    - preserve language output
    - expose image / point / task hidden states
    - avoid hard-coding PTv3 or Qwen internals here

Dense change / segmentation decoders are intentionally NOT added yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


SUPPORTED_TASK_MODES = ("2d", "3d", "2d3d")


@dataclass
class PAIROutput:
    """
    Unified output container for the PAIR multimodal backbone.

    The exact tensor shapes depend on the active modality.

    Typical examples
    ----------------
    image_hidden:
        [B, N_img, D_llm] or None

    point_hidden:
        [B, N_point, D_llm] or None

    task_hidden:
        [B, D_llm] or [B, 1, D_llm]

    logits:
        Native language-model logits when requested.

    generated_ids / generated_text:
        Optional native language generation outputs.

    aux:
        Intermediate features useful for debugging or future decoders.
    """

    image_hidden: Optional[torch.Tensor] = None
    point_hidden: Optional[torch.Tensor] = None
    task_hidden: Optional[torch.Tensor] = None

    logits: Optional[torch.Tensor] = None

    generated_ids: Optional[torch.Tensor] = None
    generated_text: Optional[Any] = None

    aux: Optional[Dict[str, Any]] = None


class PAIRModel(nn.Module):
    """
    Top-level PAIR orchestration model.

    Architecture
    ------------
    2D image:
        image -> Qwen3-VL native vision path -> image tokens

    3D point cloud:
        points -> point_encoder (PTv3)
               -> point_adapter
               -> multimodal injector
               -> Qwen3-VL

    Language:
        prompt + <TASK> -> Qwen3-VL

    Outputs:
        contextualized image hidden states
        contextualized point hidden states
        contextualized <TASK> hidden state
        optional native text logits / generation

    Notes
    -----
    This is deliberately a coordination layer. It should not know PTv3
    implementation details and should not directly manipulate Qwen internals.
    """

    def __init__(
        self,
        qwen_backbone: nn.Module,
        point_encoder: Optional[nn.Module] = None,
        point_adapter: Optional[nn.Module] = None,
        multimodal_injector: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.qwen_backbone = qwen_backbone
        self.point_encoder = point_encoder
        self.point_adapter = point_adapter
        self.multimodal_injector = multimodal_injector

    # ------------------------------------------------------------------
    # Validation / routing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_task_mode(task_mode: str) -> str:
        task_mode = task_mode.lower().strip()

        aliases = {
            "2d": "2d",
            "image": "2d",
            "image_only": "2d",

            "3d": "3d",
            "point": "3d",
            "point_only": "3d",

            "2d3d": "2d3d",
            "2d+3d": "2d3d",
            "image_point": "2d3d",
            "multimodal": "2d3d",
        }

        if task_mode not in aliases:
            raise ValueError(
                f"Unsupported task_mode={task_mode!r}. "
                f"Supported modes: {SUPPORTED_TASK_MODES}"
            )

        return aliases[task_mode]

    def _validate_inputs(
        self,
        task_mode: str,
        images: Optional[Any],
        point_dict: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        if task_mode in ("2d", "2d3d") and images is None:
            raise ValueError(
                f"task_mode={task_mode!r} requires image input."
            )

        if task_mode in ("3d", "2d3d") and point_dict is None:
            raise ValueError(
                f"task_mode={task_mode!r} requires point_dict."
            )

        if task_mode in ("3d", "2d3d"):
            if self.point_encoder is None:
                raise RuntimeError(
                    "3D input requested but point_encoder is not configured."
                )

            if self.point_adapter is None:
                raise RuntimeError(
                    "3D input requested but point_adapter is not configured."
                )

    # ------------------------------------------------------------------
    # 3D branch
    # ------------------------------------------------------------------

    def encode_points(
        self,
        point_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Encode raw point cloud into Qwen-compatible point tokens.

        Expected future contract
        ------------------------
        point_encoder(point_dict)
            -> dense point representation

        point_adapter(...)
            -> point tokens in Qwen hidden dimension

        The exact resampling/tokenization policy belongs inside
        point_adapter.py, not in PAIR.py.
        """

        if self.point_encoder is None:
            raise RuntimeError("point_encoder is not configured.")

        if self.point_adapter is None:
            raise RuntimeError("point_adapter is not configured.")

        point_encoded = self.point_encoder(point_dict)

        point_tokens = self.point_adapter(point_encoded)

        return {
            "point_encoded": point_encoded,
            "point_tokens": point_tokens,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        task_mode: str,
        prompt: Any,
        images: Optional[Any] = None,
        point_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_logits: bool = True,
        return_hidden_states: bool = True,
        **kwargs,
    ) -> PAIROutput:
        """
        Unified PAIR forward.

        Parameters
        ----------
        task_mode:
            "2d", "3d", or "2d3d".

        prompt:
            Language instruction / task description.

        images:
            Image input required by 2D and 2D+3D modes.

        point_dict:
            PTv3-style point input dictionary required by 3D and 2D+3D.

        return_logits:
            Whether to expose native Qwen LM logits.

        return_hidden_states:
            Whether to return contextualized image / point / <TASK> states.

        kwargs:
            Reserved for Qwen-specific runtime options.

        Returns
        -------
        PAIROutput
        """

        task_mode = self._normalize_task_mode(task_mode)

        self._validate_inputs(
            task_mode=task_mode,
            images=images,
            point_dict=point_dict,
        )

        point_encoded = None
        point_tokens = None

        if task_mode in ("3d", "2d3d"):
            point_outputs = self.encode_points(point_dict)
            point_encoded = point_outputs["point_encoded"]
            point_tokens = point_outputs["point_tokens"]

        # The Qwen wrapper will own:
        #   - processor / tokenizer
        #   - native Qwen image path
        #   - <POINT> / <TASK> construction
        #   - external point-token injection
        #   - hidden-state extraction
        #
        # Keeping those mechanics out of PAIR.py makes this top-level module
        # stable even if the Qwen implementation changes later.
        qwen_outputs = self.qwen_backbone(
            prompt=prompt,
            images=images if task_mode in ("2d", "2d3d") else None,
            point_tokens=point_tokens if task_mode in ("3d", "2d3d") else None,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            **kwargs,
        )

        if isinstance(qwen_outputs, dict):
            get_value = qwen_outputs.get
        else:
            get_value = lambda key, default=None: getattr(
                qwen_outputs, key, default
            )

        aux = {
            "task_mode": task_mode,
            "point_encoded": point_encoded,
            "point_tokens": point_tokens,
        }

        qwen_aux = get_value("aux", None)
        if qwen_aux is not None:
            aux["qwen"] = qwen_aux

        return PAIROutput(
            image_hidden=get_value("image_hidden", None),
            point_hidden=get_value("point_hidden", None),
            task_hidden=get_value("task_hidden", None),
            logits=get_value("logits", None),
            aux=aux,
        )

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        *,
        task_mode: str,
        prompt: Any,
        images: Optional[Any] = None,
        point_dict: Optional[Dict[str, torch.Tensor]] = None,
        **generate_kwargs,
    ) -> PAIROutput:
        """
        Native language generation through the Qwen backbone.

        Dense change / segmentation prediction will be added separately
        later and should not be conflated with this method.
        """

        task_mode = self._normalize_task_mode(task_mode)

        self._validate_inputs(
            task_mode=task_mode,
            images=images,
            point_dict=point_dict,
        )

        point_encoded = None
        point_tokens = None

        if task_mode in ("3d", "2d3d"):
            point_outputs = self.encode_points(point_dict)
            point_encoded = point_outputs["point_encoded"]
            point_tokens = point_outputs["point_tokens"]

        generation_outputs = self.qwen_backbone.generate(
            prompt=prompt,
            images=images if task_mode in ("2d", "2d3d") else None,
            point_tokens=point_tokens if task_mode in ("3d", "2d3d") else None,
            **generate_kwargs,
        )

        if isinstance(generation_outputs, dict):
            get_value = generation_outputs.get
        else:
            get_value = lambda key, default=None: getattr(
                generation_outputs, key, default
            )

        return PAIROutput(
            image_hidden=get_value("image_hidden", None),
            point_hidden=get_value("point_hidden", None),
            task_hidden=get_value("task_hidden", None),
            generated_ids=get_value("generated_ids", None),
            generated_text=get_value("generated_text", None),
            aux={
                "task_mode": task_mode,
                "point_encoded": point_encoded,
                "point_tokens": point_tokens,
            },
        )


# Alias kept intentionally short for later config / registry use.
PAIR = PAIRModel


if __name__ == "__main__":
    print("PAIR.py import scaffold OK")
    print("Supported task modes:", SUPPORTED_TASK_MODES)
