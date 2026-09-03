"""
PAIR: Prompt-Aware Image-Point Reasoning.

Shared temporal backbone for 2d / 3d / 2d3d.

PAIR exposes two kinds of features for the unified decoder:
1) dense perception features before LLM reasoning
   - image_dense_t*: Qwen vision merged embeddings
   - point_dense_t*: PTv3 per-point features
2) contextual reasoning features after LLM
   - image_hidden_t*
   - point_hidden_t*
   - task_hidden

Current limitation: batch size = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


SUPPORTED_TASK_MODES = ("2d", "3d", "2d3d")


@dataclass
class PAIROutput:
    image_dense_t1: Optional[torch.Tensor] = None
    image_dense_t2: Optional[torch.Tensor] = None
    point_dense_t1: Optional[torch.Tensor] = None
    point_dense_t2: Optional[torch.Tensor] = None

    image_hidden_t1: Optional[torch.Tensor] = None
    image_hidden_t2: Optional[torch.Tensor] = None
    point_hidden_t1: Optional[torch.Tensor] = None
    point_hidden_t2: Optional[torch.Tensor] = None
    task_hidden: Optional[torch.Tensor] = None

    logits: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    generated_text: Optional[Any] = None
    aux: Optional[Dict[str, Any]] = None


class PAIRModel(nn.Module):
    def __init__(self, qwen_backbone: nn.Module,
                 point_encoder: Optional[nn.Module] = None,
                 point_adapter: Optional[nn.Module] = None):
        super().__init__()
        self.qwen_backbone = qwen_backbone
        self.point_encoder = point_encoder
        self.point_adapter = point_adapter

    # ------------------------------------------------------------------
    # Task routing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_task_mode(task_mode: str) -> str:
        aliases = {
            "2d": "2d", "image": "2d", "image_only": "2d",
            "3d": "3d", "point": "3d", "point_only": "3d",
            "2d3d": "2d3d", "2d+3d": "2d3d",
            "image_point": "2d3d", "multimodal": "2d3d",
        }
        task_mode = task_mode.lower().strip()
        if task_mode not in aliases:
            raise ValueError(
                f"Unsupported task_mode={task_mode!r}. Supported: {SUPPORTED_TASK_MODES}"
            )
        return aliases[task_mode]

    def _validate_inputs(self, *, task_mode, images_t1, images_t2,
                         point_dict_t1, point_dict_t2):
        if task_mode in ("2d", "2d3d") and (images_t1 is None or images_t2 is None):
            raise ValueError(f"task_mode={task_mode!r} requires images_t1 and images_t2")

        if task_mode in ("3d", "2d3d"):
            if point_dict_t1 is None or point_dict_t2 is None:
                raise ValueError(f"task_mode={task_mode!r} requires point_dict_t1 and point_dict_t2")
            if self.point_encoder is None:
                raise RuntimeError("3D input requested but point_encoder is not configured")
            if self.point_adapter is None:
                raise RuntimeError("3D input requested but point_adapter is not configured")

    # ------------------------------------------------------------------
    # Shared 3D branch
    # ------------------------------------------------------------------

    def encode_points(self, point_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        point_encoded = self.point_encoder(point_dict)
        adapter_out = self.point_adapter.forward_with_metadata(point_encoded)

        return {
            "point_encoded": point_encoded,
            "point_adapter_output": adapter_out,
            "point_tokens": adapter_out.tokens,
            "point_token_coord": adapter_out.sampled_coord,
            "point_token_indices": adapter_out.sampled_indices,
        }

    @staticmethod
    def _dense_point_fields(point_encoded):
        if point_encoded is None:
            return None, None, None
        return point_encoded.features, point_encoded.coord, point_encoded.batch

    # ------------------------------------------------------------------
    # Point-token helpers
    # ------------------------------------------------------------------

    def _normalize_single_point_tokens(self, point_tokens):
        if point_tokens is None:
            return None
        if not torch.is_tensor(point_tokens):
            raise TypeError("point_tokens must be a torch.Tensor")
        if point_tokens.ndim == 3:
            if point_tokens.shape[0] != 1:
                raise NotImplementedError("PAIR V1 supports batch size 1")
            point_tokens = point_tokens[0]
        if point_tokens.ndim != 2:
            raise ValueError(f"point_tokens must be [N,D] or [1,N,D], got {tuple(point_tokens.shape)}")
        if point_tokens.shape[-1] != self.qwen_backbone.hidden_size:
            raise ValueError(
                f"Point token dim must equal Qwen hidden size "
                f"{self.qwen_backbone.hidden_size}, got {point_tokens.shape[-1]}"
            )
        return point_tokens

    def _concat_point_tokens(self, point_tokens_t1, point_tokens_t2):
        pieces = [
            x for x in (
                self._normalize_single_point_tokens(point_tokens_t1),
                self._normalize_single_point_tokens(point_tokens_t2),
            ) if x is not None
        ]
        return None if not pieces else torch.cat(pieces, dim=0).unsqueeze(0)

    def _validate_point_layout(self, point_tokens, point_mask):
        expected = point_tokens.shape[1]
        actual = int(point_mask.sum().item())
        if expected != actual:
            raise RuntimeError(
                f"Prepared {expected} point tokens but prompt contains {actual} <POINT> placeholders"
            )

    def _make_point_injection_hook(self, *, point_tokens, point_mask,
                                   full_seq_len, stats):
        self._validate_point_layout(point_tokens, point_mask)
        hidden_size = self.qwen_backbone.hidden_size

        def hook(module, args, output):
            stats["calls"] += 1
            if (torch.is_tensor(output) and output.ndim == 3 and
                    output.shape == (1, full_seq_len, hidden_size)):
                out = output.clone()
                mask = point_mask[0].to(out.device)
                out[0, mask] = point_tokens[0].to(out.device, out.dtype)
                stats["replaced"] = True
                return out
            return output

        return hook

    # ------------------------------------------------------------------
    # Qwen capture hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _make_visual_capture_hook(store):
        def hook(module, args, output):
            # Qwen3VL vision returns:
            #   (merged_image_embeddings, deepstack_visual_features)
            if isinstance(output, (tuple, list)) and len(output) > 0:
                store["dense"] = output[0]
            elif torch.is_tensor(output):
                store["dense"] = output
        return hook

    @staticmethod
    def _make_language_capture_hook(store):
        def hook(module, args, output):
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, (tuple, list)) and len(output) > 0:
                hidden = output[0]
            if hidden is not None:
                store["last_hidden"] = hidden
        return hook

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _image_token_shape(self, inputs, grid_index):
        if grid_index is None or "image_grid_thw" not in inputs:
            return None

        grid = inputs["image_grid_thw"][grid_index]
        t, h_patch, w_patch = [int(x.item()) for x in grid]
        merge = int(self.qwen_backbone.vision_spatial_merge_size)

        if h_patch % merge != 0 or w_patch % merge != 0:
            raise RuntimeError("Qwen image grid is not divisible by spatial_merge_size")

        return t, h_patch // merge, w_patch // merge

    def _reshape_image_tokens(self, tokens, inputs, grid_index):
        if tokens is None:
            return None
        shape = self._image_token_shape(inputs, grid_index)
        if shape is None:
            return None

        t, h, w = shape
        expected = t * h * w
        if tokens.shape[0] != expected:
            raise RuntimeError(
                f"Image token count {tokens.shape[0]} does not match grid {shape} ({expected})"
            )
        return tokens.reshape(t, h, w, tokens.shape[-1])

    @staticmethod
    def _split_image_dense(vision_dense, prepared):
        if vision_dense is None:
            return None, None

        n1 = int(prepared["image_count_t1"])
        n2 = int(prepared["image_count_t2"])
        if vision_dense.shape[0] != n1 + n2:
            raise RuntimeError(
                f"Vision dense count {vision_dense.shape[0]} != expected {n1+n2}"
            )

        cursor = 0
        t1 = vision_dense[cursor:cursor + n1] if n1 else None
        cursor += n1
        t2 = vision_dense[cursor:cursor + n2] if n2 else None
        return t1, t2

    # ------------------------------------------------------------------
    # Contextual hidden extraction
    # ------------------------------------------------------------------

    def _extract_multimodal_hidden(self, *, last_hidden, prepared, inputs):
        device = last_hidden.device

        def mask(name):
            return prepared[name].to(device)

        image_mask_t1, image_mask_t2 = mask("image_mask_t1"), mask("image_mask_t2")
        point_mask_t1, point_mask_t2 = mask("point_mask_t1"), mask("point_mask_t2")
        task_mask = mask("task_mask")

        if int(task_mask.sum().item()) != 1:
            raise RuntimeError("PAIR expects exactly one <TASK> token")

        def select(m):
            return last_hidden[0][m[0]] if int(m.sum().item()) > 0 else None

        image_hidden_t1 = select(image_mask_t1)
        image_hidden_t2 = select(image_mask_t2)
        point_hidden_t1 = select(point_mask_t1)
        point_hidden_t2 = select(point_mask_t2)
        task_hidden = last_hidden[0][task_mask[0]].reshape(
            1, self.qwen_backbone.hidden_size
        )

        return {
            "image_hidden_t1": image_hidden_t1,
            "image_hidden_t2": image_hidden_t2,
            "point_hidden_t1": point_hidden_t1,
            "point_hidden_t2": point_hidden_t2,
            "task_hidden": task_hidden,
            # Backward-compatible contextual 2D views.
            "image_hidden_2d_t1": self._reshape_image_tokens(
                image_hidden_t1, inputs, prepared["image_grid_index_t1"]
            ),
            "image_hidden_2d_t2": self._reshape_image_tokens(
                image_hidden_t2, inputs, prepared["image_grid_index_t2"]
            ),
        }

    # ------------------------------------------------------------------
    # Qwen preparation
    # ------------------------------------------------------------------

    def _prepare_qwen(self, *, prompt, images_t1, images_t2,
                      point_tokens_t1, point_tokens_t2):
        prepared = self.qwen_backbone.prepare_inputs(
            prompt=prompt, images_t1=images_t1, images_t2=images_t2,
            point_tokens_t1=point_tokens_t1, point_tokens_t2=point_tokens_t2,
        )

        device = self.qwen_backbone.model_device
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in prepared["inputs"].items()
        }

        point_tokens = self._concat_point_tokens(point_tokens_t1, point_tokens_t2)
        return {**prepared, "inputs": inputs, "point_tokens": point_tokens}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *, task_mode: str, prompt: str,
                images_t1=None, images_t2=None,
                point_dict_t1=None, point_dict_t2=None,
                return_logits: bool = True,
                return_hidden_states: bool = True,
                return_dense_features: bool = True,
                use_cache: bool = False, **qwen_kwargs) -> PAIROutput:

        task_mode = self._normalize_task_mode(task_mode)
        self._validate_inputs(
            task_mode=task_mode, images_t1=images_t1, images_t2=images_t2,
            point_dict_t1=point_dict_t1, point_dict_t2=point_dict_t2,
        )

        out_t1 = out_t2 = None
        point_encoded_t1 = point_encoded_t2 = None
        point_tokens_t1 = point_tokens_t2 = None

        if task_mode in ("3d", "2d3d"):
            out_t1 = self.encode_points(point_dict_t1)
            out_t2 = self.encode_points(point_dict_t2)
            point_encoded_t1, point_encoded_t2 = out_t1["point_encoded"], out_t2["point_encoded"]
            point_tokens_t1, point_tokens_t2 = out_t1["point_tokens"], out_t2["point_tokens"]

        prepared = self._prepare_qwen(
            prompt=prompt,
            images_t1=images_t1 if task_mode in ("2d", "2d3d") else None,
            images_t2=images_t2 if task_mode in ("2d", "2d3d") else None,
            point_tokens_t1=point_tokens_t1 if task_mode in ("3d", "2d3d") else None,
            point_tokens_t2=point_tokens_t2 if task_mode in ("3d", "2d3d") else None,
        )

        inputs = prepared["inputs"]
        point_tokens = prepared["point_tokens"]
        injection_stats = {"calls": 0, "replaced": False}

        handles = []
        visual_capture, language_capture = {}, {}

        if point_tokens is not None:
            handles.append(
                self.qwen_backbone.model.get_input_embeddings().register_forward_hook(
                    self._make_point_injection_hook(
                        point_tokens=point_tokens,
                        point_mask=prepared["point_mask"],
                        full_seq_len=inputs["input_ids"].shape[1],
                        stats=injection_stats,
                    )
                )
            )

        if return_dense_features and task_mode in ("2d", "2d3d"):
            handles.append(
                self.qwen_backbone.model.visual.register_forward_hook(
                    self._make_visual_capture_hook(visual_capture)
                )
            )

        if return_hidden_states:
            handles.append(
                self.qwen_backbone.model.language_model.register_forward_hook(
                    self._make_language_capture_hook(language_capture)
                )
            )

        if not return_logits and "logits_to_keep" not in qwen_kwargs:
            qwen_kwargs["logits_to_keep"] = 1

        try:
            qwen_outputs = self.qwen_backbone.model(
                **inputs, return_dict=True, use_cache=use_cache, **qwen_kwargs
            )
        finally:
            for handle in handles:
                handle.remove()

        if point_tokens is not None and not injection_stats["replaced"]:
            raise RuntimeError("Temporal point tokens were prepared but not injected into Qwen")

        image_dense_t1 = image_dense_t2 = None
        if return_dense_features and task_mode in ("2d", "2d3d"):
            if "dense" not in visual_capture:
                raise RuntimeError("Qwen vision forward ran but pre-LLM image features were not captured")
            image_dense_t1, image_dense_t2 = self._split_image_dense(
                visual_capture["dense"], prepared
            )

        image_hidden_t1 = image_hidden_t2 = None
        point_hidden_t1 = point_hidden_t2 = None
        task_hidden = None
        hidden_extra = {}

        if return_hidden_states:
            if "last_hidden" not in language_capture:
                raise RuntimeError("Qwen language hidden state was not captured")

            hidden = self._extract_multimodal_hidden(
                last_hidden=language_capture["last_hidden"],
                prepared=prepared, inputs=inputs,
            )
            image_hidden_t1, image_hidden_t2 = hidden["image_hidden_t1"], hidden["image_hidden_t2"]
            point_hidden_t1, point_hidden_t2 = hidden["point_hidden_t1"], hidden["point_hidden_t2"]
            task_hidden = hidden["task_hidden"]
            hidden_extra = {
                "image_hidden_2d_t1": hidden["image_hidden_2d_t1"],
                "image_hidden_2d_t2": hidden["image_hidden_2d_t2"],
            }

        point_dense_t1, point_coord_t1, point_batch_t1 = self._dense_point_fields(point_encoded_t1)
        point_dense_t2, point_coord_t2, point_batch_t2 = self._dense_point_fields(point_encoded_t2)

        aux = {
            "task_mode": task_mode,
            "point_encoded_t1": point_encoded_t1,
            "point_encoded_t2": point_encoded_t2,
            "point_tokens_t1": point_tokens_t1,
            "point_tokens_t2": point_tokens_t2,
            "point_tokens_concat": point_tokens,

            "point_dense_coord_t1": point_coord_t1,
            "point_dense_coord_t2": point_coord_t2,
            "point_dense_batch_t1": point_batch_t1,
            "point_dense_batch_t2": point_batch_t2,

            "point_token_coord_t1": None if out_t1 is None else out_t1["point_token_coord"],
            "point_token_coord_t2": None if out_t2 is None else out_t2["point_token_coord"],
            "point_token_indices_t1": None if out_t1 is None else out_t1["point_token_indices"],
            "point_token_indices_t2": None if out_t2 is None else out_t2["point_token_indices"],

            "image_token_shape_t1": self._image_token_shape(inputs, prepared["image_grid_index_t1"]),
            "image_token_shape_t2": self._image_token_shape(inputs, prepared["image_grid_index_t2"]),
            "image_dense_2d_t1": self._reshape_image_tokens(
                image_dense_t1, inputs, prepared["image_grid_index_t1"]
            ),
            "image_dense_2d_t2": self._reshape_image_tokens(
                image_dense_t2, inputs, prepared["image_grid_index_t2"]
            ),

            "prompt_text": prepared["prompt_text"],
            "image_count_t1": prepared["image_count_t1"],
            "image_count_t2": prepared["image_count_t2"],
            "point_count_t1": prepared["point_count_t1"],
            "point_count_t2": prepared["point_count_t2"],
            "task_count": prepared["task_count"],
            "point_injection_calls": injection_stats["calls"],
            "point_injection_replaced": injection_stats["replaced"],
            **hidden_extra,
        }

        return PAIROutput(
            image_dense_t1=image_dense_t1,
            image_dense_t2=image_dense_t2,
            point_dense_t1=point_dense_t1,
            point_dense_t2=point_dense_t2,
            image_hidden_t1=image_hidden_t1,
            image_hidden_t2=image_hidden_t2,
            point_hidden_t1=point_hidden_t1,
            point_hidden_t2=point_hidden_t2,
            task_hidden=task_hidden,
            logits=qwen_outputs.logits if return_logits else None,
            aux=aux,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, *, task_mode: str, prompt: str,
                 images_t1=None, images_t2=None,
                 point_dict_t1=None, point_dict_t2=None,
                 max_new_tokens: int = 64, do_sample: bool = False,
                 **generate_kwargs) -> PAIROutput:

        task_mode = self._normalize_task_mode(task_mode)
        self._validate_inputs(
            task_mode=task_mode, images_t1=images_t1, images_t2=images_t2,
            point_dict_t1=point_dict_t1, point_dict_t2=point_dict_t2,
        )

        out_t1 = out_t2 = None
        point_tokens_t1 = point_tokens_t2 = None

        if task_mode in ("3d", "2d3d"):
            out_t1 = self.encode_points(point_dict_t1)
            out_t2 = self.encode_points(point_dict_t2)
            point_tokens_t1, point_tokens_t2 = out_t1["point_tokens"], out_t2["point_tokens"]

        prepared = self._prepare_qwen(
            prompt=prompt,
            images_t1=images_t1 if task_mode in ("2d", "2d3d") else None,
            images_t2=images_t2 if task_mode in ("2d", "2d3d") else None,
            point_tokens_t1=point_tokens_t1 if task_mode in ("3d", "2d3d") else None,
            point_tokens_t2=point_tokens_t2 if task_mode in ("3d", "2d3d") else None,
        )

        inputs, point_tokens = prepared["inputs"], prepared["point_tokens"]
        stats = {"calls": 0, "replaced": False}
        handle = None

        if point_tokens is not None:
            handle = self.qwen_backbone.model.get_input_embeddings().register_forward_hook(
                self._make_point_injection_hook(
                    point_tokens=point_tokens,
                    point_mask=prepared["point_mask"],
                    full_seq_len=inputs["input_ids"].shape[1],
                    stats=stats,
                )
            )

        try:
            generated_ids = self.qwen_backbone.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, **generate_kwargs
            )
        finally:
            if handle is not None:
                handle.remove()

        if point_tokens is not None and not stats["replaced"]:
            raise RuntimeError("Temporal point tokens were not injected during generation")

        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[:, prompt_len:]
        text = self.qwen_backbone.processor.batch_decode(
            new_token_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        text = text[0] if len(text) == 1 else text

        return PAIROutput(
            generated_ids=generated_ids,
            generated_text=text,
            aux={
                "task_mode": task_mode,
                "point_encoded_t1": None if out_t1 is None else out_t1["point_encoded"],
                "point_encoded_t2": None if out_t2 is None else out_t2["point_encoded"],
                "point_tokens_t1": point_tokens_t1,
                "point_tokens_t2": point_tokens_t2,
                "point_token_coord_t1": None if out_t1 is None else out_t1["point_token_coord"],
                "point_token_coord_t2": None if out_t2 is None else out_t2["point_token_coord"],
                "prompt_text": prepared["prompt_text"],
                "point_injection_calls": stats["calls"],
                "point_injection_replaced": stats["replaced"],
            },
        )
