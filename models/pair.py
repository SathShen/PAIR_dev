"""
PAIR: Prompt-Aware Image-Point Reasoning.

Batch-aware temporal backbone for 2d / 3d / 2d3d.

True vectorized batching is enabled for the 2D Qwen path. Dense and reasoning
tokens are returned as flat ragged token sets with matching batch IDs in aux.

Single-sample behavior remains backward compatible.
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
            if self.point_encoder is None or self.point_adapter is None:
                raise RuntimeError("3D input requested but point encoder/adapter is not configured")

    # ------------------------------------------------------------------
    # Qwen module resolution (works before and after PEFT wrapping)
    # ------------------------------------------------------------------

    def _find_submodule(self, name: str):
        queue = [self.qwen_backbone.model]
        seen = set()
        while queue:
            module = queue.pop(0)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, name):
                return getattr(module, name)
            for attr in ("base_model", "model"):
                child = getattr(module, attr, None)
                if child is not None and child is not module:
                    queue.append(child)
        raise AttributeError(f"Could not resolve Qwen submodule {name!r}")

    def visual_module(self):
        return self._find_submodule("visual")

    def language_model_module(self):
        return self._find_submodule("language_model")

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
    # Point injection
    # ------------------------------------------------------------------

    def _point_batch(self, point_tokens, batch_size):
        return self.qwen_backbone._point_token_batch(
            point_tokens, batch_size, "point_tokens"
        )

    def _concat_point_tokens(self, t1, t2, batch_size):
        a = self._point_batch(t1, batch_size)
        b = self._point_batch(t2, batch_size)
        out = []
        for x, y in zip(a, b):
            parts = [z for z in (x, y) if z is not None]
            out.append(None if not parts else torch.cat(parts, dim=0))
        return out

    @staticmethod
    def _validate_point_layout(point_tokens, point_mask):
        if len(point_tokens) != point_mask.shape[0]:
            raise RuntimeError("Point token batch size does not match point mask")
        for b, tokens in enumerate(point_tokens):
            expected = 0 if tokens is None else int(tokens.shape[0])
            actual = int(point_mask[b].sum().item())
            if expected != actual:
                raise RuntimeError(
                    f"Batch {b}: prepared {expected} point tokens but prompt has {actual}"
                )

    def _make_point_injection_hook(self, *, point_tokens, point_mask,
                                   full_seq_len, stats):
        self._validate_point_layout(point_tokens, point_mask)
        hidden_size = self.qwen_backbone.hidden_size
        batch_size = len(point_tokens)

        def hook(module, args, output):
            stats["calls"] += 1
            if not (torch.is_tensor(output) and output.ndim == 3):
                return output
            if output.shape != (batch_size, full_seq_len, hidden_size):
                return output

            out = output.clone()
            replaced = False
            for b, tokens in enumerate(point_tokens):
                if tokens is None:
                    continue
                mask = point_mask[b].to(out.device)
                out[b, mask] = tokens.to(device=out.device, dtype=out.dtype)
                replaced = True
            stats["replaced"] = replaced
            return out

        return hook

    # ------------------------------------------------------------------
    # Capture hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _make_visual_capture_hook(store):
        def hook(module, args, output):
            if isinstance(output, (tuple, list)) and output:
                store["dense"] = output[0]
            elif torch.is_tensor(output):
                store["dense"] = output
        return hook

    @staticmethod
    def _make_language_capture_hook(store):
        def hook(module, args, output):
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, (tuple, list)) and output:
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
        t, hp, wp = [int(x.item()) for x in grid]
        merge = int(self.qwen_backbone.vision_spatial_merge_size)
        if hp % merge or wp % merge:
            raise RuntimeError("Qwen image grid is not divisible by spatial_merge_size")
        return t, hp // merge, wp // merge

    def _image_token_shapes(self, inputs, indices):
        return [self._image_token_shape(inputs, idx) for idx in indices]

    @staticmethod
    def _split_visual_dense(vision_dense, prepared):
        if vision_dense is None:
            return None, None, None, None

        records = prepared["image_records"]
        counts = prepared["image_counts_in_order"]
        batch_size = prepared["batch_size"]
        t1_parts, t2_parts = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
        cursor = 0

        for (b, label), count in zip(records, counts):
            part = vision_dense[cursor:cursor + count]
            if part.shape[0] != count:
                raise RuntimeError("Vision dense token accounting mismatch")
            (t1_parts if label == "t1" else t2_parts)[b].append(part)
            cursor += count

        if cursor != vision_dense.shape[0]:
            raise RuntimeError(
                f"Vision dense count {vision_dense.shape[0]} != consumed {cursor}"
            )

        def flatten(parts):
            features, ids = [], []
            for b, chunks in enumerate(parts):
                if not chunks:
                    continue
                x = torch.cat(chunks, dim=0)
                features.append(x)
                ids.append(torch.full(
                    (x.shape[0],), b, dtype=torch.long, device=x.device
                ))
            if not features:
                return None, None
            return torch.cat(features, dim=0), torch.cat(ids, dim=0)

        f1, b1 = flatten(t1_parts)
        f2, b2 = flatten(t2_parts)
        return f1, f2, b1, b2

    @staticmethod
    def _reshape_single(tokens, shape):
        if tokens is None or shape is None:
            return None
        t, h, w = shape
        if tokens.shape[0] != t * h * w:
            raise RuntimeError(
                f"Image token count {tokens.shape[0]} does not match grid {shape}"
            )
        return tokens.reshape(t, h, w, tokens.shape[-1])

    def _split_by_batch(self, tokens, batch_ids, shapes):
        if tokens is None:
            return [None] * len(shapes)
        return [
            self._reshape_single(tokens[batch_ids == b], shape)
            for b, shape in enumerate(shapes)
        ]

    # ------------------------------------------------------------------
    # Hidden extraction
    # ------------------------------------------------------------------

    def _extract_multimodal_hidden(self, *, last_hidden, prepared, inputs):
        device = last_hidden.device
        bsz = prepared["batch_size"]

        image_mask_t1 = prepared["image_mask_t1"].to(device)
        image_mask_t2 = prepared["image_mask_t2"].to(device)
        point_mask_t1 = prepared["point_mask_t1"].to(device)
        point_mask_t2 = prepared["point_mask_t2"].to(device)
        task_mask = prepared["task_mask"].to(device)

        def flatten(mask):
            parts, batch_ids = [], []
            for b in range(bsz):
                selected = last_hidden[b][mask[b]]
                if selected.numel() == 0:
                    continue
                parts.append(selected)
                batch_ids.append(torch.full(
                    (selected.shape[0],), b, dtype=torch.long, device=device
                ))
            if not parts:
                return None, None
            return torch.cat(parts, 0), torch.cat(batch_ids, 0)

        image_hidden_t1, image_batch_t1 = flatten(image_mask_t1)
        image_hidden_t2, image_batch_t2 = flatten(image_mask_t2)
        point_hidden_t1, point_batch_t1 = flatten(point_mask_t1)
        point_hidden_t2, point_batch_t2 = flatten(point_mask_t2)

        task_parts = []
        for b in range(bsz):
            x = last_hidden[b][task_mask[b]]
            if x.shape[0] != 1:
                raise RuntimeError(
                    f"Batch {b}: expected one task hidden, got {x.shape[0]}"
                )
            task_parts.append(x[0])
        task_hidden = torch.stack(task_parts, 0)

        shapes1 = self._image_token_shapes(
            inputs, prepared["image_grid_indices_t1"]
        )
        shapes2 = self._image_token_shapes(
            inputs, prepared["image_grid_indices_t2"]
        )

        return {
            "image_hidden_t1": image_hidden_t1,
            "image_hidden_t2": image_hidden_t2,
            "point_hidden_t1": point_hidden_t1,
            "point_hidden_t2": point_hidden_t2,
            "task_hidden": task_hidden,
            "image_batch_ids_t1": image_batch_t1,
            "image_batch_ids_t2": image_batch_t2,
            "point_reasoning_batch_ids_t1": point_batch_t1,
            "point_reasoning_batch_ids_t2": point_batch_t2,
            "image_hidden_2d_t1_list": self._split_by_batch(
                image_hidden_t1, image_batch_t1, shapes1
            ),
            "image_hidden_2d_t2_list": self._split_by_batch(
                image_hidden_t2, image_batch_t2, shapes2
            ),
        }

    def _prepare_qwen(self, *, prompt, images_t1, images_t2,
                      point_tokens_t1, point_tokens_t2):
        prepared = self.qwen_backbone.prepare_inputs(
            prompt=prompt, images_t1=images_t1, images_t2=images_t2,
            point_tokens_t1=point_tokens_t1, point_tokens_t2=point_tokens_t2,
        )
        device = self.qwen_backbone.model_device
        inputs = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in prepared["inputs"].items()
        }
        point_tokens = self._concat_point_tokens(
            point_tokens_t1, point_tokens_t2, prepared["batch_size"]
        )
        return {**prepared, "inputs": inputs, "point_tokens": point_tokens}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *, task_mode: str, prompt,
                images_t1=None, images_t2=None,
                point_dict_t1=None, point_dict_t2=None,
                return_logits: bool = True,
                return_hidden_states: bool = True,
                return_dense_features: bool = True,
                use_cache: bool = False, **qwen_kwargs):

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
        stats = {"calls": 0, "replaced": False}
        visual_capture, language_capture, handles = {}, {}, []

        if any(x is not None for x in point_tokens):
            handles.append(
                self.qwen_backbone.model.get_input_embeddings().register_forward_hook(
                    self._make_point_injection_hook(
                        point_tokens=point_tokens,
                        point_mask=prepared["point_mask"],
                        full_seq_len=inputs["input_ids"].shape[1],
                        stats=stats,
                    )
                )
            )

        if return_dense_features and task_mode in ("2d", "2d3d"):
            handles.append(
                self.visual_module().register_forward_hook(
                    self._make_visual_capture_hook(visual_capture)
                )
            )

        if return_hidden_states:
            handles.append(
                self.language_model_module().register_forward_hook(
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
            for h in handles:
                h.remove()

        if any(x is not None for x in point_tokens) and not stats["replaced"]:
            raise RuntimeError("Temporal point tokens were prepared but not injected into Qwen")

        image_dense_t1 = image_dense_t2 = None
        image_dense_batch_t1 = image_dense_batch_t2 = None
        if return_dense_features and task_mode in ("2d", "2d3d"):
            if "dense" not in visual_capture:
                raise RuntimeError("Qwen vision forward ran but image dense features were not captured")
            (image_dense_t1, image_dense_t2,
             image_dense_batch_t1, image_dense_batch_t2) = self._split_visual_dense(
                visual_capture["dense"], prepared
            )

        hidden = {}
        if return_hidden_states:
            if "last_hidden" not in language_capture:
                raise RuntimeError("Qwen language hidden state was not captured")
            hidden = self._extract_multimodal_hidden(
                last_hidden=language_capture["last_hidden"],
                prepared=prepared, inputs=inputs,
            )

        point_dense_t1, point_coord_t1, point_batch_t1 = self._dense_point_fields(point_encoded_t1)
        point_dense_t2, point_coord_t2, point_batch_t2 = self._dense_point_fields(point_encoded_t2)

        shapes1 = self._image_token_shapes(inputs, prepared["image_grid_indices_t1"])
        shapes2 = self._image_token_shapes(inputs, prepared["image_grid_indices_t2"])
        single = prepared["single_input"]

        dense2d1 = self._split_by_batch(image_dense_t1, image_dense_batch_t1, shapes1)
        dense2d2 = self._split_by_batch(image_dense_t2, image_dense_batch_t2, shapes2)

        aux = {
            "task_mode": task_mode,
            "batch_size": prepared["batch_size"],

            "image_dense_batch_ids_t1": image_dense_batch_t1,
            "image_dense_batch_ids_t2": image_dense_batch_t2,
            "image_reasoning_batch_ids_t1": hidden.get("image_batch_ids_t1"),
            "image_reasoning_batch_ids_t2": hidden.get("image_batch_ids_t2"),

            "image_token_shapes_t1": shapes1,
            "image_token_shapes_t2": shapes2,
            "image_dense_2d_t1_list": dense2d1,
            "image_dense_2d_t2_list": dense2d2,
            "image_hidden_2d_t1_list": hidden.get("image_hidden_2d_t1_list"),
            "image_hidden_2d_t2_list": hidden.get("image_hidden_2d_t2_list"),

            "point_encoded_t1": point_encoded_t1,
            "point_encoded_t2": point_encoded_t2,
            "point_tokens_t1": point_tokens_t1,
            "point_tokens_t2": point_tokens_t2,
            "point_dense_coord_t1": point_coord_t1,
            "point_dense_coord_t2": point_coord_t2,
            "point_dense_batch_t1": point_batch_t1,
            "point_dense_batch_t2": point_batch_t2,

            "point_token_coord_t1": None if out_t1 is None else out_t1["point_token_coord"],
            "point_token_coord_t2": None if out_t2 is None else out_t2["point_token_coord"],
            "point_token_indices_t1": None if out_t1 is None else out_t1["point_token_indices"],
            "point_token_indices_t2": None if out_t2 is None else out_t2["point_token_indices"],

            "prompt_text": prepared["prompt_text"],
            "prompt_texts": prepared["prompt_texts"],
            "image_counts_t1": prepared["image_counts_t1"],
            "image_counts_t2": prepared["image_counts_t2"],
            "point_counts_t1": prepared["point_counts_t1"],
            "point_counts_t2": prepared["point_counts_t2"],
            "task_count": prepared["task_count"],
            "point_injection_calls": stats["calls"],
            "point_injection_replaced": stats["replaced"],
        }

        # Single-sample aliases keep existing smoke tests working.
        if single:
            aux.update({
                "image_token_shape_t1": shapes1[0],
                "image_token_shape_t2": shapes2[0],
                "image_dense_2d_t1": dense2d1[0],
                "image_dense_2d_t2": dense2d2[0],
                "image_hidden_2d_t1": (
                    hidden.get("image_hidden_2d_t1_list") or [None]
                )[0],
                "image_hidden_2d_t2": (
                    hidden.get("image_hidden_2d_t2_list") or [None]
                )[0],
            })

        return PAIROutput(
            image_dense_t1=image_dense_t1,
            image_dense_t2=image_dense_t2,
            point_dense_t1=point_dense_t1,
            point_dense_t2=point_dense_t2,
            image_hidden_t1=hidden.get("image_hidden_t1"),
            image_hidden_t2=hidden.get("image_hidden_t2"),
            point_hidden_t1=hidden.get("point_hidden_t1"),
            point_hidden_t2=hidden.get("point_hidden_t2"),
            task_hidden=hidden.get("task_hidden"),
            logits=qwen_outputs.logits if return_logits else None,
            aux=aux,
        )

    @torch.no_grad()
    def generate(self, *, task_mode: str, prompt,
                 images_t1=None, images_t2=None,
                 point_dict_t1=None, point_dict_t2=None,
                 max_new_tokens: int = 64, do_sample: bool = False,
                 **generate_kwargs):
        # Keep generation simple and compatible. Batch generation is supported
        # by Qwen after prepare_inputs; point injection uses the same hook.
        task_mode = self._normalize_task_mode(task_mode)
        self._validate_inputs(
            task_mode=task_mode, images_t1=images_t1, images_t2=images_t2,
            point_dict_t1=point_dict_t1, point_dict_t2=point_dict_t2,
        )

        point_tokens_t1 = point_tokens_t2 = None
        if task_mode in ("3d", "2d3d"):
            out1 = self.encode_points(point_dict_t1)
            out2 = self.encode_points(point_dict_t2)
            point_tokens_t1, point_tokens_t2 = out1["point_tokens"], out2["point_tokens"]

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
        if any(x is not None for x in point_tokens):
            handle = self.qwen_backbone.model.get_input_embeddings().register_forward_hook(
                self._make_point_injection_hook(
                    point_tokens=point_tokens,
                    point_mask=prepared["point_mask"],
                    full_seq_len=inputs["input_ids"].shape[1],
                    stats=stats,
                )
            )

        try:
            ids = self.qwen_backbone.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, **generate_kwargs
            )
        finally:
            if handle is not None:
                handle.remove()

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompt_lens = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
        else:
            prompt_lens = attention_mask.sum(1).tolist()

        texts = []
        for b, prompt_len in enumerate(prompt_lens):
            new_ids = ids[b, int(prompt_len):]
            texts.append(self.qwen_backbone.processor.decode(
                new_ids, skip_special_tokens=True
            ))

        return PAIROutput(
            generated_ids=ids,
            generated_text=texts[0] if prepared["single_input"] else texts,
            aux={"task_mode": task_mode, "batch_size": prepared["batch_size"]},
        )


PAIR = PAIRModel
