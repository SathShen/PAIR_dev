"""PAIR shared decoder, task queries and final prediction logits.

- QwenClassPrototypeEncoder: class descriptions + task descriptions + sample TASK hidden.
- TransformerQueryDecoder: shared 2D/3D task query cross-attention over spatial memory.
- Image/Point heads: modality-specific, within-modality T1/T2 weights shared.
- PAIRChangeDecoder: complete predictions BEFORE the existing loss.py.

ImageAdapter and PointAdapter only build spatial memories and high-resolution /
original-point features. They DO NOT hold classification heads.

2D loss-ready logits: semantic_t1/t2 [N,C] (optional for BCD), change [N].
3D loss-ready logits: semantic_t1/t2 [N1,C], [N2,C],
                      event_t1/t2 [N1,3], [N2,3].

3D event protocol: 0 unchanged, 1 removed (T1 only), 2 added (T2 only).

Optional multi-box spatial guidance:
  2D boxes [B,K,4] normalized xyxy coordinates, valid mask [B,K].
  3D boxes [B,K,6] in the same XYZ reference frame as input points;
      optional box_scene_bounds [B,2,3] normalizes 3D box query positions.
  K may be zero. Region queries read the same time-compatible memory, and
  soft gates yield residual FEATURE correction only (never crop or hard mask).
  Box proposal generation/supervision is upstream, NOT implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QueryEncodingOutput:
    """Outputs from a single Qwen text-encoding pass.

    task_queries: [B,Q,D], sample-conditioned queries in decoder space.
    semantic_prototypes: [C,D] or None, L2-normalized text prototypes.
    raw_class_ids / class_names: semantic channel order; empty for BCD.
    """

    task_queries: torch.Tensor
    semantic_prototypes: Optional[torch.Tensor]
    raw_class_ids: Tuple[int, ...]
    class_names: Tuple[str, ...]


class QwenClassPrototypeEncoder(nn.Module):
    """Use one Qwen text pass to encode class and task descriptions.

    Class prototypes and task prompts share a projection but differ in use:
      - class vectors [C,D] are L2-normalized for semantic classification;
      - task vectors [Q,D] combine with the current sample's Qwen <TASK>
        hidden [B,qwen_dim], yielding sample-conditioned queries [B,Q,D].

    The Qwen text forward can be detached without detaching gradients from
    the learnable projections or the sample's multimodal <TASK> hidden.
    """

    def __init__(
        self,
        *,
        qwen_dim: int,
        decoder_dim: int = 256,
        prompt_template: str = "A remote sensing semantic class: {name}.",
        task_prompt_template: str = "A remote sensing change detection task: {name}.",
    ) -> None:
        super().__init__()
        if qwen_dim <= 0 or decoder_dim <= 0:
            raise ValueError("qwen_dim and decoder_dim must be positive")
        if "{name}" not in prompt_template or "{name}" not in task_prompt_template:
            raise ValueError("Both prompt templates must contain '{name}'")

        self.qwen_dim = int(qwen_dim)
        self.decoder_dim = int(decoder_dim)
        self.prompt_template = str(prompt_template)
        self.task_prompt_template = str(task_prompt_template)

        self.projection = nn.Sequential(
            nn.Linear(self.qwen_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
        )
        self.sample_projection = nn.Linear(self.qwen_dim, self.decoder_dim)
        self.query_norm = nn.LayerNorm(self.decoder_dim)

    @staticmethod
    def normalize_class_dict(
        class_names: Dict[int, str],
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        if not isinstance(class_names, dict) or not class_names:
            raise ValueError("class_names must be a nonempty Dict[int, str]")
        normalized = {}
        for raw_id, name in class_names.items():
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise TypeError(f"class ID must be int, got {raw_id!r}")
            if not isinstance(name, str) or not name.strip():
                raise TypeError(f"class name for ID {raw_id} must be nonempty str")
            normalized[raw_id] = name.strip()
        raw_ids = tuple(sorted(normalized))
        return raw_ids, tuple(normalized[raw_id] for raw_id in raw_ids)

    @staticmethod
    def _last_valid_hidden(
        hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("Expected hidden [P,S,D] and attention_mask [P,S]")
        valid = attention_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Every text prompt must contain at least one valid token")
        positions = torch.arange(valid.shape[1], device=valid.device)
        last = positions.masked_fill(~valid, -1).amax(dim=1)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last]

    @staticmethod
    def _run_text_qwen(
        qwen_backbone, prompts: Sequence[str], *, detach_qwen: bool
    ) -> torch.Tensor:
        tokenizer = qwen_backbone.tokenizer
        qwen_model = qwen_backbone.model
        if tokenizer.eos_token is None:
            raise RuntimeError("Qwen tokenizer must define an EOS token")
        encoded = tokenizer(
            [text + tokenizer.eos_token for text in prompts],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        device = next(qwen_model.parameters()).device
        ids = encoded["input_ids"].to(device=device)
        mask = encoded["attention_mask"].to(device=device)

        def execute() -> torch.Tensor:
            output = qwen_model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            return QwenClassPrototypeEncoder._last_valid_hidden(
                output.hidden_states[-1], mask
            )

        if not detach_qwen:
            return execute()

        # A text-only, read-only pass must not alter the configured train/eval
        # behavior of the multimodal model (including LoRA dropout modules).
        training_states = [(m, m.training) for m in qwen_model.modules()]
        qwen_model.eval()
        try:
            with torch.no_grad():
                result = execute().detach()
        finally:
            for module, was_training in training_states:
                module.training = was_training
        return result

    def forward(
        self,
        *,
        qwen_backbone,
        task_hidden: torch.Tensor,
        task_descriptions: Sequence[str],
        class_names: Optional[Dict[int, str]] = None,
        detach_qwen: bool = True,
    ) -> QueryEncodingOutput:
        if task_hidden.ndim == 3 and task_hidden.shape[1] == 1:
            task_hidden = task_hidden[:, 0, :]
        if task_hidden.ndim != 2 or task_hidden.shape[1] != self.qwen_dim:
            raise ValueError(
                f"task_hidden must be [B,{self.qwen_dim}], got "
                f"{tuple(task_hidden.shape)}"
            )
        if task_hidden.shape[0] == 0:
            raise ValueError("task_hidden must have a nonempty batch")
        if isinstance(task_descriptions, str) or not task_descriptions:
            raise ValueError("task_descriptions must be a nonempty sequence of strings")
        tasks = tuple(task_descriptions)
        if any(not isinstance(t, str) or not t.strip() for t in tasks):
            raise ValueError("Every task description must be nonempty text")

        if class_names is None:
            raw_ids, ordered_names = (), ()
        else:
            raw_ids, ordered_names = self.normalize_class_dict(class_names)

        prompts = [self.prompt_template.format(name=n) for n in ordered_names]
        prompts.extend(self.task_prompt_template.format(name=t.strip()) for t in tasks)
        hidden = self._run_text_qwen(
            qwen_backbone, prompts, detach_qwen=detach_qwen
        )
        if hidden.shape != (len(prompts), self.qwen_dim):
            raise RuntimeError(
                f"Qwen text hidden shape {tuple(hidden.shape)} does not match "
                f"({len(prompts)},{self.qwen_dim})"
            )

        # Projected text features and sample features remain trainable even
        # when the text-only Qwen pass is detached.
        projection_dtype = self.projection[0].weight.dtype
        text_features = self.projection(hidden.to(dtype=projection_dtype))
        num_classes = len(raw_ids)
        semantic_prototypes = (
            F.normalize(text_features[:num_classes].float(), dim=-1)
            if num_classes > 0
            else None
        )
        task_vectors = text_features[num_classes:]
        sample_vectors = self.sample_projection(
            task_hidden.to(device=task_vectors.device, dtype=projection_dtype)
        )
        task_queries = self.query_norm(
            task_vectors.unsqueeze(0) + sample_vectors.unsqueeze(1)
        )
        return QueryEncodingOutput(
            task_queries=task_queries,
            semantic_prototypes=semantic_prototypes,
            raw_class_ids=raw_ids,
            class_names=ordered_names,
        )


class TransformerDecoderLayer(nn.Module):
    """Pre-norm cross-attention + FFN. Query is never a dense point/pixel map.

    MHA is batch_first; query [B,Q,D], memory [B,L,D]. Boolean masks follow
    the PAIR convention True=ALLOWED (inverted internally for PyTorch MHA).
    """

    def __init__(
        self,
        *,
        decoder_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if decoder_dim <= 0 or num_heads <= 0 or decoder_dim % num_heads != 0:
            raise ValueError("decoder_dim must be positive and divisible by num_heads")
        if mlp_ratio <= 0 or not (0.0 <= dropout < 1.0):
            raise ValueError("mlp_ratio must be positive and dropout in [0,1)")
        self.num_heads = int(num_heads)
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.memory_norm = nn.LayerNorm(decoder_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(decoder_dim)
        hidden_dim = max(1, int(round(decoder_dim * mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decoder_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        context, _ = self.cross_attention(
            normed,
            normalized_memory,
            normalized_memory,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        query = query + self.dropout(context)
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query


class TransformerQueryDecoder(nn.Module):
    """One weight-shared query decoder for both 2D and 3D PAIR adapters.

    memory_mask: [B,L], True for a real/unpadded memory token.
    query_memory_mask: [B,Q,L], True where a particular Query may attend.
    This allows T1-only / T2-only / both-time temporal attention without
    introducing separate modality-specific decoders.

    Returns [B,Q,D] updated queries. Final prediction heads are in this file.
    """

    def __init__(
        self,
        *,
        decoder_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if decoder_dim <= 0 or num_layers <= 0:
            raise ValueError("decoder_dim and num_layers must be positive")
        self.decoder_dim = int(decoder_dim)
        self.num_heads = int(num_heads)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    decoder_dim=decoder_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(decoder_dim)

    def forward(
        self,
        *,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        query_memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("query and memory must be [B,Q,D] and [B,L,D]")
        b, q, d = query.shape
        bm, l, dm = memory.shape
        if b != bm or d != self.decoder_dim or dm != self.decoder_dim:
            raise ValueError(
                f"Expected query [B,Q,{self.decoder_dim}] and "
                f"memory [B,L,{self.decoder_dim}] with matching B, got "
                f"{tuple(query.shape)} and {tuple(memory.shape)}"
            )
        if b == 0 or q == 0 or l == 0:
            raise ValueError("Batch, query, and memory lengths must all be nonzero")
        if query.device != memory.device:
            raise ValueError("query and memory must be on the same device")
        if memory.dtype != query.dtype:
            # Vision / point memories can arrive in BF16 while the learnable
            # text-query projection runs in FP32. MHA requires matching dtypes.
            memory = memory.to(dtype=query.dtype)

        if memory_mask is not None:
            if memory_mask.shape != (b, l) or memory_mask.dtype != torch.bool:
                raise ValueError("memory_mask must be bool [B,L], True=valid")
            if memory_mask.device != memory.device:
                raise ValueError("memory_mask must be on memory.device")
        if query_memory_mask is not None:
            if (
                query_memory_mask.shape != (b, q, l)
                or query_memory_mask.dtype != torch.bool
            ):
                raise ValueError("query_memory_mask must be bool [B,Q,L], True=allowed")
            if query_memory_mask.device != memory.device:
                raise ValueError("query_memory_mask must be on memory.device")

        key_padding_mask = None
        attn_mask = None
        if query_memory_mask is None:
            if memory_mask is not None:
                if not memory_mask.any(dim=-1).all():
                    raise ValueError("Every sample needs at least one valid memory token")
                key_padding_mask = ~memory_mask
        else:
            allowed = query_memory_mask
            if memory_mask is not None:
                allowed = allowed & memory_mask.unsqueeze(1)
            if not allowed.any(dim=-1).all():
                raise ValueError("Every query must have at least one allowed memory token")
            attn_mask = (
                (~allowed)
                .unsqueeze(1)
                .expand(b, self.num_heads, q, l)
                .reshape(b * self.num_heads, q, l)
            )

        for layer in self.layers:
            query = layer(
                query,
                memory,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        return self.final_norm(query)


# -----------------------------------------------------------------------------
# Prediction heads. They live HERE, not inside the modality adapters.
# -----------------------------------------------------------------------------


@dataclass
class PredictionLogits:
    """Loss-ready flattened logits unless forward_2d(return_maps=True).

    Semantic channels follow sorted raw class IDs; raw_class_ids is metadata
    only (loss.py still receives its dataset class_names mapping).
    """

    semantic_logits_t1: Optional[torch.Tensor] = None
    semantic_logits_t2: Optional[torch.Tensor] = None
    change_logits: Optional[torch.Tensor] = None
    event_logits_t1: Optional[torch.Tensor] = None
    event_logits_t2: Optional[torch.Tensor] = None
    raw_class_ids: Tuple[int, ...] = ()
    class_names: Tuple[str, ...] = ()
    updated_queries: Optional[torch.Tensor] = None
    box_guidance_applied: bool = False


class QueryConditionedClassHead(nn.Module):
    """One task query + C prototypes -> C spatial classifiers.

    For each class c, w_c = MLP([prototype_c, query,
    prototype_c * query]). The multiplicative interaction ensures the query
    affects class-specific weights, rather than adding the same constant to
    every class logit (which would cancel under softmax).

    Receives dense features as [B,P,D] (images) or [N,D] (ragged points).
    This class is instantiated SEPARATELY for 2D semantics, 3D semantics,
    and 3D events; only T1/T2 of a task share its instance.
    """

    def __init__(self, decoder_dim: int = 256, logit_scale_init: float = 10.0):
        super().__init__()
        if decoder_dim <= 0 or logit_scale_init <= 0:
            raise ValueError('decoder_dim and logit_scale_init must be positive')
        self.decoder_dim = int(decoder_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(3 * decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))

    def class_weights(
        self, query: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        if query.ndim != 2 or query.shape[-1] != self.decoder_dim:
            raise ValueError(f'query must be [B,{self.decoder_dim}]')
        if (
            prototypes.ndim != 2
            or prototypes.shape[-1] != self.decoder_dim
            or prototypes.shape[0] == 0
        ):
            raise ValueError(f'prototypes must be nonempty [C,{self.decoder_dim}]')
        query = query.to(dtype=self.weight_mlp[0].weight.dtype)
        b, d = query.shape
        c = prototypes.shape[0]
        p = prototypes.to(device=query.device, dtype=query.dtype).unsqueeze(0).expand(b, c, d)
        q = query.unsqueeze(1).expand(b, c, d)
        weights = self.weight_mlp(torch.cat([p, q, p * q], dim=-1))
        return F.normalize(weights.float(), dim=-1)

    def _features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.decoder_dim:
            raise ValueError('feature width does not match decoder_dim')
        features = features.to(dtype=self.feature_projection[0].weight.dtype)
        return F.normalize(self.feature_projection(features).float(), dim=-1)

    def forward_image(
        self, features: torch.Tensor, query: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """features [B,D,H,W] -> logits [B,C,H,W]."""
        if features.ndim != 4 or features.shape[1] != self.decoder_dim:
            raise ValueError(f'image features must be [B,{self.decoder_dim},H,W]')
        b, _, h, w = features.shape
        if query.shape != (b, self.decoder_dim):
            raise ValueError('image query batch/width does not match features')
        pixels = features.flatten(2).transpose(1, 2)
        weights = self.class_weights(query, prototypes)
        scale = self.logit_scale.float().clamp(max=math.log(50.0)).exp()
        logits = scale * torch.einsum('bpd,bcd->bpc', self._features(pixels), weights)
        return logits.transpose(1, 2).reshape(b, -1, h, w)

    def forward_points(
        self,
        features: torch.Tensor,
        batch_ids: torch.Tensor,
        query: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """features [N,D] + batch_ids [N] -> logits [N,C]."""
        if features.ndim != 2 or features.shape[1] != self.decoder_dim:
            raise ValueError(f'point features must be [N,{self.decoder_dim}]')
        if batch_ids.ndim != 1 or batch_ids.numel() != features.shape[0]:
            raise ValueError('batch_ids must be [N] and match point count')
        weights = self.class_weights(query, prototypes)
        scale = self.logit_scale.float().clamp(max=math.log(50.0)).exp()
        return scale * torch.einsum(
            'nd,ncd->nc', self._features(features), weights.index_select(0, batch_ids)
        )


class QueryConditionedBinaryHead(nn.Module):
    """Updated change Q generates one spatial classifier, not two channels."""

    def __init__(self, decoder_dim: int = 256, logit_scale_init: float = 10.0):
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim), nn.LayerNorm(decoder_dim)
        )
        self.query_to_weight = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.query_to_bias = nn.Linear(decoder_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))

    def forward(self, features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """[B,D,H,W] + [B,D] -> [B,H,W] raw (pre-sigmoid) logits."""
        if features.ndim != 4 or features.shape[1] != self.decoder_dim:
            raise ValueError(f'change features must be [B,{self.decoder_dim},H,W]')
        b, _, h, w = features.shape
        if query.shape != (b, self.decoder_dim):
            raise ValueError('change query shape mismatch')
        feature_dtype = self.feature_projection[0].weight.dtype
        pixels = F.normalize(
            self.feature_projection(
                features.flatten(2).transpose(1, 2).to(dtype=feature_dtype)
            ).float(), dim=-1,
        )
        query = query.to(dtype=self.query_to_weight[0].weight.dtype)
        weights = F.normalize(self.query_to_weight(query).float(), dim=-1)
        bias = self.query_to_bias(query).float()
        scale = self.logit_scale.float().clamp(max=math.log(50.0)).exp()
        logits = scale * torch.einsum('bpd,bd->bp', pixels, weights) + bias
        return logits.reshape(b, h, w)


class MultiBoxGuidance(nn.Module):
    """Optional 2D/3D multi-region guidance with NO hard restriction.

    An instance represents both modalities but encodes 2D/3D positions with
    separate small MLPs. It does not predict boxes. Coordinates must share
    the SAME spatial reference frame as the dense pixels / point coordinates.

    A box proposes where to refine, not where predictions are allowed.
    The base image/point features always survive via the residual skip.
    """

    def __init__(
        self,
        decoder_dim: int = 256,
        *,
        fourier_bands: int = 4,
        max_boxes: int = 64,
        edge_softness: float = 0.08,
        box_chunk_size: int = 8,
        init_residual_strength: float = 0.10,
    ) -> None:
        super().__init__()
        if fourier_bands < 1 or max_boxes < 1 or box_chunk_size < 1:
            raise ValueError('fourier_bands, max_boxes, and chunk size must be positive')
        if edge_softness <= 0 or not 0 < init_residual_strength < 1:
            raise ValueError('edge_softness must be positive and residual strength in (0,1)')
        self.decoder_dim = int(decoder_dim)
        self.fourier_bands = int(fourier_bands)
        self.max_boxes = int(max_boxes)
        self.edge_softness = float(edge_softness)
        self.box_chunk_size = int(box_chunk_size)
        frequencies = (2.0 ** torch.arange(fourier_bands, dtype=torch.float32)) * math.pi
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.pos_2d = nn.Sequential(
            nn.Linear(4 * 2 * fourier_bands, decoder_dim),
            nn.GELU(), nn.Linear(decoder_dim, decoder_dim),
        )
        self.pos_3d = nn.Sequential(
            nn.Linear(6 * 2 * fourier_bands, decoder_dim),
            nn.GELU(), nn.Linear(decoder_dim, decoder_dim),
        )
        # Separate 2D/3D adapters; unchanged 2D/3D heads remain independent.
        self.feature_2d = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.feature_3d = nn.Linear(decoder_dim, decoder_dim, bias=False)
        start = math.log(init_residual_strength / (1.0 - init_residual_strength))
        self.strength_2d = nn.Parameter(torch.tensor(start, dtype=torch.float32))
        self.strength_3d = nn.Parameter(torch.tensor(start, dtype=torch.float32))

    def check_boxes(
        self,
        boxes: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        scores: Optional[torch.Tensor],
        *,
        batch_size: int,
        dimension: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if boxes is None:
            if valid is not None or scores is not None:
                raise ValueError('box_valid and box_scores require boxes')
            return None, None, None
        expected = (4 if dimension == 2 else 6)
        if boxes.ndim != 3 or boxes.shape[0] != batch_size or boxes.shape[2] != expected:
            raise ValueError(f'boxes must be [B,K,{expected}]')
        if boxes.device != device or not torch.is_floating_point(boxes):
            raise ValueError('boxes must be floating-point on memory.device')
        if boxes.shape[1] > self.max_boxes:
            raise ValueError(f'K={boxes.shape[1]} exceeds max_boxes={self.max_boxes}; '
                             'select proposals upstream; do not silently truncate')
        if not torch.isfinite(boxes).all():
            raise ValueError('boxes must be finite')
        b, k, _ = boxes.shape
        if valid is None:
            valid = torch.ones((b,k), device=device, dtype=torch.bool)
        if valid.dtype != torch.bool or valid.shape != (b,k) or valid.device != device:
            raise ValueError('box_valid must be bool [B,K] on memory.device')
        if scores is None:
            scores = torch.ones((b,k),device=device,dtype=torch.float32)
        if scores.shape != (b,k) or scores.device != device or not torch.is_floating_point(scores):
            raise ValueError('box_scores must be floating-point [B,K] on memory.device')
        if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('box_scores must be finite probabilities in [0,1]')
        # Validate only marked-valid boxes. Padded invalid slots may be zero.
        half = expected // 2
        if valid.any():
            kept = boxes[valid]
            if not (kept[:,half:] > kept[:,:half]).all():
                raise ValueError('valid boxes must have min < max in every axis')
            if dimension == 2 and ((kept < 0).any() or (kept > 1).any()):
                raise ValueError('2D boxes must use normalized xyxy coordinates in [0,1]')
        return boxes, valid, scores.float() * valid.float()

    def encode(self, boxes: torch.Tensor, *, dimension: int) -> torch.Tensor:
        freqs = self.frequencies.to(device=boxes.device)
        scaled = boxes.float().unsqueeze(-1) * freqs
        embedding = torch.cat((scaled.sin(), scaled.cos()), dim=-1).flatten(-2)
        linear = self.pos_2d if dimension == 2 else self.pos_3d
        return linear(embedding.to(dtype=linear[0].weight.dtype))

    def _soft_gate(self, xyz: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """xyz [...,1,D], boxes [...,K,D] endpoints (min,max) -> [...,K].

        Gates use box-relative edge width rather than absolute pixels/meters.
        The 3D scene-coordinate units can therefore be metres or normalized.
        """
        dimensions = xyz.shape[-1]
        lo, hi = boxes[..., :dimensions], boxes[..., dimensions:]
        width = (hi - lo).clamp_min(1e-6)
        epsilon = width * self.edge_softness
        left = torch.sigmoid((xyz - lo) / epsilon)
        right = torch.sigmoid((hi - xyz) / epsilon)
        return (left * right).prod(dim=-1)

    def refine_image(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        scores: torch.Tensor,
        region_queries: torch.Tensor,
    ) -> torch.Tensor:
        """[B,D,H,W] + [B,K,4] + [B,K,D] -> [B,D,H,W]."""
        b, d, h, w = features.shape
        if boxes.shape[:2] != region_queries.shape[:2] or region_queries.shape[-1] != d:
            raise ValueError('region query count/width does not match boxes/features')
        if not (valid & (scores > 0)).any():
            return features
        # Only small [B,chunk,H,W] gate tensors are constructed. Avoid [B,K,D,H,W].
        yy = (torch.arange(h, device=features.device, dtype=torch.float32) + .5) / h
        xx = (torch.arange(w, device=features.device, dtype=torch.float32) + .5) / w
        y, x = torch.meshgrid(yy,xx,indexing='ij')
        xy = torch.stack((x,y),dim=-1).view(1,1,h,w,2)
        total = torch.zeros((b,1,h,w),device=features.device,dtype=torch.float32)
        union = torch.zeros_like(total)
        pooled = torch.zeros((b,d,h,w),device=features.device,dtype=torch.float32)
        transforms = self.feature_2d(region_queries.to(dtype=self.feature_2d.weight.dtype)).float()
        for j in range(0,boxes.shape[1],self.box_chunk_size):
            sl = slice(j,min(j+self.box_chunk_size,boxes.shape[1]))
            box = boxes[:,sl].float().view(b,-1,1,1,4)
            gate = self._soft_gate(xy,box).mul(scores[:,sl,None,None])
            union = torch.maximum(union,gate.amax(dim=1,keepdim=True))
            total = total + gate.sum(dim=1,keepdim=True)
            pooled = pooled + torch.einsum('bkhw,bkd->bdhw',gate,transforms[:,sl])
        context = pooled / total.clamp_min(1e-6)
        delta = union * torch.tanh(context) * torch.sigmoid(self.strength_2d)
        return features + delta.to(dtype=features.dtype)

    def refine_points(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        batch_ids: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        scores: torch.Tensor,
        region_queries: torch.Tensor,
    ) -> torch.Tensor:
        """[N,D] + XYZ [N,3] + boxes [B,K,6] -> [N,D].

        Split by sample to avoid giant [N,K,D] tensors; only [point_chunk,K]
        pairwise gates are materialized. No matching of T1/T2 point indices.
        """
        if xyz.shape != (features.shape[0],3) or xyz.device != features.device:
            raise ValueError('point XYZ must be [N,3] on the feature device')
        if not torch.isfinite(xyz).all():
            raise ValueError('point XYZ must be finite')
        if not (valid & (scores > 0)).any():
            return features
        weights = self.feature_3d(region_queries.to(dtype=self.feature_3d.weight.dtype)).float()
        output = features.clone()
        for ib in range(boxes.shape[0]):
            sample_index = torch.nonzero(batch_ids == ib,as_tuple=False).flatten()
            if sample_index.numel() == 0 or not (valid[ib] & (scores[ib] > 0)).any():
                continue
            kept = valid[ib] & (scores[ib] > 0)
            local_boxes=boxes[ib,kept].float()
            local_scores=scores[ib,kept]
            local_weights=weights[ib,kept]
            # Bounded number of points per iteration; preserves ordering.
            for start in range(0,len(sample_index),65536):
                select=sample_index[start:start+65536]
                coordinates=xyz[select].float()
                total=torch.zeros((len(select),1),device=xyz.device)
                union=torch.zeros_like(total)
                pooled=torch.zeros((len(select),features.shape[1]),device=xyz.device)
                for j in range(0,local_boxes.shape[0],self.box_chunk_size):
                    sl=slice(j,min(j+self.box_chunk_size,local_boxes.shape[0]))
                    gate=self._soft_gate(
                        coordinates[:,None,:],local_boxes[sl][None,:,:]
                    ) * local_scores[None,sl]
                    union=torch.maximum(union,gate.max(dim=1,keepdim=True).values)
                    total=total+gate.sum(dim=1,keepdim=True)
                    pooled=pooled+gate @ local_weights[sl]
                context=pooled / total.clamp_min(1e-6)
                correction=union * torch.tanh(context) * torch.sigmoid(self.strength_3d)
                output=output.index_add(0,select,correction.to(dtype=output.dtype))
        return output


class PAIRChangeDecoder(nn.Module):
    """Shared Q decoder and complete modality-specific prediction heads.

    2D input:
      memory [B,L,D], task_hidden [B,qwen_dim],
      pixel_features_t1/t2 [B,D,H,W]. Decoder builds F_change.

    3D input:
      memory [B,L,D], task_hidden [B,qwen_dim],
      point_features_t1/t2 [N1,D]/[N2,D],
      point_batch_t1/t2 and XYZ; Decoder builds F_event T1/T2.

    query_memory_mask is optional [B,Q,L], True=allowed. Alternatively,
    memory_time_ids [B,L] (1=T1, 2=T2, 0=padding) builds task-aware masks:
      2D sem1->T1, sem2->T2, change->both;
      3D sem1->T1, sem2->T2, events->both.

    Adapters only construct F_pixel/F_point, spatial memory and coordinates.
    This Decoder owns all learnable temporal fusion, Event matching, boxes,
    queries and logits. No losses or post-sigmoid/softmax live here.
    """

    TASKS_2D_SCD = ('Semantic T1', 'Semantic T2', 'Binary Change')
    TASKS_2D_BCD = ('Binary Change',)
    TASKS_3D = ('Semantic T1', 'Semantic T2', 'Event T1', 'Event T2')
    EVENT_SUPPORT_T1 = (0, 1)
    EVENT_SUPPORT_T2 = (0, 2)

    def __init__(
        self,
        *,
        qwen_dim: int,
        decoder_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_box_proposals: int = 64,
        box_edge_softness: float = 0.08,
        box_chunk_size: int = 8,
        box_residual_init: float = 0.10,
        temporal_channels: int = 64,
        temporal_cell_size: float = 0.5,
        event_hidden_dim: int = 64,
        event_chunk_size: int = 8192,
        event_checkpoint: bool = True,
        event_residual_init: float = 0.10,
    ) -> None:
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.qwen_dim = int(qwen_dim)
        self.class_encoder = QwenClassPrototypeEncoder(
            qwen_dim=qwen_dim, decoder_dim=decoder_dim
        )
        self.query_decoder = TransformerQueryDecoder(
            decoder_dim=decoder_dim, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        # Shared query decoder, but NOT shared image and point prediction heads.
        self.image_semantic_head = QueryConditionedClassHead(decoder_dim)
        self.image_change_head = QueryConditionedBinaryHead(decoder_dim)
        self.point_semantic_head = QueryConditionedClassHead(decoder_dim)
        self.point_event_head = QueryConditionedClassHead(decoder_dim)
        self.event_prototypes = nn.Parameter(torch.randn(3, decoder_dim) * 0.02)
        # 2D temporal fusion formerly in ImageAdapter (same operators/widths).
        if temporal_channels <= 0:
            raise ValueError('temporal_channels must be positive')
        image_norm_groups = max(g for g in range(min(16, decoder_dim), 0, -1)
                                if decoder_dim % g == 0)
        self.temporal_reduce = nn.Conv2d(decoder_dim, temporal_channels, kernel_size=1)
        self.change_fuse = nn.Sequential(
            nn.Conv2d(4 * temporal_channels, decoder_dim, kernel_size=1, bias=False),
            nn.GroupNorm(image_norm_groups, decoder_dim), nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1,
                      groups=decoder_dim, bias=False),
            nn.GroupNorm(image_norm_groups, decoder_dim), nn.GELU(),
        )
        # 3D temporal fusion formerly in PointAdapter.
        if event_hidden_dim <= 0 or event_chunk_size <= 0 or temporal_cell_size <= 0:
            raise ValueError('3D temporal settings must be positive')
        if not 0 < event_residual_init < 1:
            raise ValueError('event_residual_init must be inside (0,1)')
        self.temporal_cell_size = float(temporal_cell_size)
        self.event_chunk_size = int(event_chunk_size)
        self.event_checkpoint = bool(event_checkpoint)
        self.temporal_event_mlp = nn.Sequential(
            nn.Linear(2 * decoder_dim + 2, event_hidden_dim), nn.GELU(),
            nn.Linear(event_hidden_dim, decoder_dim),
        )
        self.temporal_event_norm = nn.LayerNorm(decoder_dim)
        self.temporal_event_strength = nn.Parameter(torch.tensor(
            math.log(event_residual_init / (1 - event_residual_init))
        ))
        self.box_guidance = MultiBoxGuidance(
            decoder_dim=decoder_dim,
            max_boxes=max_box_proposals,
            edge_softness=box_edge_softness,
            box_chunk_size=box_chunk_size,
            init_residual_strength=box_residual_init,
        )

    def _fuse_temporal_features(self, features_t1: torch.Tensor,
                                features_t2: torch.Tensor) -> torch.Tensor:
        """2D F_change from aligned F_pixel T1/T2, shared for SCD and BCD."""
        a = self.temporal_reduce(features_t1)
        b = self.temporal_reduce(features_t2)
        return self.change_fuse(torch.cat((a, b, (b-a).abs(), a*b), dim=1))

    @staticmethod
    def _shared_grid_keys(xyz_t1: torch.Tensor, xyz_t2: torch.Tensor,
                          cell_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match by a common XYZ voxel grid (NOT exact nearest-neighbor matching)."""
        a, b = xyz_t1.detach().float(), xyz_t2.detach().float()
        origin = torch.minimum(a.amin(0), b.amin(0))
        maximum = torch.maximum(a.amax(0), b.amax(0))
        sizes = torch.floor((maximum - origin) / cell_size).long() + 2
        nx, ny, nz = (int(v) for v in sizes.tolist())
        if nx * ny * nz >= (1 << 62):
            raise OverflowError('3D temporal grid exceeds int64 range')
        def keys(x):
            grid = torch.floor((x - origin) / cell_size).long()
            if (grid < 0).any():
                raise ValueError('inconsistent 3D coordinate reference')
            return grid[:, 0] * (ny * nz) + grid[:, 1] * nz + grid[:, 2]
        return keys(a), keys(b)

    @staticmethod
    def _same_voxel_lookup(target_key: torch.Tensor, source_key: torch.Tensor):
        """Return one representative source index per matching voxel; O(N log N)."""
        ordered, perm = torch.sort(source_key)
        ix = torch.searchsorted(ordered.contiguous(), target_key.contiguous())
        ix = ix.clamp(max=ordered.numel() - 1)
        return perm[ix], ordered[ix] == target_key

    def _encode_event_features(self, local: torch.Tensor, other: torch.Tensor,
                               local_xyz: torch.Tensor, other_xyz: torch.Tensor,
                               source_indices: torch.Tensor,
                               matched: torch.Tensor) -> torch.Tensor:
        """Chunked local cross-temporal F_event, preserving one logit per input point."""
        parts = []
        dtype = self.temporal_event_mlp[0].weight.dtype
        for start in range(0, local.shape[0], self.event_chunk_size):
            end = min(start + self.event_chunk_size, local.shape[0])
            a = local[start:end]
            pick = source_indices[start:end]
            mask = matched[start:end, None]
            other_features = other[pick] * mask.to(dtype=other.dtype)
            distance = ((local_xyz[start:end].float() - other_xyz[pick].float())
                        .square().sum(-1, keepdim=True).sqrt())
            distance = (distance / self.temporal_cell_size).clamp(max=4.0) * mask.float()
            fields = torch.cat((a, other_features - a, distance.to(dtype=a.dtype),
                                mask.to(dtype=a.dtype)), dim=-1).to(dtype)
            if self.training and self.event_checkpoint:
                correction = checkpoint(self.temporal_event_mlp, fields,
                                        use_reentrant=False)
            else:
                correction = self.temporal_event_mlp(fields)
            parts.append(a + torch.sigmoid(self.temporal_event_strength) *
                         self.temporal_event_norm(correction.to(dtype=a.dtype)))
        return torch.cat(parts, dim=0)

    def _fuse_temporal_points(
        self,
        point_features_t1: torch.Tensor,
        point_features_t2: torch.Tensor,
        point_xyz_t1: torch.Tensor,
        point_xyz_t2: torch.Tensor,
        point_batch_t1: torch.Tensor,
        point_batch_t2: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """3D matching and Event feature construction are owned by Decoder."""
        event1, event2 = [], []
        for bid in range(batch_size):
            ix1 = torch.nonzero(point_batch_t1 == bid, as_tuple=False).flatten()
            ix2 = torch.nonzero(point_batch_t2 == bid, as_tuple=False).flatten()
            if ix1.numel() == 0 or ix2.numel() == 0:
                raise ValueError('each 3D batch item must contain T1 and T2 points')
            a, b = point_features_t1[ix1], point_features_t2[ix2]
            xyz1, xyz2 = point_xyz_t1[ix1], point_xyz_t2[ix2]
            with torch.no_grad():
                key1, key2 = self._shared_grid_keys(xyz1, xyz2, self.temporal_cell_size)
                to2, valid1 = self._same_voxel_lookup(key1, key2)
                to1, valid2 = self._same_voxel_lookup(key2, key1)
            event1.append(self._encode_event_features(a, b, xyz1, xyz2, to2, valid1))
            event2.append(self._encode_event_features(b, a, xyz2, xyz1, to1, valid2))
        return torch.cat(event1, dim=0), torch.cat(event2, dim=0)

    @staticmethod
    def _make_temporal_mask(
        memory_time_ids: torch.Tensor,
        query_count: int,
        *,
        is_3d: bool,
    ) -> torch.Tensor:
        if memory_time_ids.ndim != 2 or memory_time_ids.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        ):
            raise ValueError('memory_time_ids must be an integer tensor [B,L]')
        if not torch.all((memory_time_ids >= 0) & (memory_time_ids <= 2)):
            raise ValueError('memory_time_ids only permits 0=pad, 1=T1, 2=T2')
        b, length = memory_time_ids.shape
        both = memory_time_ids != 0
        if is_3d:
            if query_count != 4:
                raise ValueError('3D requires exactly four task queries')
            masks = (memory_time_ids == 1, memory_time_ids == 2, both, both)
        elif query_count == 3:
            masks = (memory_time_ids == 1, memory_time_ids == 2, both)
        elif query_count == 1:
            masks = (both,)
        else:
            raise ValueError('2D requires one BCD query or three SCD queries')
        mask = torch.stack(masks, dim=1)
        if mask.shape != (b, query_count, length):
            raise AssertionError('unexpected temporal mask dimensions')
        return mask

    def _decode(
        self,
        *,
        qwen_backbone,
        task_hidden: torch.Tensor,
        task_descriptions: Sequence[str],
        class_names: Optional[Dict[int, str]],
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        query_memory_mask: Optional[torch.Tensor],
        memory_time_ids: Optional[torch.Tensor],
        is_3d: bool,
        detach_qwen_class_encoder: bool,
    ) -> Tuple[QueryEncodingOutput, torch.Tensor]:
        if memory_time_ids is not None:
            if query_memory_mask is not None:
                raise ValueError('Pass only one of memory_time_ids / query_memory_mask')
            if memory_time_ids.shape != memory.shape[:2]:
                raise ValueError('memory_time_ids shape must be [B,L]')
            if memory_time_ids.device != memory.device:
                raise ValueError('memory_time_ids must be on memory.device')
            query_memory_mask = self._make_temporal_mask(
                memory_time_ids, len(task_descriptions), is_3d=is_3d,
            )
            if memory_mask is not None:
                if not torch.equal(memory_mask, memory_time_ids != 0):
                    raise ValueError('memory_mask must match nonzero memory_time_ids')
            else:
                memory_mask = memory_time_ids != 0
        encoding = self.class_encoder(
            qwen_backbone=qwen_backbone,
            task_hidden=task_hidden,
            task_descriptions=task_descriptions,
            class_names=class_names,
            detach_qwen=detach_qwen_class_encoder,
        )
        queries = self.query_decoder(
            query=encoding.task_queries,
            memory=memory,
            memory_mask=memory_mask,
            query_memory_mask=query_memory_mask,
        )
        return encoding, queries

    def _regional_queries(
        self,
        *,
        global_query: torch.Tensor,
        boxes_for_encoding: torch.Tensor,
        dimension: int,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        memory_time_ids: Optional[torch.Tensor],
        query_memory_mask: Optional[torch.Tensor],
        task_index: int,
        allowed_time: Optional[int] = None,
    ) -> torch.Tensor:
        """Box+task Q -> same SHARED decoder; masks mirror corresponding task Q."""
        b, k, _ = boxes_for_encoding.shape
        initial = global_query.unsqueeze(1) + self.box_guidance.encode(
            boxes_for_encoding, dimension=dimension
        )
        scoped = None
        if query_memory_mask is not None:
            scoped = query_memory_mask[:,task_index:task_index+1].expand(-1,k,-1)
        elif memory_time_ids is not None:
            allowed = (memory_time_ids == allowed_time) if allowed_time in (1,2) else (memory_time_ids != 0)
            scoped = allowed.unsqueeze(1).expand(-1,k,-1)
        return self.query_decoder(
            query=initial,
            memory=memory,
            memory_mask=memory_mask if memory_time_ids is None else (memory_time_ids != 0),
            query_memory_mask=scoped,
        )

    @staticmethod
    def _normalize_box3d(
        boxes: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize world XYZ box boundaries for stable Fourier encoding."""
        if bounds.shape != (boxes.shape[0],2,3):
            raise ValueError('box_scene_bounds must be [B,2,3]')
        if bounds.device != boxes.device or not torch.isfinite(bounds).all():
            raise ValueError('box_scene_bounds must be finite and on memory.device')
        extent = bounds[:,1]-bounds[:,0]
        if (extent <= 0).any():
            raise ValueError('box_scene_bounds max must exceed min in all axes')
        return torch.cat(((boxes[:,:,:3]-bounds[:,None,0,:]) / extent[:,None,:],
                          (boxes[:,:,3:]-bounds[:,None,0,:]) / extent[:,None,:]),dim=-1)

    def forward_2d(
        self,
        *,
        qwen_backbone,
        task_hidden: torch.Tensor,
        memory: torch.Tensor,
        pixel_features_t1: torch.Tensor,
        pixel_features_t2: torch.Tensor,
        prediction_mode: str = 'scd',
        class_names: Optional[Dict[int, str]] = None,
        output_sizes: Optional[Sequence[Tuple[int, int]]] = None,
        return_maps: bool = False,
        memory_mask: Optional[torch.Tensor] = None,
        query_memory_mask: Optional[torch.Tensor] = None,
        memory_time_ids: Optional[torch.Tensor] = None,
        detach_qwen_class_encoder: bool = True,
        boxes_2d: Optional[torch.Tensor] = None,
        box_valid: Optional[torch.Tensor] = None,
        box_scores: Optional[torch.Tensor] = None,
    ) -> PredictionLogits:
        """Return flat logits for loss.py, or dense logits if return_maps=True.

        output_sizes handles variable per-sample target raster H,W with bilinear
        logit interpolation (not interpolating class IDs/probabilities).
        """
        mode = prediction_mode.strip().lower()
        if mode not in ('scd', 'bcd'):
            raise ValueError("prediction_mode must be 'scd' or 'bcd'")
        if mode == 'scd' and not class_names:
            raise ValueError('SCD requires a nonempty class_names dict')
        if mode == 'bcd' and class_names:
            raise ValueError('BCD cannot receive semantic class definitions')
        if mode == 'bcd':
            class_names = None
        if (
            pixel_features_t1.ndim != 4
            or pixel_features_t1.shape != pixel_features_t2.shape
        ):
            raise ValueError('2D feature maps must have matching [B,D,H,W] shapes')
        b, d, h, w = pixel_features_t1.shape
        if d != self.decoder_dim or memory.shape[0] != b:
            raise ValueError('2D image feature width / memory batch mismatch')
        if output_sizes is not None:
            if len(output_sizes) != b or any(
                len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0
                for size in output_sizes
            ):
                raise ValueError('output_sizes must contain B positive (H,W) pairs')
            if return_maps and any(tuple(map(int,s)) != (h,w) for s in output_sizes):
                raise ValueError('return_maps=True requires output_sizes to match feature size')

        change_features = self._fuse_temporal_features(pixel_features_t1, pixel_features_t2)
        descriptions = self.TASKS_2D_SCD if mode == 'scd' else self.TASKS_2D_BCD
        encoded, queries = self._decode(
            qwen_backbone=qwen_backbone, task_hidden=task_hidden,
            task_descriptions=descriptions,
            class_names=class_names, memory=memory,
            memory_mask=memory_mask,
            query_memory_mask=query_memory_mask,
            memory_time_ids=memory_time_ids,
            is_3d=False,
            detach_qwen_class_encoder=detach_qwen_class_encoder,
        )
        boxes_2d, box_valid, box_scores = self.box_guidance.check_boxes(
            boxes_2d, box_valid, box_scores, batch_size=b,
            dimension=2, device=memory.device,
        )
        use_boxes = (boxes_2d is not None and boxes_2d.shape[1] > 0
                     and bool((box_valid & (box_scores > 0)).any()))
        if use_boxes:
            idx = 2 if mode == 'scd' else 0
            change_regions = self._regional_queries(
                global_query=queries[:,idx],
                boxes_for_encoding=boxes_2d,
                dimension=2, memory=memory,
                memory_mask=memory_mask,
                memory_time_ids=memory_time_ids,
                query_memory_mask=query_memory_mask,
                task_index=idx,
            )
            change_features = self.box_guidance.refine_image(
                change_features, boxes_2d, box_valid, box_scores, change_regions
            )
            if mode == 'scd':
                sem1_regions = self._regional_queries(
                    global_query=queries[:,0], boxes_for_encoding=boxes_2d,
                    dimension=2, memory=memory, memory_mask=memory_mask,
                    memory_time_ids=memory_time_ids,
                    query_memory_mask=query_memory_mask, task_index=0,allowed_time=1,
                )
                sem2_regions = self._regional_queries(
                    global_query=queries[:,1], boxes_for_encoding=boxes_2d,
                    dimension=2, memory=memory, memory_mask=memory_mask,
                    memory_time_ids=memory_time_ids,
                    query_memory_mask=query_memory_mask, task_index=1,allowed_time=2,
                )
                pixel_features_t1 = self.box_guidance.refine_image(
                    pixel_features_t1,boxes_2d,box_valid,box_scores,sem1_regions,
                )
                pixel_features_t2 = self.box_guidance.refine_image(
                    pixel_features_t2,boxes_2d,box_valid,box_scores,sem2_regions,
                )
        semantic_t1 = semantic_t2 = None
        if mode == 'scd':
            proto = encoded.semantic_prototypes
            semantic_t1 = self.image_semantic_head.forward_image(
                pixel_features_t1, queries[:, 0], proto
            )
            semantic_t2 = self.image_semantic_head.forward_image(
                pixel_features_t2, queries[:, 1], proto
            )
        change_query = queries[:, 2] if mode == 'scd' else queries[:, 0]
        change = self.image_change_head(change_features, change_query)
        if not return_maps:
            semantic_t1, semantic_t2, change = self._flatten_2d(
                semantic_t1, semantic_t2, change, output_sizes
            )
        return PredictionLogits(
            semantic_logits_t1=semantic_t1,
            semantic_logits_t2=semantic_t2,
            change_logits=change,
            raw_class_ids=encoded.raw_class_ids,
            class_names=encoded.class_names,
            updated_queries=queries,
            box_guidance_applied=use_boxes,
        )

    @staticmethod
    def _flatten_2d(
        semantic_t1: Optional[torch.Tensor],
        semantic_t2: Optional[torch.Tensor],
        change: torch.Tensor,
        output_sizes: Optional[Sequence[Tuple[int,int]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        b, h, w = change.shape
        if output_sizes is None:
            output_sizes = ((h, w),) * b
        sem1_parts, sem2_parts, change_parts = [], [], []
        for index, size in enumerate(output_sizes):
            target_hw = (int(size[0]), int(size[1]))
            def sem_part(sem):
                if sem is None:
                    return None
                x = sem[index:index+1]
                if tuple(x.shape[-2:]) != target_hw:
                    x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
                return x[0].permute(1,2,0).reshape(-1, x.shape[1])
            a, bb = sem_part(semantic_t1), sem_part(semantic_t2)
            if a is not None: sem1_parts.append(a)
            if bb is not None: sem2_parts.append(bb)
            c = change[index:index+1].unsqueeze(1)
            if tuple(c.shape[-2:]) != target_hw:
                c = F.interpolate(c, size=target_hw, mode='bilinear', align_corners=False)
            change_parts.append(c.reshape(-1))
        return (
            torch.cat(sem1_parts, dim=0) if sem1_parts else None,
            torch.cat(sem2_parts, dim=0) if sem2_parts else None,
            torch.cat(change_parts, dim=0),
        )

    @staticmethod
    def _check_point_features(
        features: torch.Tensor, batch_ids: torch.Tensor, batch_size: int,
        decoder_dim: int, name: str,
    ) -> None:
        if (
            features.ndim != 2 or features.shape[1] != decoder_dim or
            batch_ids.ndim != 1 or batch_ids.shape[0] != features.shape[0] or
            batch_ids.dtype != torch.long or batch_ids.device != features.device
        ):
            raise ValueError(f'{name}: expected features [N,{decoder_dim}] and long batch_ids [N]')
        if features.shape[0] == 0:
            raise ValueError(f'{name}: empty point cloud')
        if (batch_ids < 0).any() or (batch_ids >= batch_size).any():
            raise ValueError(f'{name}: batch_ids out of [0,B)')

    @staticmethod
    def _mask_event_logits(logits: torch.Tensor, *, phase: int) -> torch.Tensor:
        """Prevent illegal event categories before both loss and inference.

        Values remain [N,3] with a finite minimum suitable for FP32/BF16.
        masked_fill also blocks gradients toward the illegal logit channel.
        Loss still performs its independent active-support CE and Dice.
        """
        illegal_index = 2 if phase == 1 else 1 if phase == 2 else None
        if illegal_index is None:
            raise ValueError('phase must be 1 or 2')
        invalid = torch.zeros_like(logits, dtype=torch.bool)
        invalid[:, illegal_index] = True
        return logits.masked_fill(invalid, -1e4)

    def forward_3d(
        self,
        *,
        qwen_backbone,
        task_hidden: torch.Tensor,
        memory: torch.Tensor,
        point_features_t1: torch.Tensor,
        point_features_t2: torch.Tensor,
        point_batch_t1: torch.Tensor,
        point_batch_t2: torch.Tensor,
        class_names: Dict[int, str],
        memory_mask: Optional[torch.Tensor] = None,
        query_memory_mask: Optional[torch.Tensor] = None,
        memory_time_ids: Optional[torch.Tensor] = None,
        detach_qwen_class_encoder: bool = True,
        boxes_3d: Optional[torch.Tensor] = None,
        box_valid: Optional[torch.Tensor] = None,
        box_scores: Optional[torch.Tensor] = None,
        box_scene_bounds: Optional[torch.Tensor] = None,
        point_xyz_t1: Optional[torch.Tensor] = None,
        point_xyz_t2: Optional[torch.Tensor] = None,
    ) -> PredictionLogits:
        """3D per-point semantic and active-support event logits, ready for loss."""
        if not class_names:
            raise ValueError('3D semantic classes are required')
        b = memory.shape[0]
        for name, features, point_batch in (
            ('point_features_t1', point_features_t1, point_batch_t1),
            ('point_features_t2', point_features_t2, point_batch_t2),
        ):
            self._check_point_features(features, point_batch, b, self.decoder_dim, name)
        # XYZ is always required for 3D temporal fusion, even without boxes.
        if point_xyz_t1 is None or point_xyz_t2 is None:
            raise ValueError('3D Event fusion requires point_xyz_t1 and point_xyz_t2')
        for xyz, features, name in (
            (point_xyz_t1, point_features_t1, 'point_xyz_t1'),
            (point_xyz_t2, point_features_t2, 'point_xyz_t2'),
        ):
            if xyz.shape != (features.shape[0], 3) or xyz.device != features.device:
                raise ValueError(f'{name} must have shape [N,3] and match point feature device')
            if not torch.isfinite(xyz).all():
                raise ValueError(f'{name} must be finite')
        # Keep F_event independent of Box: Box-guided refinements are residuals.
        event_features_t1, event_features_t2 = self._fuse_temporal_points(
            point_features_t1, point_features_t2,
            point_xyz_t1, point_xyz_t2,
            point_batch_t1, point_batch_t2, b,
        )
        encoded, queries = self._decode(
            qwen_backbone=qwen_backbone, task_hidden=task_hidden,
            task_descriptions=self.TASKS_3D, class_names=class_names,
            memory=memory, memory_mask=memory_mask,
            query_memory_mask=query_memory_mask,
            memory_time_ids=memory_time_ids, is_3d=True,
            detach_qwen_class_encoder=detach_qwen_class_encoder,
        )
        boxes_3d, box_valid, box_scores = self.box_guidance.check_boxes(
            boxes_3d, box_valid, box_scores, batch_size=b,
            dimension=3, device=memory.device,
        )
        use_boxes = (boxes_3d is not None and boxes_3d.shape[1] > 0
                     and bool((box_valid & (box_scores > 0)).any()))
        if use_boxes:
            if point_xyz_t1 is None or point_xyz_t2 is None or box_scene_bounds is None:
                raise ValueError('3D box guidance requires point XYZ (T1/T2) and '
                                 'box_scene_bounds in the same spatial reference frame')
            normalized = self._normalize_box3d(boxes_3d,box_scene_bounds)
            e1_regions = self._regional_queries(
                global_query=queries[:,2], boxes_for_encoding=normalized,
                dimension=3, memory=memory, memory_mask=memory_mask,
                memory_time_ids=memory_time_ids,
                query_memory_mask=query_memory_mask,task_index=2,
            )
            e2_regions = self._regional_queries(
                global_query=queries[:,3], boxes_for_encoding=normalized,
                dimension=3, memory=memory, memory_mask=memory_mask,
                memory_time_ids=memory_time_ids,
                query_memory_mask=query_memory_mask,task_index=3,
            )
            event_features_t1 = self.box_guidance.refine_points(
                event_features_t1,point_xyz_t1,point_batch_t1,
                boxes_3d,box_valid,box_scores,e1_regions,
            )
            event_features_t2 = self.box_guidance.refine_points(
                event_features_t2,point_xyz_t2,point_batch_t2,
                boxes_3d,box_valid,box_scores,e2_regions,
            )
            s1_regions = self._regional_queries(
                global_query=queries[:,0], boxes_for_encoding=normalized,
                dimension=3, memory=memory, memory_mask=memory_mask,
                memory_time_ids=memory_time_ids,
                query_memory_mask=query_memory_mask,task_index=0,allowed_time=1,
            )
            s2_regions = self._regional_queries(
                global_query=queries[:,1], boxes_for_encoding=normalized,
                dimension=3, memory=memory, memory_mask=memory_mask,
                memory_time_ids=memory_time_ids,
                query_memory_mask=query_memory_mask,task_index=1,allowed_time=2,
            )
            point_features_t1 = self.box_guidance.refine_points(
                point_features_t1,point_xyz_t1,point_batch_t1,
                boxes_3d,box_valid,box_scores,s1_regions,
            )
            point_features_t2 = self.box_guidance.refine_points(
                point_features_t2,point_xyz_t2,point_batch_t2,
                boxes_3d,box_valid,box_scores,s2_regions,
            )
            # Semantic logits are recomputed below after the refinement.
        sem1 = self.point_semantic_head.forward_points(
            point_features_t1, point_batch_t1, queries[:,0], encoded.semantic_prototypes
        )
        sem2 = self.point_semantic_head.forward_points(
            point_features_t2, point_batch_t2, queries[:,1], encoded.semantic_prototypes
        )
        evt_proto = F.normalize(self.event_prototypes.float(), dim=-1)
        evt1 = self.point_event_head.forward_points(
            event_features_t1, point_batch_t1, queries[:,2], evt_proto
        )
        evt2 = self.point_event_head.forward_points(
            event_features_t2, point_batch_t2, queries[:,3], evt_proto
        )
        evt1 = self._mask_event_logits(evt1, phase=1)
        evt2 = self._mask_event_logits(evt2, phase=2)
        return PredictionLogits(
            semantic_logits_t1=sem1, semantic_logits_t2=sem2,
            event_logits_t1=evt1, event_logits_t2=evt2,
            raw_class_ids=encoded.raw_class_ids,
            class_names=encoded.class_names,
            updated_queries=queries,
            box_guidance_applied=use_boxes,
        )
