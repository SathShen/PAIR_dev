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
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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


class PAIRChangeDecoder(nn.Module):
    """Shared Q decoder and complete modality-specific prediction heads.

    2D input:
      memory [B,L,D], task_hidden [B,qwen_dim],
      pixel_features_t1/t2/change [B,D,H,W].

    3D input:
      memory [B,L,D], task_hidden [B,qwen_dim],
      point_features_t1/t2 [N1,D]/[N2,D],
      event_features_t1/t2 [N1,D]/[N2,D],
      point_batch_t1/t2 [N1]/[N2].

    query_memory_mask is optional [B,Q,L], True=allowed. Alternatively,
    memory_time_ids [B,L] (1=T1, 2=T2, 0=padding) builds task-aware masks:
      2D sem1->T1, sem2->T2, change->both;
      3D sem1->T1, sem2->T2, events->both.

    The adapter is responsible for constructing cross-temporal change/event
    features and spatial memory. The entire path from these features to logits
    is owned by this module. No losses or post-sigmoid/softmax live here.
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

    def forward_2d(
        self,
        *,
        qwen_backbone,
        task_hidden: torch.Tensor,
        memory: torch.Tensor,
        pixel_features_t1: torch.Tensor,
        pixel_features_t2: torch.Tensor,
        change_features: torch.Tensor,
        prediction_mode: str = 'scd',
        class_names: Optional[Dict[int, str]] = None,
        output_sizes: Optional[Sequence[Tuple[int, int]]] = None,
        return_maps: bool = False,
        memory_mask: Optional[torch.Tensor] = None,
        query_memory_mask: Optional[torch.Tensor] = None,
        memory_time_ids: Optional[torch.Tensor] = None,
        detach_qwen_class_encoder: bool = True,
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
            or pixel_features_t1.shape != change_features.shape
        ):
            raise ValueError('2D feature maps must have matching [B,D,H,W] shapes')
        b, d, h, w = change_features.shape
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
        event_features_t1: torch.Tensor,
        event_features_t2: torch.Tensor,
        point_batch_t1: torch.Tensor,
        point_batch_t2: torch.Tensor,
        class_names: Dict[int, str],
        memory_mask: Optional[torch.Tensor] = None,
        query_memory_mask: Optional[torch.Tensor] = None,
        memory_time_ids: Optional[torch.Tensor] = None,
        detach_qwen_class_encoder: bool = True,
    ) -> PredictionLogits:
        """3D per-point semantic and active-support event logits, ready for loss."""
        if not class_names:
            raise ValueError('3D semantic classes are required')
        b = memory.shape[0]
        for name, features, point_batch in (
            ('point_features_t1', point_features_t1, point_batch_t1),
            ('point_features_t2', point_features_t2, point_batch_t2),
            ('event_features_t1', event_features_t1, point_batch_t1),
            ('event_features_t2', event_features_t2, point_batch_t2),
        ):
            self._check_point_features(features, point_batch, b, self.decoder_dim, name)
        encoded, queries = self._decode(
            qwen_backbone=qwen_backbone, task_hidden=task_hidden,
            task_descriptions=self.TASKS_3D, class_names=class_names,
            memory=memory, memory_mask=memory_mask,
            query_memory_mask=query_memory_mask,
            memory_time_ids=memory_time_ids, is_3d=True,
            detach_qwen_class_encoder=detach_qwen_class_encoder,
        )
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
        )
