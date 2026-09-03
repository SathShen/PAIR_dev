"""
PAIR Unified Dense Change Decoder
=================================

ViT / PTv3: dense perception
Qwen LLM: multimodal / temporal / task reasoning
Unified decoder: inject reasoning back into dense tokens and predict dense
semantic/change representations.

The decoder core is modality-agnostic. It never accepts fields such as
`image_hidden_2d_t1` or `point_hidden_t1`.

Expected decoder-side representation:
    dense token      = feature + xyz position + modality id + batch id
    reasoning token  = LLM hidden + xyz position + modality id + batch id

2D uses xyz=[x,y,0].
3D uses xyz=[x,y,z].
2D+3D concatenates both token types after thin modality adapters.

Temporal correspondence is supplied by sparse TemporalLinks:
    2D   -> same-grid / local-grid links
    3D   -> voxel / radius / KNN links
    2D3D -> world-coordinate links

The core therefore stays identical for all modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Unified structures
# =============================================================================


@dataclass
class UnifiedTokenSet:
    """
    Flat ragged token representation shared by 2D / 3D / 2D3D.

    features:     [N, D]
    positions:    [N, 3]
    modality_ids: [N]     0=image, 1=point
    batch_ids:    [N]
    """

    features: torch.Tensor
    positions: torch.Tensor
    modality_ids: torch.Tensor
    batch_ids: torch.Tensor

    def validate(
        self,
        *,
        feature_dim: Optional[int] = None,
        name: str = "tokens",
    ) -> None:
        if self.features.ndim != 2:
            raise ValueError(
                f"{name}.features must be [N,D], got {tuple(self.features.shape)}"
            )

        n = self.features.shape[0]

        if self.positions.shape != (n, 3):
            raise ValueError(
                f"{name}.positions must be [N,3], got {tuple(self.positions.shape)}"
            )

        if self.modality_ids.shape != (n,):
            raise ValueError(
                f"{name}.modality_ids must be [N], got {tuple(self.modality_ids.shape)}"
            )

        if self.batch_ids.shape != (n,):
            raise ValueError(
                f"{name}.batch_ids must be [N], got {tuple(self.batch_ids.shape)}"
            )

        if feature_dim is not None and self.features.shape[1] != int(feature_dim):
            raise ValueError(
                f"{name}.features dim must be {feature_dim}, "
                f"got {self.features.shape[1]}"
            )

        if self.modality_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name}.modality_ids must be integer tensor")

        if self.batch_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name}.batch_ids must be integer tensor")

        if not torch.isfinite(self.features).all():
            raise ValueError(f"{name}.features contains NaN/Inf")

        if not torch.isfinite(self.positions).all():
            raise ValueError(f"{name}.positions contains NaN/Inf")


@dataclass
class TemporalLinks:
    """
    Sparse cross-time neighborhood links.

    source_indices: [N_target, K]
    weights:        [N_target, K] or None

    Invalid source entries use -1.

    The decoder does not know whether links came from:
      - aligned image grids
      - point-cloud KNN/radius search
      - 2D/3D world-coordinate matching
    """

    source_indices: torch.Tensor
    weights: Optional[torch.Tensor] = None

    def validate(
        self,
        *,
        num_target: int,
        num_source: int,
        name: str = "links",
    ) -> None:
        if self.source_indices.ndim != 2:
            raise ValueError(
                f"{name}.source_indices must be [N,K]"
            )

        if self.source_indices.shape[0] != int(num_target):
            raise ValueError(
                f"{name}: expected {num_target} rows, "
                f"got {self.source_indices.shape[0]}"
            )

        if self.source_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"{name}.source_indices must be integer tensor"
            )

        valid = self.source_indices >= 0

        if valid.any():
            max_index = int(
                self.source_indices[valid].max().item()
            )

            if max_index >= int(num_source):
                raise IndexError(
                    f"{name}: source index {max_index} exceeds "
                    f"source token count {num_source}"
                )

        if self.weights is not None:
            if self.weights.shape != self.source_indices.shape:
                raise ValueError(
                    f"{name}.weights must match source_indices shape"
                )

            if not torch.isfinite(self.weights).all():
                raise ValueError(
                    f"{name}.weights contains NaN/Inf"
                )


@dataclass
class UnifiedDecoderOutput:
    semantic_feature_t1: torch.Tensor
    semantic_feature_t2: torch.Tensor

    change_feature_t1: torch.Tensor
    change_feature_t2: torch.Tensor

    semantic_logits_t1: torch.Tensor
    semantic_logits_t2: torch.Tensor

    change_logits_t1: torch.Tensor
    change_logits_t2: torch.Tensor

    raw_class_ids: Tuple[int, ...]
    class_names: Tuple[str, ...]


# =============================================================================
# Unified token embedding
# =============================================================================


class CoordinateEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(xyz.float())


class UnifiedTokenEmbedding(nn.Module):
    """
    feature + position + modality + time
    """

    def __init__(
        self,
        dim: int,
        num_modalities: int = 2,
    ):
        super().__init__()

        self.position_encoder = CoordinateEncoder(dim)
        self.modality_embedding = nn.Embedding(
            num_modalities,
            dim,
        )
        self.time_embedding = nn.Embedding(
            2,
            dim,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: UnifiedTokenSet,
        *,
        time_id: int,
    ) -> torch.Tensor:
        x = tokens.features

        pos = self.position_encoder(
            tokens.positions
        ).to(
            dtype=x.dtype,
            device=x.device,
        )

        mod = self.modality_embedding(
            tokens.modality_ids.long()
        ).to(dtype=x.dtype)

        time_ids = torch.full(
            (x.shape[0],),
            fill_value=int(time_id),
            dtype=torch.long,
            device=x.device,
        )

        time = self.time_embedding(
            time_ids
        ).to(dtype=x.dtype)

        return self.norm(
            x + pos + mod + time
        )


# =============================================================================
# Dense <- LLM reasoning injection
# =============================================================================


class ReasoningInjection(nn.Module):
    """
    Dense queries attend to a much smaller reasoning-token set.

    Query chunking prevents one huge attention matrix for large point clouds.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        query_chunk_size: int = 4096,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.query_chunk_size = int(query_chunk_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

    def _one_batch(
        self,
        dense: torch.Tensor,
        reasoning: torch.Tensor,
    ) -> torch.Tensor:
        if dense.shape[0] == 0 or reasoning.shape[0] == 0:
            return dense

        kv = self.norm_kv(
            reasoning
        ).unsqueeze(0)

        outputs = []

        for start in range(
            0,
            dense.shape[0],
            self.query_chunk_size,
        ):
            end = min(
                start + self.query_chunk_size,
                dense.shape[0],
            )

            q = self.norm_q(
                dense[start:end]
            ).unsqueeze(0)

            attended, _ = self.attention(
                q,
                kv,
                kv,
                need_weights=False,
            )

            outputs.append(
                attended[0]
            )

        injected = torch.cat(
            outputs,
            dim=0,
        )

        return self.out_norm(
            dense + injected
        )

    def forward(
        self,
        *,
        dense: torch.Tensor,
        dense_batch_ids: torch.Tensor,
        reasoning: torch.Tensor,
        reasoning_batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(dense)

        for batch_id in torch.unique(
            dense_batch_ids
        ).tolist():
            dense_mask = (
                dense_batch_ids == int(batch_id)
            )

            reasoning_mask = (
                reasoning_batch_ids == int(batch_id)
            )

            output[dense_mask] = self._one_batch(
                dense[dense_mask],
                reasoning[reasoning_mask],
            )

        return output


# =============================================================================
# Sparse T1 <-> T2 fusion
# =============================================================================


class SparseTemporalFusion(nn.Module):
    """
    Every target dense token receives a weighted local context from the
    other time using externally supplied sparse links.

    Complexity is O(N*K), not O(N^2).
    """

    def __init__(self, dim: int):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def gather_cross_context(
        *,
        source: torch.Tensor,
        links: TemporalLinks,
    ) -> torch.Tensor:
        index = links.source_indices.long()
        valid = index >= 0
        safe_index = index.clamp(min=0)

        gathered = source[
            safe_index
        ]

        if links.weights is None:
            weights = valid.to(
                dtype=source.dtype
            )
        else:
            weights = links.weights.to(
                dtype=source.dtype,
                device=source.device,
            )

            weights = weights * valid.to(
                dtype=weights.dtype
            )

        denominator = weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        normalized_weights = (
            weights / denominator
        )

        cross = (
            gathered
            * normalized_weights.unsqueeze(-1)
        ).sum(dim=1)

        has_neighbor = valid.any(
            dim=1,
            keepdim=True,
        )

        return torch.where(
            has_neighbor,
            cross,
            torch.zeros_like(cross),
        )

    def forward(
        self,
        *,
        target: torch.Tensor,
        source: torch.Tensor,
        links: TemporalLinks,
    ) -> torch.Tensor:
        links.validate(
            num_target=target.shape[0],
            num_source=source.shape[0],
            name="temporal_links",
        )

        cross = self.gather_cross_context(
            source=source,
            links=links,
        )

        fused = self.fuse(
            torch.cat(
                [
                    target,
                    cross,
                    torch.abs(target - cross),
                    target * cross,
                ],
                dim=-1,
            )
        )

        return self.norm(
            target + fused
        )


# =============================================================================
# Task conditioning
# =============================================================================


class TaskConditioning(nn.Module):
    """
    <TASK> hidden -> FiLM conditioning over every dense token.
    """

    def __init__(
        self,
        qwen_dim: int,
        decoder_dim: int,
    ):
        super().__init__()

        self.to_film = nn.Linear(
            qwen_dim,
            decoder_dim * 2,
        )

        self.norm = nn.LayerNorm(
            decoder_dim
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        batch_ids: torch.Tensor,
        task_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if task_hidden.ndim != 2:
            raise ValueError(
                f"task_hidden must be [B,D], got {tuple(task_hidden.shape)}"
            )

        gamma_beta = self.to_film(
            task_hidden
        )

        gamma, beta = gamma_beta.chunk(
            2,
            dim=-1,
        )

        gamma = torch.tanh(
            gamma
        )

        token_gamma = gamma[
            batch_ids.long()
        ]

        token_beta = beta[
            batch_ids.long()
        ]

        return self.norm(
            x * (1.0 + token_gamma)
            + token_beta
        )


# =============================================================================
# Shared decoder block
# =============================================================================


class SharedDenseBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(
            dim * float(mlp_ratio)
        )

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(
                dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (
            x
            + self.mlp(
                self.norm(x)
            )
        )


# =============================================================================
# Qwen class prototype encoder
# =============================================================================


class QwenClassPrototypeEncoder(nn.Module):
    """
    DatasetSpec.class_names
        -> text prompts
        -> Qwen Transformer
        -> semantic prototypes

    Example:
        {0: "unchanged", 10: "water", 50: "building"}

    gives only 3 prototypes.
    Raw IDs are kept only for target/prediction mapping.
    """

    def __init__(
        self,
        *,
        qwen_dim: int,
        decoder_dim: int,
        prompt_template: str = (
            "A remote sensing semantic class: {name}."
        ),
    ):
        super().__init__()

        self.qwen_dim = int(qwen_dim)
        self.decoder_dim = int(decoder_dim)
        self.prompt_template = str(
            prompt_template
        )

        self.projection = nn.Sequential(
            nn.Linear(
                self.qwen_dim,
                self.decoder_dim,
            ),
            nn.LayerNorm(
                self.decoder_dim
            ),
        )

    @staticmethod
    def normalize_class_dict(
        class_names: Dict[int, str],
    ) -> Tuple[
        Tuple[int, ...],
        Tuple[str, ...],
    ]:
        if not isinstance(
            class_names,
            dict,
        ):
            raise TypeError(
                "DatasetSpec.class_names must be Dict[int, str]"
            )

        if not class_names:
            raise ValueError(
                "class_names cannot be empty"
            )

        normalized = {}

        for raw_id, name in class_names.items():
            if isinstance(raw_id, bool) or not isinstance(
                raw_id,
                int,
            ):
                raise TypeError(
                    f"class ID must be int, got {raw_id!r}"
                )

            if not isinstance(name, str) or not name.strip():
                raise TypeError(
                    f"class name for ID {raw_id} must be non-empty str"
                )

            normalized[int(raw_id)] = (
                name.strip()
            )

        raw_ids = tuple(
            sorted(
                normalized.keys()
            )
        )

        names = tuple(
            normalized[raw_id]
            for raw_id in raw_ids
        )

        return raw_ids, names

    @staticmethod
    def _masked_mean(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.to(
            dtype=hidden.dtype
        ).unsqueeze(-1)

        return (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )

    def forward(
        self,
        *,
        class_names: Dict[int, str],
        qwen_backbone,
        detach_qwen: bool = True,
    ):
        raw_ids, names = (
            self.normalize_class_dict(
                class_names
            )
        )

        prompts = [
            self.prompt_template.format(
                name=name
            )
            for name in names
        ]

        tokenizer = qwen_backbone.tokenizer
        qwen_model = qwen_backbone.model

        device = next(
            qwen_model.parameters()
        ).device

        encoded = tokenizer(
            prompts,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        def run_qwen():
            output = qwen_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            hidden = output.hidden_states[-1]

            return self._masked_mean(
                hidden,
                encoded["attention_mask"],
            )

        if detach_qwen:
            with torch.no_grad():
                language_hidden = (
                    run_qwen()
                ).detach()
        else:
            language_hidden = run_qwen()

        projection_dtype = (
            self.projection[0].weight.dtype
        )

        language_hidden = language_hidden.to(
            dtype=projection_dtype
        )

        prototypes = self.projection(
            language_hidden
        )

        prototypes = F.normalize(
            prototypes.float(),
            dim=-1,
        )

        return (
            raw_ids,
            names,
            prototypes,
        )


# =============================================================================
# Unified decoder
# =============================================================================


class UnifiedChangeDecoder(nn.Module):
    """
    One decoder core for 2d / 3d / 2d3d.

    Dense features are expected to have already passed through a thin
    modality adapter into decoder_dim.

    Reasoning features are raw Qwen hidden states with qwen_dim.
    """

    def __init__(
        self,
        *,
        decoder_dim: int = 256,
        qwen_dim: int = 2560,
        num_heads: int = 8,
        num_shared_blocks: int = 2,
        reasoning_chunk_size: int = 4096,
        dropout: float = 0.0,
        num_modalities: int = 2,
        initial_logit_scale: float = 10.0,
    ):
        super().__init__()

        self.decoder_dim = int(
            decoder_dim
        )
        self.qwen_dim = int(
            qwen_dim
        )

        self.token_embedding = UnifiedTokenEmbedding(
            dim=self.decoder_dim,
            num_modalities=num_modalities,
        )

        self.reasoning_projection = (
            nn.Sequential(
                nn.Linear(
                    self.qwen_dim,
                    self.decoder_dim,
                ),
                nn.LayerNorm(
                    self.decoder_dim
                ),
            )
        )

        self.reasoning_injection = (
            ReasoningInjection(
                dim=self.decoder_dim,
                num_heads=num_heads,
                dropout=dropout,
                query_chunk_size=reasoning_chunk_size,
            )
        )

        self.temporal_fusion = (
            SparseTemporalFusion(
                dim=self.decoder_dim
            )
        )

        self.task_conditioning = (
            TaskConditioning(
                qwen_dim=self.qwen_dim,
                decoder_dim=self.decoder_dim,
            )
        )

        self.shared_blocks = nn.ModuleList(
            [
                SharedDenseBlock(
                    dim=self.decoder_dim,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(
                    int(num_shared_blocks)
                )
            ]
        )

        # Modality-independent latent branches.
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(
                self.decoder_dim
            ),
            nn.Linear(
                self.decoder_dim,
                self.decoder_dim,
            ),
            nn.GELU(),
        )

        self.change_head = nn.Sequential(
            nn.LayerNorm(
                self.decoder_dim
            ),
            nn.Linear(
                self.decoder_dim,
                self.decoder_dim,
            ),
            nn.GELU(),
        )

        self.binary_change_classifier = (
            nn.Linear(
                self.decoder_dim,
                1,
            )
        )

        self.class_encoder = (
            QwenClassPrototypeEncoder(
                qwen_dim=self.qwen_dim,
                decoder_dim=self.decoder_dim,
            )
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(
                float(
                    initial_logit_scale
                )
            ).log()
        )

    def _prepare_reasoning(
        self,
        tokens: UnifiedTokenSet,
        *,
        time_id: int,
    ):
        tokens.validate(
            feature_dim=self.qwen_dim,
            name="reasoning_tokens",
        )

        projected = UnifiedTokenSet(
            features=self.reasoning_projection(
                tokens.features
            ),
            positions=tokens.positions,
            modality_ids=tokens.modality_ids,
            batch_ids=tokens.batch_ids,
        )

        embedded = self.token_embedding(
            projected,
            time_id=time_id,
        )

        return (
            embedded,
            tokens.batch_ids,
        )

    def _semantic_logits(
        self,
        feature: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        feature = F.normalize(
            feature.float(),
            dim=-1,
        )

        scale = self.logit_scale.exp().clamp(
            min=1.0,
            max=100.0,
        )

        return scale * (
            feature @ prototypes.T
        )

    def _decode_one_time(
        self,
        *,
        dense_tokens: UnifiedTokenSet,
        dense_time_id: int,
        reasoning_tokens: UnifiedTokenSet,
        reasoning_time_id: int,
        task_hidden: torch.Tensor,
    ) -> torch.Tensor:
        dense_tokens.validate(
            feature_dim=self.decoder_dim,
            name="dense_tokens",
        )

        x = self.token_embedding(
            dense_tokens,
            time_id=dense_time_id,
        )

        reasoning, reasoning_batch_ids = (
            self._prepare_reasoning(
                reasoning_tokens,
                time_id=reasoning_time_id,
            )
        )

        x = self.reasoning_injection(
            dense=x,
            dense_batch_ids=dense_tokens.batch_ids,
            reasoning=reasoning,
            reasoning_batch_ids=reasoning_batch_ids,
        )

        x = self.task_conditioning(
            x=x,
            batch_ids=dense_tokens.batch_ids,
            task_hidden=task_hidden,
        )

        return x

    def forward(
        self,
        *,
        dense_t1: UnifiedTokenSet,
        dense_t2: UnifiedTokenSet,

        reasoning_t1: UnifiedTokenSet,
        reasoning_t2: UnifiedTokenSet,

        task_hidden: torch.Tensor,

        links_t1_to_t2: TemporalLinks,
        links_t2_to_t1: TemporalLinks,

        class_names: Dict[int, str],
        qwen_backbone,

        detach_qwen_class_encoder: bool = True,
    ) -> UnifiedDecoderOutput:

        dense_t1.validate(
            feature_dim=self.decoder_dim,
            name="dense_t1",
        )
        dense_t2.validate(
            feature_dim=self.decoder_dim,
            name="dense_t2",
        )

        reasoning_t1.validate(
            feature_dim=self.qwen_dim,
            name="reasoning_t1",
        )
        reasoning_t2.validate(
            feature_dim=self.qwen_dim,
            name="reasoning_t2",
        )

        # 1) LLM reasoning -> full dense space.
        x1 = self._decode_one_time(
            dense_tokens=dense_t1,
            dense_time_id=0,
            reasoning_tokens=reasoning_t1,
            reasoning_time_id=0,
            task_hidden=task_hidden,
        )

        x2 = self._decode_one_time(
            dense_tokens=dense_t2,
            dense_time_id=1,
            reasoning_tokens=reasoning_t2,
            reasoning_time_id=1,
            task_hidden=task_hidden,
        )

        # 2) Sparse local T1 <-> T2 interaction.
        # Both directions use pre-fusion x1/x2.
        temporal_x1 = self.temporal_fusion(
            target=x1,
            source=x2,
            links=links_t1_to_t2,
        )

        temporal_x2 = self.temporal_fusion(
            target=x2,
            source=x1,
            links=links_t2_to_t1,
        )

        x1 = temporal_x1
        x2 = temporal_x2

        # 3) Same shared blocks regardless of modality.
        for block in self.shared_blocks:
            x1 = block(x1)
            x2 = block(x2)

        # 4) Unified latent outputs.
        semantic_feature_t1 = (
            self.semantic_head(x1)
        )
        semantic_feature_t2 = (
            self.semantic_head(x2)
        )

        change_feature_t1 = (
            self.change_head(x1)
        )
        change_feature_t2 = (
            self.change_head(x2)
        )

        # 5) Dataset-specific language prototypes.
        (
            raw_class_ids,
            ordered_class_names,
            prototypes,
        ) = self.class_encoder(
            class_names=class_names,
            qwen_backbone=qwen_backbone,
            detach_qwen=detach_qwen_class_encoder,
        )

        semantic_logits_t1 = (
            self._semantic_logits(
                semantic_feature_t1,
                prototypes,
            )
        )
        semantic_logits_t2 = (
            self._semantic_logits(
                semantic_feature_t2,
                prototypes,
            )
        )

        # 6) Dataset-independent binary change logits.
        change_logits_t1 = (
            self.binary_change_classifier(
                change_feature_t1
            )[:, 0]
        )
        change_logits_t2 = (
            self.binary_change_classifier(
                change_feature_t2
            )[:, 0]
        )

        return UnifiedDecoderOutput(
            semantic_feature_t1=semantic_feature_t1,
            semantic_feature_t2=semantic_feature_t2,
            change_feature_t1=change_feature_t1,
            change_feature_t2=change_feature_t2,
            semantic_logits_t1=semantic_logits_t1,
            semantic_logits_t2=semantic_logits_t2,
            change_logits_t1=change_logits_t1,
            change_logits_t2=change_logits_t2,
            raw_class_ids=raw_class_ids,
            class_names=ordered_class_names,
        )


# =============================================================================
# Small helper for aligned 2D testing
# =============================================================================


def build_identity_temporal_links(
    num_tokens: int,
    *,
    device: torch.device,
) -> TemporalLinks:
    """
    For an already co-registered 2D grid:
    each T1 token points to the same-position T2 token.

    3D will later use its own voxel/radius/KNN link builder.
    """

    source_indices = torch.arange(
        int(num_tokens),
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)

    weights = torch.ones(
        (
            int(num_tokens),
            1,
        ),
        dtype=torch.float32,
        device=device,
    )

    return TemporalLinks(
        source_indices=source_indices,
        weights=weights,
    )
