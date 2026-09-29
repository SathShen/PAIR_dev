"""
PAIR Unified Dense Change Decoder
=================================

Shared dense decoder for 2D / 3D / future 2D+3D PAIR paths.

Design:
- dense token = adapted dense feature + xyz + modality id + time id
- reasoning token = Qwen hidden + xyz + modality id + time id
- sparse TemporalLinks provide cross-time correspondence
- semantic prediction uses Qwen language class prototypes
- 2D keeps the binary change head
- 3D uses one shared 3-class event head:
    0 unchanged
    1 added
    2 removed

The decoder core is modality-agnostic. Dataset-specific active event support
belongs to the loss/metrics layer, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Unified structures
# =============================================================================


@dataclass
class UnifiedTokenSet:
    """Flat ragged token representation shared by 2D / 3D / 2D3D."""

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
                f"{name}.features dim must be {feature_dim}, got {self.features.shape[1]}"
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

    source_indices: [N_target,K], invalid source entries use -1
    weights: [N_target,K] or None
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
            raise ValueError(f"{name}.source_indices must be [N,K]")
        if self.source_indices.shape[0] != int(num_target):
            raise ValueError(
                f"{name}: expected {num_target} rows, got {self.source_indices.shape[0]}"
            )
        if self.source_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name}.source_indices must be integer tensor")

        valid = self.source_indices >= 0
        if valid.any():
            max_index = int(self.source_indices[valid].max().item())
            if max_index >= int(num_source):
                raise IndexError(
                    f"{name}: source index {max_index} exceeds source token count {num_source}"
                )

        if self.weights is not None:
            if self.weights.shape != self.source_indices.shape:
                raise ValueError(f"{name}.weights must match source_indices shape")
            if not torch.isfinite(self.weights).all():
                raise ValueError(f"{name}.weights contains NaN/Inf")


@dataclass
class UnifiedDecoderOutput:
    semantic_feature_t1: torch.Tensor
    semantic_feature_t2: torch.Tensor
    change_feature_t1: torch.Tensor
    change_feature_t2: torch.Tensor

    semantic_logits_t1: torch.Tensor
    semantic_logits_t2: torch.Tensor

    # The already-computed normalized dataset language prototypes [K,D].
    # 2D uses these again AFTER spatial feature upsampling, avoiding a second
    # Qwen text pass while leaving the shared decoder representation unchanged.
    semantic_prototypes: torch.Tensor

    # 2D binary path.
    change_logits_t1: Optional[torch.Tensor]
    change_logits_t2: Optional[torch.Tensor]

    # 3D event path.
    event_logits_t1: Optional[torch.Tensor]
    event_logits_t2: Optional[torch.Tensor]

    raw_class_ids: Tuple[int, ...]
    class_names: Tuple[str, ...]


@dataclass
class Cascade2DDecoderOutput:
    """Final prediction output of the PAIR V2 2D Cascade Gated Decoder.

    Keep the same branch-discovery protocol as UnifiedDecoderOutput so the
    unified PAIR loss/metrics can inspect optional branches without special
    casing the 2D CG route. 2D never produces 3D event logits.
    """

    semantic_logits_t1: Optional[torch.Tensor]
    semantic_logits_t2: Optional[torch.Tensor]
    change_logits_t1: torch.Tensor
    change_logits_t2: torch.Tensor

    # Keep the same semantic-class ordering metadata as UnifiedDecoderOutput.
    # Metrics uses raw_class_ids to verify that prototype/logit channel order
    # matches DatasetSpec.class_names.
    raw_class_ids: Tuple[int, ...]
    class_names: Tuple[str, ...]

    event_logits_t1: Optional[torch.Tensor] = None
    event_logits_t2: Optional[torch.Tensor] = None


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
    """feature + position + modality + time"""

    def __init__(self, dim: int, num_modalities: int = 2):
        super().__init__()
        self.position_encoder = CoordinateEncoder(dim)
        self.modality_embedding = nn.Embedding(num_modalities, dim)
        self.time_embedding = nn.Embedding(2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: UnifiedTokenSet, *, time_id: int) -> torch.Tensor:
        x = tokens.features
        pos = self.position_encoder(tokens.positions).to(
            dtype=x.dtype,
            device=x.device,
        )
        mod = self.modality_embedding(tokens.modality_ids.long()).to(dtype=x.dtype)
        time_ids = torch.full(
            (x.shape[0],),
            int(time_id),
            dtype=torch.long,
            device=x.device,
        )
        time = self.time_embedding(time_ids).to(dtype=x.dtype)
        return self.norm(x + pos + mod + time)


# =============================================================================
# Dense <- Qwen reasoning injection
# =============================================================================


class ReasoningInjection(nn.Module):
    """Dense queries attend to a much smaller reasoning-token set."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        query_chunk_size: int = 4096,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.query_chunk_size = int(query_chunk_size)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
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

        kv = self.norm_kv(reasoning).unsqueeze(0)
        outputs = []
        for start in range(0, dense.shape[0], self.query_chunk_size):
            end = min(start + self.query_chunk_size, dense.shape[0])
            q = self.norm_q(dense[start:end]).unsqueeze(0)
            attended, _ = self.attention(q, kv, kv, need_weights=False)
            outputs.append(attended[0])

        injected = torch.cat(outputs, dim=0)
        return self.out_norm(dense + injected)

    def forward(
        self,
        *,
        dense: torch.Tensor,
        dense_batch_ids: torch.Tensor,
        reasoning: torch.Tensor,
        reasoning_batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(dense)
        for batch_id in torch.unique(dense_batch_ids).tolist():
            dense_mask = dense_batch_ids == int(batch_id)
            reasoning_mask = reasoning_batch_ids == int(batch_id)
            output[dense_mask] = self._one_batch(
                dense[dense_mask],
                reasoning[reasoning_mask],
            )
        return output


# =============================================================================
# Sparse T1 <-> T2 fusion
# =============================================================================


class SparseTemporalFusion(nn.Module):
    """O(N*K) local temporal interaction using externally supplied links."""

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

        if source.shape[0] == 0:
            if valid.any():
                raise IndexError(
                    "Temporal links reference a source tensor with zero tokens"
                )
            return source.new_zeros((index.shape[0], source.shape[-1]))

        safe_index = index.clamp(min=0)
        gathered = source[safe_index]

        if links.weights is None:
            weights = valid.to(dtype=source.dtype)
        else:
            weights = links.weights.to(
                dtype=source.dtype,
                device=source.device,
            )
            weights = weights * valid.to(dtype=weights.dtype)

        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        normalized_weights = weights / denominator
        cross = (gathered * normalized_weights.unsqueeze(-1)).sum(dim=1)
        has_neighbor = valid.any(dim=1, keepdim=True)
        return torch.where(has_neighbor, cross, torch.zeros_like(cross))

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
        cross = self.gather_cross_context(source=source, links=links)
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
        return self.norm(target + fused)


# =============================================================================
# Task conditioning
# =============================================================================


class TaskConditioning(nn.Module):
    """<TASK> hidden -> FiLM conditioning over every dense token."""

    def __init__(self, qwen_dim: int, decoder_dim: int):
        super().__init__()
        self.to_film = nn.Linear(qwen_dim, decoder_dim * 2)
        self.norm = nn.LayerNorm(decoder_dim)

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

        gamma, beta = self.to_film(task_hidden).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        token_gamma = gamma[batch_ids.long()]
        token_beta = beta[batch_ids.long()]
        return self.norm(x * (1.0 + token_gamma) + token_beta)


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
        hidden_dim = int(dim * float(mlp_ratio))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class SelfAttentionBlock(nn.Module):
    """Standard pre-norm self-attention + MLP block for a short token set."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % int(num_heads) != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )
        hidden_dim = int(dim * float(mlp_ratio))
        self.norm_attn = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"SelfAttentionBlock expects [B,N,D], got {tuple(x.shape)}"
            )
        qkv = self.norm_attn(x)
        attended, _ = self.attention(
            qkv,
            qkv,
            qkv,
            need_weights=False,
        )
        x = x + attended
        x = x + self.mlp(self.norm_mlp(x))
        return x


class FixedTokenAttention(nn.Module):
    """Efficient shared global attention for variable-length dense tokens.

    The module keeps M learned tokens [M,D]. For each sample:

        dense [N,D]
          -> learned tokens read dense with cross-attention
          -> self-attention runs only on M learned tokens
          -> dense tokens read the updated learned tokens
          -> dense [N,D]

    This makes the expensive global interaction O(N*M + M^2) instead of O(N^2).
    The same parameters are used by 2D and 3D.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_tokens: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim = int(dim)
        num_tokens = int(num_tokens)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        if dim <= 0 or num_tokens <= 0 or num_layers <= 0:
            raise ValueError(
                "dim, num_tokens and num_layers must all be positive"
            )
        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.learned_tokens = nn.Parameter(
            torch.empty(num_tokens, dim)
        )
        nn.init.trunc_normal_(self.learned_tokens, std=0.02)

        # Learned tokens read the full dense set.
        self.read_query_norm = nn.LayerNorm(dim)
        self.read_dense_norm = nn.LayerNorm(dim)
        self.read_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.read_out_norm = nn.LayerNorm(dim)

        # Global reasoning happens only on the short learned-token set.
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Dense tokens read the globally mixed learned tokens back.
        self.write_dense_norm = nn.LayerNorm(dim)
        self.write_token_norm = nn.LayerNorm(dim)
        self.write_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.write_out_norm = nn.LayerNorm(dim)

    def _one_sample(self, dense: torch.Tensor) -> torch.Tensor:
        if dense.ndim != 2 or dense.shape[-1] != self.dim:
            raise ValueError(
                f"dense must be [N,{self.dim}], got {tuple(dense.shape)}"
            )
        if dense.shape[0] == 0:
            return dense

        dense_b = dense.unsqueeze(0)
        learned = self.learned_tokens.to(
            device=dense.device,
            dtype=dense.dtype,
        ).unsqueeze(0)

        q = self.read_query_norm(learned)
        kv = self.read_dense_norm(dense_b)
        read, _ = self.read_attention(q, kv, kv, need_weights=False)
        learned = self.read_out_norm(learned + read)

        for block in self.self_blocks:
            learned = block(learned)

        q = self.write_dense_norm(dense_b)
        kv = self.write_token_norm(learned)
        write, _ = self.write_attention(q, kv, kv, need_weights=False)
        return self.write_out_norm(dense_b + write)[0]

    def forward_pair(
        self,
        *,
        x1: torch.Tensor,
        batch_ids1: torch.Tensor,
        x2: torch.Tensor,
        batch_ids2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jointly mix T1/T2 tokens within each sample, then split them back."""
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(
                f"x1/x2 must be [N,D], got {tuple(x1.shape)} / {tuple(x2.shape)}"
            )
        if x1.shape[-1] != self.dim or x2.shape[-1] != self.dim:
            raise ValueError(
                f"x1/x2 feature dim must be {self.dim}"
            )
        if batch_ids1.shape != (x1.shape[0],):
            raise ValueError(
                f"batch_ids1 must be [{x1.shape[0]}], got {tuple(batch_ids1.shape)}"
            )
        if batch_ids2.shape != (x2.shape[0],):
            raise ValueError(
                f"batch_ids2 must be [{x2.shape[0]}], got {tuple(batch_ids2.shape)}"
            )

        ids1 = set(int(x) for x in torch.unique(batch_ids1).tolist())
        ids2 = set(int(x) for x in torch.unique(batch_ids2).tolist())
        if ids1 != ids2:
            raise ValueError(
                f"T1/T2 batch id sets differ: {sorted(ids1)} vs {sorted(ids2)}"
            )

        out1 = torch.empty_like(x1)
        out2 = torch.empty_like(x2)
        for batch_id in sorted(ids1):
            mask1 = batch_ids1 == int(batch_id)
            mask2 = batch_ids2 == int(batch_id)
            n1 = int(mask1.sum().item())
            n2 = int(mask2.sum().item())
            if n1 == 0 or n2 == 0:
                raise ValueError(
                    f"batch {batch_id} must contain both T1 and T2 tokens"
                )

            joined = torch.cat([x1[mask1], x2[mask2]], dim=0)
            joined = self._one_sample(joined)
            out1[mask1] = joined[:n1]
            out2[mask2] = joined[n1:n1 + n2]

        return out1, out2


# =============================================================================
# PAIR V2 2D Cascade Gated Decoder
# Ported from PerASCD models/common.py.
# The module always keeps three streams:
#   x0 = T1 semantic
#   x1 = T2 semantic
#   xc = explicit change
# =============================================================================


class CBAMconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction=16):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x):
        avg_out = self.mlp1(self.avg_pool(x))
        max_out = self.mlp1(self.max_pool(x))
        channel_w = self.sigmoid(avg_out + max_out)
        x = x * channel_w

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_w = self.sigmoid(
            self.conv1x1(torch.cat([avg_out, max_out], dim=1))
        )
        x = x * spatial_w
        return self.conv2d(x)


class ChangeAwareGatingModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels // 4,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.conv_local = nn.Conv2d(in_channels // 4, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels // 4, 2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        avg = self.avg_pool(x)
        avg = self.conv_global(avg)
        global_weight = self.sigmoid(avg)

        logit = self.conv_local(x)
        local_weight = self.sigmoid(logit)
        return local_weight * (1 + global_weight)


class CascadeGatedBlock(nn.Module):
    def __init__(
        self,
        feat_channels,
        out_channels,
        drop_rate=0.0,
        use_lateral=True,
    ):
        super().__init__()
        self.use_lateral = use_lateral

        self.feat_conv0 = nn.Sequential(
            CBAMconv2d(feat_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.feat_convc = nn.Sequential(
            CBAMconv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.highconv0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.highconv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.highconvc = nn.Sequential(
            nn.Conv2d(
                out_channels * 3,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.lowconv0 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.lowconv1 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.lowconvc = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.cagm = ChangeAwareGatingModule(out_channels * 2)
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x0, x1, xc, feat0=None, feat1=None):
        x0 = self.highconv0(x0)
        x1 = self.highconv1(x1)
        xc = self.highconvc(torch.cat([xc, x0, x1], dim=1))

        if self.use_lateral:
            if feat0 is None or feat1 is None:
                raise ValueError(
                    "use_lateral=True requires feat0 and feat1."
                )
            x0 = F.interpolate(
                x0,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            x1 = F.interpolate(
                x1,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            xc = F.interpolate(
                xc,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            f0 = self.feat_conv0(feat0)
            f1 = self.feat_conv0(feat1)
        else:
            f0 = self.feat_conv0(x0)
            f1 = self.feat_conv0(x1)

        fc = self.feat_convc(torch.abs(f0 - f1))
        hardship_map = self.cagm(torch.cat([xc, fc], dim=1))
        w_high = hardship_map[:, 0].unsqueeze(1)
        w_low = hardship_map[:, 1].unsqueeze(1)

        x0 = self.lowconv0((w_high * x0) + (w_low * f0))
        x1 = self.lowconv1((w_high * x1) + (w_low * f1))
        xc = self.lowconvc((w_high * xc) + (w_low * fc))

        x0 = self.dropout(x0)
        x1 = self.dropout(x1)
        xc = self.dropout(xc)
        return x0, x1, xc


class CascadeGatedDecoder(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channels,
        drop_rate=0.0,
        use_refinement_block=False,
    ):
        super().__init__()
        self.use_refinement_block = use_refinement_block

        self.first_feat_conv0 = nn.Sequential(
            CBAMconv2d(
                in_channel_list[-1],
                out_channels,
                kernel_size=3,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        fusion_blocks = []
        for i in range(len(in_channel_list) - 1):
            fusion_blocks.append(
                CascadeGatedBlock(
                    feat_channels=in_channel_list[
                        len(in_channel_list) - i - 2
                    ],
                    out_channels=out_channels,
                    drop_rate=drop_rate,
                    use_lateral=True,
                )
            )
        self.fusion_blocks = nn.ModuleList(fusion_blocks)

        if use_refinement_block:
            self.refinement_block = CascadeGatedBlock(
                feat_channels=out_channels,
                out_channels=out_channels,
                drop_rate=drop_rate,
                use_lateral=False,
            )
        else:
            self.refinement_block = None

    def forward(self, feat_list_a, feat_list_b):
        x0 = self.first_feat_conv0(feat_list_a[-1])
        x1 = self.first_feat_conv0(feat_list_b[-1])
        xc = torch.abs(x0 - x1)

        for i, block in enumerate(self.fusion_blocks):
            feat0 = feat_list_a[len(feat_list_a) - i - 2]
            feat1 = feat_list_b[len(feat_list_b) - i - 2]
            x0, x1, xc = block(
                x0,
                x1,
                xc,
                feat0,
                feat1,
            )

        if self.refinement_block is not None:
            x0, x1, xc = self.refinement_block(x0, x1, xc)

        return x0, x1, xc


# =============================================================================
# Qwen class prototype encoder
# =============================================================================


class QwenClassPrototypeEncoder(nn.Module):
    """
    DatasetSpec.class_names -> text prompts -> Qwen -> semantic prototypes.

    Raw dataset IDs are kept only for target/prediction mapping. Prototype
    positions are compact and ordered by sorted raw ID.
    """

    def __init__(
        self,
        *,
        qwen_dim: int,
        decoder_dim: int,
        prompt_template: str = "A remote sensing semantic class: {name}.",
    ):
        super().__init__()
        self.qwen_dim = int(qwen_dim)
        self.decoder_dim = int(decoder_dim)
        self.prompt_template = str(prompt_template)
        self.projection = nn.Sequential(
            nn.Linear(self.qwen_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
        )

    @staticmethod
    def normalize_class_dict(
        class_names: Dict[int, str],
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        if not isinstance(class_names, dict):
            raise TypeError("DatasetSpec.class_names must be Dict[int, str]")
        if not class_names:
            raise ValueError("class_names cannot be empty")

        normalized = {}
        for raw_id, name in class_names.items():
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise TypeError(f"class ID must be int, got {raw_id!r}")
            if not isinstance(name, str) or not name.strip():
                raise TypeError(
                    f"class name for ID {raw_id} must be non-empty str"
                )
            normalized[int(raw_id)] = name.strip()

        raw_ids = tuple(sorted(normalized))
        names = tuple(normalized[raw_id] for raw_id in raw_ids)
        return raw_ids, names

    @staticmethod
    def _last_valid_hidden(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.to(dtype=torch.long)
        positions = torch.arange(
            mask.shape[1], device=mask.device, dtype=torch.long
        ).unsqueeze(0)
        last_index = (positions * mask).amax(dim=1)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, last_index]

    def forward(
        self,
        *,
        class_names: Dict[int, str],
        qwen_backbone,
        detach_qwen: bool = True,
    ):
        raw_ids, names = self.normalize_class_dict(class_names)

        tokenizer = qwen_backbone.tokenizer
        if tokenizer.eos_token is None:
            raise RuntimeError("Qwen tokenizer must define an EOS token")
        prompts = [
            self.prompt_template.format(name=name) + tokenizer.eos_token
            for name in names
        ]

        qwen_model = qwen_backbone.model
        device = next(qwen_model.parameters()).device
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
            return self._last_valid_hidden(
                output.hidden_states[-1],
                encoded["attention_mask"],
            )

        if detach_qwen:
            # Prototype text encoding is a read-only Qwen pass. Disable dropout
            # for deterministic class features, then restore the caller's mode.
            qwen_was_training = qwen_model.training
            qwen_model.eval()
            try:
                with torch.no_grad():
                    language_hidden = run_qwen().detach()
            finally:
                qwen_model.train(qwen_was_training)
        else:
            language_hidden = run_qwen()

        projection_dtype = self.projection[0].weight.dtype
        prototypes = self.projection(
            language_hidden.to(dtype=projection_dtype)
        )
        prototypes = F.normalize(prototypes.float(), dim=-1)
        return raw_ids, names, prototypes


# =============================================================================
# Unified decoder
# =============================================================================


class UnifiedChangeDecoder(nn.Module):
    """
    One shared decoder core for 2D / 3D / 2D3D.

    prediction_type:
        "binary" -> 2D binary change logits
        "event"  -> 3D three-class event logits

    This is an internal routing argument, not a user config field.
    """

    EVENT_CLASSES = (
        "unchanged",
        "added",
        "removed",
    )

    def __init__(
        self,
        *,
        decoder_dim: int = 256,
        qwen_dim: int = 2560,
        vision_dim: int = 1024,
        num_heads: int = 8,
        num_shared_blocks: int = 2,
        reasoning_chunk_size: int = 4096,
        dropout: float = 0.0,
        num_modalities: int = 2,
        initial_logit_scale: float = 10.0,
        unified_num_tokens: int = 64,
        unified_num_layers: int = 2,
        unified_num_heads: int = 8,
        unified_mlp_ratio: float = 2.0,
        unified_dropout: float = 0.0,
    ):
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.qwen_dim = int(qwen_dim)
        self.vision_dim = int(vision_dim)

        self.token_embedding = UnifiedTokenEmbedding(
            self.decoder_dim,
            num_modalities=num_modalities,
        )
        self.reasoning_projection = nn.Sequential(
            nn.Linear(self.qwen_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
        )
        self.reasoning_injection = ReasoningInjection(
            self.decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            query_chunk_size=reasoning_chunk_size,
        )

        # Shared 2D/3D global decoder directly after dense<-LLM cross-attention.
        # It compresses each sample's joint T1/T2 dense tokens to a fixed set
        # of learned tokens, performs self-attention only on that short set,
        # and writes the global information back to every dense token.
        self.unified_attention = FixedTokenAttention(
            self.decoder_dim,
            num_tokens=unified_num_tokens,
            num_layers=unified_num_layers,
            num_heads=unified_num_heads,
            mlp_ratio=unified_mlp_ratio,
            dropout=unified_dropout,
        )

        self.temporal_fusion = SparseTemporalFusion(self.decoder_dim)
        self.task_conditioning = TaskConditioning(
            self.qwen_dim,
            self.decoder_dim,
        )
        self.shared_blocks = nn.ModuleList(
            [
                SharedDenseBlock(
                    self.decoder_dim,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(int(num_shared_blocks))
            ]
        )

        # Shared latent representations.
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(self.decoder_dim),
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.GELU(),
        )
        self.change_head = nn.Sequential(
            nn.LayerNorm(self.decoder_dim),
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.GELU(),
        )

        # PAIR V2 2D route.
        # Input order is shallow -> deep:
        # [ViT DeepStack @ 1/4, ViT DeepStack @ 1/8,
        #  ViT DeepStack @ 1/16, dense<-LLM fused feature @ 1/32].
        # The deepest feature has already been projected into decoder_dim and
        # fused by dense-query / LLM-key-value cross-attention.
        self.cg_decoder_2d = CascadeGatedDecoder(
            in_channel_list=(
                self.vision_dim,
                self.vision_dim,
                self.vision_dim,
                self.decoder_dim,
            ),
            out_channels=self.decoder_dim,
            drop_rate=dropout,
            use_refinement_block=False,
        )

        # Output heads.
        # Legacy/unified binary classifier is kept for the original token route.
        self.binary_change_classifier = nn.Linear(self.decoder_dim, 1)

        # PAIR V2 2D CG change classifier.
        # One shared xc stream produces two temporal binary logits:
        #   channel 0 -> change_logits_t1
        #   channel 1 -> change_logits_t2
        # These are two independent output channels, not unchanged/change classes.
        self.classifier_cd = nn.Sequential(
            nn.Conv2d(self.decoder_dim, self.decoder_dim // 2, kernel_size=1),
            nn.BatchNorm2d(self.decoder_dim // 2),
            nn.ReLU(),
            nn.Conv2d(self.decoder_dim // 2, 2, kernel_size=1),
        )

        self.event_head = nn.Linear(self.decoder_dim, 3)

        self.class_encoder = QwenClassPrototypeEncoder(
            qwen_dim=self.qwen_dim,
            decoder_dim=self.decoder_dim,
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(float(initial_logit_scale)).log()
        )

    @staticmethod
    def _normalize_2d_prediction_mode(prediction_mode: str) -> str:
        prediction_mode = str(prediction_mode).lower().strip()
        aliases = {
            "scd": "scd",
            "semantic": "scd",
            "semantic_change": "scd",
            "bcd": "bcd",
            "binary": "bcd",
            "binary_change": "bcd",
        }
        if prediction_mode not in aliases:
            raise ValueError(
                "prediction_mode must be 'scd' or 'bcd', "
                f"got {prediction_mode!r}"
            )
        return aliases[prediction_mode]

    def _validate_2d_pyramid(
        self,
        feat_list: Sequence[torch.Tensor],
        *,
        name: str,
    ) -> None:
        if not isinstance(feat_list, (list, tuple)):
            raise TypeError(f"{name} must be a list/tuple of four tensors")
        if len(feat_list) != 4:
            raise ValueError(
                f"{name} must contain four scales [1/4,1/8,1/16,1/32], "
                f"got {len(feat_list)}"
            )

        expected_channels = (
            self.vision_dim,
            self.vision_dim,
            self.vision_dim,
            self.decoder_dim,
        )
        batch_size = None
        previous_hw = None

        for i, (feat, expected_c) in enumerate(
            zip(feat_list, expected_channels)
        ):
            if not torch.is_tensor(feat) or feat.ndim != 4:
                raise ValueError(
                    f"{name}[{i}] must be [B,C,H,W], "
                    f"got {type(feat)!r} / "
                    f"{getattr(feat, 'shape', None)}"
                )
            if feat.shape[1] != expected_c:
                raise ValueError(
                    f"{name}[{i}] channel dim must be {expected_c}, "
                    f"got {feat.shape[1]}"
                )
            if batch_size is None:
                batch_size = feat.shape[0]
            elif feat.shape[0] != batch_size:
                raise ValueError(
                    f"{name} has inconsistent batch dimensions"
                )
            if not torch.isfinite(feat).all():
                raise ValueError(f"{name}[{i}] contains NaN/Inf")

            hw = tuple(feat.shape[-2:])
            if hw[0] <= 0 or hw[1] <= 0:
                raise ValueError(f"{name}[{i}] has invalid spatial size {hw}")

            if previous_hw is not None:
                expected_hw = (
                    previous_hw[0] // 2,
                    previous_hw[1] // 2,
                )
                if hw != expected_hw:
                    raise ValueError(
                        f"{name} must be a strict x2 pyramid shallow->deep; "
                        f"scale {i-1}={previous_hw}, scale {i}={hw}, "
                        f"expected {expected_hw}"
                    )
            previous_hw = hw

    def _semantic_logits_2d(
        self,
        feature: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        feature = F.normalize(feature.float(), dim=1)
        scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)
        return scale * torch.einsum(
            "bdhw,kd->bkhw",
            feature,
            prototypes,
        )

    def _binary_logits_2d(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier_cd(feature)
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(
                "2D CG change classifier must return [B,2,H,W], "
                f"got {tuple(logits.shape)}"
            )
        return logits[:, 0], logits[:, 1]

    def forward_2d_cg(
        self,
        *,
        feat_pyramid_t1: Sequence[torch.Tensor],
        feat_pyramid_t2: Sequence[torch.Tensor],
        prediction_mode: str,
        class_names: Optional[Dict[int, str]] = None,
        qwen_backbone=None,
        detach_qwen_class_encoder: bool = True,
    ) -> Cascade2DDecoderOutput:
        """
        PAIR V2 2D route.

        Expected feature order for both times:
            [1/4, 1/8, 1/16, 1/32]

        For Qwen3-VL-4B at 512x512 this is conceptually:
            DeepStack shallow @ 128x128, vision_dim channels
            DeepStack middle  @  64x64, vision_dim channels
            DeepStack deep    @  32x32, vision_dim channels
            dense<-LLM fused  @  16x16, decoder_dim channels

        The 1/32 feature is produced before this function by:
            Q = final merged Qwen-ViT dense feature
            K,V = Qwen LLM image hidden states
            + position / modality / time embeddings on both sides
            -> shared ReasoningInjection cross-attention

        SCD and BCD share the exact same CG decoder and explicit xc stream.
        Only the final prediction route differs.
        """
        prediction_mode = self._normalize_2d_prediction_mode(
            prediction_mode
        )
        self._validate_2d_pyramid(
            feat_pyramid_t1,
            name="feat_pyramid_t1",
        )
        self._validate_2d_pyramid(
            feat_pyramid_t2,
            name="feat_pyramid_t2",
        )

        for i, (feat1, feat2) in enumerate(
            zip(feat_pyramid_t1, feat_pyramid_t2)
        ):
            if feat1.shape != feat2.shape:
                raise ValueError(
                    f"T1/T2 pyramid shape mismatch at scale {i}: "
                    f"{tuple(feat1.shape)} vs {tuple(feat2.shape)}"
                )

        x0, x1, xc = self.cg_decoder_2d(
            feat_pyramid_t1,
            feat_pyramid_t2,
        )

        change_logits_t1, change_logits_t2 = self._binary_logits_2d(xc)

        if prediction_mode == "scd":
            if class_names is None:
                raise ValueError("SCD route requires class_names")
            if qwen_backbone is None:
                raise ValueError("SCD route requires qwen_backbone")

            raw_class_ids, ordered_class_names, prototypes = self.class_encoder(
                class_names=class_names,
                qwen_backbone=qwen_backbone,
                detach_qwen=detach_qwen_class_encoder,
            )
            semantic_logits_t1 = self._semantic_logits_2d(x0, prototypes)
            semantic_logits_t2 = self._semantic_logits_2d(x1, prototypes)
        else:
            semantic_logits_t1 = None
            semantic_logits_t2 = None
            raw_class_ids = tuple()
            ordered_class_names = tuple()

        return Cascade2DDecoderOutput(
            semantic_logits_t1=semantic_logits_t1,
            semantic_logits_t2=semantic_logits_t2,
            change_logits_t1=change_logits_t1,
            change_logits_t2=change_logits_t2,
            raw_class_ids=raw_class_ids,
            class_names=ordered_class_names,
        )

    @staticmethod
    def _normalize_prediction_type(prediction_type: str) -> str:
        prediction_type = str(prediction_type).lower().strip()
        if prediction_type not in ("binary", "event"):
            raise ValueError(
                "prediction_type must be 'binary' or 'event', "
                f"got {prediction_type!r}"
            )
        return prediction_type

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
            features=self.reasoning_projection(tokens.features),
            positions=tokens.positions,
            modality_ids=tokens.modality_ids,
            batch_ids=tokens.batch_ids,
        )
        embedded = self.token_embedding(projected, time_id=time_id)
        return embedded, tokens.batch_ids

    def _semantic_logits(
        self,
        feature: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        feature = F.normalize(feature.float(), dim=-1)
        scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)
        return scale * (feature @ prototypes.T)

    def _inject_reasoning_one_time(
        self,
        *,
        dense_tokens: UnifiedTokenSet,
        dense_time_id: int,
        reasoning_tokens: UnifiedTokenSet,
        reasoning_time_id: int,
    ) -> torch.Tensor:
        """Fuse one temporal dense stream with Qwen reasoning.

        Both Q and K/V receive the shared PAIR token embedding:
            feature + position + modality + time.

        This is used by both the 3D unified route and the restored 2D
        dense<-LLM cross-attention route.
        """
        dense_tokens.validate(
            feature_dim=self.decoder_dim,
            name="dense_tokens",
        )
        x = self.token_embedding(
            dense_tokens,
            time_id=dense_time_id,
        )
        reasoning, reasoning_batch_ids = self._prepare_reasoning(
            reasoning_tokens,
            time_id=reasoning_time_id,
        )
        return self.reasoning_injection(
            dense=x,
            dense_batch_ids=dense_tokens.batch_ids,
            reasoning=reasoning,
            reasoning_batch_ids=reasoning_batch_ids,
        )

    def fuse_2d_dense_reasoning(
        self,
        *,
        dense_t1: UnifiedTokenSet,
        dense_t2: UnifiedTokenSet,
        reasoning_t1: UnifiedTokenSet,
        reasoning_t2: UnifiedTokenSet,
        task_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restore PAIR's 2D dense-query / LLM-KV fusion.

        T1 and T2 share exactly the same projection, positional/modality/time
        embeddings and cross-attention parameters.  The only temporal
        distinction is the learned time embedding (0 for T1, 1 for T2).

        Returns flat fused dense features [N1,D] and [N2,D].  The caller
        restores them to their 2D grids and uses them as the 1/32 CGDecoder
        features.
        """
        dense_t1.validate(feature_dim=self.decoder_dim, name="dense_2d_t1")
        dense_t2.validate(feature_dim=self.decoder_dim, name="dense_2d_t2")
        reasoning_t1.validate(feature_dim=self.qwen_dim, name="reasoning_2d_t1")
        reasoning_t2.validate(feature_dim=self.qwen_dim, name="reasoning_2d_t2")

        x1 = self._inject_reasoning_one_time(
            dense_tokens=dense_t1,
            dense_time_id=0,
            reasoning_tokens=reasoning_t1,
            reasoning_time_id=0,
        )
        x2 = self._inject_reasoning_one_time(
            dense_tokens=dense_t2,
            dense_time_id=1,
            reasoning_tokens=reasoning_t2,
            reasoning_time_id=1,
        )

        # Shared Unified Decoder: T1/T2 are mixed jointly inside each sample.
        # The same module/parameters are also used by the 3D route.
        x1, x2 = self.unified_attention.forward_pair(
            x1=x1,
            batch_ids1=dense_t1.batch_ids,
            x2=x2,
            batch_ids2=dense_t2.batch_ids,
        )

        # <TASK> is the shared full-context readout: it appears after both
        # temporal inputs in the causal sequence, so condition both 2D dense
        # streams with the same complete bi-temporal task representation.
        x1 = self.task_conditioning(
            x=x1,
            batch_ids=dense_t1.batch_ids,
            task_hidden=task_hidden,
        )
        x2 = self.task_conditioning(
            x=x2,
            batch_ids=dense_t2.batch_ids,
            task_hidden=task_hidden,
        )
        return x1, x2

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
        prediction_type: str = "binary",
    ) -> UnifiedDecoderOutput:
        prediction_type = self._normalize_prediction_type(prediction_type)

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

        # 1) Dense Q attends to Qwen LLM K/V independently for T1/T2.
        x1 = self._inject_reasoning_one_time(
            dense_tokens=dense_t1,
            dense_time_id=0,
            reasoning_tokens=reasoning_t1,
            reasoning_time_id=0,
        )
        x2 = self._inject_reasoning_one_time(
            dense_tokens=dense_t2,
            dense_time_id=1,
            reasoning_tokens=reasoning_t2,
            reasoning_time_id=1,
        )

        # 2) Shared 2D/3D Unified Decoder. T1 and T2 tokens from the same
        # sample are mixed jointly through the fixed learned-token bottleneck.
        x1, x2 = self.unified_attention.forward_pair(
            x1=x1,
            batch_ids1=dense_t1.batch_ids,
            x2=x2,
            batch_ids2=dense_t2.batch_ids,
        )

        # Shared task conditioning after the global decoder.  The 2D route
        # applies the same module at the corresponding point before CGDecoder.
        x1 = self.task_conditioning(
            x=x1,
            batch_ids=dense_t1.batch_ids,
            task_hidden=task_hidden,
        )
        x2 = self.task_conditioning(
            x=x2,
            batch_ids=dense_t2.batch_ids,
            task_hidden=task_hidden,
        )

        # 3) Sparse T1 <-> T2 interaction. Both directions use pre-fusion x1/x2.
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
        x1, x2 = temporal_x1, temporal_x2

        # 4) Shared dense refinement.
        for block in self.shared_blocks:
            x1 = block(x1)
            x2 = block(x2)

        # 5) Shared semantic/change representations.
        semantic_feature_t1 = self.semantic_head(x1)
        semantic_feature_t2 = self.semantic_head(x2)
        change_feature_t1 = self.change_head(x1)
        change_feature_t2 = self.change_head(x2)

        # 6) Dataset-specific semantic language prototypes.
        raw_class_ids, ordered_class_names, prototypes = self.class_encoder(
            class_names=class_names,
            qwen_backbone=qwen_backbone,
            detach_qwen=detach_qwen_class_encoder,
        )
        semantic_logits_t1 = self._semantic_logits(
            semantic_feature_t1,
            prototypes,
        )
        semantic_logits_t2 = self._semantic_logits(
            semantic_feature_t2,
            prototypes,
        )

        # 7) Route only the final change/event classifier.
        if prediction_type == "binary":
            change_logits_t1 = self.binary_change_classifier(
                change_feature_t1
            )[:, 0]
            change_logits_t2 = self.binary_change_classifier(
                change_feature_t2
            )[:, 0]
            event_logits_t1 = None
            event_logits_t2 = None
        else:
            change_logits_t1 = None
            change_logits_t2 = None
            event_logits_t1 = self.event_head(change_feature_t1)
            event_logits_t2 = self.event_head(change_feature_t2)

        return UnifiedDecoderOutput(
            semantic_feature_t1=semantic_feature_t1,
            semantic_feature_t2=semantic_feature_t2,
            change_feature_t1=change_feature_t1,
            change_feature_t2=change_feature_t2,
            semantic_logits_t1=semantic_logits_t1,
            semantic_logits_t2=semantic_logits_t2,
            semantic_prototypes=prototypes,
            change_logits_t1=change_logits_t1,
            change_logits_t2=change_logits_t2,
            event_logits_t1=event_logits_t1,
            event_logits_t2=event_logits_t2,
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
    """Same-position T1/T2 links for an already co-registered flat 2D token grid."""
    source_indices = torch.arange(
        int(num_tokens),
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)
    weights = torch.ones(
        (int(num_tokens), 1),
        dtype=torch.float32,
        device=device,
    )
    return TemporalLinks(
        source_indices=source_indices,
        weights=weights,
    )