from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed(embed_dim: int, length: int, cls_token: bool = False) -> torch.Tensor:
    """?먮낯 ReMasker媛 ?곕뜕 1D sin-cos ?꾩튂 ?꾨쿋?⑹쓣 PyTorch ?먯꽌濡??ш뎄?꾪븳??"""
    assert embed_dim % 2 == 0, "embed_dim must be even."
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    position = np.arange(length, dtype=np.float32)
    out = np.einsum("m,d->md", position, omega)
    pos_embed = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).float()


class ObservedSetSelfBlock(nn.Module):
    """Contextualizes observed feature tokens as an unordered conditioning set."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    @staticmethod
    def safe_padding_mask(observed_bool: torch.Tensor) -> torch.Tensor:
        padding_mask = ~observed_bool.bool()
        all_padded = padding_mask.all(dim=1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded] = False
        return padding_mask

    def forward(self, tokens: torch.Tensor, observed_bool: torch.Tensor) -> torch.Tensor:
        padding_mask = self.safe_padding_mask(observed_bool)
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=padding_mask, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        return tokens


class TargetConditionalBlock(nn.Module):
    """Lets each target feature query retrieve evidence from observed features."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_q_norm = nn.LayerNorm(embed_dim)
        self.cross_kv_norm = nn.LayerNorm(embed_dim)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        observed_tokens: torch.Tensor,
        observed_bool: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = ObservedSetSelfBlock.safe_padding_mask(observed_bool)
        observed_norm = self.cross_kv_norm(observed_tokens)
        cross_out, _ = self.cross_attn(
            query=self.cross_q_norm(target_tokens),
            key=observed_norm,
            value=observed_norm,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        target_tokens = target_tokens + self.dropout(cross_out)
        target_norm = self.self_norm(target_tokens)
        self_out, _ = self.self_attn(target_norm, target_norm, target_norm, need_weights=False)
        target_tokens = target_tokens + self.dropout(self_out)
        target_tokens = target_tokens + self.dropout(self.ffn(self.ffn_norm(target_tokens)))
        return target_tokens


class TargetConditionalObservedSetEncoder(nn.Module):
    """
    Target-conditional observed-set encoder for imputation.

    Every feature has a target query. The query cross-attends to the currently
    observed feature set, so the representation for feature j is directly
    optimized for reconstructing feature j rather than being a generic row
    embedding.
    """

    def __init__(
        self,
        input_dim: int,
        num_dim: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mechanism_hidden_dim: int,
        cross_attn_layers: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_dim = int(num_dim)
        self.embed_dim = embed_dim
        self.feature_embedding = nn.Embedding(input_dim, embed_dim)
        # Kept for checkpoint compatibility with earlier public forward-only runs.
        self.type_embedding = nn.Embedding(2, embed_dim)
        feature_type_ids = torch.zeros(input_dim, dtype=torch.long)
        feature_type_ids[self.num_dim :] = 1
        self.register_buffer("feature_type_ids", feature_type_ids, persistent=False)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.observed_state_embedding = nn.Embedding(2, embed_dim)
        self.target_state_embedding = nn.Embedding(2, embed_dim)
        self.query_seed = nn.Parameter(torch.zeros(1, input_dim, embed_dim))
        self.register_buffer("feature_ids", torch.arange(input_dim), persistent=False)
        self.mechanism_mlp = nn.Sequential(
            nn.Linear(5, mechanism_hidden_dim),
            nn.GELU(),
            nn.Linear(mechanism_hidden_dim, embed_dim),
        )
        observed_depth = max(1, depth // 2)
        target_depth = max(1, depth + max(0, cross_attn_layers - 1))
        self.observed_blocks = nn.ModuleList(
            [ObservedSetSelfBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(observed_depth)]
        )
        self.target_blocks = nn.ModuleList(
            [TargetConditionalBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(target_depth)]
        )
        self.field_norm = nn.LayerNorm(embed_dim)
        self.value_summary_norm = nn.LayerNorm(embed_dim)
        self.mask_summary_norm = nn.LayerNorm(embed_dim)
        self.summary_norm = nn.LayerNorm(embed_dim)
        self.observed_pool_score = nn.Linear(embed_dim, 1)
        self.target_pool_score = nn.Sequential(
            nn.Linear(embed_dim * 3, mechanism_hidden_dim),
            nn.GELU(),
            nn.Linear(mechanism_hidden_dim, 1),
        )
        self.mask_summary_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._latest_fusion_stats = {
            "target_pool_entropy": 0.0,
            "observed_ratio": 1.0,
            "target_depth": float(target_depth),
        }
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_seed, std=0.02)
        nn.init.xavier_uniform_(self.observed_pool_score.weight)
        nn.init.zeros_(self.observed_pool_score.bias)

    def latest_fusion_stats(self) -> dict[str, float]:
        return dict(self._latest_fusion_stats)

    def _mechanism_context(self, input_x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        observed_mask = 1.0 - missing_mask
        missing_rate = missing_mask.float().mean(dim=1, keepdim=True)
        observed_rate = observed_mask.float().mean(dim=1, keepdim=True)
        entropy = -(
            missing_rate * torch.log(missing_rate.clamp_min(1e-6))
            + (1.0 - missing_rate) * torch.log((1.0 - missing_rate).clamp_min(1e-6))
        )
        observed_count = observed_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        observed_abs_mean = (input_x.abs() * observed_mask).sum(dim=1, keepdim=True) / observed_count
        observed_mean = (input_x * observed_mask).sum(dim=1, keepdim=True) / observed_count
        observed_var = (((input_x - observed_mean) * observed_mask) ** 2).sum(dim=1, keepdim=True) / observed_count
        observed_std = torch.sqrt(observed_var + 1e-6)
        stats = torch.cat([missing_rate, observed_rate, entropy, observed_abs_mean, observed_std], dim=-1)
        return self.mechanism_mlp(stats)

    def _build_observed_tokens(
        self,
        feature_embed: torch.Tensor,
        value_embed: torch.Tensor,
        state_embed: torch.Tensor,
        mechanism_ctx: torch.Tensor,
    ) -> torch.Tensor:
        return feature_embed + value_embed + state_embed + mechanism_ctx

    def _build_target_tokens(
        self,
        query_seed: torch.Tensor,
        feature_embed: torch.Tensor,
        target_state: torch.Tensor,
        mechanism_ctx: torch.Tensor,
    ) -> torch.Tensor:
        return query_seed + feature_embed + target_state + mechanism_ctx

    def _record_stats(self, target_weights: torch.Tensor, observed_bool: torch.Tensor) -> None:
        entropy = -(target_weights * torch.log(target_weights.clamp_min(1e-8))).sum(dim=1)
        max_entropy = math.log(max(2, target_weights.size(1)))
        self._latest_fusion_stats = {
            "target_pool_entropy": float((entropy / max_entropy).mean().detach().cpu()),
            "observed_ratio": float(observed_bool.float().mean().detach().cpu()),
            "target_depth": float(len(self.target_blocks)),
        }

    def forward(self, input_x: torch.Tensor, missing_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = input_x.size(0)
        observed_bool = missing_mask < 0.5
        feature_embed = self.feature_embedding(self.feature_ids).unsqueeze(0).expand(batch_size, -1, -1)
        mechanism_repr = self._mechanism_context(input_x=input_x, missing_mask=missing_mask)
        mechanism_ctx = mechanism_repr.unsqueeze(1).expand(-1, self.input_dim, -1)
        observed_tokens = self._build_observed_tokens(
            feature_embed=feature_embed,
            value_embed=self.value_encoder(input_x.unsqueeze(-1)),
            state_embed=self.observed_state_embedding(observed_bool.long()),
            mechanism_ctx=mechanism_ctx,
        )
        for block in self.observed_blocks:
            observed_tokens = block(observed_tokens, observed_bool)

        target_state = self.target_state_embedding(missing_mask.long())
        target_tokens = self._build_target_tokens(
            query_seed=self.query_seed.expand(batch_size, -1, -1),
            feature_embed=feature_embed,
            target_state=target_state,
            mechanism_ctx=mechanism_ctx,
        )
        for block in self.target_blocks:
            target_tokens = block(target_tokens, observed_tokens, observed_bool)
        field_tokens = self.field_norm(target_tokens)

        obs_logits = self.observed_pool_score(observed_tokens).squeeze(-1).masked_fill(~observed_bool, -1e4)
        obs_weights = torch.softmax(obs_logits, dim=1)
        value_summary = torch.sum(obs_weights.unsqueeze(-1) * observed_tokens, dim=1)
        value_summary = self.value_summary_norm(value_summary + 0.1 * mechanism_repr)

        mask_context = self.mask_summary_mlp(torch.cat([feature_embed, target_state], dim=-1))
        mask_summary = self.mask_summary_norm(mask_context.mean(dim=1) + 0.1 * mechanism_repr)

        target_logits = self.target_pool_score(torch.cat([field_tokens, target_state, mechanism_ctx], dim=-1)).squeeze(-1)
        target_weights = torch.softmax(target_logits, dim=1)
        pooled_repr = torch.sum(target_weights.unsqueeze(-1) * field_tokens, dim=1)
        z_init = self.summary_norm(pooled_repr + 0.1 * mechanism_repr)
        self._record_stats(target_weights=target_weights, observed_bool=observed_bool)
        return {
            "field_tokens": field_tokens,
            "cls_repr": z_init,
            "pooled_repr": pooled_repr,
            "value_summary": value_summary,
            "mask_summary": mask_summary,
            "z_init": z_init,
            "mechanism_repr": mechanism_repr,
        }


class SimpleFieldwiseHead(nn.Module):
    """Lightweight field-wise decoder used by the public Eunseo model."""

    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.row_proj = nn.Linear(embed_dim, embed_dim)
        self.mask_embedding = nn.Embedding(2, embed_dim)
        self.task_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_embed = get_1d_sincos_pos_embed(embed_dim, input_dim, cls_token=False)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.row_proj.weight)
        nn.init.zeros_(self.row_proj.bias)
        nn.init.normal_(self.task_bias, std=0.02)
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        row_repr: torch.Tensor,
        field_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        row_context = self.row_proj(row_repr).unsqueeze(1)
        tokens = (
            field_tokens
            + row_context
            + self.mask_embedding(condition_mask.long())
            + self.pos_embed
            + self.task_bias
        )
        return self.head(tokens).squeeze(-1)


@dataclass
class EunseoOutputs:
    z_init: torch.Tensor
    z_final: torch.Tensor
    mechanism_repr: torch.Tensor
    forward_num: torch.Tensor
    forward_cat_logits: torch.Tensor
    forward_cat_class_logits: list[torch.Tensor]


class TargetWiseCategoricalHead(nn.Module):
    """Predicts each categorical feature directly from TOSE target tokens."""

    def __init__(
        self,
        num_dim: int,
        cat_bin_num: torch.Tensor | list[int] | tuple[int, ...],
        cat_codebook_sizes: list[int] | tuple[int, ...],
        embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_dim = int(num_dim)
        self.cat_bin_num = [int(value) for value in cat_bin_num]
        self.cat_bit_slices: list[tuple[int, int]] = []
        start = self.num_dim
        for bit_len in self.cat_bin_num:
            end = start + bit_len
            self.cat_bit_slices.append((start, end))
            start = end

        self.row_proj = nn.Linear(embed_dim, embed_dim)
        self.mask_embedding = nn.Embedding(2, embed_dim)
        self.type_embedding = nn.Embedding(1, embed_dim)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, int(num_classes)),
                )
                for num_classes in cat_codebook_sizes
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.row_proj.weight)
        nn.init.zeros_(self.row_proj.bias)
        for head in self.heads:
            final = head[-1]
            nn.init.xavier_uniform_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        row_repr: torch.Tensor,
        field_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        if len(self.heads) == 0:
            return []

        row_context = self.row_proj(row_repr)
        outputs: list[torch.Tensor] = []
        for column_idx, (start, end) in enumerate(self.cat_bit_slices):
            token_j = field_tokens[:, start:end].mean(dim=1)
            mask_j = condition_mask[:, start:end].amax(dim=1).long()
            token_j = token_j + row_context + self.mask_embedding(mask_j) + self.type_embedding.weight[0]
            outputs.append(self.heads[column_idx](token_j))
        return outputs


class EunseoModel(nn.Module):
    """Eunseo forward-only tabular imputation model."""

    def __init__(
        self,
        input_dim: int,
        num_dim: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mechanism_hidden_dim: int,
        cross_attn_layers: int,
        cat_bin_num: torch.Tensor | list[int] | tuple[int, ...] = (),
        cat_codebook_sizes: list[int] | tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_dim = num_dim
        self.cat_dim = input_dim - num_dim

        self.encoder = TargetConditionalObservedSetEncoder(
            input_dim=input_dim,
            num_dim=num_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mechanism_hidden_dim=mechanism_hidden_dim,
            cross_attn_layers=cross_attn_layers,
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        self.forward_decoder = SimpleFieldwiseHead(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.forward_cat_classifier = TargetWiseCategoricalHead(
            num_dim=num_dim,
            cat_bin_num=cat_bin_num,
            cat_codebook_sizes=cat_codebook_sizes,
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_x: torch.Tensor,
        missing_mask: torch.Tensor,
        decoder_forward_mask: torch.Tensor | None = None,
    ) -> EunseoOutputs:
        enc = self.encoder(input_x=input_x, missing_mask=missing_mask)
        z = self.final_norm(enc["z_init"] + 0.1 * enc["mechanism_repr"])
        field_tokens = enc["field_tokens"]

        if decoder_forward_mask is None:
            decoder_forward_mask = missing_mask

        forward_fields = self.forward_decoder(
            row_repr=z,
            field_tokens=field_tokens,
            condition_mask=decoder_forward_mask,
        )
        forward_num = forward_fields[:, : self.num_dim]
        forward_cat_logits = forward_fields[:, self.num_dim :]
        forward_cat_class_logits = self.forward_cat_classifier(
            row_repr=z,
            field_tokens=field_tokens,
            condition_mask=decoder_forward_mask,
        )

        return EunseoOutputs(
            z_init=enc["z_init"],
            z_final=z,
            mechanism_repr=enc["mechanism_repr"],
            forward_num=forward_num,
            forward_cat_logits=forward_cat_logits,
            forward_cat_class_logits=forward_cat_class_logits,
        )

