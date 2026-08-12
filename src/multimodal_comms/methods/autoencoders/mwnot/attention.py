from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn, _ = self.attn(self.q_norm(q), self.kv_norm(tokens), self.kv_norm(tokens), key_padding_mask=key_padding_mask)
        q = q + self.drop(attn)
        q = q + self.ff(self.ff_norm(q))
        return q


class CrossAttentionDecoder(nn.Module):
    """Learnable-query cross-attention decoder."""

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads, ff_mult=ff_mult, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, tokens, key_padding_mask)
        return self.norm(q)
