from __future__ import annotations

import torch
from torch import nn


class MultiscaleTokenizer(nn.Module):
    """Flatten multiscale coefficient maps into transformer tokens."""

    def __init__(self, in_channels: int, embed_dim: int, max_levels: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.coarse_proj = nn.Linear(in_channels, embed_dim)
        self.detail_proj = nn.Linear(3 * in_channels, embed_dim)
        self.coord_proj = nn.Linear(2, embed_dim)
        self.level_embed = nn.Embedding(max_levels, embed_dim)
        self.type_embed = nn.Embedding(2, embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _coords(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([yy, xx], dim=-1).view(1, H * W, 2).expand(B, -1, -1)

    def _map_to_tokens(
        self,
        x: torch.Tensor,
        level: int,
        typ: int,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        proj = self.detail_proj if typ == 1 else self.coarse_proj
        tok = proj(seq)
        tok = tok + self.coord_proj(self._coords(B, H, W, x.device))
        tok = tok + self.level_embed(torch.full((B, H * W), level, dtype=torch.long, device=x.device))
        tok = tok + self.type_embed(torch.full((B, H * W), typ, dtype=torch.long, device=x.device))
        flat_mask = None if mask is None else mask.flatten(1)
        return self.dropout(tok), flat_mask

    def forward(
        self,
        details: list[torch.Tensor],
        coarse: torch.Tensor,
        detail_masks: list[torch.Tensor] | None = None,
        coarse_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for i, d in enumerate(details):
            m = None if detail_masks is None else detail_masks[i]
            t, fm = self._map_to_tokens(d, i, 1, m)
            tokens.append(t)
            if fm is not None:
                masks.append(fm)
        t, fm = self._map_to_tokens(coarse, len(details), 0, coarse_mask)
        tokens.append(t)
        if fm is not None:
            masks.append(fm)
        key_padding_mask = None
        if masks:
            valid = torch.cat(masks, dim=1)
            key_padding_mask = ~valid
        return torch.cat(tokens, dim=1), key_padding_mask
