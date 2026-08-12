from __future__ import annotations

import torch
from torch import nn

from .attention import CrossAttentionDecoder
from .configs import MWNOTConfig
from .lifting import LegendrePatchProjector, LinearPatchProjector
from .tokenizer import MultiscaleTokenizer
from .utils import sort_adjacency_by_degree
from .wavelets import WaveletDecomposition2D, downsample_mask


class MWNOTEncoder(nn.Module):
    """Preprocess, lift, wavelet-decompose, and tokenize adjacency matrices."""

    def __init__(self, cfg: MWNOTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.lifting == "legendre":
            self.lifting = LegendrePatchProjector(cfg.patch_size, cfg.poly_order)
            in_channels = self.lifting.out_channels
        elif cfg.lifting == "linear":
            self.lifting = LinearPatchProjector(cfg.patch_size, cfg.linear_lift_channels)
            in_channels = cfg.linear_lift_channels
        else:
            raise ValueError(f"unknown lifting {cfg.lifting}")
        self.in_channels = in_channels
        self.wavelets = WaveletDecomposition2D(cfg.wavelet_levels)
        self.tokenizer = MultiscaleTokenizer(in_channels, cfg.embed_dim, cfg.max_levels, cfg.dropout)

    def preprocess(self, A: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = A.float()
        if self.cfg.symmetrize_input:
            x = 0.5 * (x + x.transpose(-1, -2))
        if self.cfg.sort_nodes:
            valid_nodes = None if mask is None else mask.any(dim=-1)
            x = sort_adjacency_by_degree(x, valid_nodes)
        if self.cfg.normalize_input:
            if self.cfg.preserve_zeros:
                nz = x > 0
                denom = x.masked_fill(~nz, 0.0).amax(dim=(-1, -2), keepdim=True).clamp_min(1.0)
                x = torch.where(nz, x / denom, x)
            else:
                denom = x.amax(dim=(-1, -2), keepdim=True).clamp_min(1.0)
                x = x / denom
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        return x

    def forward(self, A: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.preprocess(A, mask)
        lifted = self.lifting(x)
        if not self.cfg.use_wavelets:
            details: list[torch.Tensor] = []
            coarse = lifted
            detail_masks: list[torch.Tensor] = []
            coarse_mask = mask
        else:
            details, coarse = self.wavelets(lifted)
            detail_masks, coarse_mask = (None, None) if mask is None else downsample_mask(mask, self.cfg.wavelet_levels)
        return self.tokenizer(details, coarse, detail_masks, coarse_mask)


class MWNOTModel(nn.Module):
    """Multiwavelet Neural Operator Transformer for base WMGM parameters."""

    def __init__(self, cfg: MWNOTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        M = cfg.M
        self.encoder = MWNOTEncoder(cfg)
        self.l_decoder = CrossAttentionDecoder(M, cfg.embed_dim, cfg.num_heads, cfg.num_layers, cfg.ff_mult, cfg.dropout)
        self.p_decoder = CrossAttentionDecoder(M * M, cfg.embed_dim, cfg.num_heads, cfg.num_layers, cfg.ff_mult, cfg.dropout)
        self.l_head = nn.Linear(cfg.embed_dim, 1)
        self.p_head = nn.Linear(cfg.embed_dim, 1)

    def forward(self, A: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens, key_padding_mask = self.encoder(A, mask)
        l_q = self.l_decoder(tokens, key_padding_mask)
        p_q = self.p_decoder(tokens, key_padding_mask)
        l_logits = self.l_head(l_q).squeeze(-1)
        p_logits = self.p_head(p_q).squeeze(-1).view(A.shape[0], self.cfg.M, self.cfg.M)
        if self.cfg.enforce_p_symmetry:
            p_logits = 0.5 * (p_logits + p_logits.transpose(-1, -2))
        return {"p_logits": p_logits, "l_logits": l_logits}

    @torch.no_grad()
    def predict(self, A: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = self.forward(A, mask)
        return {"p1": torch.sigmoid(out["p_logits"]), "l1": torch.softmax(out["l_logits"], dim=-1), **out}
