"""Neural-operator "generator" compressor for embedding sequences.

MWNOT-portable (../MWNOT-portable) infers the compact base parameters
(p1, l1) of a Weighted Multiplicative Graph Model from an observed weighted
adjacency matrix -- treating the matrix as an "image", lifting local
patches, multiwavelet-decomposing across scales, and cross-attention
decoding into a small, fixed-size generator. Only the generator needs to be
transmitted; the receiver reconstructs from it.

This module applies the same idea to the continuous-latent autoencoder: instead of transmitting
a large model's per-token hidden states (S x H, S = message length), or
naively sampling K of those S positions as done today, find a small,
fixed-size "generator" for the whole hidden-state matrix and transmit that.
The receiver expands it back to text by injecting it as prefix embeddings
into the same shared decoder LM, exactly like the existing K-sampled-latent
scheme -- so this is a drop-in replacement for the encode side only.

Adaptation from square adjacency matrices to token sequences:
  - hidden_size plays the role of MWNOT's lifted feature channels. It is
    projected once through a pointwise linear "lift", not a 2D patch basis:
    there is no spatial neighborhood structure across hidden dims to
    exploit (unlike pixels or graph-node neighborhoods), so a per-position
    linear projection is the natural analogue.
  - the sequence axis plays the role of the single spatial axis that gets
    multiscale-decomposed: nearby tokens are correlated at many scales,
    same as nearby graph nodes after degree sorting.
  - WaveletDecomposition2D expects two spatial axes; the second is a
    singleton dummy. Replicate-padding it from 1 to 2 makes the LH/HH
    detail bands identically zero (see the derivation in
    `training.programs.pretrain_mwnot_autoencoder`) and collapses the transform to a
    plain 1D Haar decomposition along the sequence axis -- verified by the
    shape/gradient smoke test in tests/test_mwnot_generator.py.
"""

from __future__ import annotations

import torch
from torch import nn

from multimodal_comms.methods.autoencoders.mwnot.attention import CrossAttentionDecoder
from multimodal_comms.methods.autoencoders.mwnot.tokenizer import MultiscaleTokenizer
from multimodal_comms.methods.autoencoders.mwnot.wavelets import WaveletDecomposition2D, downsample_mask


class SequenceGeneratorEncoder(nn.Module):
    """Compress a variable-length (S, H) hidden-state sequence into
    `num_latents` fixed-size H-dim generator vectors."""

    def __init__(
        self,
        hidden_size: int,
        num_latents: int = 4,
        lift_channels: int = 32,
        embed_dim: int = 256,
        wavelet_levels: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.lift_channels = lift_channels
        self.embed_dim = embed_dim
        self.wavelet_levels = wavelet_levels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_mult = ff_mult
        self.dropout = dropout

        self.lift = nn.Linear(hidden_size, lift_channels)
        self.wavelets = WaveletDecomposition2D(wavelet_levels)
        self.tokenizer = MultiscaleTokenizer(
            lift_channels, embed_dim, max_levels=max(16, wavelet_levels + 1), dropout=dropout
        )
        self.generator_decoder = CrossAttentionDecoder(
            num_latents, embed_dim, num_heads, num_layers, ff_mult=ff_mult, dropout=dropout
        )
        self.to_hidden = nn.Linear(embed_dim, hidden_size)

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """hidden: (B, S, H) float32. valid_mask: (B, S) bool, True = real token.

        Returns generator: (B, num_latents, H) float32.
        """
        x = self.lift(hidden)  # (B, S, C)
        x = x.transpose(1, 2).unsqueeze(-1)  # (B, C, S, 1)

        detail_masks, coarse_mask = None, None
        if valid_mask is not None:
            mask2d = valid_mask[:, :, None]  # (B, S, 1)
            detail_masks, coarse_mask = downsample_mask(mask2d, self.wavelet_levels)

        details, coarse = self.wavelets(x)
        tokens, key_padding_mask = self.tokenizer(details, coarse, detail_masks, coarse_mask)
        gen = self.generator_decoder(tokens, key_padding_mask)
        return self.to_hidden(gen)

    def config_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_latents": self.num_latents,
            "lift_channels": self.lift_channels,
            "embed_dim": self.embed_dim,
            "wavelet_levels": self.wavelet_levels,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_mult": self.ff_mult,
            "dropout": self.dropout,
        }
