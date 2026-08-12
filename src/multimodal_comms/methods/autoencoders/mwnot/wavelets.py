from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WaveletDecomposition2D(nn.Module):
    """2D Haar decomposition approximation to multiwavelet decomposition.

    Input x: B x C x H x W.
    Each level returns detail coefficients B x (3C) x ceil(H/2) x ceil(W/2)
    in [LH, HL, HH] channel order and updates the coarse LL stream.
    Final coarse has shape B x C x h_L x w_L.
    """

    def __init__(self, levels: int = 3) -> None:
        super().__init__()
        self.levels = max(1, levels)

    @staticmethod
    def _pad_even(x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        details: list[torch.Tensor] = []
        coarse = x
        for _ in range(self.levels):
            coarse = self._pad_even(coarse)
            x00 = coarse[:, :, 0::2, 0::2]
            x01 = coarse[:, :, 0::2, 1::2]
            x10 = coarse[:, :, 1::2, 0::2]
            x11 = coarse[:, :, 1::2, 1::2]
            ll = (x00 + x01 + x10 + x11) * 0.5
            lh = (x00 - x01 + x10 - x11) * 0.5
            hl = (x00 + x01 - x10 - x11) * 0.5
            hh = (x00 - x01 - x10 + x11) * 0.5
            details.append(torch.cat([lh, hl, hh], dim=1))
            coarse = ll
        return details, coarse


def downsample_mask(mask: torch.Tensor, levels: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Downsample B x H x W boolean masks with max pooling."""
    masks: list[torch.Tensor] = []
    cur = mask[:, None].float()
    for _ in range(max(1, levels)):
        _, _, H, W = cur.shape
        if H % 2 or W % 2:
            cur = F.pad(cur, (0, W % 2, 0, H % 2), value=0.0)
        cur = F.max_pool2d(cur, kernel_size=2, stride=2)
        masks.append(cur[:, 0].bool())
    return masks, cur[:, 0].bool()
