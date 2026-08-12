from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _legendre_values(x: torch.Tensor, order: int) -> torch.Tensor:
    vals = [torch.ones_like(x)]
    if order > 1:
        vals.append(x)
    for n in range(2, order):
        vals.append(((2 * n - 1) * x * vals[-1] - (n - 1) * vals[-2]) / n)
    return torch.stack(vals, dim=-1)


class LegendrePatchProjector(nn.Module):
    """Project local A patches onto tensor-product Legendre basis.

    Input: A with shape B x N x N.
    Output: feature map with shape B x k^2 x N x N.

    This is a practical local polynomial lifting approximation; exact
    multiwavelet scaling functions can be substituted behind this interface.
    """

    def __init__(self, patch_size: int = 5, poly_order: int = 3) -> None:
        super().__init__()
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        self.patch_size = patch_size
        self.poly_order = poly_order
        grid = torch.linspace(-1.0, 1.0, patch_size)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        by = _legendre_values(yy.reshape(-1), poly_order)
        bx = _legendre_values(xx.reshape(-1), poly_order)
        basis = torch.einsum("pi,pj->pij", by, bx).reshape(patch_size * patch_size, poly_order * poly_order)
        basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", basis)

    @property
    def out_channels(self) -> int:
        return self.poly_order * self.poly_order

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        B, H, W = A.shape
        patches = F.unfold(A[:, None], kernel_size=self.patch_size, padding=self.patch_size // 2)
        coeff = torch.einsum("bpl,pc->bcl", patches, self.basis)
        return coeff.view(B, self.out_channels, H, W)


class LinearPatchProjector(nn.Module):
    """Learned patch lifting fallback.

    Input: B x N x N. Output: B x out_channels x N x N.
    """

    def __init__(self, patch_size: int = 5, out_channels: int = 16) -> None:
        super().__init__()
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(patch_size * patch_size, out_channels)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        B, H, W = A.shape
        patches = F.unfold(A[:, None], kernel_size=self.patch_size, padding=self.patch_size // 2)
        x = patches.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2).contiguous().view(B, self.out_channels, H, W)
