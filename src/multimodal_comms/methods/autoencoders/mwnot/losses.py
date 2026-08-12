from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MWNOTLoss(nn.Module):
    """Composite base-parameter loss: BCE for p1 logits and KL for l1."""

    def __init__(self, p_weight: float = 1.0, l_weight: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.p_weight = p_weight
        self.l_weight = l_weight
        self.eps = eps

    def forward(self, p_logits: torch.Tensor, l_logits: torch.Tensor, p_target: torch.Tensor, l_target: torch.Tensor) -> dict[str, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(p_logits, p_target)
        log_q = F.log_softmax(l_logits, dim=-1)
        target = l_target.clamp_min(self.eps)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        kl = F.kl_div(log_q, target, reduction="batchmean")
        total = self.p_weight * bce + self.l_weight * kl
        return {"loss": total, "bce": bce.detach(), "kl": kl.detach()}


@torch.no_grad()
def metrics(p_logits: torch.Tensor, l_logits: torch.Tensor, p_target: torch.Tensor, l_target: torch.Tensor) -> dict[str, float]:
    p_mae = (torch.sigmoid(p_logits) - p_target).abs().mean().item()
    l_mae = (torch.softmax(l_logits, dim=-1) - l_target).abs().mean().item()
    return {"p_mae": p_mae, "l_mae": l_mae}
