from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CorruptionConfig:
    missing_edge_prob: float = 0.0
    spurious_edge_prob: float = 0.0
    missing_node_prob: float = 0.0


def kron_power(x: torch.Tensor, K: int) -> torch.Tensor:
    out = x
    for _ in range(K - 1):
        out = torch.kron(out, x)
    return out


def sample_base_params(
    M: int,
    p_low: float = 0.05,
    p_high: float = 0.95,
    dirichlet_alpha: float = 1.0,
    symmetric_p: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if generator is not None and abs(dirichlet_alpha - 1.0) < 1e-12:
        l1 = -torch.rand(M, generator=generator).clamp_min(1e-8).log()
    else:
        gamma = torch.distributions.Gamma(torch.full((M,), dirichlet_alpha), torch.ones(M))
        l1 = gamma.sample()
    l1 = l1 / l1.sum().clamp_min(1e-8)
    p1 = p_low + (p_high - p_low) * torch.rand(M, M, generator=generator)
    if symmetric_p:
        p1 = 0.5 * (p1 + p1.t())
    return p1.clamp(1e-4, 1 - 1e-4), l1


def log_bin_weights(levels: torch.Tensor) -> torch.Tensor:
    """Map nonnegative edge levels to observed log-binned adjacency."""
    positive = levels > 0
    out = torch.zeros_like(levels, dtype=torch.float32)
    out[positive] = torch.floor(torch.log2(levels[positive].float() + 1.0))
    return out


def generate_wmgm_graph(
    p1: torch.Tensor,
    l1: torch.Tensor,
    K: int,
    num_nodes: int | None = None,
    directed: bool = False,
    corruption: CorruptionConfig | None = None,
    sort_nodes: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Generate one synthetic WMGM graph.

    pK and lK are Kronecker powers. Nodes are sampled from lK. Edge levels are
    sampled from Geometric(1 - pK[class_i, class_j]) - 1, giving nonnegative
    levels with larger p producing heavier edges. Observed A is log2-binned.
    """
    M = l1.numel()
    lK = kron_power(l1, K)
    pK = kron_power(p1, K).clamp(1e-6, 1 - 1e-6)
    if num_nodes is None:
        num_nodes = M**K

    classes = torch.multinomial(lK, num_nodes, replacement=True, generator=generator)
    probs = pK[classes[:, None], classes[None, :]]
    geom = torch.distributions.Geometric(probs=(1.0 - probs).clamp(1e-6, 1.0))
    levels = (geom.sample() - 1.0).to(torch.float32)
    levels.fill_diagonal_(0.0)
    if not directed:
        levels = torch.triu(levels, diagonal=1)
        levels = levels + levels.t()

    if corruption is not None:
        if corruption.missing_edge_prob > 0:
            keep = torch.rand_like(levels) >= corruption.missing_edge_prob
            levels = levels * keep
        if corruption.spurious_edge_prob > 0:
            add = torch.rand_like(levels) < corruption.spurious_edge_prob
            noise = torch.randint(1, 4, levels.shape, device=levels.device).float()
            levels = torch.where(add, torch.maximum(levels, noise), levels)
            levels.fill_diagonal_(0.0)
        if corruption.missing_node_prob > 0:
            keep_nodes = torch.rand(num_nodes, device=levels.device) >= corruption.missing_node_prob
            levels = levels * keep_nodes[:, None] * keep_nodes[None, :]
        if not directed:
            levels = torch.triu(levels, diagonal=1)
            levels = levels + levels.t()

    A = log_bin_weights(levels)
    if sort_nodes:
        degree = A.sum(dim=0) + A.sum(dim=1)
        idx = degree.argsort(descending=True)
        A = A[idx][:, idx]
        classes = classes[idx]

    return {"A": A, "p1": p1.float(), "l1": l1.float(), "classes": classes, "levels": levels}
