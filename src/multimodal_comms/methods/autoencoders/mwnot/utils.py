from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device(name: str = "cuda") -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def linear_warmup_cosine(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def sort_adjacency_by_degree(A: torch.Tensor, valid_nodes: torch.Tensor | None = None) -> torch.Tensor:
    """Sort nodes by descending weighted degree.

    A has shape (..., N, N). valid_nodes, if given, has shape (..., N).
    """
    degree = A.sum(dim=-1) + A.sum(dim=-2)
    if valid_nodes is not None:
        degree = degree.masked_fill(~valid_nodes, -1.0)
    idx = degree.argsort(dim=-1, descending=True)
    A = torch.gather(A, -2, idx.unsqueeze(-1).expand(*idx.shape, A.size(-1)))
    A = torch.gather(A, -1, idx.unsqueeze(-2).expand(*A.shape[:-2], A.size(-2), idx.size(-1)))
    return A
