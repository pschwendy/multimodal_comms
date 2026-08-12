from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .wmgm import CorruptionConfig, generate_wmgm_graph, sample_base_params


@dataclass
class WMGMDatasetConfig:
    num_samples: int = 1024
    M: int = 4
    K: int = 3
    min_nodes: int | None = None
    max_nodes: int | None = None
    symmetric_p: bool = True
    sort_nodes: bool = False
    corruption: CorruptionConfig | None = None


class WMGMDataset(Dataset[dict[str, torch.Tensor]]):
    """On-the-fly synthetic WMGM dataset predicting base p1 and l1."""

    def __init__(self, cfg: WMGMDatasetConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        p1, l1 = sample_base_params(self.cfg.M, symmetric_p=self.cfg.symmetric_p, generator=gen)
        n = None
        if self.cfg.min_nodes is not None and self.cfg.max_nodes is not None:
            n = int(torch.randint(self.cfg.min_nodes, self.cfg.max_nodes + 1, (1,), generator=gen).item())
        return generate_wmgm_graph(
            p1=p1,
            l1=l1,
            K=self.cfg.K,
            num_nodes=n,
            corruption=self.cfg.corruption,
            sort_nodes=self.cfg.sort_nodes,
            generator=gen,
        )


def collate_wmgm(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-size adjacency matrices.

    Returns A: B x Nmax x Nmax and mask: B x Nmax x Nmax where True marks
    real cells and False marks padding.
    """
    bsz = len(batch)
    nmax = max(item["A"].shape[0] for item in batch)
    A = torch.zeros(bsz, nmax, nmax, dtype=torch.float32)
    mask = torch.zeros(bsz, nmax, nmax, dtype=torch.bool)
    p1 = torch.stack([item["p1"] for item in batch])
    l1 = torch.stack([item["l1"] for item in batch])
    for b, item in enumerate(batch):
        n = item["A"].shape[0]
        A[b, :n, :n] = item["A"]
        mask[b, :n, :n] = True
    return {"A": A, "mask": mask, "p1": p1, "l1": l1}
