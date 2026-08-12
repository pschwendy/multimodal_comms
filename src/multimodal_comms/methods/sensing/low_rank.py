from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(slots=True)
class SVDCodec:
    rank: int

    def encode(self, value: ArrayLike, *, seed: int = 0) -> dict[str, NDArray[np.float64]]:
        matrix = np.asarray(value, dtype=np.float64)
        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        rank = min(self.rank, len(singular))
        return {"u": u[:, :rank], "s": singular[:rank], "vt": vt[:rank]}

    def decode(self, code, *, seed: int = 0) -> NDArray[np.float64]:
        return (code["u"] * code["s"]) @ code["vt"]


@dataclass(slots=True)
class PCACodec:
    rank: int

    def encode(self, value: ArrayLike, *, seed: int = 0):
        matrix = np.asarray(value, dtype=np.float64)
        mean = matrix.mean(axis=0)
        u, singular, vt = np.linalg.svd(matrix - mean, full_matrices=False)
        rank = min(self.rank, len(singular))
        return {"scores": u[:, :rank] * singular[:rank], "components": vt[:rank], "mean": mean}

    def decode(self, code, *, seed: int = 0):
        return code["scores"] @ code["components"] + code["mean"]


@dataclass(slots=True)
class CURCodec:
    rank: int

    def encode(self, value: ArrayLike, *, seed: int = 0):
        matrix = np.asarray(value, dtype=np.float64)
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(matrix.shape[0], min(self.rank, matrix.shape[0]), replace=False))
        cols = np.sort(rng.choice(matrix.shape[1], min(self.rank, matrix.shape[1]), replace=False))
        c = matrix[:, cols]
        r = matrix[rows, :]
        intersection = matrix[np.ix_(rows, cols)]
        return {"c": c, "u": np.linalg.pinv(intersection), "r": r}

    def decode(self, code, *, seed: int = 0):
        return code["c"] @ code["u"] @ code["r"]


class StreamDiagnostics:
    @staticmethod
    def effective_rank(value: ArrayLike, tolerance: float = 1e-8) -> int:
        singular = np.linalg.svd(np.asarray(value), compute_uv=False)
        return int(np.count_nonzero(singular > tolerance * singular[0])) if singular.size else 0

    @staticmethod
    def relative_error(original: ArrayLike, reconstructed: ArrayLike) -> float:
        original = np.asarray(original)
        denominator = float(np.linalg.norm(original))
        numerator = float(np.linalg.norm(original - np.asarray(reconstructed)))
        return numerator / (denominator if denominator else 1.0)
