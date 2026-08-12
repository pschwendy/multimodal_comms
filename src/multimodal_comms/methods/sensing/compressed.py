from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Recovery = Literal["omp", "ridge"]


def make_sensing_matrix(
    measurements: int, dimension: int, seed: int = 0
) -> NDArray[np.float64]:
    """Return a reproducible Gaussian sensing matrix with unit-norm columns."""
    if not 0 < measurements <= dimension:
        raise ValueError("require 0 < measurements <= dimension")
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((measurements, dimension))
    norms = np.linalg.norm(phi, axis=0, keepdims=True)
    return phi / np.clip(norms, 1e-12, None)


def _omp(
    design: NDArray[np.float64],
    observations: NDArray[np.float64],
    sparsity: int,
) -> NDArray[np.float64]:
    """Small NumPy OMP implementation used by the public dependency-light codec."""
    coefficients = np.zeros((observations.shape[0], design.shape[1]), dtype=np.float64)
    column_norms = np.linalg.norm(design, axis=0)
    if np.any(column_norms <= 1e-12):
        raise ValueError("the sensed dictionary contains a zero-norm atom")

    for row_index, target in enumerate(observations):
        residual = target.copy()
        support: list[int] = []
        for _ in range(sparsity):
            scores = np.abs(design.T @ residual) / column_norms
            if support:
                scores[support] = -np.inf
            atom = int(np.argmax(scores))
            if not np.isfinite(scores[atom]):
                break
            support.append(atom)
            active = design[:, support]
            active_coefficients, *_ = np.linalg.lstsq(active, target, rcond=None)
            residual = target - active @ active_coefficients
            if np.linalg.norm(residual) <= 1e-10 * max(np.linalg.norm(target), 1.0):
                break
        if support:
            coefficients[row_index, support] = active_coefficients
    return coefficients


@dataclass(slots=True)
class CompressedSensingCodec:
    """Shared sensing/dictionary codec with explicit sparse recovery.

    ``dictionary`` stores atoms in columns. Signals must therefore be sparse
    (or at least compressible) as ``coefficients @ dictionary.T`` for OMP to
    be appropriate. Smooth image blocks in a DCT basis often satisfy this;
    arbitrary dense embedding vectors generally do not. ``ridge`` is retained
    as an explicit minimum-norm projection baseline, not called compressed
    sensing.
    """

    phi: NDArray[np.float64]
    dictionary: NDArray[np.float64]
    sparsity: int | None = None
    recovery: Recovery = "omp"
    ridge: float = 1e-8
    _design: NDArray[np.float64] = field(init=False, repr=False)
    _ridge_decoder: NDArray[np.float64] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.phi = np.asarray(self.phi, dtype=np.float64)
        self.dictionary = np.asarray(self.dictionary, dtype=np.float64)
        if self.phi.ndim != 2 or self.dictionary.ndim != 2:
            raise ValueError("phi and dictionary must be matrices")
        if self.phi.shape[1] != self.dictionary.shape[0]:
            raise ValueError("phi columns must equal dictionary rows")
        if self.recovery not in {"omp", "ridge"}:
            raise ValueError("recovery must be 'omp' or 'ridge'")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")

        self._design = self.phi @ self.dictionary
        maximum = min(self._design.shape)
        if self.sparsity is None:
            # Full-rank square transforms retain exact round trips by default.
            # Undersampled use gets a conservative sparse prior which callers
            # should normally set from their domain knowledge or validation.
            self.sparsity = (
                maximum
                if self.phi.shape[0] == self.phi.shape[1]
                else max(1, self.phi.shape[0] // 4)
            )
        if not 1 <= self.sparsity <= maximum:
            raise ValueError(f"sparsity must be between 1 and {maximum}")

        self._ridge_decoder = None
        if self.recovery == "ridge":
            gram = self._design.T @ self._design
            self._ridge_decoder = np.linalg.solve(
                gram + self.ridge * np.eye(gram.shape[0]), self._design.T
            )

    @property
    def compression_ratio(self) -> float:
        return self.phi.shape[0] / self.phi.shape[1]

    def encode(self, value: ArrayLike, *, seed: int = 0) -> NDArray[np.float64]:
        value_array = np.asarray(value, dtype=np.float64)
        if value_array.shape[-1] != self.phi.shape[1]:
            raise ValueError("input's final dimension must match phi columns")
        return value_array @ self.phi.T

    def decode(self, code: ArrayLike, *, seed: int = 0) -> NDArray[np.float64]:
        code_array = np.asarray(code, dtype=np.float64)
        if code_array.shape[-1] != self.phi.shape[0]:
            raise ValueError("code's final dimension must match phi rows")
        original_shape = code_array.shape[:-1]
        rows = np.atleast_2d(code_array).reshape(-1, code_array.shape[-1])
        if self.recovery == "omp":
            coefficients = _omp(self._design, rows, self.sparsity)
        else:
            assert self._ridge_decoder is not None
            coefficients = rows @ self._ridge_decoder.T
        recovered = coefficients @ self.dictionary.T
        return recovered.reshape(*original_shape, self.dictionary.shape[0])

    def roundtrip(self, value: ArrayLike, *, seed: int = 0) -> NDArray[np.float64]:
        return self.decode(self.encode(value, seed=seed), seed=seed)
