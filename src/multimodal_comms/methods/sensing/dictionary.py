"""Compressed-sensing codec for continuous token/hidden-state packets.

Tier-1 (signal-level) evaluation lives here: given a packet of row-vectors
(e.g. a (100, 2048) slice of per-token hidden states), compress each row to
M << N measurements with a fixed sensing matrix, then reconstruct via sparse
recovery against a dictionary that both endpoints already hold. Nothing here
talks to an LLM -- see experiments.compressed_sensing.programs.eval_compressed_sensing for the
rate-distortion / sparsity sweep, and training.data.harvest_cs_packets for
pulling real hidden-state packets to test against.

The sensing matrix and dictionary are the two things sender and receiver
must share in advance (a fixed seed reproduces the sensing matrix; the
dictionary is fit offline once and frozen). Only the M-dimensional
measurement vector per token crosses the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import Lasso, orthogonal_mp

RecoveryMethod = Literal["omp", "lasso"]
DictionaryKind = Literal["identity", "random", "dct", "learned"]


def make_sensing_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    """i.i.d. Gaussian sensing matrix (m, n), columns normalized to unit norm.

    Reproducible from `seed` alone, so it never has to be transmitted --
    both endpoints regenerate it locally.
    """
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((m, n))
    phi /= np.linalg.norm(phi, axis=0, keepdims=True)
    return phi


def make_dictionary(
    kind: DictionaryKind,
    dim: int,
    n_atoms: int,
    data: np.ndarray | None = None,
    spatial_shape: tuple[int, int] | None = None,
    seed: int = 0,
    alpha: float = 1.0,
    max_iter: int = 500,
) -> np.ndarray:
    """Build the shared dictionary D, shape (dim, n_atoms), unit-norm columns.

    - "identity": standard basis (requires n_atoms == dim).
    - "random": random Gaussian atoms -- a generic incoherent basis with no
      structure, i.e. the null hypothesis that packets aren't sparse in
      anything in particular.
    - "dct": orthonormal DCT synthesis basis (requires n_atoms == dim).
      Pass ``spatial_shape=(height, width)`` for the separable 2-D basis used
      by image blocks. Without it, this is a 1-D vector basis. Hidden-state
      vectors have no obvious reason to be sparse in either basis, which is
      exactly worth checking empirically.
    - "learned": online dictionary learning (MiniBatchDictionaryLearning,
      the sklearn K-SVD-style algorithm) fit on `data`, shape
      (n_samples, dim). This is the realistic "shared dictionary" case --
      fit once offline on representative traffic, then frozen and shipped
      to both endpoints.
    """
    if kind == "identity":
        if n_atoms != dim:
            raise ValueError("identity dictionary requires n_atoms == dim")
        return np.eye(dim)

    if kind == "random":
        rng = np.random.default_rng(seed)
        d = rng.standard_normal((dim, n_atoms))
        return d / np.linalg.norm(d, axis=0, keepdims=True)

    if kind == "dct":
        if n_atoms != dim:
            raise ValueError("dct dictionary requires n_atoms == dim")
        from scipy.fft import dct, idctn

        if spatial_shape is None:
            # scipy's DCT matrix is the analysis transform; dictionary atoms
            # must be columns of its inverse (the synthesis transform).
            d = dct(np.eye(dim), axis=0, norm="ortho").T
        else:
            if spatial_shape[0] * spatial_shape[1] != dim:
                raise ValueError("spatial_shape product must equal dim")
            atoms = []
            for atom in range(dim):
                coefficient = np.zeros(spatial_shape, dtype=np.float64)
                coefficient.flat[atom] = 1.0
                atoms.append(idctn(coefficient, norm="ortho").reshape(-1))
            d = np.stack(atoms, axis=1)
        return d / np.linalg.norm(d, axis=0, keepdims=True)

    if kind == "learned":
        if data is None:
            raise ValueError("learned dictionary requires `data` (n_samples, dim)")
        learner = MiniBatchDictionaryLearning(
            n_components=n_atoms,
            alpha=alpha,
            max_iter=max_iter,
            transform_algorithm="omp",
            random_state=seed,
        )
        learner.fit(data)
        d = learner.components_.T  # (dim, n_atoms)
        return d / np.linalg.norm(d, axis=0, keepdims=True)

    raise ValueError(f"unknown dictionary kind: {kind}")


def _lasso_code(a: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
    model.fit(a, y)
    return model.coef_


@dataclass
class CSCodec:
    """A fixed (sensing matrix, dictionary) pair -- the shared codec state."""

    phi: np.ndarray  # (m, n) sensing matrix
    dictionary: np.ndarray  # (n, n_atoms) shared dictionary, unit-norm columns
    method: RecoveryMethod = "omp"
    sparsity: int = 10  # target nonzero coefs for OMP
    lasso_alpha: float = 0.01  # regularization strength for lasso

    def __post_init__(self) -> None:
        if self.phi.shape[1] != self.dictionary.shape[0]:
            raise ValueError("phi columns must match dictionary rows (ambient dim)")
        self._a = self.phi @ self.dictionary  # (m, n_atoms), cached

    @property
    def m(self) -> int:
        return self.phi.shape[0]

    @property
    def n(self) -> int:
        return self.phi.shape[1]

    @property
    def compression_ratio(self) -> float:
        """M / N -- fraction of the ambient dimension actually transmitted."""
        return self.m / self.n

    def encode(self, x: np.ndarray) -> np.ndarray:
        """x: (n,) or (n_samples, n) row-vectors -> (m,) or (n_samples, m)."""
        return x @ self.phi.T

    def decode(self, y: np.ndarray) -> np.ndarray:
        """Sparse-recover in the shared dictionary and reconstruct x_hat."""
        y2 = np.atleast_2d(y)
        if self.method == "omp":
            codes = orthogonal_mp(self._a, y2.T, n_nonzero_coefs=self.sparsity).T
        elif self.method == "lasso":
            codes = np.stack([_lasso_code(self._a, yi, self.lasso_alpha) for yi in y2])
        else:
            raise ValueError(f"unknown recovery method: {self.method}")
        x_hat = codes @ self.dictionary.T
        return x_hat if y.ndim == 2 else x_hat[0]

    def roundtrip(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))


def relative_l2_error(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-row relative reconstruction error ||x - x_hat|| / ||x||."""
    num = np.linalg.norm(x - x_hat, axis=-1)
    den = np.linalg.norm(x, axis=-1)
    return num / np.clip(den, 1e-12, None)


def cosine_similarity_rows(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between original and reconstructed vectors."""
    num = np.sum(x * x_hat, axis=-1)
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(x_hat, axis=-1)
    return num / np.clip(den, 1e-12, None)


def oracle_sparse_fit(dictionary: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """Best k-term OMP fit of x directly in `dictionary`, no sensing/CS step.

    This is the oracle upper bound on what any M-measurement CS scheme could
    hope to recover at sparsity k -- use it to check whether the packets are
    actually sparse in a candidate dictionary before spending a sweep on
    recovery hyperparameters.
    """
    codes = orthogonal_mp(dictionary, x.T, n_nonzero_coefs=k).T
    return codes @ dictionary.T


@dataclass
class CodebookCS:
    """CS over packets whose atoms are a known, finite codebook -- e.g. a
    model's token embedding table.

    Each packet row is *exactly* 1-sparse in the codebook: it literally IS
    one specific atom (a token's embedding), coefficient 1. Recovery is then
    nearest-neighbor search in the M-dimensional measurement space rather
    than iterative sparse coding: project the whole codebook once with the
    shared sensing matrix (both sides can do this locally -- nothing new is
    transmitted), then match each compressed measurement against it.

    This is compressed sensing via the Johnson-Lindenstrauss lemma: a random
    projection to M = O(log(n_atoms)/eps^2) dimensions approximately
    preserves pairwise distances, so nearest-neighbor identity in the
    projected space tracks nearest-neighbor identity in the original space
    with high probability -- exact identification should need far fewer
    measurements than generic k-sparse recovery in an unstructured
    dictionary.
    """

    phi: np.ndarray  # (m, dim)
    codebook: np.ndarray  # (n_atoms, dim), e.g. the embedding table

    def __post_init__(self) -> None:
        self.codebook = self.codebook.astype(np.float32, copy=False)
        self.phi = self.phi.astype(np.float32, copy=False)
        self._projected = self.codebook @ self.phi.T  # (n_atoms, m)
        self._proj_sq = np.sum(self._projected**2, axis=1)  # (n_atoms,)

    @property
    def m(self) -> int:
        return self.phi.shape[0]

    def encode(self, x: np.ndarray) -> np.ndarray:
        return x @ self.phi.T

    def decode_indices(self, y: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Nearest-neighbor codebook index for each measurement row in y."""
        y2 = np.atleast_2d(y).astype(np.float32, copy=False)
        idx = np.empty(y2.shape[0], dtype=np.int64)
        for start in range(0, y2.shape[0], batch_size):
            batch = y2[start:start + batch_size]
            # squared L2 distance via ||a||^2 - 2a.b + ||b||^2, argmin drops the ||y||^2 term
            scores = batch @ self._projected.T  # (batch, n_atoms)
            dist = self._proj_sq[None, :] - 2 * scores
            idx[start:start + batch_size] = np.argmin(dist, axis=1)
        return idx if y.ndim == 2 else idx[0]

    def decode(self, y: np.ndarray) -> np.ndarray:
        idx = self.decode_indices(y)
        return self.codebook[idx]

    def roundtrip(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))


def token_recovery_accuracy(true_indices: np.ndarray, recovered_indices: np.ndarray) -> float:
    return float(np.mean(true_indices == recovered_indices))


def leverage_scores(x: np.ndarray, rank: int) -> np.ndarray:
    """Per-row statistical leverage scores of packet `x` w.r.t. its OWN
    top-`rank` left-singular subspace.

    A row with high leverage is poorly explained by a low-rank fit to the
    REST of the packet -- it carries information the other rows can't
    substitute for, so it's a natural pick for "must be measured exactly."
    Standard in randomized numerical linear algebra (CUR decomposition,
    matrix sketching / column-subset selection) -- computed purely from the
    packet's own SVD, no learned encoder involved.
    """
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    u_r = u[:, :rank]
    return np.sum(u_r**2, axis=1)


def select_landmarks(scores: np.ndarray, n_landmarks: int) -> np.ndarray:
    """Indices of the `n_landmarks` highest-scoring rows."""
    return np.argsort(-scores)[:n_landmarks]


def cur_reconstruct(x: np.ndarray, landmark_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct every row of `x` as a least-squares linear combination of
    the exactly-known landmark rows (CUR-style: a handful of ACTUAL rows of
    the matrix serve as the reconstruction basis for the rest).

    Landmark rows are copied through exactly (they're assumed transmitted
    losslessly, e.g. via CodebookCS); every other row is approximated. The
    coefficients returned are what a receiver would need per non-landmark
    row to reconstruct it -- the actual "compressed representation".
    """
    landmarks = x[landmark_idx]  # (r, dim)
    coeffs, *_ = np.linalg.lstsq(landmarks.T, x.T, rcond=None)  # (r, n_rows)
    x_hat = coeffs.T @ landmarks
    x_hat[landmark_idx] = x[landmark_idx]
    coeffs = coeffs.T  # (n_rows, r)
    coeffs[landmark_idx] = 0.0
    return x_hat, coeffs


def fit_global_basis(rows: np.ndarray, rank: int) -> np.ndarray:
    """Fit a shared, packet-agnostic reconstruction basis (rank, dim) via
    PCA on pooled rows from a held-out fit corpus of real packets.

    This is the "optimization phase" dictionary: fit once offline on
    representative traffic, then frozen and known to both endpoints ahead
    of time -- no per-packet landmarks need to be transmitted at all, only
    each row's coefficients against this fixed basis.
    """
    mean = rows.mean(axis=0)
    _, _, vt = np.linalg.svd(rows - mean, full_matrices=False)
    return vt[:rank]  # (rank, dim), orthonormal rows


def project_reconstruct(x: np.ndarray, basis: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct every row of `x` by projecting onto a shared, pre-fit
    `basis` (rank, dim) -- e.g. from fit_global_basis. Returns (x_hat,
    coeffs) where coeffs (n_rows, rank) is what actually needs transmitting.
    """
    coeffs = (x - mean) @ basis.T  # (n_rows, rank)
    x_hat = coeffs @ basis + mean
    return x_hat, coeffs


def energy_captured_curve(dictionary: np.ndarray, x: np.ndarray, ks: list[int]) -> dict[int, float]:
    """Fraction of packet energy captured by the oracle k-term fit, per k.

    A dictionary the packets are genuinely sparse in should show this
    climbing steeply at small k; a flat, slow climb means CS on this
    dictionary won't beat naive dimensionality reduction (e.g. PCA/random
    projection without a sparsity assumption).
    """
    total_energy = np.sum(x**2)
    out: dict[int, float] = {}
    for k in ks:
        x_hat = oracle_sparse_fit(dictionary, x, k)
        residual_energy = np.sum((x - x_hat) ** 2)
        out[k] = 1.0 - residual_energy / total_energy
    return out
