"""Diagnostic: is an embedding stream compressible along the STREAM axis?

compressed_sensing.py asks whether a single (T, dim) packet's per-vector
FEATURE axis (dim) can be shrunk -- CS on the token dimension. That was
tested and lost to direct token indexing (see memory: compressed_sensing
_pivot). This module asks the transposed question: given a stream of T
embedding vectors x_1..x_T in R^dim (any modality -- there is nothing
text-specific here, x_t is just "the embedding at stream position t"), can
T itself be shrunk? I.e. can a receiver reconstruct all T vectors from only
k << T transmitted quantities, exploiting redundancy ACROSS stream
positions rather than within a vector's coordinates?

This only pays off if consecutive/nearby stream positions are informative
about each other -- the temporal (or positional) analogue of "sparse in a
basis" from compressed sensing, and of low Renyi information dimension
from Wu's Shannon-theoretic treatment (dimension per unit of the stream
axis, not per coordinate). Three independent, complementary checks:

1. temporal_energy_captured_curve / effective_temporal_rank -- linear
   oracle bound. Truncated SVD of the (T, dim) packet is the best possible
   rank-r linear reconstruction (Eckart-Young); no learned or fixed
   temporal dictionary can beat it. Needs only one or a few packets.

2. causal_predictability_curve -- how well a short linear predictor from
   the previous `order` vectors explains x_t. Cheap, causal (works online,
   unlike the SVD check which needs the whole window), and directly
   interpretable: near-zero residual means the stream is smoothly evolving
   and interpolatable; near-one residual means every step is fresh
   information, the same regime that sank the per-vector CS pivot.

3. windowed_intrinsic_dimension -- nonlinear oracle bound. Levina-Bickel
   MLE intrinsic dimension of the point cloud formed by many length-T
   windows (each window flattened to one point in R^(T*dim)). Catches
   redundancy that lives on a curved manifold rather than a linear
   subspace, which check 1 would miss entirely. Needs many window samples
   (many packets), not just one.

Gate: if none of these show structure well below T, stop -- temporal CS
cannot beat sending every vector, for the same reason the token-index
pivot failed (signal near its own entropy rate, nothing to exploit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


def temporal_energy_captured_curve(x: np.ndarray, rs: list[int]) -> dict[int, float]:
    """Fraction of a (T, dim) packet's energy captured by the best rank-r
    linear reconstruction ACROSS TIME, for each r in `rs`.

    Truncated SVD is the Eckart-Young oracle: no temporal dictionary (fixed
    or learned, DCT/wavelet/PCA/anything) can beat it at a given rank. If
    this curve doesn't climb steeply well below r=T, no temporal-CS scheme
    exists that would help on this data -- check before building one.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x.shape[0]
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    total_energy = np.sum(s**2)
    out: dict[int, float] = {}
    for r in rs:
        r_eff = min(r, t, len(s))
        captured = np.sum(s[:r_eff] ** 2)
        out[r] = float(captured / total_energy) if total_energy > 0 else 1.0
    return out


def effective_temporal_rank(x: np.ndarray, energy_threshold: float = 0.95) -> int:
    """Smallest r such that a rank-r linear reconstruction across time
    captures `energy_threshold` fraction of the packet's energy."""
    x = np.asarray(x, dtype=np.float64)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    total = np.sum(s**2)
    if total == 0:
        return 0
    cumulative = np.cumsum(s**2) / total
    return int(np.searchsorted(cumulative, energy_threshold) + 1)


def causal_predictability_curve(x: np.ndarray, orders: list[int]) -> dict[int, float]:
    """Residual-to-signal energy ratio of the best causal linear predictor
    of x_t from its previous `order` vectors, fit by least squares over the
    whole packet, for each order in `orders`.

    order=0 is the "no temporal structure" baseline (predict the running
    mean); ratio should be ~1 there. A ratio that drops sharply at small
    order means the stream is smoothly evolving -- interpolatable from a
    sparse temporal sampling. A ratio that stays near 1 at every order
    means each step is close to fresh information: the same failure mode
    as i.i.d. token embeddings, just along the other axis.
    """
    x = np.asarray(x, dtype=np.float64)
    t, dim = x.shape
    out: dict[int, float] = {}
    total_energy = np.sum(x**2)
    for order in orders:
        if order == 0:
            pred = np.broadcast_to(x.mean(axis=0), x.shape)
            residual_energy = np.sum((x - pred) ** 2)
            out[order] = float(residual_energy / total_energy) if total_energy > 0 else 0.0
            continue
        if order >= t:
            out[order] = float("nan")
            continue
        # Predict x_t from [x_{t-1}, ..., x_{t-order}] via a single shared
        # per-lag scalar weight per channel, fit by least squares: stack
        # lagged copies as regressors, target is x_t, one fit per channel.
        n_rows = t - order
        regressors = np.stack(
            [x[order - lag : t - lag] for lag in range(1, order + 1)], axis=-1
        )  # (n_rows, dim, order)
        target = x[order:]  # (n_rows, dim)
        residual_energy = 0.0
        for d in range(dim):
            a = regressors[:, d, :]  # (n_rows, order)
            b = target[:, d]  # (n_rows,)
            coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
            residual_energy += float(np.sum((b - a @ coeffs) ** 2))
        out[order] = residual_energy / total_energy if total_energy > 0 else 0.0
        _ = n_rows
    return out


def _levina_bickel_mle_dimension(points: np.ndarray, k: int) -> float:
    """Levina-Bickel (2004) maximum-likelihood intrinsic dimension
    estimator, averaged over all points at neighborhood size k.

    For each point, uses the k nearest neighbors' distances r_1<=...<=r_k
    to estimate a local dimension m_k(x) = [1/(k-1) * sum_j log(r_k/r_j)]^-1,
    then averages m_k over all points (the standard "average of inverses"
    variant). Ambient dimension is treated as an upper bound only; this
    estimates the dimension of the manifold/set the points concentrate on,
    which is the fractal/Renyi-type quantity Wu's thesis ties to achievable
    linear compression rate.
    """
    n = points.shape[0]
    if n <= k:
        raise ValueError(f"need more than k={k} points, got {n}")
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    dist, _ = nbrs.kneighbors(points)  # (n, k+1), column 0 is self (dist 0)
    dist = dist[:, 1:]  # drop self, (n, k)
    dist = np.clip(dist, 1e-12, None)
    log_ratios = np.log(dist[:, -1:] / dist)  # log(r_k / r_j), (n, k)
    # j runs 1..k-1 in the original formula (excludes the k-th term itself)
    inv_local_dim = np.mean(log_ratios[:, :-1], axis=1)
    inv_local_dim = np.clip(inv_local_dim, 1e-12, None)
    local_dim = 1.0 / inv_local_dim
    return float(np.mean(local_dim))


def windowed_intrinsic_dimension(
    windows: np.ndarray, ks: list[int]
) -> dict[int, float]:
    """Levina-Bickel intrinsic dimension of the point cloud formed by many
    length-T windows, each flattened to one point in R^(T*dim).

    `windows`: (n_windows, T, dim), independently-drawn stream segments
    (e.g. strided samples from many long streams, or one window per item
    in a corpus). Reports d_hat(k) for each neighborhood size k in `ks` --
    Levina-Bickel is somewhat k-sensitive, so report a small sweep rather
    than trusting one value.

    Compare d_hat / T against dim: if d_hat/T << dim, each additional
    timestep of the window carries much less genuine new information than
    a full free vector's worth, which is exactly the condition under which
    temporal CS (few measurements across T, shared temporal dictionary)
    should beat sending every vector.
    """
    n_windows, t, dim = windows.shape
    flat = windows.reshape(n_windows, t * dim).astype(np.float64)
    out: dict[int, float] = {}
    for k in ks:
        out[k] = _levina_bickel_mle_dimension(flat, k=k)
    return out


@dataclass
class StreamDiagnosis:
    t: int
    dim: int
    effective_rank_95: int
    effective_rank_99: int
    causal_residual_ratio: dict[int, float]
    intrinsic_dim_per_window: float | None
    intrinsic_dim_per_step: float | None
    n_windows: int | None = None

    @property
    def trivial_rank_ceiling(self) -> float:
        """min(T, dim) / T -- the rank test can never report a rate above
        this regardless of redundancy, since a (T, dim) matrix has rank
        <= min(T, dim) trivially. When dim < T this ceiling can be well
        below 1.0 even for genuinely unstructured data, so predicted_rate
        _linear should be read relative to this ceiling, not to 1.0."""
        return min(self.t, self.dim) / self.t

    @property
    def predicted_rate_linear(self) -> float:
        """Predicted k/T from the linear (SVD) oracle bound at 95% energy."""
        return self.effective_rank_95 / self.t

    @property
    def predicted_rate_intrinsic(self) -> float | None:
        """Predicted k/T from the nonlinear (Levina-Bickel) oracle bound,
        i.e. intrinsic dimension per step, normalized by the per-vector
        ambient width dim -- this is the temporal analogue of Wu's
        information-dimension-as-achievable-rate result."""
        if self.intrinsic_dim_per_step is None:
            return None
        return self.intrinsic_dim_per_step / self.dim

    @property
    def intrinsic_estimate_reliable(self) -> bool | None:
        """kNN-based dimension estimators (Levina-Bickel included) are
        reliable at confirming LOW dimension but are systematically biased
        DOWNWARD -- not merely noisy -- when the true dimension is large
        relative to the sample count: nearest-neighbor distances stop being
        informative and the estimate saturates well below the truth instead
        of growing with it. Empirically (see stream_dimension tests) this
        plateau shows up even as n_windows is increased 8x, so it is not a
        simple "add more samples" fix within a reasonable budget. Flags
        unreliable whenever the raw estimate isn't comfortably smaller than
        the sample count -- a necessary, not sufficient, sanity check.
        Returns None if no windows were supplied.
        """
        if self.intrinsic_dim_per_window is None or self.n_windows is None:
            return None
        return self.intrinsic_dim_per_window < self.n_windows / 10

    def verdict(self, threshold: float = 0.5) -> str:
        """Go/no-go: BOTH the linear and (when available and reliable) the
        intrinsic-dimension rate must clear the bar. This is deliberately
        conservative rather than optimistic (max of the two rates, not
        min): the intrinsic-dimension estimator's failure mode is a
        downward-biased false positive (see intrinsic_estimate_reliable),
        so letting it override a pessimistic linear result would let a
        systematic underestimation artifact repeat the false "compressible"
        conclusion that sank the per-vector CS pivot on token data."""
        rates = [self.predicted_rate_linear]
        note = ""
        if self.predicted_rate_intrinsic is not None:
            if self.intrinsic_estimate_reliable:
                rates.append(self.predicted_rate_intrinsic)
            else:
                note = (" (intrinsic-dimension estimate discarded: unreliable at this "
                        "sample count, see intrinsic_estimate_reliable)")
        worst_rate = max(rates)
        if worst_rate < threshold:
            return f"GO: predicted rate {worst_rate:.3f} < {threshold} -- structure exists, worth prototyping{note}"
        return f"NO-GO: predicted rate {worst_rate:.3f} >= {threshold} -- stream looks near-full-rate along time{note}"


def diagnose_stream(
    packet: np.ndarray,
    windows: np.ndarray | None = None,
    energy_thresholds: tuple[float, float] = (0.95, 0.99),
    causal_orders: list[int] | None = None,
    intrinsic_ks: list[int] | None = None,
) -> StreamDiagnosis:
    """Run the full diagnostic battery on one representative packet
    (T, dim), optionally with a larger `windows` array (n_windows, T, dim)
    for the intrinsic-dimension check, which needs many samples."""
    t, dim = packet.shape
    causal_orders = causal_orders if causal_orders is not None else [0, 1, 2, 4, 8]
    intrinsic_ks = intrinsic_ks if intrinsic_ks is not None else [10, 20]

    rank_95 = effective_temporal_rank(packet, energy_thresholds[0])
    rank_99 = effective_temporal_rank(packet, energy_thresholds[1])
    causal = causal_predictability_curve(packet, causal_orders)

    intrinsic_per_window = None
    intrinsic_per_step = None
    n_windows = None
    if windows is not None:
        n_windows = windows.shape[0]
        d_by_k = windowed_intrinsic_dimension(windows, intrinsic_ks)
        intrinsic_per_window = float(np.median(list(d_by_k.values())))
        intrinsic_per_step = intrinsic_per_window / t

    return StreamDiagnosis(
        t=t,
        dim=dim,
        effective_rank_95=rank_95,
        effective_rank_99=rank_99,
        causal_residual_ratio=causal,
        intrinsic_dim_per_window=intrinsic_per_window,
        intrinsic_dim_per_step=intrinsic_per_step,
        n_windows=n_windows,
    )


def synthetic_stream(
    t: int,
    dim: int,
    latent_rank: int,
    noise_std: float = 0.05,
    smoothness: float = 0.9,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic (T, dim) stream with KNOWN temporal redundancy, for
    validating the estimators above against ground truth before trusting
    them on real data.

    Construction: a `latent_rank`-dimensional latent trajectory z_t is
    generated as a smoothed random walk (AR(1) with coefficient
    `smoothness`, so it evolves gradually rather than jumping every step),
    then linearly mapped into R^dim and perturbed with i.i.d. Gaussian
    noise. Ground truth: temporal rank/intrinsic dimension should recover
    ~latent_rank as noise_std -> 0, and should saturate near dim (or near
    min(t, dim)) as smoothness -> 0 (independent steps, nothing to
    exploit) -- both directions are checked in tests.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros((t, latent_rank))
    innovation_scale = np.sqrt(1 - smoothness**2) if smoothness < 1 else 1.0
    for i in range(1, t):
        z[i] = smoothness * z[i - 1] + innovation_scale * rng.standard_normal(latent_rank)
    a = rng.standard_normal((latent_rank, dim))
    x = z @ a
    x += noise_std * rng.standard_normal((t, dim))
    return x


def synthetic_stream_batch(
    n_windows: int,
    t: int,
    dim: int,
    latent_rank: int,
    noise_std: float = 0.05,
    smoothness: float = 0.9,
    seed: int = 0,
) -> np.ndarray:
    """Many independent draws of `synthetic_stream`, stacked to
    (n_windows, t, dim) -- the input shape `windowed_intrinsic_dimension`
    and `diagnose_stream`'s `windows` argument expect."""
    return np.stack(
        [
            synthetic_stream(t, dim, latent_rank, noise_std, smoothness, seed=seed * 10_000 + i)
            for i in range(n_windows)
        ]
    )
