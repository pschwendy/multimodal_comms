#!/usr/bin/env python3
"""Stage-0 gate: is this embedding stream compressible along the STREAM
axis (fewer transmitted vectors, T -> k) rather than the per-vector
feature axis that compressed_sensing.py already tested and lost on?

Modality-agnostic: point --input at any (T, dim) or (n_windows, T, dim)
array of embedding vectors -- text hidden states, audio frame embeddings,
whatever. Nothing here assumes tokens or a vocabulary.

Run with --synthetic for a self-contained demo against data with known,
tunable temporal redundancy (no input file needed) -- useful to see what a
clear GO vs NO-GO case looks like before pointing this at real data.

Usage:
    python experiments/compressed_sensing/programs/diagnose_stream_dimension.py --synthetic --latent-rank 3 --smoothness 0.8
    python experiments/compressed_sensing/programs/diagnose_stream_dimension.py --input path/to/stream.npy --window 64 --stride 16
"""

import argparse
import json
import os
import sys

import numpy as np


from multimodal_comms.benchmarks.hiddenbench.runtime.stream_dimension import (  # noqa: E402
    causal_predictability_curve,
    diagnose_stream,
    synthetic_stream,
    synthetic_stream_batch,
    temporal_energy_captured_curve,
)


def make_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Slide a (T_total, dim) stream into overlapping (window, dim)
    segments -- an approximate i.i.d. sample of "what a length-`window`
    chunk of this stream looks like", for the intrinsic-dimension check."""
    t_total, dim = x.shape
    starts = range(0, t_total - window + 1, stride)
    return np.stack([x[s : s + window] for s in starts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                     help="path to a .npy file: (T, dim) single stream, or (n_windows, T, dim) pre-windowed")
    ap.add_argument("--window", type=int, default=64,
                     help="window length T for the diagnostic when --input is a single long (T_total, dim) stream")
    ap.add_argument("--stride", type=int, default=16,
                     help="stride for slicing windows out of a single long stream")
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data instead of --input")
    ap.add_argument("--synthetic-t", type=int, default=60)
    ap.add_argument("--synthetic-dim", type=int, default=256)
    ap.add_argument("--latent-rank", type=int, default=4)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--smoothness", type=float, default=0.8)
    ap.add_argument("--n-windows", type=int, default=400,
                     help="number of windows to draw for the intrinsic-dimension check")
    ap.add_argument("--intrinsic-ks", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--causal-orders", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--verdict-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.synthetic:
        print(f"Synthetic stream: T={args.synthetic_t} dim={args.synthetic_dim} "
              f"latent_rank={args.latent_rank} noise_std={args.noise_std} smoothness={args.smoothness}")
        packet = synthetic_stream(
            t=args.synthetic_t, dim=args.synthetic_dim, latent_rank=args.latent_rank,
            noise_std=args.noise_std, smoothness=args.smoothness, seed=0,
        )
        windows = synthetic_stream_batch(
            n_windows=args.n_windows, t=args.synthetic_t, dim=args.synthetic_dim,
            latent_rank=args.latent_rank, noise_std=args.noise_std, smoothness=args.smoothness, seed=1,
        )
    else:
        if args.input is None:
            ap.error("must pass --input or --synthetic")
        arr = np.load(args.input)
        if arr.ndim == 2:
            t_total, dim = arr.shape
            print(f"Loaded single stream: T_total={t_total} dim={dim}; "
                  f"slicing into window={args.window} stride={args.stride} for the intrinsic-dim check")
            windows = make_windows(arr, args.window, args.stride)
            if windows.shape[0] < max(args.intrinsic_ks) + 5:
                print(f"WARNING: only {windows.shape[0]} windows from this stream -- "
                      f"intrinsic-dimension estimate will be unreliable. Use a longer stream, "
                      f"a smaller --window, or point --input at a real (n_windows, T, dim) array.")
            packet = arr[: args.window] if t_total >= args.window else arr
        elif arr.ndim == 3:
            print(f"Loaded pre-windowed streams: n_windows={arr.shape[0]} T={arr.shape[1]} dim={arr.shape[2]}")
            windows = arr
            packet = arr[0]
        else:
            raise ValueError(f"expected a 2D (T, dim) or 3D (n_windows, T, dim) array, got shape {arr.shape}")

    t, dim = packet.shape
    if t > dim:
        print(f"NOTE: T={t} > dim={dim} -- the linear (SVD) rank check has a trivial ceiling of "
              f"min(T,dim)/T = {min(t, dim) / t:.3f} regardless of redundancy; read predicted_rate_linear "
              f"relative to that ceiling, not to 1.0.")

    diag = diagnose_stream(
        packet, windows=windows, causal_orders=args.causal_orders, intrinsic_ks=args.intrinsic_ks,
    )

    print("\n== Linear (SVD) oracle bound: energy captured vs. rank r, across time ==")
    curve = temporal_energy_captured_curve(packet, sorted(set([1, 2, 4, 8, 16, 32, 64, t] + [t // 4, t // 2])))
    for r in sorted(curve):
        if r <= t:
            print(f"  r={r:4d}  energy={curve[r]:.4f}")
    print(f"  effective_rank_95 = {diag.effective_rank_95} / T={t}  (trivial ceiling min(T,dim)={min(t, dim)})")
    print(f"  effective_rank_99 = {diag.effective_rank_99} / T={t}")

    print("\n== Causal predictability: residual/signal energy vs. predictor order ==")
    _ = causal_predictability_curve(packet, args.causal_orders)  # already in diag, recomputed here just to log
    for order in args.causal_orders:
        v = diag.causal_residual_ratio.get(order, float("nan"))
        print(f"  order={order:3d}  residual_ratio={v:.4f}")

    if diag.intrinsic_dim_per_window is not None:
        print("\n== Nonlinear (Levina-Bickel) oracle bound ==")
        print(f"  intrinsic_dim_per_window (ambient T*dim={t * dim}) = {diag.intrinsic_dim_per_window:.2f}")
        print(f"  intrinsic_dim_per_step (compare against dim={dim}) = {diag.intrinsic_dim_per_step:.2f}")

    print(f"\nPredicted rate (linear)    k/T ~= {diag.predicted_rate_linear:.3f}  "
          f"(trivial ceiling {diag.trivial_rank_ceiling:.3f})")
    if diag.predicted_rate_intrinsic is not None:
        print(f"Predicted rate (intrinsic) k/T ~= {diag.predicted_rate_intrinsic:.3f}")
    print(f"\n{diag.verdict(args.verdict_threshold)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "t": diag.t,
                    "dim": diag.dim,
                    "effective_rank_95": diag.effective_rank_95,
                    "effective_rank_99": diag.effective_rank_99,
                    "trivial_rank_ceiling": diag.trivial_rank_ceiling,
                    "causal_residual_ratio": diag.causal_residual_ratio,
                    "intrinsic_dim_per_window": diag.intrinsic_dim_per_window,
                    "intrinsic_dim_per_step": diag.intrinsic_dim_per_step,
                    "predicted_rate_linear": diag.predicted_rate_linear,
                    "predicted_rate_intrinsic": diag.predicted_rate_intrinsic,
                    "verdict": diag.verdict(args.verdict_threshold),
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
