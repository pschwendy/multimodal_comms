#!/usr/bin/env python3
"""Proper CS test of an SVD-derived dictionary: random measurement + OMP
sparse recovery, the same pipeline validated on images (eval_image_cs.py),
now applied per-token instead of a dense/direct PCA projection (which is
what the earlier `global_pca` mechanism in eval_matrix_cs.py actually did
-- no random Phi, no sparse coding, just truncated projection on the
uncompressed row).

Dictionary: top singular directions of real token-embedding rows from a
held-out fit corpus (mean-centered), used as an orthonormal basis. Unlike
CodebookCS (dictionary = exact embedding table, each row is exactly
1-sparse by construction), this tests whether real embeddings are
approximately sparse in their OWN principal-component basis -- a genuinely
different, weaker structural assumption, and the fair apples-to-apples
counterpart to the DCT/learned dictionaries used for images.
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from multimodal_comms.benchmarks.hiddenbench.runtime.compressed_sensing import (
    CodebookCS,
    CSCodec,
    cosine_similarity_rows,
    make_sensing_matrix,
    relative_l2_error,
)


def summarize(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cs_packets")
    ap.add_argument("--packet-size", type=int, default=100)
    ap.add_argument("--n-fit-packets", type=int, default=20)
    ap.add_argument("--n-eval-rows", type=int, default=1500)
    ap.add_argument("--ms", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    embedding_table = torch.load(os.path.join(args.data_dir, "embedding_table.pt"), weights_only=True).numpy()
    token_ids = torch.load(os.path.join(args.data_dir, "eval_token_ids.pt"), weights_only=True).numpy()
    dim = embedding_table.shape[1]
    n_packets = len(token_ids) // args.packet_size
    packet_ids = token_ids[:n_packets * args.packet_size].reshape(n_packets, args.packet_size)
    fit_ids, eval_ids = packet_ids[:args.n_fit_packets], packet_ids[args.n_fit_packets:].reshape(-1)
    fit_rows = embedding_table[fit_ids.reshape(-1)]

    mean = fit_rows.mean(axis=0)
    _, s, vt = np.linalg.svd(fit_rows - mean, full_matrices=False)
    n_atoms = vt.shape[0]
    dictionary = vt.T  # (dim, n_atoms), orthonormal columns
    energy = np.cumsum(s**2) / np.sum(s**2)
    print(f"SVD dictionary: {n_atoms} atoms from {len(fit_rows)} fit rows "
          f"(rank needed for 95% energy: {int(np.searchsorted(energy, 0.95) + 1)}/{n_atoms})")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(eval_ids), size=min(args.n_eval_rows, len(eval_ids)), replace=False)
    true_ids = eval_ids[sample_idx]
    x = embedding_table[true_ids] - mean

    print("Building full-space nearest-neighbor index (once) ...")
    nn_index = CodebookCS(phi=np.eye(dim, dtype=np.float32), codebook=embedding_table)

    index_bits = math.ceil(math.log2(embedding_table.shape[0]))
    results = {"n_atoms": n_atoms, "index_bits": index_bits, "sweep": []}
    header = f"{'M':>5} {'bits':>6} {'vs_index':>9} {'k':>4} {'top1_acc':>9}  {'rel_err(mean/median)':<22} {'cos_sim(mean/median)':<22}"
    print("\n== SVD dictionary + random Phi + OMP sweep (per-token) ==")
    print(header)
    print("-" * len(header))

    for m in args.ms:
        k = max(1, min(m - 1, n_atoms, 30))
        bits = m * 16
        phi = make_sensing_matrix(m, dim, seed=args.seed)
        codec = CSCodec(phi=phi, dictionary=dictionary, method="omp", sparsity=k)

        x_hat = codec.roundtrip(x) + mean
        x_true = x + mean
        recovered_idx = nn_index.decode_indices(x_hat)
        acc = float(np.mean(recovered_idx == true_ids))
        err = relative_l2_error(x_true, x_hat)
        sim = cosine_similarity_rows(x_true, x_hat)
        err_s, sim_s = summarize(err), summarize(sim)

        print(f"{m:5d} {bits:6d} {bits/index_bits:8.2f}x {k:4d} {acc:9.4f}  "
              f"{err_s['mean']:.4f}/{err_s['median']:.4f}          "
              f"{sim_s['mean']:.4f}/{sim_s['median']:.4f}")
        results["sweep"].append({
            "m": m, "bits": bits, "vs_index_bits": bits / index_bits, "k": k,
            "top1_accuracy": acc, "rel_err": err_s, "cos_sim": sim_s,
        })

    out_path = os.path.join(args.data_dir, "eval_svd_dictionary_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
