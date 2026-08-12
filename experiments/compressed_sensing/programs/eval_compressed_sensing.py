#!/usr/bin/env python3
"""Tier-1 (signal-level) evaluation of CS on real token-embedding packets.

Packets are raw token embeddings, so the "shared dictionary" is exactly the
model's embedding table -- already known to both endpoints, nothing to fit
or transmit. Every packet row is exactly 1-sparse in that table (it IS one
specific atom), which reframes tier-1 evaluation as compressive token
identification (see the reusable CodebookCS implementation):
project the codebook once with a shared random sensing matrix Phi, then
recover each compressed token by nearest-neighbor match.

Sweeps the number of measurements M and reports, for each M:
  - top-1 token recovery accuracy against the FULL vocabulary codebook
  - top-1 accuracy against a codebook restricted to just the tokens that
    actually occur in the eval corpus (the realistic "domain-restricted
    dictionary" case)
  - embedding-level fidelity (relative L2 error, cosine similarity) of the
    recovered vector vs. the true one

No LLM calls, no task accuracy -- this is the cheap proxy tier for picking
CS hyperparameters (M, in particular) before anything touches the
multi-agent benchmark harness for tier-2 (functional) evaluation.

Output: prints a rate-identification table and writes full results to
data/cs_packets/eval_results.json.
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from multimodal_comms.benchmarks.hiddenbench.runtime.compressed_sensing import (
    CodebookCS,
    cosine_similarity_rows,
    make_sensing_matrix,
    relative_l2_error,
    token_recovery_accuracy,
)


def summarize(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cs_packets")
    ap.add_argument("--packet-size", type=int, default=100)
    ap.add_argument("--n-eval-tokens", type=int, default=2000,
                     help="subsample this many held-out tokens for the sweep (decode cost scales with this x vocab)")
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--measurement-bits", type=int, default=16,
                     help="bits per CS measurement, for the direct-index-transmission comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    embedding_table = torch.load(os.path.join(args.data_dir, "embedding_table.pt")).numpy()
    token_ids = torch.load(os.path.join(args.data_dir, "eval_token_ids.pt")).numpy()
    vocab_size, dim = embedding_table.shape
    n_packets = len(token_ids) // args.packet_size
    distinct = np.unique(token_ids)
    print(f"embedding_table: {embedding_table.shape}, eval tokens: {len(token_ids)} "
          f"({n_packets} packets of {args.packet_size}), {len(distinct)} distinct tokens in corpus")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(token_ids), size=min(args.n_eval_tokens, len(token_ids)), replace=False)
    true_ids = token_ids[sample_idx]
    x = embedding_table[true_ids]

    restricted_table = embedding_table[distinct]
    restricted_pos = {tok: i for i, tok in enumerate(distinct)}
    true_restricted_idx = np.array([restricted_pos[t] for t in true_ids])

    codebooks = {
        "full_vocab": (embedding_table, true_ids, vocab_size),
        "corpus_restricted": (restricted_table, true_restricted_idx, len(distinct)),
    }

    index_bits = math.ceil(math.log2(vocab_size))
    restricted_index_bits = math.ceil(math.log2(max(len(distinct), 2)))
    results = {
        "vocab_size": int(vocab_size),
        "dim": int(dim),
        "n_eval_tokens": int(len(sample_idx)),
        "distinct_tokens_in_corpus": int(len(distinct)),
        "index_bits_full_vocab": index_bits,
        "index_bits_corpus_restricted": restricted_index_bits,
        "sweep": [],
    }
    print(f"\nDirect index transmission cost: {index_bits} bits/token (full vocab), "
          f"{restricted_index_bits} bits/token (corpus-restricted) -- the baseline any CS "
          f"scheme for single, exact, discretely-known tokens has to beat.")

    header = (f"{'M':>5} {'M/dim':>7} {'codebook':>17} {'n_atoms':>8} {'top1_acc':>9}  "
              f"{'cs_bits':>8} {'vs_index':>9}  {'rel_err(mean/median)':<22} {'cos_sim(mean/median)':<22}")
    print("\n== Rate-identification sweep ==")
    print(header)
    print("-" * len(header))

    for m in args.ms:
        phi = make_sensing_matrix(m, dim, seed=args.seed)
        for name, (table, gt_idx, n_atoms) in codebooks.items():
            codec = CodebookCS(phi=phi, codebook=table)
            y = codec.encode(x)
            recovered_idx = codec.decode_indices(y)
            acc = token_recovery_accuracy(gt_idx, recovered_idx)

            x_hat = table[recovered_idx]
            err = relative_l2_error(x, x_hat)
            sim = cosine_similarity_rows(x, x_hat)
            err_s, sim_s = summarize(err), summarize(sim)

            cs_bits = m * args.measurement_bits
            ref_bits = index_bits if name == "full_vocab" else restricted_index_bits
            vs_index = f"{cs_bits / ref_bits:.2f}x"

            print(f"{m:5d} {m/dim:7.4f} {name:>17} {n_atoms:8d} {acc:9.4f}  "
                  f"{cs_bits:8d} {vs_index:>9}  "
                  f"{err_s['mean']:.4f}/{err_s['median']:.4f}          "
                  f"{sim_s['mean']:.4f}/{sim_s['median']:.4f}")
            results["sweep"].append({
                "m": m, "ratio": m / dim, "codebook": name, "n_atoms": n_atoms,
                "top1_accuracy": acc, "cs_bits": cs_bits, "index_bits": ref_bits,
                "rel_err": err_s, "cos_sim": sim_s,
            })

    out_path = args.out or os.path.join(args.data_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
