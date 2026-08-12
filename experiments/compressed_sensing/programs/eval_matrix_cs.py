#!/usr/bin/env python3
"""Tier-1 evaluation: does importance-weighted matrix CS beat uniform CS?

Earlier tier-1 checks found real 100-token packets have only modest
exploitable structure under naive priors -- neither cross-token low rank
(~63/100 for 95% energy) nor cross-feature low rank (~1576/2048 across the
full vocab) is dramatic, and per-token exact identification already needs
only ~18 bits (the vocab index), which beats any CS scheme that spends more
bits per token. So the open question is whether *importance-weighted*
budget allocation -- spending exact-identification bits only on the tokens
that matter, and getting the rest for free from packet structure -- can
beat the flat 100 x 18-bit baseline. Three mechanisms, all pure linear
algebra (no trained encoder):

  1. Geometric leverage landmarks: pick the r rows least explained by the
     packet's own low-rank structure (leverage_scores), transmit those
     exactly, reconstruct the rest via least-squares against them
     (cur_reconstruct). "Importance" = geometric uniqueness.
  2. Corpus-frequency landmarks: pick the r rarest tokens in the packet (by
     frequency in a held-out fit corpus), same CUR reconstruction.
     "Importance" = informational rarity, a classical NLP notion.
  3. Global PCA dictionary: fit a shared low-rank basis once on a held-out
     fit corpus of real packets (fit_global_basis) -- the "optimization
     phase" -- then reconstruct new packets purely from per-row
     coefficients against that fixed, pre-shared basis (project_reconstruct).
     No per-packet landmarks needed at all.

All three are compared against the flat direct-index baseline (100 x 18
bits, exact) on bits/packet vs. reconstruction fidelity and discrete top-1
accuracy (does nearest-neighbor-snapping a reconstructed row land on the
correct token?).
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
    cur_reconstruct,
    fit_global_basis,
    leverage_scores,
    make_sensing_matrix,
    project_reconstruct,
    relative_l2_error,
    select_landmarks,
    token_recovery_accuracy,
)

POS_BITS = 7  # ceil(log2(100)) -- marking which of 100 positions is a landmark
COEFF_BITS = 8  # bits per quantized reconstruction coefficient


def summarize(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values))}


def build_nearest_neighbor_index(embedding_table: np.ndarray) -> CodebookCS:
    """Build the full-space (unprojected) nearest-neighbor index once, reused
    across every mechanism/r/packet -- rebuilding it per call would redundantly
    redo a large matmul each time."""
    identity_phi = np.eye(embedding_table.shape[1], dtype=np.float32)
    return CodebookCS(phi=identity_phi, codebook=embedding_table)


def nearest_token_accuracy(x_hat: np.ndarray, true_ids: np.ndarray, nn: CodebookCS) -> np.ndarray:
    """Snap each reconstructed row to its nearest vocab embedding, return correctness mask."""
    recovered = nn.decode_indices(x_hat)
    return recovered == true_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cs_packets")
    ap.add_argument("--packet-size", type=int, default=100)
    ap.add_argument("--n-fit-packets", type=int, default=20)
    ap.add_argument("--rs", type=int, nargs="+", default=[5, 10, 20, 30, 50, 70])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    embedding_table = torch.load(os.path.join(args.data_dir, "embedding_table.pt"), weights_only=True).numpy()
    token_ids = torch.load(os.path.join(args.data_dir, "eval_token_ids.pt"), weights_only=True).numpy()
    dim = embedding_table.shape[1]
    n_packets = len(token_ids) // args.packet_size
    packet_ids = token_ids[:n_packets * args.packet_size].reshape(n_packets, args.packet_size)

    fit_ids, eval_ids_mat = packet_ids[:args.n_fit_packets], packet_ids[args.n_fit_packets:]
    fit_rows = embedding_table[fit_ids.reshape(-1)]
    print(f"{len(fit_ids)} fit packets, {len(eval_ids_mat)} eval packets, packet_size={args.packet_size}, dim={dim}")

    freq = np.bincount(fit_ids.reshape(-1), minlength=embedding_table.shape[0])
    print("Building full-space nearest-neighbor index (once) ...")
    nn_index = build_nearest_neighbor_index(embedding_table)

    index_bits = math.ceil(math.log2(embedding_table.shape[0]))
    baseline_bits = args.packet_size * index_bits
    print(f"\nBaseline (direct index, exact, per packet): {baseline_bits} bits "
          f"({args.packet_size} tokens x {index_bits} bits)")

    results = {"index_bits": index_bits, "baseline_bits_per_packet": baseline_bits, "sweep": []}
    header = (f"{'mechanism':>12} {'r':>4} {'bits/pkt':>9} {'vs_baseline':>11}  "
              f"{'non_landmark_top1_acc':>22}  {'rel_err(mean/median)':<22} {'cos_sim(mean/median)':<22}")
    print("\n== Importance-weighted matrix-CS sweep ==")
    print(header)
    print("-" * len(header))

    for r in args.rs:
        basis = fit_global_basis(fit_rows, rank=r)
        mean = fit_rows.mean(axis=0)

        for mech in ["leverage", "frequency", "global_pca"]:
            accs, errs, sims, bits_per_packet = [], [], [], []
            for ids_row in eval_ids_mat:
                x = embedding_table[ids_row]  # (packet_size, dim)

                if mech == "leverage":
                    scores = leverage_scores(x, rank=r)
                    landmarks = select_landmarks(scores, r)
                    x_hat, _ = cur_reconstruct(x, landmarks)
                    non_landmark = np.setdiff1d(np.arange(args.packet_size), landmarks)
                    bits = r * (index_bits + POS_BITS) + (args.packet_size - r) * r * COEFF_BITS
                elif mech == "frequency":
                    rarity = -freq[ids_row]  # lower freq -> higher score
                    landmarks = select_landmarks(rarity, r)
                    x_hat, _ = cur_reconstruct(x, landmarks)
                    non_landmark = np.setdiff1d(np.arange(args.packet_size), landmarks)
                    bits = r * (index_bits + POS_BITS) + (args.packet_size - r) * r * COEFF_BITS
                else:  # global_pca: no landmarks, every row is approximated
                    x_hat, _ = project_reconstruct(x, basis, mean)
                    non_landmark = np.arange(args.packet_size)
                    bits = args.packet_size * r * COEFF_BITS

                correct = nearest_token_accuracy(x_hat[non_landmark], ids_row[non_landmark], nn_index)
                accs.append(np.mean(correct))
                errs.append(relative_l2_error(x[non_landmark], x_hat[non_landmark]))
                sims.append(cosine_similarity_rows(x[non_landmark], x_hat[non_landmark]))
                bits_per_packet.append(bits)

            err_s = summarize(np.concatenate(errs))
            sim_s = summarize(np.concatenate(sims))
            mean_bits = float(np.mean(bits_per_packet))
            mean_acc = float(np.mean(accs))
            print(f"{mech:>12} {r:4d} {mean_bits:9.0f} {mean_bits/baseline_bits:10.2f}x  "
                  f"{mean_acc:22.4f}  {err_s['mean']:.4f}/{err_s['median']:.4f}          "
                  f"{sim_s['mean']:.4f}/{sim_s['median']:.4f}")
            results["sweep"].append({
                "mechanism": mech, "r": r, "bits_per_packet": mean_bits,
                "vs_baseline": mean_bits / baseline_bits, "non_landmark_top1_accuracy": mean_acc,
                "rel_err": err_s, "cos_sim": sim_s,
            })

    out_path = args.out or os.path.join(args.data_dir, "eval_matrix_cs_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
