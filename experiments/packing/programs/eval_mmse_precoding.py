#!/usr/bin/env python3
"""Sender-side joint MMSE precoding vs. naive superposition, at FIXED
bandwidth (one D-dim packet, no repetition).

Naive superposition (everything tested so far) builds the packet as
  m = sum_i bind(x_i, i) = sum_i B_i B_i^T x_i
with no attempt to account for how the N private bases B_i (each D x w,
orthonormal columns, drawn from a private per-receiver seed) interact.
Joint MMSE precoding uses the fact that the packet CONSTRUCTOR sees every
x_i and every B_i simultaneously, and asks: what's the best *linear*
correction to that packet, in closed form?

  C = (sum_i B_i B_i^T + lambda I_D)^{-1}          (D x D, computed ONCE
                                                     per active key set)
  m_mmse = m_naive @ C                              (same bandwidth as m_naive)

Receivers decode exactly as before (unbind = project onto their own
private B_i), just applied to m_mmse instead of m_naive. This is only
non-trivial when keys are "thin" (w < D): for full-rank orthogonal keys
(rotation_qr / exact_subspace, our no-bottleneck checkpoint's w=D case),
sum_i B_i B_i^T = N * I exactly, so the correction is a scalar and buys
nothing -- there's no crosstalk structure to exploit without a real
dimension bottleneck. This script therefore only sweeps random_subspace
(private, possibly-overlapping thin keys), where sum_i B_i B_i^T has real
off-diagonal structure.

Example:
  python experiments/packing/programs/eval_mmse_precoding.py \
      --loads 4 16 --widths 640 1280 --lambdas 0.001 0.01 0.1 1 10 1000
"""

import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))

from training.programs.pretrain_autoencoder import (  # noqa: E402
    load_jsonl,
    encode_batch,
    decode_prompt_ids,
    latent_token_positions,
    decode_batch_loss,
)
from multimodal_comms.methods.superposition.latent import RandomSubspaceKeyring  # noqa: E402


def mmse_correction(keyring: RandomSubspaceKeyring, n: int, D: int, lam: float):
    """C = (sum_i B_i B_i^T + lambda I_D)^{-1}, computed once per (keyring, n)."""
    KKt = torch.zeros(D, D)
    for j in range(n):
        b = keyring.basis(j, n)  # (D, w)
        KKt += b @ b.T
    return torch.linalg.inv(KKt + lam * torch.eye(D))


@torch.no_grad()
def eval_precoding_config(model, tokenizer, dev_texts, num_latents, prompt_ids,
                           li_positions, max_len, device, last_layer, embed_layer,
                           keyring, n, lam, D,
                           n_groups=12, max_batch_texts=64, max_decode_batch=16):
    C = mmse_correction(keyring, n, D, lam) if lam is not None else None
    accs, losses, weights = [], [], []
    idx, groups_done = 0, 0
    while groups_done < n_groups and idx + n <= len(dev_texts):
        groups_in_call = max(1, min(max(1, max_batch_texts // n), n_groups - groups_done))
        batch = dev_texts[idx: idx + groups_in_call * n]
        idx += len(batch)
        groups_done += groups_in_call

        latents = encode_batch(model, tokenizer, batch, num_latents, max_len,
                                device, last_layer).cpu()  # (B, K, H)
        noisy_list = []
        for g in range(groups_in_call):
            grp = latents[g * n:(g + 1) * n]  # (n, K, H)
            packet = None
            for j in range(n):
                bound = keyring.bind(grp[j], j, n)
                packet = bound if packet is None else packet + bound
            if C is not None:
                packet = packet @ C  # MMSE correction, same bandwidth
            for j in range(n):
                noisy_list.append(keyring.unbind(packet, j, n))
        noisy = torch.stack(noisy_list).to(device)  # (B, K, H)

        for c in range(0, len(batch), max_decode_batch):
            sub_batch = batch[c:c + max_decode_batch]
            sub_noisy = noisy[c:c + max_decode_batch].to(device)
            loss, acc = decode_batch_loss(model, tokenizer, sub_batch, sub_noisy,
                                           prompt_ids, li_positions, max_len,
                                           device, embed_layer)
            accs.append(acc)
            losses.append(loss.item())
            weights.append(len(sub_batch))
    if not weights:
        return float("nan"), float("nan"), 0
    tot = sum(weights)
    return (sum(l * w for l, w in zip(losses, weights)) / tot,
            sum(a * w for a, w in zip(accs, weights)) / tot, groups_done * n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="data/autoencoder_pretrain_large/final")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--loads", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--widths", type=int, nargs="+", default=[640, 1280])
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                    help="lambda->0 approaches zero-forcing/least-squares "
                         "inversion of sum_i B_i B_i^T; lambda->inf approaches "
                         "transmitting nothing (C->0), NOT the naive baseline "
                         "-- naive (C=I, no correction) is reported separately.")
    ap.add_argument("--key-seed", type=int, default=1234)
    ap.add_argument("--n-groups", type=int, default=12)
    ap.add_argument("--max-batch-texts", type=int, default=64)
    ap.add_argument("--max-decode-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/mmse_precoding_sweep.json")
    args = ap.parse_args()

    print(f"Loading {args.model_path} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(args.device).eval()

    ae_cfg = json.load(open(os.path.join(args.model_path, "ae_config.json")))
    num_latents = ae_cfg["num_latents"]
    D = model.config.hidden_size
    last_layer = model.config.num_hidden_layers - 1
    embed_layer = model.get_input_embeddings()
    prompt_ids = decode_prompt_ids(tok, num_latents)
    li_positions = latent_token_positions(tok, prompt_ids, num_latents)

    dev_texts = load_jsonl(args.dev_data)
    random.Random(args.seed).shuffle(dev_texts)
    print(f"{len(dev_texts)} dev texts, num_latents={num_latents}, hidden_size={D}")

    results = {}
    for w in args.widths:
        label = f"random_subspace_{w}"
        results[label] = {}
        keyring = RandomSubspaceKeyring(D, width=w, seed=args.key_seed)
        for n in args.loads:
            results[label][n] = {}
            # naive baseline: no correction (matches earlier crosstalk sweep)
            loss, acc, n_eval = eval_precoding_config(
                model, tok, dev_texts, num_latents, prompt_ids, li_positions,
                args.max_len, args.device, last_layer, embed_layer, keyring, n,
                None, D, n_groups=args.n_groups, max_batch_texts=args.max_batch_texts,
                max_decode_batch=args.max_decode_batch,
            )
            results[label][n]["naive"] = {"loss": loss, "tok_acc": acc, "n_eval": n_eval}
            print(f"  {label:22s} N={n:<3d} naive        loss={loss:.4f} "
                  f"tok_acc={acc:.4f} (n={n_eval})")
            for lam in args.lambdas:
                loss, acc, n_eval = eval_precoding_config(
                    model, tok, dev_texts, num_latents, prompt_ids, li_positions,
                    args.max_len, args.device, last_layer, embed_layer, keyring, n,
                    lam, D, n_groups=args.n_groups, max_batch_texts=args.max_batch_texts,
                    max_decode_batch=args.max_decode_batch,
                )
                results[label][n][f"lam={lam}"] = {"loss": loss, "tok_acc": acc, "n_eval": n_eval}
                print(f"  {label:22s} N={n:<3d} lam={lam:<8g} loss={loss:.4f} "
                      f"tok_acc={acc:.4f} (n={n_eval})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "hidden_size": D,
                   "num_latents": num_latents, "results": results}, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
