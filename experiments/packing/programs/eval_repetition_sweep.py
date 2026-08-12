#!/usr/bin/env python3
"""M-fold diversity combining: does sending M independently-keyed packets of
the SAME N messages, then combining per-slot unbind estimates, reduce
crosstalk (and, for width-restricted schemes, recover truncated capacity)
on the plain non-superpose-finetuned autoencoder?

For each of M packets, a fresh independently-seeded keyring binds the same
N messages. A receiver only ever needs its own M private per-packet keys
(never another slot's), so this preserves the same privacy property as the
single-packet schemes -- it trades bandwidth (O(M), independent of N) for
crosstalk/width recovery.

Combining rule:
  rotation_qr:     z_hat = mean_m( unbind_m(P_m, j) )
                   -- z_j is recovered exactly every repetition (Q_j Q_j^T=I),
                      crosstalk is i.i.d. zero-mean across m, so averaging
                      shrinks its std by 1/sqrt(M).
  subspace/random: z_hat = mean_m( (D / width_m) * unbind_m(P_m, j) )
                   -- E[(D/w) B B^T] = I for a Haar-random w-dim basis, so
                      this is an unbiased ensemble estimator of z_j itself
                      (not just of a cleaner noisy version of it): with
                      independent bases per repetition, growing M recovers
                      truncated dimensions in addition to averaging out any
                      residual crosstalk.

Example:
  python experiments/packing/programs/eval_repetition_sweep.py \
      --model-path data/autoencoder_pretrain_large/final \
      --loads 4 16 --reps 1 2 4 8 16 \
      --schemes rotation_qr exact_subspace random_subspace_640
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
from multimodal_comms.methods.superposition.latent import (  # noqa: E402
    OrthogonalKeyring,
    SubspaceKeyring,
    RandomSubspaceKeyring,
)


def make_keyring(scheme: str, D: int, width: int | None, seed: int):
    if scheme == "rotation_qr":
        return OrthogonalKeyring(D, seed=seed, mode="qr")
    if scheme == "exact_subspace":
        return SubspaceKeyring(D, seed=seed)
    if scheme == "random_subspace":
        return RandomSubspaceKeyring(D, width=width, seed=seed)
    raise ValueError(f"Unknown scheme {scheme!r}")


@torch.no_grad()
def eval_repetition_config(model, tokenizer, dev_texts, num_latents, prompt_ids,
                            li_positions, max_len, device, last_layer, embed_layer,
                            D, scheme, width, n, M, base_seed,
                            n_groups=12, max_batch_texts=64, max_decode_batch=16):
    keyrings = [make_keyring(scheme, D, width, base_seed + 97 * m) for m in range(M)]
    trace_scale = None
    if scheme == "exact_subspace":
        trace_scale = D / (D // n)
    elif scheme == "random_subspace":
        trace_scale = D / width

    accs, losses, weights = [], [], []
    idx, groups_done = 0, 0
    while groups_done < n_groups and idx + n <= len(dev_texts):
        groups_in_call = max(1, min(max(1, max_batch_texts // n), n_groups - groups_done))
        batch = dev_texts[idx: idx + groups_in_call * n]
        idx += len(batch)
        groups_done += groups_in_call

        latents = encode_batch(model, tokenizer, batch, num_latents, max_len,
                                device, last_layer).cpu()  # (B, K, H)
        combined_list = []
        for g in range(groups_in_call):
            grp = latents[g * n:(g + 1) * n]  # (n, K, H)
            combined = torch.zeros_like(grp)
            for kr in keyrings:
                packet = None
                for j in range(n):
                    bound = kr.bind(grp[j], j, n)
                    packet = bound if packet is None else packet + bound
                for j in range(n):
                    est = kr.unbind(packet, j, n)
                    if trace_scale is not None:
                        est = est * trace_scale
                    combined[j] += est
            combined /= M
            combined_list.append(combined)
        noisy = torch.cat(combined_list, dim=0)  # (B, K, H), CPU

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
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--schemes", type=str, nargs="+",
                    default=["rotation_qr", "exact_subspace", "random_subspace_640"],
                    help="'random_subspace_W' encodes width W directly.")
    ap.add_argument("--key-seed", type=int, default=1234)
    ap.add_argument("--n-groups", type=int, default=12)
    ap.add_argument("--max-batch-texts", type=int, default=64)
    ap.add_argument("--max-decode-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/repetition_sweep.json")
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

    parsed_schemes = []
    for s in args.schemes:
        if s.startswith("random_subspace_"):
            w = int(s.rsplit("_", 1)[1])
            parsed_schemes.append(("random_subspace", w, s))
        else:
            parsed_schemes.append((s, None, s))

    results = {}
    for scheme, width, label in parsed_schemes:
        results[label] = {}
        for n in args.loads:
            results[label][n] = {}
            for M in args.reps:
                loss, acc, n_eval = eval_repetition_config(
                    model, tok, dev_texts, num_latents, prompt_ids, li_positions,
                    args.max_len, args.device, last_layer, embed_layer, D,
                    scheme, width, n, M, args.key_seed,
                    n_groups=args.n_groups, max_batch_texts=args.max_batch_texts,
                    max_decode_batch=args.max_decode_batch,
                )
                results[label][n][M] = {"loss": loss, "tok_acc": acc, "n_eval": n_eval}
                print(f"  {label:24s} N={n:<3d} M={M:<3d} loss={loss:.4f} "
                      f"tok_acc={acc:.4f} (n={n_eval})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "hidden_size": D,
                   "num_latents": num_latents, "results": results}, f, indent=2)
    print(f"\nWritten to {args.out}")

    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(f"# M-fold repetition combining sweep on `{args.model_path}`\n\n")
        for label, _, _ in [(l, w, l) for _, w, l in parsed_schemes]:
            pass
        for scheme, width, label in parsed_schemes:
            f.write(f"\n## {label}\n\n")
            f.write("| N \\ M | " + " | ".join(f"M={m}" for m in args.reps) + " |\n")
            f.write("|---|" + "---|" * len(args.reps) + "\n")
            for n in args.loads:
                row = [f"{results[label][n][m]['tok_acc']:.3f}" for m in args.reps]
                f.write(f"| N={n} | " + " | ".join(row) + " |\n")
    print(f"Summary table written to {md_path}")


if __name__ == "__main__":
    main()
