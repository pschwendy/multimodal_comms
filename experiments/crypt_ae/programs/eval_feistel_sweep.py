#!/usr/bin/env python3
"""Crosstalk/reconstruction-scaling sweep for FeistelKeyring (keyed nonlinear
coupling bind/unbind), alongside the existing linear schemes for reference.

Same harness as eval_crosstalk_sweep.py (teacher-forced token accuracy as N
messages are bound with a per-slot key and summed into one packet, then each
slot is unbound with ITS OWN key and decoded). Extended to also run against
an already superpose-fine-tuned "robust" checkpoint (e.g.
data/superpose_subspace_16lat/final), not just a plain base checkpoint --
that checkpoint's decoder was fine-tuned to tolerate LINEAR (subspace)
crosstalk, so this is a zero-shot generalization test to a crosstalk
statistic it never saw in training, not an in-distribution eval.

Example:
  python experiments/crypt_ae/programs/eval_feistel_sweep.py \
      --model-path data/autoencoder_pretrain_large/final \
      --dev-data data/fineweb_ae_large/dev.jsonl \
      --loads 1 2 4 8 16 --rounds 8 --device cuda:0
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
    FeistelKeyring,
)


@torch.no_grad()
def eval_config(model, tokenizer, dev_texts, num_latents, prompt_ids, li_positions,
                 max_len, device, last_layer, embed_layer, keyring, n,
                 n_groups=16, max_batch_texts=64, max_decode_batch=8):
    """Same protocol as eval_crosstalk_sweep.py: bind+sum n messages per
    group, unbind each slot with its own key, decode all n teacher-forced,
    average token accuracy over n_groups independent groups."""
    accs, losses, weights = [], [], []
    idx = 0
    groups_done = 0
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
            for j in range(n):
                noisy_list.append(keyring.unbind(packet, j, n))
        noisy = torch.stack(noisy_list)  # (B, K, H), CPU

        for c in range(0, len(batch), max_decode_batch):
            sub_batch = batch[c:c + max_decode_batch]
            sub_noisy = noisy[c:c + max_decode_batch].to(device)
            loss, acc = decode_batch_loss(model, tokenizer, sub_batch, sub_noisy,
                                           prompt_ids, li_positions, max_len,
                                           device, embed_layer)
            accs.append(acc)
            losses.append(loss.item())
            weights.append(len(sub_batch))
    n_eval = groups_done * n
    if not weights:
        return float("nan"), float("nan"), 0
    tot = sum(weights)
    return (sum(l * w for l, w in zip(losses, weights)) / tot,
            sum(a * w for a, w in zip(accs, weights)) / tot, n_eval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="data/autoencoder_pretrain_large/final")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--loads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--rounds", type=int, nargs="+", default=[8],
                     help="FeistelKeyring round counts to sweep.")
    ap.add_argument("--feistel-weights", type=str, default=None,
                     help="Path to training.programs.train_feistel_keyring output "
                          "(key-decorrelation-trained round fns). Applied to "
                          "every --rounds value that matches its n_rounds; "
                          "others fall back to fixed-random-init.")
    ap.add_argument("--key-seed", type=int, default=1234)
    ap.add_argument("--n-groups", type=int, default=16)
    ap.add_argument("--max-batch-texts", type=int, default=64)
    ap.add_argument("--max-decode-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/feistel_sweep.json")
    ap.add_argument("--include-linear-refs", action="store_true", default=True)
    ap.add_argument("--no-linear-refs", dest="include_linear_refs", action="store_false")
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

    configs = []
    if args.include_linear_refs:
        configs.append(("rotation_qr", OrthogonalKeyring(D, seed=args.key_seed, mode="qr")))
        configs.append(("exact_subspace", SubspaceKeyring(D, seed=args.key_seed)))
    for r in args.rounds:
        kr = FeistelKeyring(D, seed=args.key_seed, n_rounds=r)
        name = f"feistel_r{r}"
        if args.feistel_weights:
            try:
                kr.load_weights(args.feistel_weights)
                name += "_trained"
            except ValueError as e:
                print(f"  (skipping trained weights for r={r}: {e})")
        configs.append((name, kr))

    results = {}
    for name, keyring in configs:
        results[name] = {}
        for n in args.loads:
            loss, acc, n_eval = eval_config(
                model, tok, dev_texts, num_latents, prompt_ids, li_positions,
                args.max_len, args.device, last_layer, embed_layer, keyring, n,
                n_groups=args.n_groups, max_batch_texts=args.max_batch_texts,
                max_decode_batch=args.max_decode_batch,
            )
            results[name][n] = {"loss": loss, "tok_acc": acc, "n_eval": n_eval}
            print(f"  {name:16s} N={n:<3d} loss={loss:.4f} tok_acc={acc:.4f} "
                  f"(n={n_eval})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "hidden_size": D,
                   "num_latents": num_latents, "key_seed": args.key_seed,
                   "results": results}, f, indent=2)
    print(f"\nWritten to {args.out}")

    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(f"# Feistel (keyed nonlinear) crosstalk sweep on `{args.model_path}`\n\n")
        f.write(f"hidden_size={D}, num_latents={num_latents}, "
                f"n_groups={args.n_groups}\n\n")
        f.write("| Scheme | " + " | ".join(f"N={n}" for n in args.loads) + " |\n")
        f.write("|---|" + "---|" * len(args.loads) + "\n")
        for name, _ in configs:
            row = [f"{results[name][n]['tok_acc']:.3f}" for n in args.loads]
            f.write(f"| {name} | " + " | ".join(row) + " |\n")
    print(f"Summary table written to {md_path}")


if __name__ == "__main__":
    main()
