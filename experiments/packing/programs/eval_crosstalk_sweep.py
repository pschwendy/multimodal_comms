#!/usr/bin/env python3
"""No-training crosstalk sweep on a plain (non-superpose-finetuned) autoencoder.

Takes a single-message AE checkpoint as-is and asks: if we naively bind N
messages with various keying schemes and sum them into one packet, how does
teacher-forced reconstruction degrade as N grows? This characterizes the raw
crosstalk sensitivity of the codec before spending any compute fine-tuning
it for superposition.

Schemes compared at each load N:
  rotation_qr     -- OrthogonalKeyring, full-rank random rotation per slot
  exact_subspace  -- SubspaceKeyring, one shared master basis sliced into
                     disjoint D/N-dim blocks (zero crosstalk, hard capacity)
  random_subspace_w{W} -- RandomSubspaceKeyring, independent private W-dim
                     subspace per slot (approximate crosstalk that shrinks
                     with W/D, no coordination or knowledge of N required)

Example:
  python experiments/packing/programs/eval_crosstalk_sweep.py \
      --model-path data/autoencoder_pretrain_large/final \
      --dev-data data/fineweb_ae_large/dev.jsonl \
      --loads 1 2 4 8 16 --device cuda:0
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


@torch.no_grad()
def eval_config(model, tokenizer, dev_texts, num_latents, prompt_ids, li_positions,
                 max_len, device, last_layer, embed_layer, keyring, n,
                 n_groups=16, max_batch_texts=64, max_decode_batch=8):
    """Bind+sum n messages per group under `keyring`, unbind each slot with
    its own key, decode all n teacher-forced. Averages token accuracy over
    n_groups independent groups (n_groups * n dev texts total).

    Encoding/binding happens in full n-sized groups (binding needs all n
    slots together), but decoding is per-example and is sub-chunked to
    max_decode_batch regardless of n -- large-vocab cross-entropy on a
    B x seq x vocab logits tensor is the actual memory bottleneck, not the
    packet math, so a high superposition load must not force a huge decode
    batch."""
    accs, losses, weights = [], [], []
    idx = 0
    groups_done = 0
    while groups_done < n_groups and idx + n <= len(dev_texts):
        groups_in_call = max(1, min(max(1, max_batch_texts // n), n_groups - groups_done))
        batch = dev_texts[idx: idx + groups_in_call * n]
        idx += len(batch)
        groups_done += groups_in_call

        latents = encode_batch(model, tokenizer, batch, num_latents, max_len,
                                device, last_layer).cpu()  # (B, K, H); keyring math is CPU/fp32
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
    ap.add_argument("--widths", type=float, nargs="+",
                    default=[1.0, 0.5, 0.25, 0.125, 0.0625],
                    help="random_subspace widths, as fractions of hidden_size.")
    ap.add_argument("--key-seed", type=int, default=1234)
    ap.add_argument("--n-groups", type=int, default=16,
                    help="Independent groups (of `load` messages each) per config.")
    ap.add_argument("--max-batch-texts", type=int, default=64)
    ap.add_argument("--max-decode-batch", type=int, default=8,
                    help="Cap on examples per cross-entropy forward call, "
                         "independent of superposition load N.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/crosstalk_sweep.json")
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

    configs = [
        ("rotation_qr", OrthogonalKeyring(D, seed=args.key_seed, mode="qr")),
        ("exact_subspace", SubspaceKeyring(D, seed=args.key_seed)),
    ]
    for frac in args.widths:
        w = max(1, int(round(D * frac)))
        configs.append((f"random_subspace_w{w}",
                        RandomSubspaceKeyring(D, width=w, seed=args.key_seed)))

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
            print(f"  {name:24s} N={n:<3d} loss={loss:.4f} tok_acc={acc:.4f} "
                  f"(n={n_eval})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "hidden_size": D,
                   "num_latents": num_latents, "key_seed": args.key_seed,
                   "results": results}, f, indent=2)
    print(f"\nWritten to {args.out}")

    # Markdown summary table.
    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(f"# Crosstalk sweep on base checkpoint `{args.model_path}`\n\n")
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
