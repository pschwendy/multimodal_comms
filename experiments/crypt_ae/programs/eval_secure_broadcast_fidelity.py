#!/usr/bin/env python3
"""Real-text, free-running fidelity of the PRIVATE (per-receiver-secret)
broadcast pipeline -- the number that actually matters for "does this
reconstruct well when entangling several messages", as opposed to the
synthetic cos-sim security sweep (eval_crypto_security.py) or
teacher-forced token accuracy (eval_feistel_sweep.py). Uses
SecureBroadcastCodec/SecureReceiverCodec end to end: real dev text ->
private-keyed packet -> free-running generation -> difflib ratio to truth.

Example:
  python experiments/crypt_ae/programs/eval_secure_broadcast_fidelity.py \
      --model-path data/superpose_pretrain_s2/final \
      --dev-data data/fineweb_ae_large/dev.jsonl \
      --loads 1 2 4 8 --n-groups 12 --device cuda:0
"""

import argparse
import difflib
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

from multimodal_comms.methods.superposition.latent import (  # noqa: E402
    LatentCodec,
    SecureBroadcastCodec,
    SecureReceiverCodec,
    mint_receiver_secrets,
)


def load_jsonl(path):
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else obj)
    return texts


def fidelity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="data/superpose_pretrain_s2/final")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--loads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-groups", type=int, default=12)
    ap.add_argument("--key-mode", type=str, default="qr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/secure_broadcast_fidelity.json")
    args = ap.parse_args()

    codec = LatentCodec(model_path=args.model_path, device=args.device)
    dev_texts = load_jsonl(args.dev_data)
    random.Random(args.seed).shuffle(dev_texts)
    print(f"{len(dev_texts)} dev texts, model={args.model_path}, key_mode={args.key_mode}")

    results = {}
    idx = 0
    for n in args.loads:
        fids = []
        for g in range(args.n_groups):
            if idx + n > len(dev_texts):
                idx = 0
            batch = dev_texts[idx: idx + n]
            idx += n
            secrets = mint_receiver_secrets(n)
            texts_by_slot = {j: batch[j] for j in range(n)}
            sender = SecureBroadcastCodec(codec, secrets_by_slot=secrets, key_mode=args.key_mode)
            packet = sender.encode_packet(texts_by_slot)
            for j in range(n):
                receiver = SecureReceiverCodec(codec, my_slot=j, my_secret=secrets[j],
                                                key_mode=args.key_mode)
                decoded = receiver.decode(packet)
                fids.append(fidelity(decoded or "", texts_by_slot[j]))
        mean_fid = sum(fids) / len(fids)
        results[n] = {"mean_fidelity": mean_fid, "n_eval": len(fids)}
        print(f"  load={n:<3d} mean_fidelity={mean_fid:.4f}  (n={len(fids)})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "key_mode": args.key_mode,
                   "loads": args.loads, "n_groups": args.n_groups,
                   "results": results}, f, indent=2)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
