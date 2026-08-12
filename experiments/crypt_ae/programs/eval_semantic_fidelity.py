#!/usr/bin/env python3
"""Latent-space and semantic-embedding fidelity of the private broadcast
pipeline -- the metrics that actually answer "is content preserved",
unlike full-string difflib (harsh on paraphrase/length, see
reports/crypto_autoencoder_security_20260721.md follow-up discussion) or
synthetic-Gaussian cos-sim (eval_crypto_security.py; doesn't use real
encoder-latent statistics).

For each load N, using REAL dev text through the real encoder:
  latent_mse / latent_cos  - unbind(packet, my_key) vs my own true latent
                              z (channel-level: how much does binding N
                              messages together and unbinding distort the
                              latent itself, before the decoder even runs).
  semantic_cos / semantic_mse - re-encode the DECODED text and compare
                              that embedding to the true latent (uses the
                              autoencoder's OWN encoder as the semantic
                              embedding function -- no sentence_transformers
                              dependency needed; also the metric is native
                              to the same space the channel operates in).
  difflib                  - kept as a secondary/legacy cross-reference.

Also reports an "identity" row per checkpoint: encode -> decode with NO
packet math at all (no bind/sum/unbind) -- isolates how lossy the raw
K-latent bottleneck is BY ITSELF, before any superposition/crypto is
involved, since that's a separate question from "does entangling multiple
messages hurt."

Example:
  python experiments/crypt_ae/programs/eval_semantic_fidelity.py \
      --model-path data/superpose_pretrain_s2/final \
      --dev-data data/fineweb_ae_large/dev.jsonl \
      --loads 1 2 4 8 --n-groups 8 --device cuda:0
"""

import argparse
import difflib
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

import torch  # noqa: E402

from multimodal_comms.methods.superposition.latent import (  # noqa: E402
    LatentCodec,
    SecureBroadcastCodec,
    SecureReceiverCodec,
    deserialize_packet,
    mint_receiver_secrets,
    _split_nonce_prefix,
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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.mse_loss(a, b).item()


def fidelity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="data/superpose_pretrain_s2/final")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--loads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-groups", type=int, default=8)
    ap.add_argument("--key-mode", type=str, default="qr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/semantic_fidelity.json")
    args = ap.parse_args()

    codec = LatentCodec(model_path=args.model_path, device=args.device)
    dev_texts = load_jsonl(args.dev_data)
    random.Random(args.seed).shuffle(dev_texts)
    print(f"{len(dev_texts)} dev texts, model={args.model_path}, key_mode={args.key_mode}",
          flush=True)

    results = {}
    idx = 0

    # Identity row: no packet math at all.
    id_lat_mse, id_lat_cos, id_sem_cos, id_sem_mse, id_fid = [], [], [], [], []
    for g in range(args.n_groups):
        t = dev_texts[idx % len(dev_texts)]
        idx += 1
        z = codec.encode(t)
        dec = codec.decode(z)
        id_lat_mse.append(0.0)
        id_lat_cos.append(1.0)
        if dec:
            z2 = codec.encode(dec)
            id_sem_cos.append(cos(z2, z))
            id_sem_mse.append(mse(z2, z))
        id_fid.append(fidelity(dec or "", t))
    results["identity"] = {
        "latent_mse": 0.0, "latent_cos": 1.0,
        "semantic_cos": sum(id_sem_cos) / len(id_sem_cos) if id_sem_cos else float("nan"),
        "semantic_mse": sum(id_sem_mse) / len(id_sem_mse) if id_sem_mse else float("nan"),
        "difflib": sum(id_fid) / len(id_fid), "n_eval": len(id_fid),
    }
    r = results["identity"]
    print(f"  identity        latent_cos=1.000 semantic_cos={r['semantic_cos']:.3f} "
          f"semantic_mse={r['semantic_mse']:.4f} difflib={r['difflib']:.3f} (n={r['n_eval']})",
          flush=True)

    for n in args.loads:
        lat_mse, lat_cos, sem_cos, sem_mse, fids = [], [], [], [], []
        for g in range(args.n_groups):
            if idx + n > len(dev_texts) * 2:
                idx = 0
            batch = [dev_texts[(idx + j) % len(dev_texts)] for j in range(n)]
            idx += n
            secrets = mint_receiver_secrets(n)
            texts_by_slot = {j: batch[j] for j in range(n)}
            true_z = {j: codec.encode(texts_by_slot[j]) for j in range(n)}

            # row_keys=False: this measures a checkpoint TRAINED under a
            # single shared per-slot key, so per-row keys would score it
            # off-distribution (and cost K QRs per slot per packet). The
            # row-key fix is a confidentiality property, not a fidelity one.
            sender = SecureBroadcastCodec(codec, secrets_by_slot=secrets,
                                          key_mode=args.key_mode, row_keys=False)
            packet = sender.encode_packet(texts_by_slot)

            for j in range(n):
                receiver = SecureReceiverCodec(codec, my_slot=j, my_secret=secrets[j],
                                                key_mode=args.key_mode, row_keys=False)
                # The packet carries its per-packet nonce as a public prefix
                # (SecureBroadcastCodec mints a fresh one per encode), so it
                # must be split off before deserialising and fed to the
                # keyring that re-derives that packet's single-use key.
                nonce, inner = _split_nonce_prefix(packet)
                pt, n_slots = deserialize_packet(inner)
                recovered = receiver._keyring_for(nonce).unbind(pt, j, n_slots)
                lat_mse.append(mse(recovered, true_z[j]))
                lat_cos.append(cos(recovered, true_z[j]))
                decoded = codec.decode(recovered)
                fids.append(fidelity(decoded or "", texts_by_slot[j]))
                if decoded:
                    z2 = codec.encode(decoded)
                    sem_cos.append(cos(z2, true_z[j]))
                    sem_mse.append(mse(z2, true_z[j]))
        results[n] = {
            "latent_mse": sum(lat_mse) / len(lat_mse),
            "latent_cos": sum(lat_cos) / len(lat_cos),
            "semantic_cos": sum(sem_cos) / len(sem_cos) if sem_cos else float("nan"),
            "semantic_mse": sum(sem_mse) / len(sem_mse) if sem_mse else float("nan"),
            "difflib": sum(fids) / len(fids),
            "n_eval": len(fids),
        }
        r = results[n]
        print(f"  load={n:<3d} latent_mse={r['latent_mse']:.4f} latent_cos={r['latent_cos']:.3f} "
              f"semantic_cos={r['semantic_cos']:.3f} semantic_mse={r['semantic_mse']:.4f} "
              f"difflib={r['difflib']:.3f} (n={r['n_eval']})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "key_mode": args.key_mode,
                   "loads": args.loads, "n_groups": args.n_groups,
                   "results": results}, f, indent=2)
    print(f"Written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
