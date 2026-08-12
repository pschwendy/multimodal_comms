#!/usr/bin/env python3
"""Does the per-row-key IND-CPA fix cost reconstruction quality?

The fix (superpose.py `row_keys=True`, see _derive_row_seed and
reports/crypto_provable_security_20260722.md Sec 3a) binds each of a
message's K latent rows under its OWN independent Haar key instead of
sharing one key across all rows. Security-wise this is mandatory at K > 1.
The question here is whether the deployed checkpoints -- which were TRAINED
with shared-key crosstalk -- still reconstruct under it.

Prior reasoning says they should: per row, the crosstalk term is
sum_i z_ik Q_i^(k) Q_j^(k)T either way, an independently-Haar-rotated sum of
foreign latents, so its MARGINAL distribution per row is identical. Only the
correlation of crosstalk ACROSS rows changes (shared key: correlated;
per-row keys: independent). If the decoder leaned on that cross-row
correlation, fidelity would drop. Measure it rather than assume.

Reports latent_cos (channel-level, pre-decode) and semantic_cos (re-encode
the decoded text through the AE's own encoder -- read against the
unrelated-pair floor of ~0.28-0.44, NOT zero; see
reports/crypto_autoencoder_security_20260721.md) for both settings.

  python experiments/crypt_ae/programs/eval_row_keys_fidelity.py --device cuda:8 --loads 1 2 4
"""

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

import torch  # noqa: E402

from multimodal_comms.methods.superposition.latent import (  # noqa: E402
    LatentCodec,
    OrthogonalKeyring,
    mint_receiver_secrets,
    superpose,
)


def load_jsonl(path, n):
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else obj)
            if len(texts) >= n:
                break
    return texts


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="data/superpose_pretrain_s2/final")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--loads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/row_keys_fidelity.json")
    args = ap.parse_args()

    codec = LatentCodec(model_path=args.model_path, device=args.device)
    texts = load_jsonl(args.dev_data, 400)
    random.Random(args.seed).shuffle(texts)
    print(f"model={args.model_path}  {len(texts)} dev texts", flush=True)

    results = {}
    for row_keys in (False, True):
        label = "row_keys=True (fixed)" if row_keys else "row_keys=False (orig)"
        results[str(row_keys)] = {}
        idx = 0
        for n in args.loads:
            lat, sem = [], []
            for _ in range(args.n_groups):
                batch = [texts[(idx + j) % len(texts)] for j in range(n)]
                idx += n
                secrets = mint_receiver_secrets(n)
                kr = OrthogonalKeyring(codec.latent_dim, seed=secrets,
                                       row_keys=row_keys, nonce=1234 + idx)
                true_z = {j: codec.encode(batch[j]) for j in range(n)}
                packet = superpose(kr, true_z)
                for j in range(n):
                    rec = kr.unbind(packet, j, n)
                    lat.append(cos(rec, true_z[j]))
                    dec = codec.decode(rec)
                    if dec:
                        sem.append(cos(codec.encode(dec), true_z[j]))
            r = {
                "latent_cos": sum(lat) / len(lat),
                "semantic_cos": sum(sem) / len(sem) if sem else float("nan"),
                "n_eval": len(lat),
            }
            results[str(row_keys)][n] = r
            print(f"  {label:24s} load={n:<3d} latent_cos={r['latent_cos']:.3f} "
                  f"semantic_cos={r['semantic_cos']:.3f} (n={r['n_eval']})", flush=True)

    print("\n=== delta (fixed - original) ===")
    for n in args.loads:
        a = results["False"][n]
        b = results["True"][n]
        print(f"  load={n:<3d} latent_cos {b['latent_cos'] - a['latent_cos']:+.3f}   "
              f"semantic_cos {b['semantic_cos'] - a['semantic_cos']:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "results": results}, f, indent=2)
    print(f"\nWritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
