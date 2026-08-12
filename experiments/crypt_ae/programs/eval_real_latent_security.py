#!/usr/bin/env python3
"""Does private-mode leakage on REAL encoder latents actually sit near the
synthetic chance floor measured in eval_crypto_security.py (torch.randn),
or does real latent structure (e.g. the ~0.28 unrelated-pair baseline
found empirically, vs ~0 for isotropic Gaussian) push it higher?

For each load N, using real dev text: bind N messages with independent
private secrets, sum into a packet. For receiver j's OWN (legitimate,
correct) unbind:
  own_cos    - cos-sim to receiver j's true latent (reconstruction).
  attack_cos - mean |cos-sim| to every OTHER slot's true latent (what this
               legitimate operation leaks about messages that aren't
               receiver j's -- the real security question).
  baseline_cos - mean |cos-sim| between INDEPENDENT real-text pairs that
               never went through the channel at all (the generic-
               language-structure floor this embedding space has, per the
               calibration in this session). attack_cos should not exceed
               this by much if the keying itself isn't leaking anything
               beyond what any two unrelated texts already share.

Example:
  python experiments/crypt_ae/programs/eval_real_latent_security.py \
      --model-path data/superpose_pretrain_s2/final --device cuda:0
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
    build_keyring,
    deserialize_packet,
    mint_receiver_secrets,
    superpose,
    serialize_packet,
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
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).abs().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="data/superpose_pretrain_s2/final")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--loads", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--n-groups", type=int, default=15)
    ap.add_argument("--key-mode", type=str, default="qr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/real_latent_security.json")
    args = ap.parse_args()

    codec = LatentCodec(model_path=args.model_path, device=args.device)
    dev_texts = load_jsonl(args.dev_data)
    random.Random(args.seed).shuffle(dev_texts)
    print(f"{len(dev_texts)} dev texts, model={args.model_path}, key_mode={args.key_mode}",
          flush=True)

    # Baseline: cos-sim between independent real-text latents, no channel at all.
    base_pool = [codec.encode(t) for t in dev_texts[:16]]
    base_cos = []
    for i in range(8):
        for j in range(8, 16):
            base_cos.append(cos(base_pool[i], base_pool[j]))
    baseline = sum(base_cos) / len(base_cos)
    print(f"  baseline (unrelated real-text pairs, no channel) = {baseline:.4f}", flush=True)

    idx = 16
    results = {"baseline_cos": baseline}
    for n in args.loads:
        own_list, attack_list = [], []
        for g in range(args.n_groups):
            if idx + n > len(dev_texts):
                idx = 16
            batch = dev_texts[idx: idx + n]
            idx += n
            secrets = mint_receiver_secrets(n)
            true_z = {j: codec.encode(batch[j]) for j in range(n)}
            keyring = build_keyring(codec.latent_dim, seed=secrets, mode=args.key_mode)
            packet = superpose(keyring, true_z)
            for j in range(n):
                recovered = keyring.unbind(packet, j, n)
                own_list.append(cos(recovered, true_z[j]))
                for i in range(n):
                    if i != j:
                        attack_list.append(cos(recovered, true_z[i]))
        own_cos = sum(own_list) / len(own_list)
        attack_cos = sum(attack_list) / len(attack_list)
        results[n] = {"own_cos": own_cos, "attack_cos": attack_cos,
                       "baseline_cos": baseline, "n_eval": len(own_list)}
        print(f"  load={n:<3d} own_cos={own_cos:.4f} attack_cos={attack_cos:.4f} "
              f"baseline_cos={baseline:.4f} "
              f"(attack {'ABOVE' if attack_cos > baseline * 1.2 else 'at/below'} baseline)",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "key_mode": args.key_mode, **results}, f, indent=2)
    print(f"Written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
