#!/usr/bin/env python3
"""Quantify the confidentiality property of superposed latent packets:
"receiver j can reconstruct its own message but cannot decode a different
receiver's message."

Pure tensor math, synthetic random latents, no GPU/model load needed (the
crosstalk/reconstruction math is identical for real encoder latents -- see
reports/multiplex_load_curve_20260719.md and reports/feistel_sweep_*.md for
the same schemes' behavior on real text through the trained autoencoder).

Compares two keying modes (see hiddenbench/superpose.py's module
docstring) across loads N=2,4,8,16:

  PUBLIC  (seed: int, e.g. the SuperposeCompressor default): key(slot) =
          f(shared_seed, PUBLIC slot index). ANY party who knows the shared
          seed (the same constant for the whole channel, e.g. 1234) can
          compute EVERY slot's key with zero additional search -- this
          mode is a multiplexing scheme, not a cipher.

  PRIVATE (seed: dict via mint_receiver_secrets): key(slot) = an
          independently-minted 63-bit secret, given ONLY to that receiver
          out of band. A party without receiver i's secret cannot compute
          key_i by any means faster than brute-force over 2**63 values.

For each scheme x mode x load, reports:
  own_cos    - cos-sim(unbind(packet, my_key), my own true latent).
               Proxy for reconstruction quality (higher is better).
  attack_cos - PUBLIC: cos-sim recovered using a DIFFERENT slot's true key
               (mean |cos-sim| to that slot's message) -- exact key, zero
               guessing cost, this is what "everyone can decode everyone"
               looks like.
               PRIVATE: mean |cos-sim| between receiver j's OWN (legitimate)
               unbind and every OTHER slot's true message -- what a
               receiver's normal, authorized unbind operation leaks about
               messages that are not theirs, with no brute-forcing at all.
  chance_floor - E[|cos-sim| between independent random D-dim vectors]
               ~ 1/sqrt(D), the reference "leaks nothing" floor.

Example:
  python experiments/crypt_ae/programs/eval_crypto_security.py --dim 2560 --loads 2 4 8 16 \
      --feistel-weights data/feistel_keyring_trained_r8.pt \
      --out reports/crypto_security_sweep.json
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

from multimodal_comms.methods.superposition.latent import (  # noqa: E402
    FeistelKeyring,
    OrthogonalKeyring,
    RandomSubspaceKeyring,
    mint_receiver_secrets,
    superpose,
)


def _latents(dim: int, k: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(k, dim, generator=gen)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).abs().item()


def eval_public_mode(keyring_cls, kwargs, dim, k, n, n_trials, base_seed):
    """Shared int seed: attacker uses slot i's TRUE key (zero-cost lookup,
    the same formula anyone can run) to unbind slot i's message out of the
    packet built by slots 0..n-1."""
    own_list, attack_list = [], []
    for t in range(n_trials):
        kr = keyring_cls(dim, seed=1234, **kwargs)
        zs = {j: _latents(dim, k, base_seed + 1000 * t + j) for j in range(n)}
        packet = superpose(kr, zs)
        for j in range(n):
            recovered = kr.unbind(packet, j, n)
            own_list.append(_cos(recovered, zs[j]))
            attack_list.append(_cos(recovered, zs[j]))  # exact key: same op
    return sum(own_list) / len(own_list), sum(attack_list) / len(attack_list)


def eval_private_mode(keyring_cls, kwargs, dim, k, n, n_trials, base_seed):
    """Independent per-slot secrets: each receiver unbinds with its OWN
    (legitimate, correct) key. own_cos is that receiver's reconstruction
    quality; attack_cos is what that SAME legitimate unbind leaks about
    every OTHER slot's true message -- no brute-forcing, this is the
    honest-but-curious-insider bound, already the worst case since nobody
    without a secret can even run this operation for a slot they don't
    own."""
    own_list, attack_list = [], []
    for t in range(n_trials):
        secrets = mint_receiver_secrets(n)
        kr = keyring_cls(dim, seed=secrets, **kwargs)
        zs = {j: _latents(dim, k, base_seed + 1000 * t + j) for j in range(n)}
        packet = superpose(kr, zs)
        for j in range(n):
            recovered = kr.unbind(packet, j, n)
            own_list.append(_cos(recovered, zs[j]))
            for i in range(n):
                if i != j:
                    attack_list.append(_cos(recovered, zs[i]))
    return sum(own_list) / len(own_list), sum(attack_list) / len(attack_list)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--num-latents", type=int, default=4)
    ap.add_argument("--loads", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--feistel-rounds", type=int, default=8)
    ap.add_argument("--feistel-weights", type=str, default=None,
                     help="Path to training.programs.train_feistel_keyring output; "
                          "if omitted, uses the fixed-random-init round fns.")
    ap.add_argument("--subspace-width", type=int, default=None,
                     help="RandomSubspaceKeyring width; default dim // 4.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/crypto_security_sweep.json")
    args = ap.parse_args()

    dim, k = args.dim, args.num_latents
    width = args.subspace_width or dim // 4
    chance_floor = 1.0 / dim ** 0.5

    def make_feistel():
        kr = FeistelKeyring(dim, seed=1234, n_rounds=args.feistel_rounds)
        if args.feistel_weights:
            kr.load_weights(args.feistel_weights)
        else:
            kr._build_public_arch()
        return kr

    schemes = [
        ("rotation_qr", OrthogonalKeyring, {}),
        ("random_subspace", RandomSubspaceKeyring, {"width": width}),
        ("feistel" + ("_trained" if args.feistel_weights else "_fixedrandom"),
         None, {}),  # constructed specially below to reuse loaded weights
    ]

    results = {}
    for name, cls, kwargs in schemes:
        results[name] = {"public": {}, "private": {}}
        for n in args.loads:
            if name.startswith("feistel"):
                # Build once, reuse across trials/modes via a fresh keyring
                # object per trial inside eval_* would reload weights each
                # time; instead monkey-patch a factory that shares weights.
                proto = make_feistel()

                def _cls(dim_, seed, n_rounds=args.feistel_rounds, **_kw):
                    kr = FeistelKeyring(dim_, seed=seed, n_rounds=n_rounds)
                    kr._round_fns = proto._round_fns
                    kr._mixes = proto._mixes
                    return kr

                own_pub, atk_pub = eval_public_mode(_cls, {}, dim, k, n, args.n_trials, args.seed)
                own_priv, atk_priv = eval_private_mode(_cls, {}, dim, k, n, args.n_trials, args.seed + 1)
            else:
                own_pub, atk_pub = eval_public_mode(cls, kwargs, dim, k, n, args.n_trials, args.seed)
                own_priv, atk_priv = eval_private_mode(cls, kwargs, dim, k, n, args.n_trials, args.seed + 1)
            results[name]["public"][n] = {"own_cos": own_pub, "attack_cos": atk_pub,
                                           "attack_keyspace": n}
            results[name]["private"][n] = {"own_cos": own_priv, "attack_cos": atk_priv,
                                            "attack_keyspace": 2 ** 63}
            print(f"{name:24s} N={n:<3d} "
                  f"PUBLIC  own={own_pub:.3f} attack={atk_pub:.3f} (exact key, {n} slots enumerable) | "
                  f"PRIVATE own={own_priv:.3f} attack={atk_priv:.3f} (chance floor {chance_floor:.3f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dim": dim, "num_latents": k, "chance_floor": chance_floor,
                   "n_trials": args.n_trials, "loads": args.loads,
                   "feistel_weights": args.feistel_weights, "results": results},
                  f, indent=2)
    print(f"\nWritten to {args.out}")

    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# Cryptographic-autoencoder security sweep\n\n")
        f.write(f"dim={dim}, num_latents={k}, chance floor (unrelated D-dim "
                f"vectors) = 1/sqrt(dim) = {chance_floor:.4f}, "
                f"n_trials={args.n_trials}\n\n")
        f.write("PUBLIC = shared seed, attacker uses the target slot's exact "
                "key (0-cost, same formula anyone can run). "
                "PRIVATE = independent per-receiver secret, `attack_cos` is "
                "what a receiver's own LEGITIMATE unbind leaks about every "
                "other slot -- no brute-forcing (brute force over the real "
                "2**63 secret space is intractable to run in this sweep, "
                "and unnecessary: this reports the worst case with zero "
                "guessing cost already).\n\n")
        f.write("| Scheme | Load | own_cos (PUBLIC) | attack_cos (PUBLIC) | "
                "own_cos (PRIVATE) | attack_cos (PRIVATE) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, _, _ in schemes:
            for n in args.loads:
                pub, priv = results[name]["public"][n], results[name]["private"][n]
                f.write(f"| {name} | {n} | {pub['own_cos']:.3f} | "
                        f"{pub['attack_cos']:.3f} | {priv['own_cos']:.3f} | "
                        f"{priv['attack_cos']:.3f} |\n")
    print(f"Summary table written to {md_path}")


if __name__ == "__main__":
    main()
