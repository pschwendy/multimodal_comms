#!/usr/bin/env python3
"""Train FeistelKeyring's round-function weights against an explicit
key-decorrelation objective.

The fixed-random-init version (superpose.py's FeistelKeyring) still leaked
badly after three architecture fixes: wrong-key unbind stayed ~0.29
cosine-similar to the true latents at 16 rounds, far above the ~0.01 chance
floor for D=2560. Rather than keep hand-tuning architecture, TRAIN the
round-function weights directly against the property we actually want:

  minimize  E_{z, key0 != key1} [ cos_sim(unbind(bind(z, key0), key1), z)^2 ]

Nothing else needs a loss term -- unbind(bind(z, key), key) with the SAME
key is exact by construction (additive coupling is invertible for any
weights), so there's no reconstruction term to balance against; the
optimizer is free to spend its entire budget on decorrelating the wrong-key
path. Only the round MLPs (w1, w2, g_w, b_w) are trained; the per-round
orthogonal mixing matrices are left as fixed random QR draws (they're a
public diffusion aid, not where the key-sensitivity has to live).

These weights are PUBLIC (like a published cipher's round function) --
training them is legitimate cryptographic design (real S-boxes are
deliberately designed for exactly this confusion property, not drawn from
a hat), not a secret leak: only the per-slot key vector stays private.

Example:
  python training/programs/train_feistel_keyring.py --dim 2560 --n-rounds 8 \
      --steps 3000 --out data/feistel_keyring_trained_r8.pt
"""

import argparse
import sys
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

from multimodal_comms.methods.superposition.latent import FeistelKeyring  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--key-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--stability-weight", type=float, default=1.0,
                     help="Weight on the right-key roundtrip stability term "
                          "(see comment at the loss -- prevents the "
                          "decorrelation objective from being solved via a "
                          "numerically chaotic map).")
    ap.add_argument("--slot-pool", type=int, default=256,
                     help="Random key0/key1 slots drawn from 0..slot-pool-1.")
    ap.add_argument("--key-seed", type=int, default=1234)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", type=str, default="data/feistel_keyring_trained.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    kr = FeistelKeyring(args.dim, seed=args.key_seed, n_rounds=args.n_rounds,
                         key_dim=args.key_dim, hidden_dim=args.hidden_dim)
    kr._build_public_arch()
    params = []
    for fn in kr._round_fns:
        for t in fn:
            t.requires_grad_(True)
            params.append(t)
    print(f"Training {sum(p.numel() for p in params)} round-function "
          f"parameters over {args.n_rounds} rounds, dim={args.dim}.")

    opt = torch.optim.Adam(params, lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed + 1)

    for step in range(1, args.steps + 1):
        z = torch.randn(args.batch_size, args.dim, generator=rng)
        key0 = int(torch.randint(0, args.slot_pool, (1,), generator=rng))
        offset = int(torch.randint(1, args.slot_pool, (1,), generator=rng))
        key1 = (key0 + offset) % args.slot_pool  # guaranteed != key0

        bound = kr.bind(z, key0)
        wrong_recovered = kr.unbind(bound, key1)
        cos = torch.nn.functional.cosine_similarity(wrong_recovered, z, dim=-1)
        decorrelation_loss = (cos ** 2).mean()

        # Right-key roundtrip is invertible BY CONSTRUCTION in exact math,
        # but a first training pass (no stability term) showed that's not
        # enough: pushing the round function to decorrelate different
        # keys' outputs also made it numerically CHAOTIC -- tiny (even
        # fp64-scale) mismatches between the forward and reverse
        # computation got amplified across 8 rounds into a ~0.2 absolute
        # error for the CORRECT key too (verified: same magnitude in fp32
        # and fp64, so not ordinary rounding -- an exponential-sensitivity
        # issue in the trained map itself). Penalizing right-key roundtrip
        # error directly gives the optimizer a reason not to solve
        # decorrelation by way of a chaotic, numerically fragile map.
        right_recovered = kr.unbind(bound, key0)
        stability_loss = ((right_recovered - z) ** 2).mean()

        loss = decorrelation_loss + args.stability_weight * stability_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d}  wrong-key mean|cos|={cos.abs().mean().item():.4f}"
                  f"  decorr_loss={decorrelation_loss.item():.5f}"
                  f"  stability_loss={stability_loss.item():.6f}")

    # Correctness check over MANY trials -- a single small batch missed the
    # chaotic-map failure mode in the first (no-stability-term) run, since
    # it only shows up for some (z, key) draws. Report the worst case, not
    # just one sample.
    with torch.no_grad():
        worst = 0.0
        for _ in range(50):
            z = torch.randn(8, args.dim, generator=rng)
            rec = kr.unbind(kr.bind(z, 0), 0)
            worst = max(worst, (rec - z).abs().max().item())
    print(f"\nPost-training right-key roundtrip WORST max abs err over 50 trials: "
          f"{worst:.2e} (should be ~1e-5 or smaller)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    kr.save_weights(args.out)
    print(f"Saved trained round-function weights to {args.out}")


if __name__ == "__main__":
    main()
