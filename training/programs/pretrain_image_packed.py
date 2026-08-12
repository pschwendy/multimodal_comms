#!/usr/bin/env python3
"""Train the image-side bottleneck so images ride the same packets as text.

Mirrors `training.programs.pretrain_packed` exactly in protocol -- matryoshka width
ladder, unit-RMS codes, crosstalk noise calibrated to the packet load -- so
the resulting codes are interchangeable with text codes inside one packet
(the packers never inspect modality).

The frozen SD-VAE is applied ONCE up front and its latents cached: the VAE
is not being trained, so keeping it in the training loop would just re-run
the same encoder over the same images every epoch. Training is then a small
conv autoencoder over (4,32,32) latents, which fits comfortably on one GPU.

  python training/programs/pretrain_image_packed.py --image-dir data/images/val2017 \
      --out data/packed_image/bottleneck.pt --device cuda:8
"""

import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

from multimodal_comms.methods.packing.learned import DEFAULT_LADDER  # noqa: E402
from multimodal_comms.methods.multimodal.latent_image import (  # noqa: E402
    VAE_SCALE,
    ImageBottleneck,
)


def load_images(paths, size=256):
    from PIL import Image

    out = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((size, size), Image.BICUBIC)
        out.append(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2)


def cache_latents(image_dir, cache_path, device, size=256, batch=16, limit=0):
    """Run the frozen VAE once over the corpus and store the latents."""
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    from diffusers import AutoencoderKL

    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                   glob.glob(os.path.join(image_dir, "*.png")))
    if limit:
        paths = paths[:limit]
    print(f"encoding {len(paths)} images through the VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    lats = []
    for i in range(0, len(paths), batch):
        imgs = load_images(paths[i:i + batch], size).to(device)
        with torch.no_grad():
            lats.append((vae.encode(imgs).latent_dist.mean * VAE_SCALE).cpu())
        if (i // batch) % 25 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    lat = torch.cat(lats)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(lat, cache_path)
    print(f"cached {tuple(lat.shape)} -> {cache_path}", flush=True)
    return lat


def sample_width(ladder, step, total):
    frac = step / max(total, 1)
    avail = (ladder[:2] if frac < 0.10 else ladder[:4] if frac < 0.25
             else ladder[:6] if frac < 0.45 else ladder)
    return random.choice(avail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="data/images/val2017")
    ap.add_argument("--latent-cache", default="data/packed_image/latents.pt")
    ap.add_argument("--out", default="data/packed_image/bottleneck.pt")
    ap.add_argument("--ladder", type=int, nargs="+", default=list(DEFAULT_LADDER))
    ap.add_argument("--packet-dim", type=int, default=10240)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-rho", type=float, default=1.5)
    ap.add_argument("--noise-prob", type=float, default=0.5,
                    help="Fraction of steps carrying FramePacker crosstalk. "
                         "The rotor path is exactly zero-crosstalk, so a high "
                         "value trades clean-channel quality for overload "
                         "robustness that rotor never needs.")
    ap.add_argument("--n-dev", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:8")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    lat = cache_latents(args.image_dir, args.latent_cache, args.device,
                        limit=args.limit)
    dev_lat = lat[:args.n_dev].to(args.device)
    train_lat = lat[args.n_dev:].to(args.device)
    ladder = sorted([d for d in args.ladder if d <= args.packet_dim], reverse=True)
    print(f"{len(train_lat)} train / {len(dev_lat)} dev latents {tuple(lat.shape[1:])}")
    print("ladder (code width -> messages/packet): " +
          ", ".join(f"{d}->{args.packet_dim // d}" for d in ladder))

    bn = ImageBottleneck(code_dim=max(ladder), ch=args.ch).to(args.device)
    print(f"bottleneck {sum(p.numel() for p in bn.parameters())/1e6:.1f}M params")
    opt = torch.optim.AdamW(bn.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    t0, run = time.time(), {}
    for step in range(args.steps):
        d = sample_width(ladder, step, args.steps)
        idx = torch.randint(0, len(train_lat), (args.batch_size,),
                            device=args.device)
        x = train_lat[idx]
        if random.random() < 0.5:                      # horizontal-flip aug
            x = torch.flip(x, dims=[-1])
        code = bn.encode(x, d)
        rho = (0.0 if random.random() >= args.noise_prob
               else random.uniform(0, args.max_rho))
        if rho > 0:
            noise = torch.zeros_like(code)
            noise[:, :d] = torch.randn(code.shape[0], d, device=code.device) * rho ** 0.5
            code = code + noise
        loss = torch.nn.functional.mse_loss(bn.decode(code), x)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bn.parameters(), 1.0)
        opt.step()
        sched.step()
        run.setdefault(d, []).append(loss.item())

        if (step + 1) % args.log_every == 0:
            msg = "  ".join(f"d{k}:{sum(v)/len(v):.4f}"
                            for k, v in sorted(run.items(), reverse=True))
            print(f"step {step+1:6d}/{args.steps} [{msg}] "
                  f"({(time.time()-t0)/args.log_every:.3f}s/step)", flush=True)
            run, t0 = {}, time.time()

    # Dev latent-space quality per width (image-space PSNR needs the VAE and
    # is measured in experiments/packing/programs/eval_multimodal.py).
    bn.eval()
    report = {}
    with torch.no_grad():
        var = dev_lat.var().item()
        for d in ladder:
            rec = bn.decode(bn.encode(dev_lat, d))
            m = torch.nn.functional.mse_loss(rec, dev_lat).item()
            report[d] = {"latent_mse": m, "latent_psnr_db": 10 * np.log10(var / m),
                         "messages": args.packet_dim // d}
            print(f"  d={d:5d} (M={args.packet_dim//d:4d}): latent_mse={m:.4f} "
                  f"snr={10*np.log10(var/m):.1f} dB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": bn.state_dict(), "config": bn.config(),
                "ladder": ladder, "dev": report}, args.out)
    json.dump(report, open(args.out.replace(".pt", "_dev.json"), "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
