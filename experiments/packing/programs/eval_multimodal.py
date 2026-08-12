#!/usr/bin/env python3
"""One packet, many messages, MIXED modalities.

Packs text messages and image messages into the SAME packet under the same
keys and measures each receiver's reconstruction. Nothing in the fusion
layer is modality-aware: both bottlenecks emit unit-RMS d-dim codes, the
packer moves d-dim vectors, so a mixed packet is not a special case -- it
is the ordinary case with two code sources.

Image quality is reported BOTH ways, and the distinction matters:
  psnr_vs_vae      against the frozen VAE's own reconstruction. This is the
                   channel's fidelity -- the only part this work controls.
  psnr_vs_original against the input pixels. Includes the VAE's own ~24 dB
                   ceiling, so it can never exceed it however good the
                   bottleneck gets.

  python experiments/packing/programs/eval_multimodal.py --text-model data/packed_matryoshka/final \
      --image-bottleneck data/packed_image/bottleneck.pt --device cuda:8
"""

import argparse
import glob
import json
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from multimodal_comms.benchmarks.hiddenbench.runtime.packing import PackedCodec, build_packer, quantize_packet  # noqa: E402
from multimodal_comms.benchmarks.hiddenbench.runtime.packing_image import ImageCodec  # noqa: E402

from training.programs.pretrain_image_packed import load_images  # noqa: E402


def psnr(a, b, peak=2.0):
    m = float(((a - b) ** 2).mean())
    return 10 * np.log10(peak ** 2 / max(m, 1e-12))


def cos(a, b):
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-model", default="data/packed_matryoshka/final")
    ap.add_argument("--image-bottleneck", default="data/packed_image/bottleneck.pt")
    ap.add_argument("--image-dir", default="data/images/val2017")
    ap.add_argument("--dev-data", default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--fusion", default="rotor")
    ap.add_argument("--widths", type=int, nargs="+", default=[640, 320, 160, 80, 40])
    ap.add_argument("--n-packets", type=int, default=4)
    ap.add_argument("--n-images", type=int, default=6, help="Images scored per packet.")
    ap.add_argument("--n-texts", type=int, default=4, help="Texts scored per packet.")
    ap.add_argument("--bits", type=int, default=32)
    ap.add_argument("--short-chars", type=int, default=400)
    ap.add_argument("--device", default="cuda:8")
    ap.add_argument("--save-images", default="reports/multimodal_recon.npz")
    ap.add_argument("--out", default="reports/multimodal_sweep.json")
    args = ap.parse_args()

    tcodec = PackedCodec(args.text_model, device=args.device)
    tcodec._load()
    P = tcodec.packet_dim
    icodec = ImageCodec(args.image_bottleneck, device=args.device)
    icodec._load()
    print(f"packet P={P} floats; fusion={args.fusion}; bits={args.bits}")

    texts = [json.loads(l)["text"] for l in open(args.dev_data)]
    random.Random(0).shuffle(texts)
    texts = [t[:args.short_chars] for t in texts if len(t) > 200][:2000]
    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    random.Random(0).shuffle(img_paths)
    img_paths = img_paths[:200]

    rows, saved = [], {}
    for d in args.widths:
        M = P // d
        n_img = min(args.n_images, M // 2)
        n_txt = min(args.n_texts, M - n_img)
        acc = {"img_psnr_vae": [], "img_psnr_orig": [], "img_code_cos": [],
               "txt_semantic_cos": [], "txt_code_cos": []}
        for p in range(args.n_packets):
            # --- build one mixed packet -------------------------------------
            imgs = load_images(img_paths[p * n_img:(p + 1) * n_img]).to(args.device)
            icodec.code_dim = d
            tcodec.code_dim = d
            img_codes = icodec.encode(imgs)[:, :d].cpu()
            vae_ref = icodec.vae_decode(icodec.vae_encode(imgs)).cpu()

            tb = [texts[(p * n_txt + i) % len(texts)] for i in range(n_txt)]
            with torch.no_grad():
                true_lat = tcodec.latents(tb).float()
                txt_codes = tcodec._bn.encode(true_lat, d)[:, :d].cpu()

            # Fill every remaining slot with filler traffic so the packet is
            # genuinely at full load M, not just carrying the scored messages.
            codes = {}
            for i in range(n_img):
                codes[i] = img_codes[i]
            for i in range(n_txt):
                codes[n_img + i] = txt_codes[i]
            for i in range(n_img + n_txt, M):
                f = torch.randn(d)
                codes[i] = f * torch.rsqrt(f.pow(2).mean())

            packer = build_packer(args.fusion, P, d, seed=99 + p, nonce=p)
            packet = packer.pack(codes)
            if args.bits < 32:
                packet, _ = quantize_packet(packet, bits=args.bits)

            # --- image receivers -------------------------------------------
            rec_codes = torch.stack([packer.unpack(packet, i) for i in range(n_img)])
            padded = torch.zeros(n_img, icodec._bn.code_dim)
            padded[:, :d] = rec_codes
            rec_imgs = icodec.decode(padded.to(args.device)).cpu()
            for i in range(n_img):
                acc["img_code_cos"].append(cos(rec_codes[i], img_codes[i]))
                acc["img_psnr_vae"].append(psnr(rec_imgs[i], vae_ref[i]))
                acc["img_psnr_orig"].append(psnr(rec_imgs[i], imgs[i].cpu()))
            if p == 0:
                saved[f"d{d}_recon"] = rec_imgs.numpy()
                saved[f"d{d}_vae"] = vae_ref.numpy()
                saved[f"d{d}_orig"] = imgs.cpu().numpy()

            # --- text receivers --------------------------------------------
            for i in range(n_txt):
                rec = packer.unpack(packet, n_img + i)
                acc["txt_code_cos"].append(cos(rec, txt_codes[i]))
                pad = torch.zeros(tcodec._bn.code_dim)
                pad[:d] = rec
                out = tcodec.decode(pad.to(args.device))
                if out:
                    with torch.no_grad():
                        re_lat = tcodec.latents([out]).float()[0]
                    acc["txt_semantic_cos"].append(cos(re_lat, true_lat[i]))
                else:
                    acc["txt_semantic_cos"].append(0.0)

        row = {"code_dim": d, "messages": M, "fusion": args.fusion,
               "bits": args.bits, "n_image_slots": n_img, "n_text_slots": n_txt}
        row.update({k: statistics.mean(v) if v else float("nan")
                    for k, v in acc.items()})
        rows.append(row)
        print(f"d={d:5d} M={M:4d} | IMAGE psnr_vs_vae={row['img_psnr_vae']:.2f}dB "
              f"psnr_vs_orig={row['img_psnr_orig']:.2f}dB code_cos={row['img_code_cos']:.3f} "
              f"| TEXT semantic_cos={row['txt_semantic_cos']:.3f} "
              f"code_cos={row['txt_code_cos']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"packet_dim": P, "rows": rows}, open(args.out, "w"), indent=2)
    if args.save_images:
        np.savez_compressed(args.save_images, **saved)
        print(f"wrote {args.save_images}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
