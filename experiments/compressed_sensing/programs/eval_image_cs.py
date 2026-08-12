#!/usr/bin/env python3
"""Block-based compressed sensing on real images -- the domain CS is
actually designed for, as a contrast point against the token-packet
findings (data/cs_packets/eval_*.json): images are continuous-valued,
genuinely sparse in a fixed basis (DCT/wavelet), and "close enough"
reconstruction (PSNR/SSIM) is the real success criterion -- unlike exact
token recovery, there's no Shannon-bound mismatch here.

Pipeline (classic single-pixel-camera / block-CS): split the image into
non-overlapping BxB blocks, flatten each to a vector, apply one shared
random sensing matrix Phi (M x B^2) per block, recover each block via OMP
against a dictionary (DCT, a dictionary learned on a DIFFERENT reference
image, or plain random Gaussian as a null-hypothesis baseline), reassemble.

Reuses the method-level CS codec and dictionary builder
unchanged -- only the image-specific plumbing (blocking, PSNR/SSIM) is new.
"""

import argparse
import json
import os

import numpy as np
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

from multimodal_comms.benchmarks.hiddenbench.runtime.compressed_sensing import CSCodec, make_dictionary, make_sensing_matrix


def to_blocks(img: np.ndarray, block: int) -> np.ndarray:
    h, w = img.shape
    blocks = img.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)
    return blocks.reshape(-1, block * block)


def from_blocks(blocks: np.ndarray, shape: tuple, block: int) -> np.ndarray:
    h, w = shape
    nh, nw = h // block, w // block
    img = blocks.reshape(nh, nw, block, block).transpose(0, 2, 1, 3)
    return img.reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out-dir", default="outputs/hiddenbench/image_cs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dim = args.block * args.block

    target = img_as_float(data.camera())  # 512x512, the eval image
    ref = img_as_float(data.astronaut().mean(axis=-1))  # different image, for the learned dictionary
    ref = resize(ref, target.shape, anti_aliasing=True)

    target_blocks = to_blocks(target, args.block)
    ref_blocks = to_blocks(ref, args.block)
    print(f"image {target.shape}, block={args.block} (dim={dim}), "
          f"{target_blocks.shape[0]} blocks, ref image for learned dict: astronaut (grayscale, resized)")

    dictionaries = {
        "dct": make_dictionary(
            "dct", dim=dim, n_atoms=dim, spatial_shape=(args.block, args.block)
        ),
        "random": make_dictionary("random", dim=dim, n_atoms=dim, seed=args.seed),
        "learned": make_dictionary("learned", dim=dim, n_atoms=2 * dim, data=ref_blocks, seed=args.seed),
    }

    results = {"image_shape": target.shape, "block": args.block, "sweep": []}
    header = f"{'dict':>8} {'ratio':>6} {'M':>4} {'k':>4} {'PSNR(dB)':>9} {'SSIM':>7}"
    print("\n== Image CS sweep ==")
    print(header)
    print("-" * len(header))

    saved_images = {}
    for ratio in args.ratios:
        m = max(1, round(ratio * dim))
        k = max(1, min(m - 1, round(0.3 * dim))) if m > 1 else 1
        phi = make_sensing_matrix(m, dim, seed=args.seed)

        for dict_name, d in dictionaries.items():
            codec = CSCodec(phi=phi, dictionary=d, method="omp", sparsity=k)
            recon_blocks = codec.roundtrip(target_blocks) if m < dim else target_blocks
            recon = from_blocks(recon_blocks, target.shape, args.block)
            recon_clipped = np.clip(recon, 0.0, 1.0)

            psnr = peak_signal_noise_ratio(target, recon_clipped, data_range=1.0)
            ssim = structural_similarity(target, recon_clipped, data_range=1.0)
            print(f"{dict_name:>8} {ratio:6.2f} {m:4d} {k:4d} {psnr:9.2f} {ssim:7.4f}")
            results["sweep"].append({
                "dictionary": dict_name, "ratio": ratio, "m": m, "k": k,
                "psnr_db": float(psnr), "ssim": float(ssim),
            })
            saved_images[f"{dict_name}_r{ratio}"] = recon_clipped

    np.savez_compressed(os.path.join(args.out_dir, "reconstructions.npz"),
                         original=target, **saved_images)
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_dir}/eval_results.json and reconstructions.npz")


if __name__ == "__main__":
    main()
