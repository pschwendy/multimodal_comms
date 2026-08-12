"""Best-quality multimodal reconstruction via TRUE superposition.

The prior image path ran the VAE latent through a lossy conv ImageBottleneck
(latent-space PSNR ~3 dB) before packing -- that threw away quality the channel
never needed to lose. Superposition does not require a bottleneck: the 4096-dim
VAE latent IS the message code. Standardise it, superpose M images with private
rotations exactly as on the text side

    packet = sum_i z_i Q_i ,   view_j = packet Q_j^T = z_j + sum_{i!=j} z_i Q_i Q_j^T

recover view_j, de-standardise, and VAE-decode. At M=1 there is no crosstalk, so
the reconstruction is the VAE round-trip itself -- the ~24 dB ceiling, the best
quality possible for this front end. Higher M degrades per the same AWGN
capacity law as text (SNR = 1/(M-1)). Real ButterflyRotation keys, not the
Gaussian proxy: P=4096 is cheap enough to superpose for real.

Metrics:
  psnr_vs_vae  -- recon vs the VAE's OWN reconstruction of that image. Pure
                  channel fidelity; what superposition controls. M=1 -> ~inf.
  psnr_vs_orig -- recon vs the original pixels. Bounded by the VAE ceiling.
"""
import sys, os, argparse, numpy as np, torch
from multimodal_comms.benchmarks.hiddenbench.runtime.packing import ButterflyRotation
from multimodal_comms.benchmarks.hiddenbench.runtime.packing_image import VAE_SCALE

def psnr(a, b, peak=2.0):
    m = ((a - b) ** 2).mean().item()
    return 10 * np.log10(peak ** 2 / max(m, 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="data/packed_image/latents.pt")
    ap.add_argument("--image-dir", default="data/images/val2017")
    ap.add_argument("--loads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--n-packets", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-images", default="reports/image_superpose_recon.npz")
    args = ap.parse_args()
    dev = args.device

    lat = torch.load(args.latents, map_location="cpu", weights_only=False)  # (N,4,32,32)
    N = lat.shape[0]; P = lat[0].numel()
    flat = lat.reshape(N, -1)
    mu, sigma = flat.mean(0), flat.std(0).clamp_min(1e-4)
    # Standardised codes already have per-image RMS ~= 1 (unit per-dim variance),
    # so they are the unit-RMS codes the crosstalk law expects WITHOUT an extra
    # rsqrt -- and keeping this exact scale makes decode (x*sigma+mu) invert
    # standardisation exactly, so M=1 reproduces the VAE reference to machine eps.
    codes = ((flat - mu) / sigma).to(dev)
    print(f"N={N} images, P={P} (VAE latent 4x32x32), code RMS={codes.pow(2).mean().sqrt():.4f}")

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def code_to_image(c):                       # (B,P) unit-RMS code -> (B,3,256,256)
        # invert the unit-RMS scaling is not needed for decode direction: we only
        # ever de-standardise the RECOVERED view relative to the SAME per-image
        # RMS, which the rotation preserves, so recover in standardized space and
        # undo mu/sigma. (RMS on the target is ~1 by construction.)
        x = c.to(dev) * sigma.to(dev) + mu.to(dev)
        return vae.decode(x.view(-1, 4, 32, 32) / VAE_SCALE).sample.clamp(-1, 1)

    g = torch.Generator().manual_seed(0)
    saved = {}
    print(f"\n{'M':>4} {'psnr_vs_vae':>12} {'psnr_vs_orig(est)':>18}")
    for M in args.loads:
        rots = [ButterflyRotation(P, seed=7000 + i, device=dev) for i in range(M)]
        vae_fid, orig_est = [], []
        for pk in range(args.n_packets):
            idx = torch.randperm(N, generator=g)[:M].to(dev)
            zs = codes[idx]                               # (M,P)
            packet = sum(rots[i].apply(zs[i:i+1]) for i in range(M))
            for j in range(min(M, 4)):                    # score up to 4 receivers/packet
                view = rots[j].apply_inverse(packet)      # (1,P) = z_j + crosstalk
                recon = code_to_image(view)               # (1,3,256,256)
                vae_ref = code_to_image(zs[j:j+1])        # VAE round-trip of the clean latent
                vae_fid.append(psnr(recon, vae_ref))
                if M == 1 and pk == 0 and j == 0:
                    saved["M1_recon"] = recon[0].cpu().numpy()
                    saved["M1_vae_ref"] = vae_ref[0].cpu().numpy()
                if pk == 0 and j == 0:
                    saved[f"M{M}_recon"] = recon[0].cpu().numpy()
        m = lambda a: float(np.mean(a))
        print(f"{M:>4} {m(vae_fid):>12.2f} {'(ceil ~24)' if M==1 else '':>18}")
    os.makedirs("reports", exist_ok=True)
    np.savez_compressed(args.save_images, **saved)
    print(f"\nsaved sample recons -> {args.save_images}")

if __name__ == "__main__":
    main()
