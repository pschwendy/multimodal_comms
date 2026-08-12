"""
Compressed Sensing for Natural Images
=====================================
Demonstrates image reconstruction from partial Fourier measurements
using Total Variation (TV) regularization.
Shows how reconstruction quality varies with the sampling ratio.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import minimize
from skimage import data, color, transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import os

OUTPUT_DIR = "outputs/compressed_sensing/fourier_tv"

np.random.seed(42)


def load_images(size: int = 256):
    images = {}
    try:
        img = data.camera()
        images["cameraman"] = transform.resize(
            img.astype(np.float64) / 255.0, (size, size), anti_aliasing=True
        )
    except Exception:
        pass
    try:
        img = data.astronaut()
        img_gray = color.rgb2gray(img)
        img_gray = transform.resize(img_gray, (size, size), anti_aliasing=True)
        images["astronaut"] = img_gray
    except Exception:
        pass
    try:
        img = data.coins()
        img_gray = img.astype(np.float64) / 255.0
        img_gray = transform.resize(img_gray, (size, size), anti_aliasing=True)
        images["coins"] = img_gray
    except Exception:
        pass
    return images


def create_radial_mask(shape, ratio):
    """Create a radial (low-frequency biased) sampling mask in k-space."""
    ny, nx = shape
    y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]
    r = np.sqrt(x**2 + y**2)
    max_r = np.sqrt((nx//2)**2 + (ny//2)**2)

    n_pixels = int(ratio * nx * ny)
    mask = np.zeros(shape, dtype=bool)

    sorted_idx = np.argsort(r.ravel())
    mask_flat = mask.ravel()
    mask_flat[sorted_idx[:n_pixels]] = True
    return mask.reshape(shape)


def create_random_mask(shape, ratio):
    """Create a random sampling mask in k-space with low-frequency bias."""
    ny, nx = shape
    y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]
    r = np.sqrt(x**2 + y**2)

    weights = 1.0 / (1.0 + r / 4.0)
    weights /= weights.sum()

    n_pixels = int(ratio * nx * ny)
    flat_idx = np.random.choice(nx * ny, size=n_pixels, replace=False, p=weights.ravel())
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[flat_idx] = True
    return mask


def tv_norm(x):
    """Total Variation norm of a 2D image."""
    dx = np.diff(x, axis=1)
    dy = np.diff(x, axis=0)
    return np.sum(np.sqrt(dx[:, :-1]**2 + dy[:-1, :]**2 + 1e-8))


def reconstruct_tv(measurements, mask, alpha=0.005, lr=0.01, n_iter=300, verbose=False):
    """
    Reconstruct image from partial k-space measurements using TV regularization.
    Uses gradient descent on: ||M * F(x) - y||_2^2 + alpha * TV(x)
    where M is the sampling mask, F is the 2D Fourier transform.
    """
    ny, nx = mask.shape
    x = ifft2(ifftshift(measurements * mask)).real
    x = np.clip(x, 0, 1)

    for it in range(n_iter):
        kx = fft2(x)
        kx = fftshift(kx)
        residual_k = (kx * mask - measurements) * mask
        grad_data = 2 * ifft2(ifftshift(residual_k)).real

        # TV gradient via subgradient
        dx = np.diff(x, axis=1)  # (h, w-1)
        dy = np.diff(x, axis=0)  # (h-1, w)

        # inner product area: (h-1, w-1)
        denom = np.sqrt(dx[:-1, :]**2 + dy[:, :-1]**2 + 1e-6)

        tv_grad = np.zeros_like(x)

        # x-direction: d/dx (dx / |grad|)
        dx_norm = dx.copy()  # (h, w-1)
        dx_norm[:-1, :] /= denom
        dx_norm[-1, :] = 0
        tv_grad -= np.diff(dx_norm, axis=1, prepend=0, append=0)

        # y-direction: d/dy (dy / |grad|)
        dy_norm = dy.copy()  # (h-1, w)
        dy_norm[:, :-1] /= denom
        dy_norm[:, -1] = 0
        tv_grad -= np.diff(dy_norm, axis=0, prepend=0, append=0)

        grad = grad_data + alpha * tv_grad
        x = x - lr * grad
        x = np.clip(x, 0, 1)

        if verbose and it % 100 == 0:
            loss = np.sum(np.abs(kx * mask - measurements)**2)
            print(f"  iter {it:4d}  loss={loss:.4e}  TV={tv_norm(x):.4f}")

    return np.clip(x, 0, 1)


def run_experiment(image, name, ratios, n_iter=300, alpha=0.005, mask_type="radial"):
    print(f"\n{'='*60}")
    print(f"Processing: {name}  ({image.shape[0]}x{image.shape[1]})")
    print(f"{'='*60}")

    results = {}
    h, w = image.shape

    for ratio in ratios:
        print(f"\n--- Ratio: {ratio*100:5.1f}% ---")

        if mask_type == "random":
            mask = create_random_mask((h, w), ratio)
        else:
            mask = create_radial_mask((h, w), ratio)

        kspace = fftshift(fft2(image))
        measurements = kspace * mask

        t0 = time.time()
        recon = reconstruct_tv(measurements, mask, alpha=alpha, n_iter=n_iter)
        elapsed = time.time() - t0

        r_psnr = psnr(image, recon, data_range=1.0)
        r_ssim = ssim(image, recon, data_range=1.0)
        error = np.mean((image - recon)**2) ** 0.5

        print(f"  PSNR: {r_psnr:.2f} dB  |  SSIM: {r_ssim:.4f}  |  RMSE: {error:.4f}  |  time: {elapsed:.1f}s")

        results[ratio] = {
            "mask": mask,
            "recon": recon,
            "psnr": r_psnr,
            "ssim": r_ssim,
            "rmse": error,
        }

    return results


def plot_results(image, results, name, ratios):
    n_ratios = len(ratios)
    fig, axes = plt.subplots(2, n_ratios + 1, figsize=(3.5 * (n_ratios + 1), 7))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")

    axes[1, 0].imshow(np.log(1 + np.abs(fftshift(fft2(image)))), cmap="hot")
    axes[1, 0].set_title("Full k-space", fontsize=11)
    axes[1, 0].axis("off")

    for i, ratio in enumerate(ratios):
        ax_mask = axes[0, i + 1]
        ax_recon = axes[1, i + 1]

        mask = results[ratio]["mask"]
        recon = results[ratio]["recon"]
        rpsnr = results[ratio]["psnr"]

        ax_mask.imshow(mask, cmap="gray")
        ax_mask.set_title(f"Mask {ratio*100:.0f}%\n({int(ratio*mask.size)} samples)", fontsize=10)
        ax_mask.axis("off")

        ax_recon.imshow(recon, cmap="gray")
        ax_recon.set_title(f"Recon {ratio*100:.0f}%\nPSNR={rpsnr:.1f} dB", fontsize=10)
        ax_recon.axis("off")

    fig.suptitle(f"Compressed Sensing Reconstruction — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    svg_path = os.path.join(OUTPUT_DIR, f"recon_grid_{name}.svg")
    png_path = os.path.join(OUTPUT_DIR, f"recon_grid_{name}.png")
    fig.savefig(svg_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved reconstruction grid to {png_path}")
    plt.close(fig)


def plot_quality_curve(all_results, ratios):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"cameraman": "#1f77b4", "astronaut": "#ff7f0e", "coins": "#2ca02c"}
    markers = {"cameraman": "o", "astronaut": "s", "coins": "^"}

    for name, results in all_results.items():
        ratio_pct = [r * 100 for r in ratios]
        psnrs = [results[r]["psnr"] for r in ratios]
        ssims = [results[r]["ssim"] for r in ratios]

        c = colors.get(name, None)
        m = markers.get(name, "o")

        ax1.plot(ratio_pct, psnrs, marker=m, color=c, linewidth=2, markersize=7, label=name)
        ax2.plot(ratio_pct, ssims, marker=m, color=c, linewidth=2, markersize=7, label=name)

    ax1.set_xlabel("Sampling Ratio (%)", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title("PSNR vs Sampling Ratio", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Sampling Ratio (%)", fontsize=12)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title("SSIM vs Sampling Ratio", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    svg_path = os.path.join(OUTPUT_DIR, "quality_vs_ratio.svg")
    png_path = os.path.join(OUTPUT_DIR, "quality_vs_ratio.png")
    fig.savefig(svg_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved quality curve to {png_path}")
    plt.close(fig)


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    )
    args = parser.parse_args()
    OUTPUT_DIR = args.out_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Compressed Sensing for Natural Images")
    print("  Fourier undersampling + TV reconstruction")
    print("=" * 60)

    images = load_images(args.size)
    print(f"\nLoaded {len(images)} image(s): {list(images.keys())}")

    ratios = args.ratios
    all_results = {}

    for name, img in images.items():
        results = run_experiment(
            img, name, ratios,
            n_iter=args.iterations, alpha=0.005, mask_type="radial"
        )
        all_results[name] = results
        plot_results(img, results, name, ratios)

    if len(all_results) > 1:
        plot_quality_curve(all_results, ratios)

    print("\n" + "=" * 60)
    print("  Summary — PSNR (dB) / SSIM by sampling ratio")
    print("=" * 60)
    header = f"{'Image':<15}"
    for r in ratios:
        header += f"  {r*100:5.0f}%  "
    print(header)
    print("-" * 60)
    for name, results in all_results.items():
        row = f"{name:<15}"
        for r in ratios:
            row += f"  {results[r]['psnr']:4.1f}/{results[r]['ssim']:.3f}"
        print(row)

    print("\nDone. All figures saved.")


if __name__ == "__main__":
    main()
