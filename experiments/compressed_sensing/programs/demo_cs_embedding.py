"""
Compressed Sensing of VLM (CLIP) Image Embeddings
==================================================
Demonstrates reconstructing CLIP embeddings from partial random projections
using L1 minimization (basis pursuit / LASSO).
Shows how reconstruction quality varies with measurement ratio.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLars
from skimage import data, transform
import torch
import os
import warnings
from multimodal_comms.methods.sensing import make_dictionary
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs/compressed_sensing/clip_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_images():
    images = {}
    try:
        img = data.camera()
        img = transform.resize(img, (224, 224), anti_aliasing=True)
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
        images["cameraman"] = img_rgb
    except Exception as e:
        print(f"  [skip] cameraman: {e}")
    try:
        img = data.astronaut()
        img = transform.resize(img, (224, 224), anti_aliasing=True)
        images["astronaut"] = img.astype(np.float32)
    except Exception as e:
        print(f"  [skip] astronaut: {e}")
    try:
        img = data.coins()
        img = transform.resize(img, (224, 224), anti_aliasing=True)
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
        images["coins"] = img_rgb
    except Exception as e:
        print(f"  [skip] coins: {e}")
    return images


def get_clip_embeddings(images_rgb):
    from transformers import CLIPModel, CLIPProcessor

    print(f"\nLoading CLIP model ({MODEL_NAME}) on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for name, img in images_rgb.items():
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            emb = model.get_image_features(**inputs)
            emb = emb.cpu().numpy().flatten()
            embeddings[name] = emb
            print(f"  {name}: embedding dim = {len(emb)}")
    return embeddings


def make_projection(dimension, ratio, seed=42):
    measurements = max(1, int(ratio * dimension))
    return np.random.default_rng(seed).standard_normal((measurements, dimension)) / np.sqrt(measurements)


def compress_and_reconstruct_l1(
    x, ratio, basis="identity", alpha_scale=0.02, phi=None
):
    N = len(x)

    M = max(1, int(ratio * N))
    Phi = make_projection(N, ratio) if phi is None else phi

    if basis == "dct":
        Psi = make_dictionary("dct", N, N)
        A = Phi @ Psi
    elif basis == "identity":
        A = Phi
        Psi = None
    else:
        raise ValueError(f"Unknown basis: {basis}")

    y = Phi @ x

    # Adaptive alpha: lower when more measurements are available
    alpha = alpha_scale * max(1e-6, (1.0 - ratio))

    model = LassoLars(alpha=alpha, max_iter=2000, fit_intercept=False)
    model.fit(A, y)
    s_hat = model.coef_

    if basis == "dct":
        x_hat = Psi @ s_hat
    else:
        x_hat = s_hat

    sparsity = np.sum(np.abs(model.coef_) > 1e-6) / N
    return x_hat, sparsity


def reconstruct_pinv(x, ratio, phi=None):
    """Minimum-norm least squares via pseudo-inverse (no sparsity prior)."""
    N = len(x)
    M = max(1, int(ratio * N))
    Phi = make_projection(N, ratio) if phi is None else phi
    y = Phi @ x
    x_hat = Phi.T @ np.linalg.solve(Phi @ Phi.T + 1e-6 * np.eye(M), y)
    return x_hat


def evaluate_reconstruction(x, x_hat):
    x_norm = x / (np.linalg.norm(x) + 1e-10)
    xh_norm = x_hat / (np.linalg.norm(x_hat) + 1e-10)
    cos_sim = np.dot(x_norm, xh_norm)

    mse = np.mean((x - x_hat) ** 2)
    psnr_db = 10 * np.log10(1.0 / (mse + 1e-10))

    relative_error = np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-10)

    return {
        "cosine_similarity": float(cos_sim),
        "psnr": float(psnr_db),
        "relative_error": float(relative_error),
        "mse": float(mse),
    }


def run_experiment(embeddings):
    ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75]
    bases = ["identity", "dct"]

    all_results = {}
    for name, x in embeddings.items():
        print(f"\n{'='*60}")
        print(f"Embedding: {name}  (dim={len(x)})")
        print(f"{'='*60}")

        all_results[name] = {}

        for basis in bases:
            print(f"\n  Basis: {basis}")
            all_results[name][basis] = []
            for ratio in ratios:
                x_hat, sparsity = compress_and_reconstruct_l1(
                    x, ratio, basis=basis, alpha_scale=0.01
                )
                metrics = evaluate_reconstruction(x, x_hat)
                metrics["sparsity"] = sparsity
                all_results[name][basis].append((ratio, metrics))
                print(f"    ratio={ratio:5.2f}  cos_sim={metrics['cosine_similarity']:.4f}  "
                      f"rel_err={metrics['relative_error']:.4f}  spar={sparsity:.3f}")

    return all_results, ratios


def plot_cosine_sim_curves(all_results, ratios, embeddings):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"cameraman": "#1f77b4", "astronaut": "#ff7f0e", "coins": "#2ca02c"}
    linestyles = {"identity_l1": "-", "dct_l1": "--",
                   "identity_pinv": ":", "dct_pinv": "-."}

    for idx, key in enumerate(["identity_l1", "dct_l1", "identity_pinv", "dct_pinv"]):
        ax = axes[idx // 2, idx % 2]
        for name in all_results:
            data_pts = all_results[name][key]
            r_vals = [d[0] * 100 for d in data_pts]
            if "l1" in key:
                cos_vals = [d[1]["cosine_similarity"] for d in data_pts]
            else:
                cos_vals = [d[1]["cosine_similarity"] for d in data_pts]
            ax.plot(r_vals, cos_vals, marker="o", color=colors.get(name),
                    linewidth=2, markersize=5, label=name)
        ax.set_xlabel("Measurement Ratio (%)", fontsize=12)
        ax.set_ylabel("Cosine Similarity", fontsize=12)
        ax.set_title(key, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("CLIP Embedding Reconstruction via Compressed Sensing",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    svg_path = os.path.join(OUTPUT_DIR, "embedding_cosine_vs_ratio.svg")
    png_path = os.path.join(OUTPUT_DIR, "embedding_cosine_vs_ratio.png")
    fig.savefig(svg_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved cosine similarity curves to {png_path}")
    plt.close(fig)



def run_experiment_v2(embeddings, ratios=None):
    ratios = ratios or [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75]
    bases = ["identity", "dct"]
    methods = ["l1", "pinv"]

    all_results = {}
    all_recons = {}

    for name, x in embeddings.items():
        print(f"\n{'='*60}")
        print(f"Embedding: {name}  (dim={len(x)})")
        print(f"{'='*60}")

        all_results[name] = {}
        all_recons[name] = {}
        projections = {
            ratio: make_projection(len(x), ratio, seed=42 + index)
            for index, ratio in enumerate(ratios)
        }

        for basis in bases:
            for method in methods:
                key = f"{basis}_{method}"
                print(f"\n  {key}:")
                all_results[name][key] = []
                all_recons[name][key] = []
                for ratio in ratios:
                    if method == "l1":
                        x_hat, sparsity = compress_and_reconstruct_l1(
                            x, ratio, basis=basis, alpha_scale=0.02,
                            phi=projections[ratio],
                        )
                    else:
                        x_hat = reconstruct_pinv(x, ratio, phi=projections[ratio])
                        sparsity = 1.0

                    metrics = evaluate_reconstruction(x, x_hat)
                    metrics["sparsity"] = sparsity
                    all_results[name][key].append((ratio, metrics))
                    all_recons[name][key].append(x_hat)
                    print(f"    ratio={ratio:.2f}  cos_sim={metrics['cosine_similarity']:.4f}  "
                          f"rel_err={metrics['relative_error']:.4f}" +
                          (f"  spar={sparsity:.3f}" if method == "l1" else ""))

    return all_results, all_recons, ratios


def plot_embedding_scatter(embeddings, all_results, all_recons, ratios):
    n_images = len(embeddings)
    n_keys = 2
    keys = ["identity_l1", "dct_l1"]
    key_labels = ["Identity + L1", "DCT + L1"]
    n_ratios_display = min(5, len(ratios))
    ratio_indices = np.linspace(0, len(ratios) - 1, n_ratios_display, dtype=int)

    fig, axes = plt.subplots(n_images * n_keys, n_ratios_display,
                             figsize=(n_ratios_display * 3, n_images * n_keys * 2.8))
    if n_images * n_keys == 1:
        axes = axes.reshape(1, -1)
    elif n_ratios_display == 1:
        axes = axes.reshape(-1, 1)

    colors = {"cameraman": "#1f77b4", "astronaut": "#ff7f0e", "coins": "#2ca02c"}

    for i, (name, x) in enumerate(embeddings.items()):
        x_norm = x / (np.linalg.norm(x) + 1e-10)
        for j, (key, klab) in enumerate(zip(keys, key_labels)):
            row = i * 2 + j
            for ki, k in enumerate(ratio_indices):
                ax = axes[row, ki]
                ratio_val = ratios[k]
                metrics = all_results[name][key][k][1]
                x_hat = all_recons[name][key][k]
                xh_norm = x_hat / (np.linalg.norm(x_hat) + 1e-10)

                ax.scatter(range(len(x)), x_norm, s=0.8, alpha=0.3, color="gray",
                          label="original", rasterized=True)
                ax.scatter(range(len(x)), xh_norm, s=0.8, alpha=0.7, color=colors[name],
                          rasterized=True)
                ax.set_title(f"{ratio_val*100:.0f}%  cos={metrics['cosine_similarity']:.3f}",
                            fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if ki == 0:
                    ax.set_ylabel(f"{name}\n{klab}", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "embedding_scatter.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved embedding scatter to {png_path}")
    plt.close(fig)


def plot_recon_vs_sparsity(all_results, ratios):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = {"cameraman": "#1f77b4", "astronaut": "#ff7f0e", "coins": "#2ca02c"}
    markers = {"identity_l1": "o", "dct_l1": "s"}

    for name in all_results:
        for key in ["identity_l1", "dct_l1"]:
            data_pts = all_results[name][key]
            spars = [d[1]["sparsity"] for d in data_pts]
            cos = [d[1]["cosine_similarity"] for d in data_pts]
            ax.scatter(spars, cos, marker=markers[key], color=colors[name],
                      label=f"{name}-{key}", s=40, alpha=0.8)

    ax.set_xlabel("Sparsity (fraction of non-zero coeffs)", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Reconstruction Quality vs Solution Sparsity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "sparsity_vs_quality.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved sparsity plot to {png_path}")
    plt.close(fig)


def main():
    global OUTPUT_DIR, DEVICE, MODEL_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75]
    )
    args = parser.parse_args()
    OUTPUT_DIR, DEVICE, MODEL_NAME = args.out_dir, args.device, args.model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Compressed Sensing of CLIP Image Embeddings")
    print("  Random Gaussian projections + L1 (LASSO) recovery")
    print("=" * 60)

    images = load_images()
    print(f"\nLoaded {len(images)} natural image(s) for embedding")

    embeddings = get_clip_embeddings(images)

    all_results, all_recons, ratios = run_experiment_v2(embeddings, args.ratios)

    plot_cosine_sim_curves(all_results, ratios, embeddings)
    plot_embedding_scatter(embeddings, all_results, all_recons, ratios)
    plot_recon_vs_sparsity(all_results, ratios)

    print("\n" + "=" * 60)
    print("  Summary — Cosine Similarity by sampling ratio")
    print("=" * 60)
    keys = ["identity_l1", "dct_l1", "identity_pinv", "dct_pinv"]
    for key in keys:
        print(f"\n  {key}:")
        header = f"{'Image':<15}"
        for r in ratios:
            header += f"  {r*100:5.0f}%  "
        print(header)
        print("-" * 60)
        for name in all_results:
            row = f"{name:<15}"
            for d in all_results[name][key]:
                row += f"  {d[1]['cosine_similarity']:.3f} "
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
