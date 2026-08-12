"""
Compressed Sensing of VLM Mixed Vision-Language Representations
================================================================
Uses Qwen3-VL to generate joint vision-language hidden states,
then applies compressed sensing to reconstruct them from partial
random projections. Demonstrates compressibility of multimodal
representations and how sampling ratio affects reconstruction.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLars
from skimage import data, color, transform
from PIL import Image
import torch
import os
import warnings
from multimodal_comms.methods.sensing import make_dictionary
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs/compressed_sensing/vlm_embeddings"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def load_images():
    images = {}
    try:
        img = data.camera()
        img = transform.resize(img, (336, 336), anti_aliasing=True)
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
        images["cameraman"] = img_rgb
    except Exception as e:
        print(f"  [skip] cameraman: {e}")
    try:
        img = data.astronaut()
        img = transform.resize(img, (336, 336), anti_aliasing=True)
        images["astronaut"] = img.astype(np.float32)
    except Exception as e:
        print(f"  [skip] astronaut: {e}")
    try:
        img = data.coins()
        img = transform.resize(img, (336, 336), anti_aliasing=True)
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
        images["coins"] = img_rgb
    except Exception as e:
        print(f"  [skip] coins: {e}")
    return images


def get_vlm_hidden_states(images):
    """Extract hidden states from Qwen3-VL after fusing vision and language."""
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"\nLoading {MODEL_NAME} on {DEVICE}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
        device_map={"": DEVICE},
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    hidden_states = {}
    prompts = [
        "Describe this image in detail.",
        "What objects and people are in this picture?",
    ]

    with torch.no_grad():
        for idx, (name, img_np) in enumerate(images.items()):
            print(f"  Processing {name}...")

            pil_img = Image.fromarray((img_np * 255).astype(np.uint8)).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompts[idx % len(prompts)]},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            image_embeds, _ = model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])

            outputs = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                image_embeds=image_embeds,
                image_grid_thw=inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            )
            hs = outputs.hidden_states

            num_layers = len(hs)
            layer_indices = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, -1]

            for li in layer_indices:
                h = hs[li].squeeze(0)  # (seq_len, hidden_dim)
                h_pooled = h.mean(dim=0).float().cpu().numpy()
                key = f"{name}_layer{li}"
                hidden_states[key] = h_pooled
                print(f"    {key}: dim={len(h_pooled)}")

            # Clean up to free VRAM
            del inputs, image_embeds, outputs, hs
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return hidden_states


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
    alpha = alpha_scale * max(1e-6, (1.0 - ratio))

    model_lasso = LassoLars(alpha=alpha, max_iter=2000, fit_intercept=False)
    model_lasso.fit(A, y)
    s_hat = model_lasso.coef_

    if basis == "dct":
        x_hat = Psi @ s_hat
    else:
        x_hat = s_hat

    sparsity = float(np.sum(np.abs(model_lasso.coef_) > 1e-6) / N)
    return x_hat, sparsity


def reconstruct_pinv(x, ratio, phi=None):
    N = len(x)
    M = max(1, int(ratio * N))
    Phi = make_projection(N, ratio) if phi is None else phi
    y = Phi @ x
    x_hat = Phi.T @ np.linalg.solve(Phi @ Phi.T + 1e-6 * np.eye(M), y)
    return x_hat


def evaluate_reconstruction(x, x_hat):
    x_norm = x / (np.linalg.norm(x) + 1e-10)
    xh_norm = x_hat / (np.linalg.norm(x_hat) + 1e-10)
    cos_sim = float(np.dot(x_norm, xh_norm))
    mse = float(np.mean((x - x_hat) ** 2))
    psnr_db = float(10 * np.log10(1.0 / (mse + 1e-10)))
    relative_error = float(np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-10))
    return {
        "cosine_similarity": cos_sim,
        "psnr": psnr_db,
        "relative_error": relative_error,
        "mse": mse,
    }


def run_full_experiment(hidden_states, ratios=None):
    ratios = ratios or [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75]

    all_results = {}
    all_recons = {}

    for key, x in hidden_states.items():
        print(f"\n  {'='*56}")
        print(f"  {key}  (dim={len(x)})")
        print(f"  {'='*56}")

        all_results[key] = {}
        all_recons[key] = {}
        projections = {
            ratio: make_projection(len(x), ratio, seed=42 + index)
            for index, ratio in enumerate(ratios)
        }

        for method, basis in [("identity_l1", "identity"), ("identity_pinv", None),
                               ("dct_l1", "dct")]:
            all_results[key][method] = []
            all_recons[key][method] = []
            print(f"    {method}:")
            for ratio in ratios:
                if "l1" in method:
                    x_hat, sparsity = compress_and_reconstruct_l1(
                        x, ratio, basis=basis, alpha_scale=0.02,
                        phi=projections[ratio],
                    )
                else:
                    x_hat = reconstruct_pinv(x, ratio, phi=projections[ratio])
                    sparsity = 1.0

                metrics = evaluate_reconstruction(x, x_hat)
                metrics["sparsity"] = sparsity
                all_results[key][method].append((ratio, metrics))
                all_recons[key][method].append(x_hat)

                extra = f"  spar={sparsity:.3f}" if "l1" in method else ""
                print(f"      ratio={ratio:.2f}  cos={metrics['cosine_similarity']:.4f}  "
                      f"rel_err={metrics['relative_error']:.4f}{extra}")

    return all_results, all_recons, ratios


def plot_layer_curves(all_results, ratios, hidden_states):
    """Plot cos_sim vs ratio, grouped by image and layer."""
    images = sorted(set(k.split("_layer")[0] for k in hidden_states.keys()))
    layers = sorted(set(int(k.split("layer")[-1]) for k in hidden_states.keys()))

    colors = {"cameraman": "#1f77b4", "astronaut": "#ff7f0e", "coins": "#2ca02c"}

    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 3.5 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for li, layer in enumerate(layers):
        for col, method in enumerate(["identity_l1", "identity_pinv"]):
            ax = axes[li, col]
            for img_name in images:
                key = f"{img_name}_layer{layer}"
                if key not in all_results:
                    continue
                data = all_results[key][method]
                r_vals = [d[0] * 100 for d in data]
                cos_vals = [d[1]["cosine_similarity"] for d in data]
                ax.plot(r_vals, cos_vals, marker="o", color=colors[img_name],
                        linewidth=2, markersize=4, label=img_name)
            ax.set_title(f"Layer {layer} — {method}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Measurement Ratio (%)", fontsize=9)
            ax.set_ylabel("Cosine Similarity", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

    fig.suptitle("VLM Hidden State Reconstruction (Qwen3-VL-8B)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    svg_path = os.path.join(OUTPUT_DIR, "vlm_layer_curves.svg")
    png_path = os.path.join(OUTPUT_DIR, "vlm_layer_curves.png")
    fig.savefig(svg_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved layer curves to {png_path}")
    plt.close(fig)


def plot_layer_vs_l1_summary(all_results, ratios):
    """Compact summary: cos_sim at 10%, 20%, 40% vs layer depth."""
    keys = [k for k in all_results.keys()]
    images = sorted(set(k.split("_layer")[0] for k in keys))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"cameraman": "#1f77b4", "astronaut": "#ff7f0e", "coins": "#2ca02c"}

    for img_name in images:
        img_keys = sorted(
            [k for k in keys if k.startswith(img_name)],
            key=lambda k: int(k.split("layer")[-1])
        )
        layer_nums = [int(k.split("layer")[-1]) for k in img_keys]

        targets = (0.10, 0.20, 0.40)
        for col, target_ratio in enumerate(targets):
            ratio_idx = min(range(len(ratios)), key=lambda index: abs(ratios[index] - target_ratio))
            label = f"{ratios[ratio_idx] * 100:.0f}%"
            ax = axes[col]
            cos_l1 = []
            cos_pinv = []
            for k in img_keys:
                cos_l1.append(all_results[k]["identity_l1"][ratio_idx][1]["cosine_similarity"])
                cos_pinv.append(all_results[k]["identity_pinv"][ratio_idx][1]["cosine_similarity"])

            ax.plot(layer_nums, cos_l1, marker="o", color=colors[img_name],
                    linewidth=2, markersize=6, linestyle="-", label=f"{img_name} L1")
            ax.plot(layer_nums, cos_pinv, marker="s", color=colors[img_name],
                    linewidth=2, markersize=6, linestyle="--", alpha=0.5, label=f"{img_name} pinv")

            ax.set_xlabel("Layer Index", fontsize=11)
            ax.set_ylabel("Cosine Similarity", fontsize=11)
            ax.set_title(f"Reconstruction at {label} Sampling", fontsize=12, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

    fig.suptitle("VLM Hidden State Compressibility by Layer Depth",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    svg_path = os.path.join(OUTPUT_DIR, "vlm_layer_depth.svg")
    png_path = os.path.join(OUTPUT_DIR, "vlm_layer_depth.png")
    fig.savefig(svg_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved layer depth summary to {png_path}")
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
    print("  Compressed Sensing of VLM Multimodal Representations")
    print("  Qwen3-VL-8B + Random Projections + L1 Recovery")
    print("=" * 60)

    images = load_images()
    print(f"\nLoaded {len(images)} image(s) for VLM processing")

    hidden_states = get_vlm_hidden_states(images)
    print(f"\nExtracted {len(hidden_states)} hidden state vectors")

    all_results, all_recons, ratios = run_full_experiment(hidden_states, args.ratios)

    plot_layer_curves(all_results, ratios, hidden_states)
    plot_layer_vs_l1_summary(all_results, ratios)

    print("\n" + "=" * 60)
    summary_index = min(range(len(ratios)), key=lambda index: abs(ratios[index] - 0.10))
    print(f"  Key Summary — cos_sim at {ratios[summary_index] * 100:.0f}% (L1 / pinv) by layer")
    print("=" * 60)
    keys_sorted = sorted(all_results.keys(),
                         key=lambda k: (k.split("_layer")[0], int(k.split("layer")[-1])))
    header = f"{'Key':<30}  {'identity_l1':>12}  {'identity_pinv':>15}"
    print(header)
    print("-" * 60)
    for key in keys_sorted:
        cos_l1 = all_results[key]["identity_l1"][summary_index][1]["cosine_similarity"]
        cos_pinv = all_results[key]["identity_pinv"][summary_index][1]["cosine_similarity"]
        print(f"{key:<30}  {cos_l1:>12.4f}  {cos_pinv:>15.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
