"""Validate compressed sensing where its sparsity assumption does and does not hold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from multimodal_comms.methods.sensing import (
    CURCodec,
    CompressedSensingCodec,
    PCACodec,
    SVDCodec,
    make_dictionary,
    make_sensing_matrix,
)
from multimodal_comms.methods.sensing.stream import (
    causal_predictability_curve,
    effective_temporal_rank,
    synthetic_stream,
)


def _relative_error(reference: np.ndarray, recovered: np.ndarray) -> float:
    return float(np.linalg.norm(reference - recovered) / np.linalg.norm(reference))


def _cosine_rows(reference: np.ndarray, recovered: np.ndarray) -> float:
    numerator = np.sum(reference * recovered, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(recovered, axis=1)
    return float(np.median(numerator / np.clip(denominator, 1e-12, None)))


def _to_blocks(image: np.ndarray, block: int) -> np.ndarray:
    height, width = image.shape
    return (
        image.reshape(height // block, block, width // block, block)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block * block)
    )


def _from_blocks(blocks: np.ndarray, shape: tuple[int, int], block: int) -> np.ndarray:
    height, width = shape
    return (
        blocks.reshape(height // block, width // block, block, block)
        .transpose(0, 2, 1, 3)
        .reshape(shape)
    )


def validate_sparse_image_coefficients() -> dict[str, float]:
    """Recover exactly sparse coefficients in the same 2-D basis used for images."""
    block, measurements, sparsity = 8, 40, 10
    dimension = block * block
    dictionary = make_dictionary(
        "dct", dimension, dimension, spatial_shape=(block, block)
    )
    phi = make_sensing_matrix(measurements, dimension, seed=19)
    rng = np.random.default_rng(4)
    coefficients = np.zeros((12, dimension))
    for row in coefficients:
        support = rng.choice(dimension, sparsity, replace=False)
        row[support] = rng.normal(size=sparsity)
    signals = coefficients @ dictionary.T

    sparse = CompressedSensingCodec(
        phi, dictionary, sparsity=sparsity, recovery="omp"
    ).roundtrip(signals)
    baseline = CompressedSensingCodec(
        phi, dictionary, sparsity=sparsity, recovery="ridge"
    ).roundtrip(signals)
    sparse_error = _relative_error(signals, sparse)
    baseline_error = _relative_error(signals, baseline)
    if not sparse_error < 0.02:
        raise AssertionError(f"sparse DCT recovery error is too high: {sparse_error:.4f}")
    if not sparse_error < baseline_error / 10:
        raise AssertionError("OMP did not beat the non-sparse projection baseline")
    return {
        "dimension": dimension,
        "measurements": measurements,
        "sparsity": sparsity,
        "relative_l2": sparse_error,
        "ridge_relative_l2": baseline_error,
    }


def validate_natural_image() -> dict[str, float | str]:
    """Run block CS on a bundled natural image; requires the full environment."""
    try:
        from skimage import data, img_as_float
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from skimage.transform import resize
    except ModuleNotFoundError:
        return {"status": "skipped", "reason": "scikit-image is not installed"}

    block, measurements, sparsity = 8, 40, 10
    image = resize(img_as_float(data.camera()), (64, 64), anti_aliasing=True)
    blocks = _to_blocks(image, block)
    dimension = block * block
    dictionary = make_dictionary(
        "dct", dimension, dimension, spatial_shape=(block, block)
    )
    phi = make_sensing_matrix(measurements, dimension, seed=7)
    sparse_blocks = CompressedSensingCodec(
        phi, dictionary, sparsity=sparsity, recovery="omp"
    ).roundtrip(blocks)
    baseline_blocks = CompressedSensingCodec(
        phi, dictionary, sparsity=sparsity, recovery="ridge"
    ).roundtrip(blocks)
    sparse_image = np.clip(_from_blocks(sparse_blocks, image.shape, block), 0.0, 1.0)
    baseline_image = np.clip(_from_blocks(baseline_blocks, image.shape, block), 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(image, sparse_image, data_range=1.0))
    ssim = float(structural_similarity(image, sparse_image, data_range=1.0))
    baseline_psnr = float(peak_signal_noise_ratio(image, baseline_image, data_range=1.0))
    if psnr < 22.0 or ssim < 0.65:
        raise AssertionError(f"natural-image CS quality is too low: {psnr:.2f} dB, {ssim:.3f}")
    if psnr < baseline_psnr + 5.0:
        raise AssertionError("image CS did not materially beat ridge projection")
    return {
        "status": "passed",
        "image": "skimage.data.camera resized to 64x64",
        "block": block,
        "measurements_per_block": measurements,
        "ambient_dimension": dimension,
        "sparsity": sparsity,
        "psnr_db": psnr,
        "ssim": ssim,
        "ridge_psnr_db": baseline_psnr,
    }


def validate_dense_embeddings() -> dict[str, float]:
    """Confirm that generic dense embeddings violate the sparse-vector prior."""
    rng = np.random.default_rng(3)
    embeddings = rng.standard_normal((256, 64))
    phi = make_sensing_matrix(32, 64, seed=7)
    dictionary = np.eye(64)
    sparse = CompressedSensingCodec(
        phi, dictionary, sparsity=8, recovery="omp"
    ).roundtrip(embeddings)
    baseline = CompressedSensingCodec(
        phi, dictionary, recovery="ridge"
    ).roundtrip(embeddings)
    sparse_error = _relative_error(embeddings, sparse)
    baseline_error = _relative_error(embeddings, baseline)
    sparse_cosine = _cosine_rows(embeddings, sparse)
    baseline_cosine = _cosine_rows(embeddings, baseline)

    # This is intentionally a negative result. If it changes, the fixture or
    # recovery assumptions changed and the scientific interpretation must be
    # revisited rather than silently declaring generic embedding CS successful.
    if sparse_error < 0.9 or sparse_error <= baseline_error:
        raise AssertionError("dense random embeddings unexpectedly look sparse")

    # Dense embeddings can still have exploitable low-rank corpus structure;
    # PCA is the appropriate comparison for that different assumption.
    factors = rng.standard_normal((256, 8))
    mixing = rng.standard_normal((8, 64))
    low_rank = factors @ mixing + 0.02 * rng.standard_normal((256, 64))
    centered = low_rank - low_rank.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    pca = centered @ right[:8].T @ right[:8] + low_rank.mean(axis=0)
    pca_error = _relative_error(low_rank, pca)
    if pca_error > 0.02:
        raise AssertionError("the low-rank embedding control was not recovered by PCA")
    return {
        "dimension": 64,
        "measurements": 32,
        "omp_relative_l2": sparse_error,
        "omp_median_cosine": sparse_cosine,
        "ridge_relative_l2": baseline_error,
        "ridge_median_cosine": baseline_cosine,
        "low_rank_pca_relative_l2": pca_error,
    }


def validate_low_rank_and_stream_methods() -> dict[str, float]:
    """Check exact low-rank controls and distinguish redundant from fresh streams."""
    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((10, 3)) @ rng.standard_normal((3, 8))
    errors = {}
    for name, codec in (
        ("svd", SVDCodec(3)),
        ("pca", PCACodec(3)),
        ("cur", CURCodec(3)),
    ):
        recovered = codec.decode(codec.encode(matrix, seed=2), seed=2)
        errors[f"{name}_relative_l2"] = _relative_error(matrix, recovered)
        if errors[f"{name}_relative_l2"] > 1e-10:
            raise AssertionError(f"{name.upper()} failed an exact rank-3 control")

    structured = synthetic_stream(
        64, 16, latent_rank=3, noise_std=0.0, smoothness=0.95, seed=2
    )
    fresh = rng.standard_normal((64, 16))
    structured_rank = effective_temporal_rank(structured, 0.95)
    fresh_rank = effective_temporal_rank(fresh, 0.95)
    structured_residual = causal_predictability_curve(structured, [1])[1]
    fresh_residual = causal_predictability_curve(fresh, [1])[1]
    if structured_rank > 3 or fresh_rank < 10:
        raise AssertionError("temporal effective-rank diagnostic inverted its controls")
    if structured_residual >= 0.2 or fresh_residual <= 0.7:
        raise AssertionError("causal stream diagnostic inverted its controls")
    return {
        **errors,
        "structured_effective_rank_95": structured_rank,
        "fresh_effective_rank_95": fresh_rank,
        "structured_order1_residual_ratio": structured_residual,
        "fresh_order1_residual_ratio": fresh_residual,
    }


def run_validation(require_natural_image: bool = False) -> dict[str, object]:
    report: dict[str, object] = {
        "sparse_image_coefficients": validate_sparse_image_coefficients(),
        "natural_image": validate_natural_image(),
        "dense_embeddings": validate_dense_embeddings(),
        "low_rank_and_stream": validate_low_rank_and_stream_methods(),
    }
    if require_natural_image and report["natural_image"].get("status") != "passed":
        raise RuntimeError("natural-image validation requires scikit-image")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="outputs/validation/sensing_behavior.json"
    )
    parser.add_argument("--require-natural-image", action="store_true")
    args = parser.parse_args()
    report = run_validation(args.require_natural_image)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
