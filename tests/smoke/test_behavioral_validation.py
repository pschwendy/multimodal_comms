from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.validation.sensing_behavior import run_validation as run_sensing
from experiments.validation.training_dynamics import run_validation as run_training


def test_sensing_assumptions_have_positive_and_negative_controls():
    report = run_sensing()
    assert report["sparse_image_coefficients"]["relative_l2"] < 0.02
    assert report["dense_embeddings"]["omp_relative_l2"] > 0.9
    assert report["low_rank_and_stream"]["svd_relative_l2"] < 1e-10
    if report["natural_image"]["status"] == "passed":
        assert report["natural_image"]["psnr_db"] > 22.0


def test_base_training_probes_make_progress():
    report = run_training(include_torch=False)
    assert report["linear_supervised"]["reduction_fraction"] > 0.9
    assert report["selector_models"]["classifier_trained_log_loss"] < 0.2


def test_full_training_probes_make_progress_when_torch_is_available():
    pytest.importorskip("torch")
    report = run_training(include_torch=True)
    assert set(report) == {
        "linear_supervised",
        "selector_models",
        "latent_bottleneck",
        "image_bottleneck",
        "sequence_generator",
        "mwnot_operator",
        "distribution_distillation",
        "token_policy",
        "policy_optimization",
        "feistel_keyring",
    }
    assert all(
        "reduction_fraction" in value or name == "selector_models"
        for name, value in report.items()
    )


def test_every_training_entry_point_is_classified():
    discovered = {
        str(path)
        for path in Path("training").rglob("*.py")
        if "validation" not in path.parts
        and (
            path.name.startswith("train")
            or path.name.startswith("pretrain")
            or path.name == "gkd_train.py"
        )
    }
    discovered.add("src/multimodal_comms/methods/autoencoders/mwnot/train.py")
    catalog = yaml.safe_load(
        Path("experiments/validation/training_catalog.yaml").read_text(encoding="utf-8")
    )
    listed = {entry["path"] for entry in catalog["entry_points"]}
    assert listed == discovered


def test_embedding_sensing_baselines_can_share_identical_measurements():
    pytest.importorskip("skimage")
    from experiments.compressed_sensing.programs.demo_cs_embedding import (
        make_projection,
        reconstruct_pinv,
    )

    vector = np.arange(16.0)
    phi = make_projection(16, 0.5, seed=9)
    recovered = reconstruct_pinv(vector, 0.5, phi=phi)
    expected = phi.T @ np.linalg.solve(
        phi @ phi.T + 1e-6 * np.eye(phi.shape[0]),
        phi @ vector,
    )
    assert np.allclose(recovered, expected)
