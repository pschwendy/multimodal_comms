"""Run deterministic, offline optimization probes for every trained model family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error

from multimodal_comms.training import LinearTrainer, TrainingBatch


def _progress(losses: list[float], maximum_ratio: float = 0.75) -> dict[str, float]:
    values = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("optimization produced a non-finite loss")
    window = max(1, min(5, len(values) // 4))
    initial = float(values[:window].mean())
    final = float(values[-window:].mean())
    if not final < initial * maximum_ratio:
        raise AssertionError(
            f"loss did not progress enough: initial={initial:.6g}, final={final:.6g}"
        )
    return {
        "initial_loss": initial,
        "final_loss": final,
        "best_loss": float(values.min()),
        "reduction_fraction": 1.0 - final / initial,
    }


def probe_linear_training() -> dict[str, float]:
    rng = np.random.default_rng(2)
    inputs = rng.standard_normal((64, 5))
    targets = inputs @ rng.standard_normal((5, 3))
    trainer = LinearTrainer(5, 3, learning_rate=0.03, seed=3)
    batch = TrainingBatch(inputs, targets)
    return _progress([trainer.step(batch) for _ in range(120)], 0.1)


def probe_selector_models() -> dict[str, float]:
    rng = np.random.default_rng(5)
    features = rng.standard_normal((320, 12))
    classifier_target = (features[:, 0] - 0.8 * features[:, 1] > 0).astype(int)
    classifier = LogisticRegression(max_iter=1000).fit(features, classifier_target)
    trained_log_loss = float(
        log_loss(classifier_target, classifier.predict_proba(features)[:, 1])
    )
    baseline_log_loss = float(
        log_loss(classifier_target, np.full(len(features), classifier_target.mean()))
    )

    regression_target = features @ rng.standard_normal(12) + 0.03 * rng.standard_normal(320)
    regressor = Ridge(alpha=0.1).fit(features, regression_target)
    trained_mse = float(mean_squared_error(regression_target, regressor.predict(features)))
    baseline_mse = float(
        mean_squared_error(regression_target, np.full(len(features), regression_target.mean()))
    )
    if trained_log_loss >= 0.35 * baseline_log_loss or trained_mse >= 0.02 * baseline_mse:
        raise AssertionError("selector classifier/regressor failed their learnable controls")
    return {
        "classifier_baseline_log_loss": baseline_log_loss,
        "classifier_trained_log_loss": trained_log_loss,
        "regressor_baseline_mse": baseline_mse,
        "regressor_trained_mse": trained_mse,
    }


def _torch() -> object:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("learned-model probes require environment-full.yml") from exc
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return torch


def probe_latent_bottleneck() -> dict[str, float]:
    torch = _torch()
    from multimodal_comms.methods.packing.learned import PackedBottleneck

    coordinates = torch.randn(48, 3)
    basis = torch.randn(3, 8)
    latents = (coordinates @ basis).reshape(48, 2, 4)
    model = PackedBottleneck(2, 4, code_dim=8, width=16)
    model.fit_stats(latents)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    losses: list[float] = []
    for _ in range(120):
        code = model.encode(latents)
        loss = model.recon_loss(latents, code)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    result = _progress(losses, 0.35)
    with torch.no_grad():
        short = model.encode(latents, code_dim=4)
    result["short_code_tail_max"] = float(short[:, 4:].abs().max())
    if result["short_code_tail_max"] != 0.0:
        raise AssertionError("matryoshka truncation left a nonzero code tail")
    return result


def probe_image_bottleneck() -> dict[str, float]:
    torch = _torch()
    from multimodal_comms.methods.multimodal.latent_image import ImageBottleneck

    grid = torch.linspace(-1.0, 1.0, 8)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    images = torch.stack(
        [torch.sin((index + 1) * xx) * torch.cos((index % 3 + 1) * yy) for index in range(12)]
    ).unsqueeze(1)
    model = ImageBottleneck(code_dim=24, ch=8, latent_shape=(1, 8, 8))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    losses: list[float] = []
    for _ in range(90):
        recovered = model.decode(model.encode(images))
        loss = torch.nn.functional.mse_loss(recovered, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return _progress(losses, 0.55)


def probe_sequence_generator() -> dict[str, float]:
    torch = _torch()
    from multimodal_comms.methods.autoencoders.mwnot_generator import SequenceGeneratorEncoder

    hidden = torch.randn(20, 8, 8)
    target = torch.stack((hidden.mean(dim=1), hidden[:, 0]), dim=1)
    model = SequenceGeneratorEncoder(
        hidden_size=8,
        num_latents=2,
        lift_channels=8,
        embed_dim=16,
        wavelet_levels=2,
        num_heads=2,
        num_layers=1,
        ff_mult=2,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    losses: list[float] = []
    valid = torch.ones(20, 8, dtype=torch.bool)
    for _ in range(90):
        loss = torch.nn.functional.mse_loss(model(hidden, valid), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return _progress(losses, 0.55)


def probe_mwnot_operator() -> dict[str, float]:
    torch = _torch()
    from multimodal_comms.methods.autoencoders.mwnot.configs import MWNOTConfig
    from multimodal_comms.methods.autoencoders.mwnot.losses import MWNOTLoss
    from multimodal_comms.methods.autoencoders.mwnot.model import MWNOTModel

    config = MWNOTConfig(
        M=2,
        patch_size=3,
        poly_order=2,
        use_wavelets=False,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_mult=2,
        dropout=0.0,
        sort_nodes=False,
    )
    model = MWNOTModel(config)
    adjacency = torch.rand(10, 8, 8)
    adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
    p_target = torch.sigmoid(torch.randn(10, 2, 2))
    p_target = 0.5 * (p_target + p_target.transpose(-1, -2))
    l_target = torch.softmax(torch.randn(10, 2), dim=-1)
    criterion = MWNOTLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses: list[float] = []
    for _ in range(240):
        output = model(adjacency)
        loss = criterion(
            output["p_logits"], output["l_logits"], p_target, l_target
        )["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return _progress(losses, 0.8)


def probe_distribution_distillation() -> dict[str, float]:
    torch = _torch()
    features = torch.randn(48, 10)
    teacher = torch.softmax(features @ torch.randn(10, 7), dim=-1)
    student = torch.nn.Linear(10, 7)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.04)
    losses: list[float] = []
    for _ in range(90):
        log_probabilities = torch.log_softmax(student(features), dim=-1)
        # This is the forward cross-entropy used by gkd_train.gkd_loss after
        # the teacher's stored top-k logits have been renormalized.
        loss = -(teacher * log_probabilities).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return _progress(losses, 0.8)


def probe_token_policy() -> dict[str, float]:
    torch = _torch()
    from multimodal_comms.methods.text.token_filter_model import PolicyHead

    hidden = torch.randn(40, 8)
    target = (hidden[:, 0] - hidden[:, 1] > 0).float()
    head = PolicyHead(hidden_dim=8, ff_dim=16)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)
    losses: list[float] = []
    for _ in range(80):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(hidden), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return _progress(losses, 0.25)


def probe_policy_optimization() -> dict[str, float]:
    torch = _torch()
    logits = torch.nn.Parameter(torch.zeros(4))
    rewards = torch.tensor([-1.0, -0.2, 0.4, 1.0])
    optimizer = torch.optim.Adam([logits], lr=0.08)
    losses: list[float] = []
    initial_reward = float((torch.softmax(logits.detach(), dim=0) * rewards).sum())
    for _ in range(70):
        probabilities = torch.softmax(logits, dim=0)
        # Exact expectation of the score-function objective used by the GRPO
        # trainers. Enumerating the tiny action set removes sampling noise.
        loss = -(probabilities * rewards).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach() + 1.01))
    final_reward = float((torch.softmax(logits.detach(), dim=0) * rewards).sum())
    result = _progress(losses, 0.35)
    result.update(initial_expected_reward=initial_reward, final_expected_reward=final_reward)
    if final_reward < 0.9:
        raise AssertionError("policy failed to concentrate on the high-reward action")
    return result


def probe_feistel_keyring() -> dict[str, float]:
    torch = _torch()
    from multimodal_comms.methods.superposition.latent import FeistelKeyring

    keyring = FeistelKeyring(8, n_rounds=2, key_dim=4, hidden_dim=12)
    keyring._build_public_arch()
    parameters = []
    for function in keyring._round_fns:
        for parameter in function:
            parameter.requires_grad_(True)
            parameters.append(parameter)
    source = torch.randn(64, 8)
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    losses: list[float] = []
    for _ in range(80):
        bound = keyring.bind(source, 0)
        wrong = keyring.unbind(bound, 1)
        cosine = torch.nn.functional.cosine_similarity(wrong, source, dim=-1)
        right = keyring.unbind(bound, 0)
        loss = cosine.square().mean() + torch.nn.functional.mse_loss(right, source)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    result = _progress(losses, 0.6)
    with torch.no_grad():
        error = (keyring.unbind(keyring.bind(source, 0), 0) - source).abs().max()
    result["right_key_max_abs_error"] = float(error)
    if result["right_key_max_abs_error"] > 2e-4:
        raise AssertionError("Feistel training damaged right-key invertibility")
    return result


PROBES: dict[str, Callable[[], dict[str, float]]] = {
    "linear_supervised": probe_linear_training,
    "selector_models": probe_selector_models,
    "latent_bottleneck": probe_latent_bottleneck,
    "image_bottleneck": probe_image_bottleneck,
    "sequence_generator": probe_sequence_generator,
    "mwnot_operator": probe_mwnot_operator,
    "distribution_distillation": probe_distribution_distillation,
    "token_policy": probe_token_policy,
    "policy_optimization": probe_policy_optimization,
    "feistel_keyring": probe_feistel_keyring,
}


def run_validation(include_torch: bool = True) -> dict[str, object]:
    selected = PROBES if include_torch else {
        key: PROBES[key] for key in ("linear_supervised", "selector_models")
    }
    return {name: probe() for name, probe in selected.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="outputs/validation/training_dynamics.json"
    )
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    report = run_validation(include_torch=not args.base_only)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
