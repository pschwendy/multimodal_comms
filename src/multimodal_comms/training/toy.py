from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    inputs: NDArray[np.float64]
    targets: NDArray[np.float64]


class LinearTrainer:
    """Tiny injectable training smoke path shared by AE/selector workflows."""

    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.01, seed: int = 0):
        self.learning_rate = learning_rate
        self.weights = np.random.default_rng(seed).standard_normal((input_dim, output_dim)) * 0.01

    def step(self, batch: TrainingBatch) -> float:
        prediction = batch.inputs @ self.weights
        error = prediction - batch.targets
        loss = float(np.mean(error**2))
        gradient = (2.0 / batch.inputs.shape[0]) * batch.inputs.T @ error
        self.weights -= self.learning_rate * gradient
        return loss

    def save(self, path: str | Path) -> None:
        # Smoke checkpoints are explicit JSON and intentionally tiny; real model state is ignored.
        Path(path).write_text(
            json.dumps({"learning_rate": self.learning_rate, "weights": self.weights.tolist()})
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearTrainer:
        value = json.loads(Path(path).read_text())
        weights = np.asarray(value["weights"], dtype=np.float64)
        trainer = cls(weights.shape[0], weights.shape[1], value["learning_rate"])
        trainer.weights = weights
        return trainer
