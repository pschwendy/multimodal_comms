"""Selector/rewriter GRPO-GKD workflow boundary.

Full frameworks are optional integrations; the package core accepts any object
with a ``step(TrainingBatch)`` method, which keeps smoke tests model-free.
"""

from .toy import LinearTrainer, TrainingBatch


def train_one_batch(trainer: LinearTrainer, batch: TrainingBatch) -> float:
    return trainer.step(batch)
