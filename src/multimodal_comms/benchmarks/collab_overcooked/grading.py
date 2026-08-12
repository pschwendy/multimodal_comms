from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CollabScore:
    success: float
    tes: float
    f1: float
    similarity: float
    redundancy: float
    collaboration: float


def grade_collaboration(
    predicted: Sequence[str], reference: Sequence[str], *, completed: bool
) -> CollabScore:
    predicted_set, reference_set = set(predicted), set(reference)
    overlap = len(predicted_set & reference_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(reference_set) if reference_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    redundancy = 1.0 - len(predicted_set) / len(predicted) if predicted else 0.0
    similarity = (
        overlap / len(predicted_set | reference_set) if predicted_set | reference_set else 1.0
    )
    success = float(completed)
    tes = recall
    collaboration = (success + f1 + similarity + (1.0 - redundancy)) / 4.0
    return CollabScore(success, tes, f1, similarity, redundancy, collaboration)
