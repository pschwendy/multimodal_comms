from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HiddenBenchScore:
    pre_accuracy: float
    post_accuracy: float
    information_gain: float
    consensus: float


def grade_hiddenbench(before: Sequence[str], after: Sequence[str], answer: str) -> HiddenBenchScore:
    if len(before) != len(after):
        raise ValueError("pre/post decisions must align")
    count = len(after)
    pre = sum(value == answer for value in before) / count if count else 0.0
    post = sum(value == answer for value in after) / count if count else 0.0
    if not count:
        consensus = 0.0
    else:
        consensus = max(Counter(after).values()) / count
    # Accuracy gain is the empirical information gain used by the lightweight adapter.
    return HiddenBenchScore(pre, post, post - pre, consensus)
