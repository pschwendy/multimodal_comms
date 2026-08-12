from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommaScore:
    completion: float
    partial_credit: float


def grade_comma(
    response: Mapping[str, object], expected: Mapping[str, object], *, telehealth=False
) -> CommaScore:
    if not expected:
        return CommaScore(1.0, 1.0)
    correct = sum(response.get(key) == value for key, value in expected.items())
    partial = correct / len(expected)
    completion = float(correct == len(expected))
    return CommaScore(completion, partial if telehealth else completion)
