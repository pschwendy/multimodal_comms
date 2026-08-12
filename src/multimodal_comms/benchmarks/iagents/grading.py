from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IAgentsScore:
    exact: float
    independent_regrade: float


def _normalize(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def grade_iagents(
    response: str, answer: str, regrader: Callable[[str, str], bool] | None = None
) -> IAgentsScore:
    exact = float(_normalize(response) == _normalize(answer))
    independent = float(regrader(response, answer)) if regrader else exact
    return IAgentsScore(exact, independent)
