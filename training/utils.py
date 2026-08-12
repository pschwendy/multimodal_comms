"""Small task-neutral helpers shared by data and training programs."""

from __future__ import annotations

import json
import re


def parse_choice(response: str, options: list[str]) -> tuple[str, str]:
    """Extract a selected option and rationale from a model response."""

    try:
        match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            vote = str(data.get("vote", "")).strip()
            rationale = str(data.get("rationale", "")).strip()
            vote_lower = vote.lower()
            for option in options:
                if option.lower() == vote_lower or option.lower() in vote_lower:
                    return option, rationale
            return vote, rationale
    except json.JSONDecodeError:
        pass

    response_lower = response.lower()
    for option in options:
        if option.lower() in response_lower:
            return option, response
    return "", response
