"""Paths and credentials for the Collab-Overcooked application.

Nothing is read from the process working directory and credentials are never
stored in the repository.  The old key-file convention is replaced
by standard environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROMPT_DIR = APP_ROOT / "prompts"


def openai_keys() -> list[str]:
    value = os.environ.get("OPENAI_API_KEYS") or os.environ.get("OPENAI_API_KEY", "")
    return [key.strip() for key in value.split(",") if key.strip()]


def require_openai_key() -> str:
    keys = openai_keys()
    if not keys:
        raise RuntimeError("set OPENAI_API_KEY (or comma-separated OPENAI_API_KEYS)")
    return keys[0]


def deepseek_settings() -> tuple[str, str]:
    return (
        os.environ.get("DEEPSEEK_API_KEY", ""),
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
