from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multimodal_comms.core import Message, MethodContext
from multimodal_comms.registry import create_method, get_method_spec

from .metrics import evaluate_roundtrip


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    method: str
    config: Mapping[str, Any] = field(default_factory=dict)
    messages: tuple[str, ...] = ("hello",)
    seed: int = 0
    output: str | None = None

    def __post_init__(self) -> None:
        get_method_spec(self.method)
        if not self.messages:
            raise ValueError("experiment requires at least one message")


def load_experiment(path: str | Path) -> ExperimentSpec:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError("YAML specs require the 'apps' extra (PyYAML)") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("experiment specification must be a mapping")
    allowed = {"method", "config", "messages", "seed", "output"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
    value["messages"] = tuple(value.get("messages", ("hello",)))
    return ExperimentSpec(**value)


def run_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    method = create_method(spec.method, spec.config)
    if get_method_spec(spec.method).kind != "communication":
        raise ValueError(f"runner requires a communication method, got {spec.method!r}")
    messages = [Message(f"agent-{i}", None, content, i) for i, content in enumerate(spec.messages)]
    result = evaluate_roundtrip(
        spec.method, method, messages, MethodContext(seed=spec.seed)
    ).to_dict()
    if spec.output:
        Path(spec.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
