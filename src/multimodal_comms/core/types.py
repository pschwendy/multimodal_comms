from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def payload_size(value: Any) -> int:
    """Return canonical UTF-8/wire bytes without confusing characters and bytes."""
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Message:
    """Canonical message. Receiver is ``None`` for broadcast."""

    sender: str
    receiver: str | None
    content: str | bytes
    round: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sender, str) or not self.sender:
            raise ValueError("sender must be a non-empty string")
        if self.receiver is not None and not isinstance(self.receiver, str):
            raise TypeError("receiver must be a string or None")
        if not isinstance(self.content, str | bytes):
            raise TypeError("content must be str or bytes")
        if self.round < 0:
            raise ValueError("round must be non-negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        is_bytes = isinstance(self.content, bytes)
        if isinstance(self.content, bytes):
            content = base64.b64encode(self.content).decode("ascii")
        else:
            content = self.content
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": content,
            "content_encoding": "base64" if is_bytes else "utf-8",
            "round": self.round,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        content = value["content"]
        if value.get("content_encoding") == "base64":
            content = base64.b64decode(content)
        return cls(
            sender=str(value["sender"]),
            receiver=value.get("receiver"),
            content=content,
            round=int(value.get("round", 0)),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class MethodContext:
    """Task-independent receiver state and deterministic seed."""

    receiver: str | None = None
    shared: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    round: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared", MappingProxyType(dict(self.shared)))

    def for_receiver(self, receiver: str, *, round: int | None = None) -> MethodContext:
        return MethodContext(
            receiver, self.shared, self.seed, self.round if round is None else round
        )


@dataclass(frozen=True, slots=True)
class Traffic:
    raw_messages: int
    raw_bytes: int
    transmitted_messages: int
    transmitted_bytes: int

    @property
    def byte_ratio(self) -> float:
        return self.transmitted_bytes / self.raw_bytes if self.raw_bytes else 0.0


@dataclass(frozen=True, slots=True)
class Transmission:
    """Wire payload and auditable traffic accounting."""

    payload: Any
    raw_bytes: int
    wire_bytes: int
    message_count: int
    media_type: str = "application/json"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(self.raw_bytes, self.wire_bytes, self.message_count) < 0:
            raise ValueError("traffic counts must be non-negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def traffic(self) -> Traffic:
        return Traffic(self.message_count, self.raw_bytes, self.message_count, self.wire_bytes)
