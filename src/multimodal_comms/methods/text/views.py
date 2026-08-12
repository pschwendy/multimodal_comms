from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from multimodal_comms.core import Message, MethodContext


class ChannelView(Protocol):
    def select(self, messages: Sequence[Message], context: MethodContext) -> list[Message]: ...
    def reset(self) -> None: ...


class FullHistoryView:
    def select(self, messages: Sequence[Message], context: MethodContext) -> list[Message]:
        return list(messages)

    def reset(self) -> None:
        pass


@dataclass(slots=True)
class DeltaView:
    """Per-receiver unseen suffix policy, deliberately separate from compression."""

    _seen: dict[str | None, int] = field(default_factory=dict)

    def select(self, messages: Sequence[Message], context: MethodContext) -> list[Message]:
        start = self._seen.get(context.receiver, 0)
        selected = list(messages[start:])
        self._seen[context.receiver] = len(messages)
        return selected

    def reset(self) -> None:
        self._seen.clear()
