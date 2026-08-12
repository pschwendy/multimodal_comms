from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from multimodal_comms.core import CommunicationMethod, Message, MethodContext, Traffic
from multimodal_comms.methods.text.views import ChannelView, FullHistoryView

Native = TypeVar("Native")


@dataclass(frozen=True, slots=True)
class AdapterResult(Generic[Native]):
    messages: list[Native]
    traffic: Traffic


class BenchmarkAdapter(ABC, Generic[Native]):
    """One conversion boundary per benchmark, shared by every method."""

    def __init__(self, method: CommunicationMethod, view: ChannelView | None = None):
        self.method = method
        self.view = view or FullHistoryView()

    @abstractmethod
    def to_messages(self, native: Sequence[Native]) -> list[Message]: ...

    @abstractmethod
    def from_messages(self, messages: Sequence[Message]) -> list[Native]: ...

    def transmit(self, native: Sequence[Native], context: MethodContext) -> AdapterResult[Native]:
        canonical = self.to_messages(native)
        selected = self.view.select(canonical, context)
        transmission = self.method.encode(selected, context)
        decoded = self.method.decode(transmission, context)
        return AdapterResult(
            self.from_messages(decoded),
            Traffic(
                raw_messages=len(canonical),
                raw_bytes=sum(
                    len(m.content.encode() if isinstance(m.content, str) else m.content)
                    for m in canonical
                ),
                transmitted_messages=transmission.message_count,
                transmitted_bytes=transmission.wire_bytes,
            ),
        )

    def reset(self) -> None:
        self.method.reset()
        self.view.reset()
