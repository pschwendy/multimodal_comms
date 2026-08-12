from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .types import Message, MethodContext, Transmission


class CommunicationMethod(ABC):
    """A task-independent communication transformation."""

    @abstractmethod
    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission: ...

    @abstractmethod
    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]: ...

    def reset(self) -> None:
        """Clear episode state. Stateless methods need no override."""
        return None


@runtime_checkable
class Codec(Protocol):
    """Modality-neutral learned representation codec."""

    def encode(self, value: Any, *, seed: int = 0) -> Any: ...
    def decode(self, code: Any, *, seed: int = 0) -> Any: ...


@runtime_checkable
class Packer(Protocol):
    def pack(self, codes: dict[int, Any], keys: dict[int, int] | None = None) -> Any: ...
    def unpack(self, packet: Any, slot: int, key: int | None = None) -> Any: ...


@runtime_checkable
class RepresentationProvider(Protocol):
    """Injected feature/model boundary used by selectors and learned codecs."""

    def embed(self, texts: Sequence[str]) -> Any: ...
