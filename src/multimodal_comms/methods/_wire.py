from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from multimodal_comms.core import Message, Transmission
from multimodal_comms.core.serialization import deserialize_messages, serialize_messages
from multimodal_comms.core.types import payload_size


def message_raw_bytes(messages: Sequence[Message]) -> int:
    return sum(payload_size(message.content) for message in messages)


def message_transmission(
    messages: Sequence[Message],
    *,
    source: Sequence[Message] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Transmission:
    payload = serialize_messages(messages)
    original = messages if source is None else source
    return Transmission(
        payload=payload,
        raw_bytes=message_raw_bytes(original),
        wire_bytes=len(payload),
        message_count=len(original),
        metadata=metadata or {},
    )


def decode_message_transmission(transmission: Transmission) -> list[Message]:
    if not isinstance(transmission.payload, bytes | str):
        raise TypeError("expected a serialized message payload")
    return deserialize_messages(transmission.payload)
