from __future__ import annotations

import json
from collections.abc import Sequence

from .types import Message


def serialize_messages(messages: Sequence[Message]) -> bytes:
    return json.dumps(
        [message.to_dict() for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def deserialize_messages(data: bytes | str) -> list[Message]:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    values = json.loads(data)
    if not isinstance(values, list):
        raise ValueError("message payload must be a JSON list")
    return [Message.from_dict(value) for value in values]
