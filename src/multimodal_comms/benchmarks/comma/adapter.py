from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from multimodal_comms.benchmarks.base import BenchmarkAdapter
from multimodal_comms.core import Message


class CommaAdapter(BenchmarkAdapter[dict[str, Any]]):
    def to_messages(self, native: Sequence[dict[str, Any]]) -> list[Message]:
        return [
            Message(
                str(item.get("role", item.get("agent_id", "agent"))),
                item.get("receiver"),
                item["content"],
                int(item.get("turn", 0)),
                {"benchmark": "comma", "modality": item.get("modality", "text")},
            )
            for item in native
        ]

    def from_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        return [
            {
                "role": message.sender,
                "receiver": message.receiver,
                "content": message.content,
                "turn": message.round,
                "modality": message.metadata.get("modality", "text"),
            }
            for message in messages
        ]
