from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from multimodal_comms.benchmarks.base import BenchmarkAdapter
from multimodal_comms.core import Message


class IAgentsAdapter(BenchmarkAdapter[dict[str, Any]]):
    def to_messages(self, native: Sequence[dict[str, Any]]) -> list[Message]:
        return [
            Message(
                str(item.get("sender", "agent")),
                item.get("receiver"),
                item["content"],
                int(item.get("round", 0)),
                {"benchmark": "iagents"},
            )
            for item in native
        ]

    def from_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        return [
            {"sender": m.sender, "receiver": m.receiver, "content": m.content, "round": m.round}
            for m in messages
        ]
