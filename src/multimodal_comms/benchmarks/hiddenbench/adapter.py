from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from multimodal_comms.benchmarks.base import BenchmarkAdapter
from multimodal_comms.core import Message


class HiddenBenchAdapter(BenchmarkAdapter[dict[str, Any]]):
    def to_messages(self, native: Sequence[dict[str, Any]]) -> list[Message]:
        return [
            Message(
                str(item["agent_id"]),
                None,
                item["content"],
                int(item.get("round_num", 0)),
                {"benchmark": "hiddenbench"},
            )
            for item in native
        ]

    def from_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        return [
            {
                "agent_id": _numeric(message.sender),
                "round_num": message.round,
                "content": message.content,
            }
            for message in messages
        ]


def _numeric(value: str):
    try:
        return int(value)
    except ValueError:
        return value
