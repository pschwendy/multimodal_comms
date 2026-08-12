from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from multimodal_comms.benchmarks.base import BenchmarkAdapter
from multimodal_comms.core import Message


class CollabOvercookedAdapter(BenchmarkAdapter[dict[str, Any]]):
    granularity = "episode"

    def to_messages(self, native: Sequence[dict[str, Any]]) -> list[Message]:
        return [
            Message(
                str(item.get("agent", "agent")),
                item.get("receiver"),
                item["content"],
                int(item.get("timestep", 0)),
                {"benchmark": "collab_overcooked", "granularity": self.granularity},
            )
            for item in native
        ]

    def from_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        return [
            {
                "agent": m.sender,
                "receiver": m.receiver,
                "content": m.content,
                "timestep": m.round,
                "granularity": self.granularity,
            }
            for m in messages
        ]


class EpisodeAdapter(CollabOvercookedAdapter):
    granularity = "episode"


class TimestepAdapter(CollabOvercookedAdapter):
    granularity = "timestep"
