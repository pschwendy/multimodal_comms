"""HiddenBench channel policy built on benchmark-neutral compressors.

The compressor implementations live under
:mod:`multimodal_comms.methods.text`; this module adds benchmark-specific
discussion-view policy.
"""

from __future__ import annotations

from typing import Any

from multimodal_comms.methods.text.compressors import *  # noqa: F401,F403
from multimodal_comms.methods.text.compressors import (
    Compressor,
    IdentityCompressor,
    build_compressor,
)

from .task import DiscussionMessage


class Channel:
    """Select and transform discussion messages for one receiver."""

    PROTOCOLS = ("full_history", "delta")

    def __init__(self, protocol: str = "full_history", compressor: Compressor | None = None):
        if protocol not in self.PROTOCOLS:
            raise ValueError(
                f"Unknown protocol: {protocol!r}. Available: {', '.join(self.PROTOCOLS)}"
            )
        self.protocol = protocol
        self.compressor = compressor or IdentityCompressor()
        self.stats = ChannelStats()

    def set_task_context(self, options: list[str]) -> None:
        self.compressor.set_task_context(options)

    def get_system_prompt_suffix(self) -> str:
        return self.compressor.get_system_prompt_suffix()

    def record_sent(self, content: str) -> None:
        self.stats.raw_messages += 1
        self.stats.raw_chars += len(content)

    def _select(
        self, history: list[DiscussionMessage], seen_count: int
    ) -> list[dict[str, Any]]:
        selected = history[seen_count:] if self.protocol == "delta" else history
        return [
            {"agent_id": item.agent_id, "round_num": item.round_num, "content": item.content}
            for item in selected
        ]

    def _finalize(
        self, view: list[dict[str, Any]], receiver_id: int
    ) -> list[dict[str, Any]]:
        compressed = self.compressor.compress(view, receiver_id)
        if view and not compressed:
            compressed = [view[-1]]
        self.stats.transmitted_messages += len(compressed)
        self.stats.transmitted_chars += sum(len(item["content"]) for item in compressed)
        return compressed

    def discussion_view(
        self,
        history: list[DiscussionMessage],
        receiver_id: int,
        seen_count: int,
    ) -> list[dict[str, Any]]:
        transmitted = self._finalize(self._select(history, seen_count), receiver_id)
        return self.compressor.decompress(transmitted, receiver_id)

    def final_view(
        self,
        history: list[DiscussionMessage],
        receiver_id: int,
        seen_count: int,
    ) -> list[dict[str, Any]]:
        transmitted = self._finalize(self._select(history, seen_count), receiver_id)
        return self.compressor.decompress(transmitted, receiver_id)


def build_channel(
    protocol: str = "full_history",
    compressor: str = "identity",
    **compressor_params: Any,
) -> Channel:
    return Channel(
        protocol=protocol,
        compressor=build_compressor(compressor, **compressor_params),
    )
