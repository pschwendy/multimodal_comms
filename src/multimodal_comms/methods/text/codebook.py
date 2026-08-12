from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from multimodal_comms.core import CommunicationMethod, Message, MethodContext, Transmission
from multimodal_comms.methods._wire import message_raw_bytes


@dataclass(frozen=True, slots=True)
class CodebookConfig:
    codebook: Mapping[str, str] | None = None
    online: bool = False
    min_occurrences: int = 2


class CodebookMethod(CommunicationMethod):
    """Lossless corpus or per-packet phrase codebook."""

    def __init__(self, config: CodebookConfig | None = None):
        self.config = config or CodebookConfig()

    def _book(self, messages: Sequence[Message]) -> dict[str, str]:
        if self.config.codebook is not None:
            return dict(self.config.codebook)
        if not self.config.online:
            return {}
        counts: dict[str, int] = {}
        for message in messages:
            if isinstance(message.content, str):
                for word in re.findall(r"\b[\w'-]{6,}\b", message.content):
                    counts[word] = counts.get(word, 0) + 1
        phrases = sorted(
            (word for word, n in counts.items() if n >= self.config.min_occurrences),
            key=lambda value: (-len(value), value),
        )
        return {phrase: f"§{i:x}§" for i, phrase in enumerate(phrases)}

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        book = self._book(messages)
        encoded = []
        for message in messages:
            value = message.to_dict()
            if isinstance(message.content, str):
                value["content"] = _replace(message.content, book)
            encoded.append(value)
        payload = json.dumps(
            {"book": book, "messages": encoded},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return Transmission(
            payload,
            message_raw_bytes(messages),
            len(payload),
            len(messages),
            metadata={"codebook_entries": len(book)},
        )

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        value = json.loads(transmission.payload)
        reverse = {token: phrase for phrase, token in value["book"].items()}
        result = []
        for item in value["messages"]:
            if item.get("content_encoding") == "utf-8":
                item["content"] = _replace(item["content"], reverse)
            result.append(Message.from_dict(item))
        return result


def _replace(text: str, replacements: Mapping[str, str]) -> str:
    for source in sorted(replacements, key=len, reverse=True):
        text = text.replace(source, replacements[source])
    return text
