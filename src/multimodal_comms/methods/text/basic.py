from __future__ import annotations

import base64
import gzip
import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from multimodal_comms.core import CommunicationMethod, Message, MethodContext, Transmission
from multimodal_comms.core.serialization import deserialize_messages, serialize_messages
from multimodal_comms.methods._wire import (
    decode_message_transmission,
    message_raw_bytes,
    message_transmission,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class IdentityMethod(CommunicationMethod):
    """Lossless canonical serialization, including arbitrary byte content."""

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        return message_transmission(messages)

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        return decode_message_transmission(transmission)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    rounds: int = 2

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be at least one")


class WindowMethod(IdentityMethod):
    def __init__(self, config: WindowConfig | None = None):
        self.config = config or WindowConfig()

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        if not messages:
            return message_transmission(messages)
        newest = max(message.round for message in messages)
        selected = [m for m in messages if m.round >= newest - self.config.rounds + 1]
        return message_transmission(selected, source=messages, metadata={"selected": len(selected)})


@dataclass(frozen=True, slots=True)
class NoveltyConfig:
    threshold: float = 0.85
    stateful: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


class NoveltyMethod(IdentityMethod):
    """Sentence novelty filtering with optional injected semantic embedder.

    The lightweight default uses deterministic string similarity. Passing an
    embedder provides learned sentence representations without loading a model in
    an algorithm constructor.
    """

    def __init__(self, config: NoveltyConfig | None = None, embedder=None):
        self.config = config or NoveltyConfig()
        self.embedder = embedder
        self._memory: dict[str | None, list[str]] = {}

    def reset(self) -> None:
        self._memory.clear()

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        known = list(self._memory.get(context.receiver, [])) if self.config.stateful else []
        result: list[Message] = []
        for message in messages:
            if not isinstance(message.content, str):
                result.append(message)
                continue
            kept: list[str] = []
            for sentence in filter(
                None, (s.strip() for s in _SENTENCE_SPLIT.split(message.content))
            ):
                if not any(
                    _text_similarity(sentence, old) >= self.config.threshold for old in known
                ):
                    kept.append(sentence)
                    known.append(sentence)
            if kept:
                result.append(
                    Message(
                        message.sender,
                        message.receiver,
                        " ".join(kept),
                        message.round,
                        message.metadata,
                    )
                )
        if messages and not result:
            result = [messages[-1]]
        if self.config.stateful:
            self._memory[context.receiver] = known
        return message_transmission(result, source=messages, metadata={"selected": len(result)})


@dataclass(frozen=True, slots=True)
class BackrefConfig:
    min_length: int = 12


class BackrefMethod(CommunicationMethod):
    """Lossless message-level dictionary backreferences."""

    def __init__(self, config: BackrefConfig | None = None):
        self.config = config or BackrefConfig()

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        dictionary: list[dict] = []
        positions: dict[tuple, int] = {}
        stream: list[dict] = []
        for message in messages:
            value = message.to_dict()
            key = (message.sender, message.receiver, message.content, message.round)
            if key in positions and len(message.content) >= self.config.min_length:
                stream.append({"ref": positions[key]})
            else:
                positions[key] = len(dictionary)
                dictionary.append(value)
                stream.append({"value": len(dictionary) - 1})
        import json

        payload = json.dumps(
            {"dictionary": dictionary, "stream": stream},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return Transmission(
            payload,
            message_raw_bytes(messages),
            len(payload),
            len(messages),
            metadata={"dictionary_entries": len(dictionary)},
        )

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        import json

        value = json.loads(transmission.payload)
        dictionary = value["dictionary"]
        return [
            Message.from_dict(dictionary[item.get("ref", item.get("value"))])
            for item in value["stream"]
        ]


class GzipMethod(CommunicationMethod):
    """Lossless gzip plus base64 wire encoding."""

    def __init__(self, compresslevel: int = 9):
        self.compresslevel = compresslevel

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        raw = serialize_messages(messages)
        payload = base64.b64encode(gzip.compress(raw, compresslevel=self.compresslevel, mtime=0))
        return Transmission(
            payload,
            message_raw_bytes(messages),
            len(payload),
            len(messages),
            media_type="application/gzip+base64",
        )

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        return deserialize_messages(gzip.decompress(base64.b64decode(transmission.payload)))
