from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from multimodal_comms.core import CommunicationMethod, Message, MethodContext, Transmission
from multimodal_comms.methods._wire import decode_message_transmission, message_transmission


@dataclass(frozen=True, slots=True)
class ExtractiveConfig:
    rate: float = 0.5
    min_chars: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")


class ExtractiveMethod(CommunicationMethod):
    """Generic deterministic extractive core for learned/token/saliency selectors.

    A score function can be injected. It receives a token and context; no task
    label or benchmark object enters this layer.
    """

    def __init__(
        self,
        config: ExtractiveConfig | None = None,
        scorer: Callable[[str, MethodContext], float] | None = None,
    ):
        self.config = config or ExtractiveConfig()
        self.scorer = scorer or (lambda token, context: float(len(token)))

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        output = []
        for message in messages:
            if not isinstance(message.content, str) or len(message.content) < self.config.min_chars:
                output.append(message)
                continue
            parts = re.findall(r"\s+|\S+", message.content)
            tokens = [part for part in parts if not part.isspace()]
            keep_n = max(1, round(len(tokens) * self.config.rate)) if tokens else 0
            ranked = sorted(range(len(tokens)), key=lambda i: (-self.scorer(tokens[i], context), i))
            kept = set(ranked[:keep_n])
            cursor = 0
            selected: list[str] = []
            for part in parts:
                if part.isspace():
                    if selected and not selected[-1].isspace():
                        selected.append(" ")
                else:
                    if cursor in kept:
                        selected.append(part)
                    cursor += 1
            content = "".join(selected).strip()
            output.append(
                Message(message.sender, message.receiver, content, message.round, message.metadata)
            )
        return message_transmission(output, source=messages, metadata={"rate": self.config.rate})

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        return decode_message_transmission(transmission)


class RewriterMethod(CommunicationMethod):
    """Injected text rewriter; the core never loads an LLM."""

    def __init__(self, rewriter: Callable[[str, MethodContext], str] | None = None):
        self.rewriter = rewriter or (lambda text, context: text)

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        output = [
            Message(
                m.sender,
                m.receiver,
                self.rewriter(m.content, context) if isinstance(m.content, str) else m.content,
                m.round,
                m.metadata,
            )
            for m in messages
        ]
        return message_transmission(output, source=messages)

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        return decode_message_transmission(transmission)
