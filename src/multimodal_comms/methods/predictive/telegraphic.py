from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from multimodal_comms.core import CommunicationMethod, Message, MethodContext, Transmission
from multimodal_comms.methods._wire import decode_message_transmission, message_transmission

_STOP = frozenset(
    (
        "a an and are as at be been by for from in is it of on or " "that the this to was were with"
    ).split()
)


@dataclass(frozen=True, slots=True)
class TelegraphicConfig:
    keep_punctuation: bool = True


class TelegraphicMethod(CommunicationMethod):
    def __init__(self, config: TelegraphicConfig | None = None):
        self.config = config or TelegraphicConfig()

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        result = []
        for message in messages:
            content: str | bytes
            if isinstance(message.content, str):
                tokens = re.findall(r"\w+|[^\w\s]", message.content, flags=re.UNICODE)
                tokens = [
                    token
                    for token in tokens
                    if token.casefold() not in _STOP
                    and (self.config.keep_punctuation or token.isalnum())
                ]
                content = " ".join(tokens)
                content = re.sub(r"\s+([,.;:!?])", r"\1", content)
            else:
                content = message.content
            result.append(
                Message(message.sender, message.receiver, content, message.round, message.metadata)
            )
        return message_transmission(result, source=messages)

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        return decode_message_transmission(transmission)
