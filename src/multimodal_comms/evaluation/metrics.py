from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from multimodal_comms.core import CommunicationMethod, Message, MethodContext


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    method: str
    messages: int
    raw_bytes: int
    wire_bytes: int
    byte_ratio: float
    exact_messages: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_roundtrip(
    identifier: str,
    method: CommunicationMethod,
    messages: Sequence[Message],
    context: MethodContext,
) -> EvaluationResult:
    transmission = method.encode(messages, context)
    decoded = method.decode(transmission, context)
    exact = sum(left == right for left, right in zip(messages, decoded, strict=False))
    return EvaluationResult(
        identifier,
        len(messages),
        transmission.raw_bytes,
        transmission.wire_bytes,
        transmission.wire_bytes / transmission.raw_bytes if transmission.raw_bytes else 0.0,
        exact,
    )
