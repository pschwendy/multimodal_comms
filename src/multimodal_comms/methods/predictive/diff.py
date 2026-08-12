from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from multimodal_comms.core import CommunicationMethod, Message, MethodContext, Transmission
from multimodal_comms.methods._wire import message_raw_bytes


class BytePredictor(Protocol):
    """Shared deterministic predictor. Learned adapters can implement this protocol."""

    def predict(self, prefix: bytes, context: MethodContext) -> int: ...


class LastBytePredictor:
    def predict(self, prefix: bytes, context: MethodContext) -> int:
        return prefix[-1] if prefix else 32


@dataclass(frozen=True, slots=True)
class PredictiveDiffConfig:
    context_key: str = "predictive_prefix"


class PredictiveDiffMethod(CommunicationMethod):
    """Lossless run/correction coding against an injected shared predictor."""

    def __init__(
        self,
        config: PredictiveDiffConfig | None = None,
        predictor: BytePredictor | None = None,
    ):
        self.config = config or PredictiveDiffConfig()
        self.predictor = predictor or LastBytePredictor()

    def _prefix(self, context: MethodContext) -> bytes:
        value = context.shared.get(self.config.context_key, b"")
        return value.encode() if isinstance(value, str) else bytes(value)

    def _ops(self, target: bytes, context: MethodContext) -> list[list[int | str]]:
        prefix = self._prefix(context)
        ops: list[list[int | str]] = []
        run = 0
        for actual in target:
            predicted = self.predictor.predict(prefix, context)
            if predicted == actual:
                run += 1
            else:
                if run:
                    ops.append(["R", run])
                    run = 0
                ops.append(["C", actual])
            prefix += bytes([actual])
        if run:
            ops.append(["R", run])
        return ops

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        entries = []
        for message in messages:
            raw = (
                message.content.encode("utf-8")
                if isinstance(message.content, str)
                else message.content
            )
            entries.append(
                {
                    "header": {k: v for k, v in message.to_dict().items() if k != "content"},
                    "text": isinstance(message.content, str),
                    "ops": self._ops(raw, context),
                }
            )
        payload = json.dumps(
            entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return Transmission(
            payload,
            message_raw_bytes(messages),
            len(payload),
            len(messages),
            metadata={"lossless": True},
        )

    def _replay(self, ops: list[list[int | str]], context: MethodContext) -> bytes:
        prefix = self._prefix(context)
        output = bytearray()
        for kind, value in ops:
            if kind == "R":
                for _ in range(int(value)):
                    predicted = self.predictor.predict(prefix, context)
                    output.append(predicted)
                    prefix += bytes([predicted])
            elif kind == "C":
                actual = int(value)
                output.append(actual)
                prefix += bytes([actual])
            else:
                raise ValueError(f"unknown predictive op: {kind}")
        return bytes(output)

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        entries = json.loads(transmission.payload)
        result = []
        for entry in entries:
            raw = self._replay(entry["ops"], context)
            content = raw.decode("utf-8", errors="replace") if entry["text"] else raw
            header = entry["header"]
            header["content"] = base64.b64encode(raw).decode() if not entry["text"] else content
            result.append(Message.from_dict(header))
        return result


@dataclass(frozen=True, slots=True)
class RateDiffConfig(PredictiveDiffConfig):
    correction_rate: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.correction_rate <= 1.0:
            raise ValueError("correction_rate must be in [0, 1]")


class RateDiffMethod(PredictiveDiffMethod):
    """Lossy rate-controlled predictive diff; omitted corrections become predictions."""

    config: RateDiffConfig

    def __init__(self, config: RateDiffConfig | None = None, predictor=None):
        config = config or RateDiffConfig()
        super().__init__(config, predictor)
        self.config = config

    def _ops(self, target: bytes, context: MethodContext) -> list[list[int | str]]:
        full = super()._ops(target, context)
        corrections = [i for i, op in enumerate(full) if op[0] == "C"]
        keep_n = round(len(corrections) * self.config.correction_rate)
        keep = set(corrections[:keep_n])
        merged: list[list[int | str]] = []
        for i, op in enumerate(full):
            if op[0] == "C" and i not in keep:
                if merged and merged[-1][0] == "R":
                    merged[-1][1] = int(merged[-1][1]) + 1
                else:
                    merged.append(["R", 1])
            else:
                merged.append(op)
        return merged

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        tx = super().encode(messages, context)
        return Transmission(
            tx.payload,
            tx.raw_bytes,
            tx.wire_bytes,
            tx.message_count,
            tx.media_type,
            {
                "lossless": self.config.correction_rate == 1.0,
                "correction_rate": self.config.correction_rate,
            },
        )
