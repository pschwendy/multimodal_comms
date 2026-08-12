"""Canonical adapter for the complete text-compressor collection.

The compressor kernels operate on small dictionaries. This adapter translates
between those dictionaries and the public ``Message``/``Transmission``
contract, so benchmarks and applications do not depend on a private wire
format.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from multimodal_comms.core import CommunicationMethod, Message, MethodContext, Transmission
from multimodal_comms.core.serialization import serialize_messages


@dataclass(frozen=True, slots=True)
class TextCompressorConfig:
    """Typed superset of parameters accepted by the text compressors.

    Only non-``None`` fields are passed to the selected constructor, so each
    implementation retains its scientifically recorded default settings.
    Constructor-signature validation prevents parameters from silently being
    accepted by the wrong method.
    """

    window_rounds: int | None = None
    threshold: float | None = None
    stateful: bool | None = None
    rate: float | None = None
    min_chars: int | None = None
    device: str | None = None
    model_path: str | None = None
    dedup: bool | None = None
    dedup_threshold: float | None = None
    tau: float | None = None
    max_new_tokens: int | None = None
    drop_floor: float | None = None
    min_words: int | None = None
    max_words: int | None = None
    min_count: int | None = None
    max_codes: int | None = None
    thr_hi: float | None = None
    thr_lo: float | None = None
    stack: str | None = None
    repserver_url: str | None = None
    timeout: float | None = None
    max_length: int | None = None
    num_latents: int | None = None
    codebook_path: str | None = None
    eps: float | None = None
    ladder: tuple[str, ...] | None = None
    max_context_tokens: int | None = None
    max_corrections_to_search: int | None = None
    key_seed: int | None = None
    key_mode: str | None = None
    max_slots: int | None = None

    def values_for(self, compressor_type: type) -> dict[str, Any]:
        values = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not None
        }
        accepted = set(inspect.signature(compressor_type).parameters)
        unknown = set(values) - accepted
        if unknown:
            raise ValueError(
                f"{compressor_type.__name__} does not accept: {', '.join(sorted(unknown))}"
            )
        return values


class TextCompressorMethod(CommunicationMethod):
    """Run one text compressor through the canonical API.

    A ready compressor may be injected for tests or custom model providers.
    Otherwise the registered constructor is used and heavy model loading
    remains lazy.
    """

    def __init__(
        self,
        method_id: str,
        config: TextCompressorConfig | None = None,
        *,
        compressor: Any | None = None,
    ) -> None:
        from .compressors import COMPRESSOR_REGISTRY

        if method_id not in COMPRESSOR_REGISTRY:
            raise ValueError(f"unknown text compressor: {method_id}")
        self.method_id = method_id
        self.config = config or TextCompressorConfig()
        self._compressor_type = COMPRESSOR_REGISTRY[method_id]
        self._parameters = self.config.values_for(self._compressor_type)
        self._injected = compressor
        self._compressor = compressor or self._compressor_type(**self._parameters)
        self._context_initialized = False
        self._episode_marker: Any = None

    def reset(self) -> None:
        self._compressor = self._injected or self._compressor_type(**self._parameters)
        self._context_initialized = False
        self._episode_marker = None

    def _prepare_episode(self, context: MethodContext) -> None:
        marker = context.shared.get("episode_id")
        if not self._context_initialized or marker != self._episode_marker:
            options = [str(value) for value in context.shared.get("options", ())]
            self._compressor.set_task_context(options)
            self._context_initialized = True
            self._episode_marker = marker

    @staticmethod
    def _sender_table(messages: Sequence[Message], receiver: str | None) -> list[str]:
        senders = list(dict.fromkeys(message.sender for message in messages))
        if receiver is not None and receiver not in senders:
            senders.append(receiver)
        return senders

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        self._prepare_episode(context)
        senders = self._sender_table(messages, context.receiver)
        sender_ids = {sender: index for index, sender in enumerate(senders)}
        receiver_id = sender_ids.get(context.receiver or "", -1)
        native: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message.content, str):
                raise TypeError(f"{self.method_id} accepts text messages only")
            native.append(
                {
                    "agent_id": sender_ids[message.sender],
                    "round_num": message.round,
                    "content": message.content,
                }
            )
        compressed = self._compressor.compress(native, receiver_id)
        envelope = {
            "method": self.method_id,
            "receiver_id": receiver_id,
            "senders": senders,
            "messages": compressed,
        }
        payload = json.dumps(
            envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return Transmission(
            payload=payload,
            raw_bytes=len(serialize_messages(messages)),
            wire_bytes=len(payload),
            message_count=len(messages),
            metadata={"method": self.method_id, "compressed_messages": len(compressed)},
        )

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        envelope = json.loads(bytes(transmission.payload).decode("utf-8"))
        native = self._compressor.decompress(
            envelope["messages"], int(envelope["receiver_id"])
        )
        senders = envelope["senders"]
        result: list[Message] = []
        for message in native:
            sender_id = int(message.get("agent_id", -1))
            sender = senders[sender_id] if 0 <= sender_id < len(senders) else f"agent_{sender_id}"
            result.append(
                Message(
                    sender=sender,
                    receiver=context.receiver,
                    content=str(message["content"]),
                    round=int(message.get("round_num", context.round)),
                )
            )
        return result


def build_text_method(
    method_id: str,
    config: TextCompressorConfig,
    injected: Mapping[str, Any],
) -> TextCompressorMethod:
    return TextCompressorMethod(method_id, config, **dict(injected))
