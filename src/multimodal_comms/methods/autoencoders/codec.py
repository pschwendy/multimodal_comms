from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from multimodal_comms.core import Codec, CommunicationMethod, Message, MethodContext, Transmission
from multimodal_comms.methods._wire import message_raw_bytes


class PassthroughCodec:
    """Tiny deterministic codec used for registry tests and dependency-free smoke runs."""

    def encode(self, value: Any, *, seed: int = 0) -> Any:
        return value

    def decode(self, code: Any, *, seed: int = 0) -> Any:
        return code


@dataclass(frozen=True, slots=True)
class AutoencoderConfig:
    samples: int = 1

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be positive")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, bytes):
        import base64

        return {"__bytes__": base64.b64encode(value).decode()}
    return value


def _from_json(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    if isinstance(value, dict) and "__bytes__" in value:
        import base64

        return base64.b64decode(value["__bytes__"])
    return value


class AutoencoderMethod(CommunicationMethod):
    """Sampled-latent wrapper around an injected modality-neutral codec."""

    variant = "sampled_latent"

    def __init__(self, config: AutoencoderConfig | None = None, codec: Codec | None = None):
        self.config = config or AutoencoderConfig()
        self.codec = codec or PassthroughCodec()

    def encode(self, messages: Sequence[Message], context: MethodContext) -> Transmission:
        entries = []
        for i, message in enumerate(messages):
            seed = context.seed + i
            codes = [
                _json_value(self.codec.encode(message.content, seed=seed + j))
                for j in range(self.config.samples)
            ]
            entries.append(
                {
                    "message": {k: v for k, v in message.to_dict().items() if k != "content"},
                    "codes": codes,
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
            metadata={"variant": self.variant, "samples": self.config.samples},
        )

    def decode(self, transmission: Transmission, context: MethodContext) -> list[Message]:
        entries = json.loads(transmission.payload)
        result = []
        for i, entry in enumerate(entries):
            code = _from_json(entry["codes"][0])
            content = self.codec.decode(code, seed=context.seed + i)
            header = entry["message"]
            if isinstance(content, bytes):
                import base64

                header["content"] = base64.b64encode(content).decode()
                header["content_encoding"] = "base64"
            else:
                header["content"] = content
                header["content_encoding"] = "utf-8"
            result.append(Message.from_dict(header))
        return result


class MWNOTAutoencoderMethod(AutoencoderMethod):
    """MWNOT sequence-codec variant, kept distinct from sampled latent AE."""

    variant = "mwnot_sequence"
