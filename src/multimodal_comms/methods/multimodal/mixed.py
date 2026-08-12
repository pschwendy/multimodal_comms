from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any


class MixedPacketCodec:
    """Typed packet for text, bytes, and JSON-compatible metadata."""

    def encode(self, value: Mapping[str, Any], *, seed: int = 0) -> bytes:
        items = {}
        for key, item in value.items():
            if isinstance(item, bytes):
                items[key] = {"type": "bytes", "value": base64.b64encode(item).decode()}
            elif isinstance(item, str):
                items[key] = {"type": "text", "value": item}
            else:
                items[key] = {"type": "json", "value": item}
        return json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

    def decode(self, code: bytes, *, seed: int = 0) -> dict[str, Any]:
        items = json.loads(code)
        return {
            key: base64.b64decode(item["value"]) if item["type"] == "bytes" else item["value"]
            for key, item in items.items()
        }
