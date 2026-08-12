from __future__ import annotations

import zlib


class ImageZlibCodec:
    """Lossless byte-image transport; image interpretation is deliberately external."""

    def __init__(self, level: int = 9):
        self.level = level

    def encode(self, value: bytes, *, seed: int = 0) -> bytes:
        if not isinstance(value, bytes):
            raise TypeError("ImageZlibCodec expects encoded image bytes")
        return zlib.compress(value, self.level)

    def decode(self, code: bytes, *, seed: int = 0) -> bytes:
        return zlib.decompress(code)
