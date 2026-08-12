from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quantize_packet(packet: ArrayLike, bits: int = 8) -> tuple[NDArray[np.signedinteger], float]:
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    values = np.asarray(packet, dtype=np.float64)
    limit = 2 ** (bits - 1) - 1
    scale = float(np.max(np.abs(values)) / limit) if values.size and np.any(values) else 1.0
    dtype = np.int8 if bits == 8 else np.int16
    quantized = np.clip(np.rint(values / scale), -limit, limit).astype(dtype)
    return quantized, scale


def dequantize_packet(packet: ArrayLike, scale: float) -> NDArray[np.float64]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return np.asarray(packet, dtype=np.float64) * scale


def packet_capacity(packet_bytes: int, code_dim: int, bits: int = 8) -> int:
    if packet_bytes < 0 or code_dim <= 0 or bits <= 0:
        raise ValueError("invalid capacity inputs")
    return (packet_bytes * 8) // (code_dim * bits)
