from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .multiplex import SuperpositionPacker


@dataclass(frozen=True, slots=True)
class SecurePacket:
    values: NDArray[np.float64]
    nonce: int
    slots: tuple[int, ...]


class SecureBroadcast:
    """Fresh-nonce convenience wrapper around private per-receiver keyrings."""

    def __init__(self, dim: int, receiver_secrets: Mapping[int, int]):
        self.dim = dim
        self.receiver_secrets = dict(receiver_secrets)
        self._issued: set[int] = set()

    def encode(self, codes: Mapping[int, ArrayLike], *, nonce: int | None = None) -> SecurePacket:
        nonce = secrets.randbits(63) if nonce is None else nonce
        if nonce in self._issued:
            raise ValueError("nonce reuse is forbidden for a receiver keyring")
        self._issued.add(nonce)
        packer = SuperpositionPacker(self.dim, self.receiver_secrets, nonce, row_keys=True)
        return SecurePacket(packer.pack(codes), nonce, tuple(sorted(codes)))

    def decode(self, packet: SecurePacket, slot: int) -> NDArray[np.float64]:
        if slot not in self.receiver_secrets:
            raise KeyError(f"receiver slot {slot} is not enrolled")
        packer = SuperpositionPacker(
            self.dim, {slot: self.receiver_secrets[slot]}, packet.nonce, row_keys=True
        )
        return packer.unpack(packet.values, slot)
