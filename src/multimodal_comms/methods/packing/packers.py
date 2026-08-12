from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(value: ArrayLike, width: int) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(result) < width:
        raise ValueError(f"code has width {len(result)}; expected at least {width}")
    return result[:width]


@dataclass(slots=True)
class BlockPacker:
    """Disjoint coordinate container: exact, zero crosstalk, hard capacity."""

    packet_dim: int
    code_dim: int

    def __post_init__(self) -> None:
        if not 0 < self.code_dim <= self.packet_dim:
            raise ValueError("require 0 < code_dim <= packet_dim")

    @property
    def capacity(self) -> int:
        return self.packet_dim // self.code_dim

    def _check(self, slot: int) -> None:
        if not 0 <= slot < self.capacity:
            raise ValueError(f"slot {slot} exceeds hard capacity {self.capacity}")

    def pack(self, codes: Mapping[int, ArrayLike], keys=None) -> NDArray[np.float64]:
        packet = np.zeros(self.packet_dim)
        for slot, code in codes.items():
            self._check(slot)
            packet[slot * self.code_dim : (slot + 1) * self.code_dim] = _vector(code, self.code_dim)
        return packet

    def unpack(self, packet: ArrayLike, slot: int, key=None) -> NDArray[np.float64]:
        self._check(slot)
        value = np.asarray(packet)
        return value[slot * self.code_dim : (slot + 1) * self.code_dim].copy()


@dataclass(slots=True)
class FramePacker:
    """Overlapping random frames: supports overload, with measurable crosstalk."""

    packet_dim: int
    code_dim: int
    seed: int = 1234
    nonce: int | None = None
    _frames: dict[tuple[int, int], NDArray[np.float64]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.code_dim <= self.packet_dim:
            raise ValueError("require 0 < code_dim <= packet_dim")
        self._frames: dict[tuple[int, int], NDArray[np.float64]] = {}

    def _seed(self, slot: int, key: int | None) -> int:
        material = f"{self.seed if key is None else key}:{slot}:{self.nonce}".encode()
        return int.from_bytes(
            hashlib.blake2b(material, digest_size=8, person=b"framepk1").digest(), "big"
        )

    def frame(self, slot: int, key: int | None = None) -> NDArray[np.float64]:
        cache_key = (slot, self._seed(slot, key))
        if cache_key not in self._frames:
            rng = np.random.default_rng(cache_key[1])
            q, r = np.linalg.qr(rng.standard_normal((self.packet_dim, self.code_dim)))
            q *= np.sign(np.diag(r))[None, :]
            self._frames[cache_key] = q.T
        return self._frames[cache_key]

    def pack(self, codes: Mapping[int, ArrayLike], keys=None) -> NDArray[np.float64]:
        packet = np.zeros(self.packet_dim)
        keys = keys or {}
        for slot, code in codes.items():
            packet += _vector(code, self.code_dim) @ self.frame(slot, keys.get(slot))
        return packet

    def unpack(self, packet: ArrayLike, slot: int, key=None) -> NDArray[np.float64]:
        return np.asarray(packet) @ self.frame(slot, key).T


@dataclass(slots=True)
class RotorPacker(BlockPacker):
    """Dense invertible rotation of block packing: exact up to hard capacity.

    It has identical fidelity and capacity to BlockPacker. This corrected
    characterization avoids treating an invertible coordinate change as extra
    information capacity.
    """

    layout_seed: int = 1234
    nonce: int | None = None
    _rotation: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super(RotorPacker, self).__post_init__()
        material = f"{self.layout_seed}:{self.nonce}".encode()
        seed = int.from_bytes(
            hashlib.blake2b(material, digest_size=8, person=b"rotorlay").digest(), "big"
        )
        rng = np.random.default_rng(seed)
        q, r = np.linalg.qr(rng.standard_normal((self.packet_dim, self.packet_dim)))
        self._rotation = q * np.sign(np.diag(r))[None, :]

    def pack(self, codes: Mapping[int, ArrayLike], keys=None) -> NDArray[np.float64]:
        return BlockPacker.pack(self, codes, keys) @ self._rotation

    def unpack(self, packet: ArrayLike, slot: int, key=None) -> NDArray[np.float64]:
        flat = np.asarray(packet) @ self._rotation.T
        return BlockPacker.unpack(self, flat, slot, key)


def build_packer(kind: str, packet_dim: int, code_dim: int, **kwargs):
    packers = {"block": BlockPacker, "frame": FramePacker, "rotor": RotorPacker}
    try:
        cls = packers[kind]
    except KeyError as error:
        raise ValueError(f"unknown packer {kind!r}; expected one of {sorted(packers)}") from error
    allowed = {item.name for item in fields(cls)}
    return cls(
        packet_dim, code_dim, **{key: value for key, value in kwargs.items() if key in allowed}
    )
