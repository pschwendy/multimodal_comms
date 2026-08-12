from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


def derive_row_seed(base_seed: int, row: int) -> int:
    """Domain-separated independent key seed for one latent row."""
    digest = hashlib.blake2b(
        f"{base_seed}:{row}".encode(), digest_size=8, person=b"rowkey-v1"
    ).digest()
    return int.from_bytes(digest, "big")


def mint_receiver_secrets(n_slots: int, start_slot: int = 0) -> dict[int, int]:
    if n_slots < 0:
        raise ValueError("n_slots must be non-negative")
    return {slot: secrets.randbits(63) for slot in range(start_slot, start_slot + n_slots)}


@dataclass(slots=True)
class OrthogonalKeyring:
    """Deterministic orthogonal keys with optional independent per-row keys."""

    dim: int
    seed: int | dict[int, int] = 1234
    nonce: int | None = None
    row_keys: bool = True
    mode: str = "qr"
    _cache: dict[tuple[int, int], NDArray[np.float64]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if self.mode not in {"qr", "sign"}:
            raise ValueError("mode must be 'qr' or 'sign'")
        self._cache: dict[tuple[int, int], NDArray[np.float64]] = {}

    def _base(self, slot: int) -> int:
        if isinstance(self.seed, dict):
            if slot not in self.seed:
                raise KeyError(f"slot {slot} is not enrolled")
            base = self.seed[slot]
        else:
            base = (self.seed * 1_000_003 + slot * 7_919) % (2**63 - 1)
        if self.nonce is None:
            return base
        return int.from_bytes(
            hashlib.blake2b(
                f"{base}:{self.nonce}".encode(), digest_size=8, person=b"nonce-v1"
            ).digest(),
            "big",
        )

    def key(self, slot: int, row: int = 0) -> NDArray[np.float64]:
        key_row = row if self.row_keys else 0
        cache_key = slot, key_row
        if cache_key not in self._cache:
            seed = derive_row_seed(self._base(slot), key_row) if self.row_keys else self._base(slot)
            rng = np.random.default_rng(seed)
            if self.mode == "sign":
                value = rng.choice((-1.0, 1.0), self.dim)
            else:
                q, r = np.linalg.qr(rng.standard_normal((self.dim, self.dim)))
                value = q * np.sign(np.diag(r))[None, :]
            self._cache[cache_key] = value
        return self._cache[cache_key]

    def bind(self, code: ArrayLike, slot: int) -> NDArray[np.float64]:
        value = np.asarray(code, dtype=np.float64)
        rows = np.atleast_2d(value)
        bound = np.stack(
            [
                row * self.key(slot, i) if self.mode == "sign" else row @ self.key(slot, i)
                for i, row in enumerate(rows)
            ]
        )
        return bound[0] if value.ndim == 1 else bound

    def unbind(self, packet: ArrayLike, slot: int) -> NDArray[np.float64]:
        value = np.asarray(packet, dtype=np.float64)
        rows = np.atleast_2d(value)
        unbound = np.stack(
            [
                row * self.key(slot, i) if self.mode == "sign" else row @ self.key(slot, i).T
                for i, row in enumerate(rows)
            ]
        )
        return unbound[0] if value.ndim == 1 else unbound
