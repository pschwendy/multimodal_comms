from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .keyrings import OrthogonalKeyring


@dataclass(slots=True)
class SuperpositionPacker:
    """Overlapping-code multiplexing by keyed addition.

    Unlike Block/Rotor, every code occupies the same dimensions and unbinding
    leaves crosstalk from all other slots. This is soft-capacity superposition.
    """

    dim: int
    seed: int | dict[int, int] = 1234
    nonce: int | None = None
    row_keys: bool = True
    keyring: OrthogonalKeyring = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.keyring = OrthogonalKeyring(self.dim, self.seed, self.nonce, self.row_keys)

    def pack(self, codes: Mapping[int, ArrayLike], keys=None) -> NDArray[np.float64]:
        packet: NDArray[np.float64] | None = None
        for slot, code in codes.items():
            bound = self.keyring.bind(code, slot)
            if packet is None:
                packet = bound.copy()
            else:
                packet += bound
        if packet is None:
            return np.zeros(self.dim)
        return packet

    def unpack(self, packet: ArrayLike, slot: int, key=None) -> NDArray[np.float64]:
        return self.keyring.unbind(packet, slot)
