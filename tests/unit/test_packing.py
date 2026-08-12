import numpy as np
import pytest
from multimodal_comms.methods.packing import (
    BlockPacker,
    FramePacker,
    RotorPacker,
    dequantize_packet,
    quantize_packet,
)


def test_block_and_rotor_recover_and_have_equal_fidelity():
    codes = {0: np.array([1.0, 2.0]), 1: np.array([-3.0, 4.0])}
    block = BlockPacker(8, 2)
    rotor = RotorPacker(8, 2, layout_seed=4)
    for slot, code in codes.items():
        assert np.array_equal(block.unpack(block.pack(codes), slot), code)
        assert np.allclose(rotor.unpack(rotor.pack(codes), slot), code, atol=1e-10)


def test_hard_capacity_and_frame_overload():
    with pytest.raises(ValueError, match="capacity"):
        BlockPacker(4, 2).pack({2: np.ones(2)})
    frame = FramePacker(4, 2, seed=3)
    packet = frame.pack({i: np.ones(2) for i in range(5)})
    assert packet.shape == (4,)
    assert frame.unpack(packet, 4).shape == (2,)


def test_quantization_bounds_and_error():
    values = np.linspace(-10, 10, 100)
    quantized, scale = quantize_packet(values, 8)
    assert quantized.min() >= -127 and quantized.max() <= 127
    reconstructed = dequantize_packet(quantized, scale)
    assert np.max(np.abs(values - reconstructed)) <= scale / 2 + 1e-12
