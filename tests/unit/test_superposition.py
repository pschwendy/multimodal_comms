import numpy as np
import pytest
from multimodal_comms.methods.superposition import OrthogonalKeyring, SecureBroadcast


def test_row_keys_break_plaintext_gram_invariant():
    latent = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    shared = OrthogonalKeyring(3, seed=2, row_keys=False).bind(latent, 0)
    separate = OrthogonalKeyring(3, seed=2, row_keys=True).bind(latent, 0)
    assert np.allclose(shared @ shared.T, latent @ latent.T)
    assert not np.allclose(separate @ separate.T, latent @ latent.T)


def test_nonce_freshness_and_single_receiver_recovery():
    broadcast = SecureBroadcast(3, {0: 123, 1: 456})
    code = np.array([1.0, 2.0, 3.0])
    packet = broadcast.encode({0: code}, nonce=10)
    assert np.allclose(broadcast.decode(packet, 0), code)
    with pytest.raises(ValueError, match="nonce reuse"):
        broadcast.encode({0: code}, nonce=10)
