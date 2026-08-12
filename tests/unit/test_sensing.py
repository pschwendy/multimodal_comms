import numpy as np
from multimodal_comms.methods.sensing import (
    CompressedSensingCodec,
    CURCodec,
    PCACodec,
    SVDCodec,
    make_sensing_matrix,
    make_dictionary,
)


def test_sensing_shapes_and_full_rank_roundtrip():
    phi = make_sensing_matrix(4, 4, seed=2)
    codec = CompressedSensingCodec(phi, np.eye(4), ridge=1e-12)
    value = np.array([1.0, 2.0, 3.0, 4.0])
    assert codec.encode(value).shape == (4,)
    assert np.allclose(codec.decode(codec.encode(value)), value, atol=1e-7)


def test_low_rank_codec_shapes():
    matrix = np.arange(20.0).reshape(4, 5)
    for codec in (SVDCodec(2), PCACodec(2), CURCodec(2)):
        decoded = codec.decode(codec.encode(matrix, seed=3))
        assert decoded.shape == matrix.shape


def test_omp_recovers_sparse_dct_signal_when_undersampled():
    dictionary = make_dictionary("dct", 64, 64, spatial_shape=(8, 8))
    assert np.allclose(dictionary.T @ dictionary, np.eye(64), atol=1e-10)
    coefficients = np.zeros(64)
    coefficients[[0, 1, 8, 9, 18]] = [4.0, -2.0, 1.5, 0.8, -0.4]
    signal = coefficients @ dictionary.T
    phi = make_sensing_matrix(32, 64, seed=11)
    codec = CompressedSensingCodec(phi, dictionary, sparsity=5)
    assert np.linalg.norm(codec.roundtrip(signal) - signal) < 1e-7


def test_dense_embedding_is_not_mislabeled_as_sparse_recovery_success():
    rng = np.random.default_rng(3)
    embeddings = rng.standard_normal((64, 64))
    phi = make_sensing_matrix(32, 64, seed=7)
    omp = CompressedSensingCodec(phi, np.eye(64), sparsity=8).roundtrip(embeddings)
    ridge = CompressedSensingCodec(
        phi, np.eye(64), recovery="ridge"
    ).roundtrip(embeddings)
    omp_error = np.linalg.norm(embeddings - omp) / np.linalg.norm(embeddings)
    ridge_error = np.linalg.norm(embeddings - ridge) / np.linalg.norm(embeddings)
    assert omp_error > 0.9
    assert omp_error > ridge_error
