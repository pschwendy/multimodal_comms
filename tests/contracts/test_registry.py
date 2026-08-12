from dataclasses import is_dataclass

import numpy as np
import pytest
from multimodal_comms.core import Message, MethodContext
from multimodal_comms.registry import create_method, get_method_spec, list_methods

EXPECTED = {
    "identity",
    "window",
    "novelty",
    "llmlingua2",
    "learned",
    "rewriter",
    "backref",
    "codebook",
    "adaptive",
    "gzip64",
    "stack",
    "counterfactual",
    "vib_sender",
    "repmatch_selector",
    "saliency",
    "repmatch_bestofk",
    "repmatch_rewriter",
    "tokenfilter",
    "autoencoder",
    "mwnot_autoencoder",
    "grammar",
    "certspan",
    "semfallback",
    "pdiff",
    "telegraphic",
    "ratediff",
    "superpose",
    "block",
    "frame",
    "rotor",
    "compressed_sensing",
    "svd",
    "pca",
    "cur",
    "image_zlib",
    "mixed_packet",
}


class FakeCompressor:
    """Dependency-free stand-in for exercising the registry adapter contract."""

    def set_task_context(self, options):
        self.options = options

    def compress(self, messages, receiver_id):
        return list(messages)

    def decompress(self, messages, receiver_id):
        return list(messages)


def test_registry_has_every_approach_and_typed_configs():
    specs = list_methods()
    assert {spec.id for spec in specs} == EXPECTED
    assert all(is_dataclass(spec.config_type) for spec in specs)
    assert all(spec.representation and spec.description for spec in specs)


@pytest.mark.parametrize("spec", list_methods(), ids=lambda spec: spec.id)
def test_every_registry_entry_instantiates_and_smokes(spec):
    method = (
        create_method(spec.id, compressor=FakeCompressor())
        if spec.kind == "communication"
        else create_method(spec.id)
    )
    if spec.kind == "communication":
        context = MethodContext(seed=5, receiver="r")
        messages = [Message("a", "r", "hello hello", 1)]
        decoded = method.decode(method.encode(messages, context), context)
        assert isinstance(decoded, list)
        assert method.decode(method.encode([], context), context) == []
    elif spec.kind == "packer":
        packet = method.pack({0: np.ones(4 if spec.id != "superpose" else 8)})
        assert method.unpack(packet, 0).shape == (4 if spec.id != "superpose" else 8,)
    else:
        if spec.id in {"image_zlib"}:
            value = b"fake image bytes"
        elif spec.id == "mixed_packet":
            value = {"text": "x", "image": b"y"}
        else:
            value = np.eye(8)
        assert method.decode(method.encode(value, seed=1), seed=1) is not None


def test_unknown_config_key_is_helpful():
    with pytest.raises(ValueError, match="unknown configuration keys"):
        create_method("window", {"bogus": 2})
    with pytest.raises(KeyError, match="unknown method"):
        get_method_spec("not-a-method")
