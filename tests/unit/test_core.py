import pytest
from multimodal_comms.core import Message, MethodContext
from multimodal_comms.core.serialization import deserialize_messages, serialize_messages


def test_message_serialization_preserves_text_and_bytes():
    messages = [
        Message("alpha", None, "meeting starts at noon", 2, {"x": 1}),
        Message("b", "c", b"\x00\xff", 3),
    ]
    assert deserialize_messages(serialize_messages(messages)) == messages


def test_message_invariants_and_immutable_metadata():
    source = {"a": 1}
    message = Message("a", None, "x", metadata=source)
    source["a"] = 2
    assert message.metadata["a"] == 1
    with pytest.raises(TypeError):
        message.metadata["a"] = 3
    with pytest.raises(ValueError):
        Message("", None, "x")


def test_context_receiver_copy_isolated():
    base = MethodContext(shared={"public": "x"}, seed=9)
    left = base.for_receiver("left")
    right = base.for_receiver("right")
    assert left.receiver == "left" and right.receiver == "right" and left.seed == right.seed == 9
