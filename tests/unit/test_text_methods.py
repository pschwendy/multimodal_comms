from multimodal_comms.core import Message, MethodContext
from multimodal_comms.methods.text import (
    BackrefMethod,
    CodebookConfig,
    CodebookMethod,
    DeltaView,
    IdentityMethod,
    NoveltyConfig,
    NoveltyMethod,
)


def test_identity_is_byte_exact():
    messages = [Message("a", None, b"\x00\xffbytes"), Message("b", "a", "meeting starts at noon")]
    method = IdentityMethod()
    tx = method.encode(messages, MethodContext())
    assert tx.raw_bytes == len(b"\x00\xffbytes") + len("meeting starts at noon".encode())
    assert method.decode(tx, MethodContext()) == messages


def test_backref_is_lossless_and_uses_dictionary():
    message = Message("a", None, "repeated content long enough")
    method = BackrefMethod()
    tx = method.encode([message, message], MethodContext())
    assert tx.metadata["dictionary_entries"] == 1
    assert method.decode(tx, MethodContext()) == [message, message]


def test_online_and_corpus_codebooks_are_lossless():
    messages = [Message("a", None, "compressionword compressionword")]
    for config in (CodebookConfig(online=True), CodebookConfig({"compressionword": "§x§"})):
        method = CodebookMethod(config)
        assert method.decode(method.encode(messages, MethodContext()), MethodContext()) == messages


def test_delta_view_receiver_isolation_and_reset():
    messages = [Message("a", None, "one"), Message("b", None, "two")]
    view = DeltaView()
    assert view.select(messages, MethodContext(receiver="r1")) == messages
    assert view.select(messages, MethodContext(receiver="r1")) == []
    assert view.select(messages, MethodContext(receiver="r2")) == messages
    view.reset()
    assert view.select(messages, MethodContext(receiver="r1")) == messages


def test_stateful_novelty_resets_between_episodes():
    method = NoveltyMethod(NoveltyConfig(threshold=1.0, stateful=True))
    message = Message("a", None, "same sentence.")
    first = method.decode(method.encode([message], MethodContext(receiver="r")), MethodContext())
    second = method.decode(method.encode([message], MethodContext(receiver="r")), MethodContext())
    # Empty views fall back to the newest message to preserve prompt coherence.
    assert first == second == [message]
    method.reset()
    assert method.decode(
        method.encode([message], MethodContext(receiver="r")), MethodContext()
    ) == [message]
