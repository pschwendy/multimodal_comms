from multimodal_comms.core import Message, MethodContext
from multimodal_comms.methods.predictive import PredictiveDiffMethod, RateDiffConfig, RateDiffMethod


def test_predictive_diff_lossless_text_and_bytes():
    messages = [Message("a", None, "meeting starts at noon"), Message("b", "a", b"\x00\xff")]
    context = MethodContext(shared={"predictive_prefix": "prior"})
    method = PredictiveDiffMethod()
    assert method.decode(method.encode(messages, context), context) == messages


def test_rate_diff_at_one_is_lossless():
    messages = [Message("a", None, "abcdef")]
    method = RateDiffMethod(RateDiffConfig(correction_rate=1.0))
    assert method.decode(method.encode(messages, MethodContext()), MethodContext()) == messages


def test_empty_predictive_input():
    method = PredictiveDiffMethod()
    assert method.decode(method.encode([], MethodContext()), MethodContext()) == []
