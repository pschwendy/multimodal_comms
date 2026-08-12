from multimodal_comms.benchmarks.collab_overcooked import EpisodeAdapter, grade_collaboration
from multimodal_comms.benchmarks.comma import CommaAdapter, grade_comma
from multimodal_comms.benchmarks.hiddenbench import HiddenBenchAdapter, grade_hiddenbench
from multimodal_comms.benchmarks.iagents import IAgentsAdapter, grade_iagents
from multimodal_comms.core import MethodContext
from multimodal_comms.methods.text import IdentityMethod


def test_each_benchmark_adapter_offline_smoke():
    cases = [
        (HiddenBenchAdapter(IdentityMethod()), [{"agent_id": 0, "round_num": 0, "content": "A"}]),
        (CommaAdapter(IdentityMethod()), [{"role": "expert", "turn": 0, "content": "cut wire"}]),
        (EpisodeAdapter(IdentityMethod()), [{"agent": "chef", "timestep": 1, "content": "serve"}]),
        (IAgentsAdapter(IdentityMethod()), [{"sender": "a", "receiver": "b", "content": "answer"}]),
    ]
    for adapter, fixture in cases:
        result = adapter.transmit(fixture, MethodContext(receiver="receiver"))
        assert len(result.messages) == 1
        assert result.traffic.raw_messages == result.traffic.transmitted_messages == 1


def test_native_graders():
    assert grade_hiddenbench(["A", "B"], ["A", "A"], "A").post_accuracy == 1.0
    assert grade_comma({"x": 1}, {"x": 1, "y": 2}, telehealth=True).partial_credit == 0.5
    assert grade_collaboration(["serve", "wait"], ["serve"], completed=True).success == 1.0
    assert grade_iagents("The Answer!", "the answer").exact == 1.0
