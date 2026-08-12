"""Channel middleware for Collab-Overcooked.

Reuses the exact compressor implementations validated on HiddenBench
(hiddenbench.channel) rather than reimplementing them, so results are
directly comparable across both benchmarks. See handoff.md for the
HiddenBench-side experiments this ports.
"""

from multimodal_comms.benchmarks.hiddenbench.runtime.channel import ChannelStats, build_compressor


class ChannelAdapter:
    """Shared per-episode object attached to both LLMAgents instances.

    Maintains one flat chronological transcript of "talk" utterances (from
    either agent) and renders a receiver's view through the attached
    compressor. `channel_scope` controls how often the compressor's internal
    state (and the transcript itself) resets:
      - "episode" (default): reset once per episode, so stateful compressors
        (backref, novelty, adaptive) can exploit coordination phrases that
        recur across timesteps even though the framework itself does not
        retransmit them.
      - "timestep": reset every game timestep, matching the scope the
        framework's own dialog_history_list already uses.
    """

    def __init__(self, compressor_name: str = "identity", channel_scope: str = "episode", **compressor_kwargs):
        self.compressor_name = compressor_name
        self.channel_scope = channel_scope
        self._compressor_kwargs = compressor_kwargs
        self.compressor = build_compressor(compressor_name, **compressor_kwargs)
        self.stats = ChannelStats()
        self.transcript: list[dict] = []
        self._turn_counter = 0

    def reset_episode(self) -> None:
        self.compressor = build_compressor(self.compressor_name, **self._compressor_kwargs)
        self.compressor.set_task_context([])
        self.transcript = []
        self._turn_counter = 0
        self.stats = ChannelStats()

    def maybe_reset_for_timestep(self) -> None:
        if self.channel_scope == "timestep":
            self.compressor.set_task_context([])
            self.transcript = []
            self._turn_counter = 0

    def record(self, agent_id: int, content: str) -> None:
        self.stats.raw_messages += 1
        self.stats.raw_chars += len(content)
        self.transcript.append({
            "agent_id": agent_id,
            "round_num": self._turn_counter,
            "content": content,
        })
        self._turn_counter += 1

    def render(self, receiver_id: int, name_for) -> str:
        compressed = self.compressor.compress(list(self.transcript), receiver_id)
        if self.transcript and not compressed:
            compressed = [self.transcript[-1]]
        self.stats.transmitted_messages += len(compressed)
        self.stats.transmitted_chars += sum(len(m["content"]) for m in compressed)
        return "".join(
            f"{name_for(m['agent_id'])} say history turn {i + 1} : {m['content']}\n"
            for i, m in enumerate(compressed)
        )

    def finalize_stats(self) -> dict:
        return self.stats.snapshot()
