"""Multi-task, many-to-many discussion over a single superposed packet.

Runs T HiddenBench tasks in lockstep with A agents each. Rounds are
SIMULTANEOUS (unlike the sequential original protocol): every agent speaks
based on what it decoded from the previous round's packet. After each
round, all T*A messages are encoded, bound to slot (task_idx * A +
agent_idx), and summed into ONE shared packet -- the entire round's
inter-agent traffic. Each agent then unbinds and decodes only its
task-mates' slots.

Because greedy decoding is deterministic, each slot is decoded once and the
text shared among that task's receivers (identical to each receiver
decoding it themselves).

The superposition load per packet is M = T * A, so scaling T (or A) scales
how many messages ride in one fixed-size packet. No early consensus
stopping: fixed rounds keep M constant for clean scaling curves.
"""

import difflib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .benchmark import HiddenBench, parse_decision
from .channel import build_channel
from .config import Config
from .prompts import (
    format_final_decision_prompt,
    format_subsequent_speaker_prompt,
    get_first_speaker_prompt,
)
from .providers.base import Message
from .superpose import SuperposedPacketCodec
from .task import AgentDecision, DiscussionMessage, Task


def reconstruction_fidelity(original: str, decoded: str | None) -> float:
    """Character-level similarity in [0, 1] between sent and decoded text."""
    if not decoded:
        return 0.0
    return difflib.SequenceMatcher(None, original, decoded).ratio()


@dataclass
class RoundRecord:
    """Accounting for one round's shared packet."""
    round_num: int
    num_slots: int
    packet_bytes: int          # serialized packet (fp32 latents, base64)
    header_bytes: int          # plaintext slot-addressing metadata
    raw_chars: int             # total plaintext chars produced by senders
    unicast_ae_bytes: int      # what per-message AE unicast would transmit
    slot_fidelity: dict[int, float] = field(default_factory=dict)
    slot_decoded: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "num_slots": self.num_slots,
            "packet_bytes": self.packet_bytes,
            "header_bytes": self.header_bytes,
            "raw_chars": self.raw_chars,
            "unicast_ae_bytes": self.unicast_ae_bytes,
            "mean_fidelity": (sum(self.slot_fidelity.values()) /
                              len(self.slot_fidelity)) if self.slot_fidelity else 0.0,
            "slot_fidelity": self.slot_fidelity,
            "slot_decoded": self.slot_decoded,
        }


class MultiplexRunner:
    """Lockstep multi-task discussion with one superposed packet per round."""

    def __init__(
        self,
        config: Config,
        packet_codec: SuperposedPacketCodec,
        num_rounds: int | None = None,
        workers: int = 1,
        provider=None,
        packing: str = "single",
    ):
        if packing not in ("single", "cross"):
            raise ValueError(f"Unknown packing mode: {packing!r}")
        # The bench instance supplies provider setup, retry logic, agent
        # initialization, and decision parsing. Its channel is forced to
        # identity: packet transport happens here, not in Channel.
        self.bench = HiddenBench(
            config, provider=provider,
            channel=build_channel("full_history", "identity"),
        )
        self.config = config
        self.codec = packet_codec
        self.num_rounds = num_rounds or config.benchmark.num_rounds
        self.workers = max(1, workers)
        # "single": all T*A messages in one packet (load = T*A).
        # "cross": A packets per round, packet a carrying agent a's message
        # from every task (slot = task index; load = T, heterogeneous
        # content). Decouples superposition load from within-task semantic
        # redundancy.
        self.packing = packing
        self.input_tokens = 0
        self.output_tokens = 0

    def _call_one(self, messages: list[Message]) -> Any:
        """One agent call. Reasoning providers (e.g. DeepSeek) sometimes
        exhaust max_tokens on reasoning_content and return empty content;
        that is not an API error, so retry it explicitly."""
        resp = self.bench._call_llm(messages)
        for _ in range(2):
            if resp.content.strip():
                break
            self.input_tokens += resp.input_tokens
            self.output_tokens += resp.output_tokens
            resp = self.bench._call_llm(messages)
        return resp

    def _call_many(self, message_lists: list[list[Message]]) -> list[Any]:
        """Call the LLM for several agents; parallel over threads if enabled."""
        if self.workers == 1:
            responses = [self._call_one(m) for m in message_lists]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                responses = list(ex.map(self._call_one, message_lists))
        for r in responses:
            self.input_tokens += r.input_tokens
            self.output_tokens += r.output_tokens
        return responses

    def run(self, tasks: list[Task]) -> dict[str, Any]:
        """Run all tasks in lockstep. Returns a results dict (JSON-safe)."""
        t_start = time.time()
        A = self.config.benchmark.num_agents
        T = len(tasks)
        load = T * A

        for task in tasks:
            self.bench.channel.set_task_context(task.options)

        # --- init agents + pre-discussion decisions (per task) ---
        agents_by_task = []
        initial_prompts_by_task = []
        pre_decisions = []
        for task in tasks:
            agents, prompts = self.bench._initialize_agents(task, A)
            agents_by_task.append(agents)
            initial_prompts_by_task.append(prompts)
            pre_decisions.append(
                self.bench._get_pre_discussion_decisions(agents, task)
            )

        # decoded_views[t][a]: list of message dicts agent (t, a) has decoded
        decoded_views: list[list[list[dict[str, Any]]]] = [
            [[] for _ in range(A)] for _ in range(T)
        ]
        discussion_by_task: list[list[DiscussionMessage]] = [[] for _ in range(T)]
        round_records: list[RoundRecord] = []

        # --- lockstep simultaneous rounds ---
        for r in range(self.num_rounds):
            # 1. Everyone speaks based on last round's decoded view.
            prompt_lists = []
            for t in range(T):
                for a in range(A):
                    agent = agents_by_task[t][a]
                    if r == 0:
                        agent.add_message(Message.user(
                            get_first_speaker_prompt(self.config.channel.message_style)
                        ))
                    else:
                        view = decoded_views[t][a]
                        agent.add_message(Message.user(
                            format_subsequent_speaker_prompt(
                                view, self.config.channel.message_style)
                        ))
                    prompt_lists.append(agent.messages)

            responses = self._call_many(prompt_lists)

            raw_texts: dict[int, str] = {}
            for t in range(T):
                for a in range(A):
                    slot = t * A + a
                    content = responses[slot].content
                    raw_texts[slot] = content
                    agents_by_task[t][a].add_message(Message.assistant(content))
                    discussion_by_task[t].append(DiscussionMessage(
                        round_num=r, agent_id=a, content=content,
                    ))
                    self.bench.channel.record_sent(content)

            # 2. Encode this round's packet(s).
            # "single": one packet, global slot t*A + a (load T*A).
            # "cross": packet per agent index a, slot = task index t (load T,
            #          one message from each task per packet).
            if self.packing == "cross":
                packet_of: dict[int, tuple[str, int]] = {}  # global slot -> (packet, in-packet slot)
                packet_strs = []
                for a in range(A):
                    texts = {t: raw_texts[t * A + a] for t in range(T)}
                    pkt = self.codec.encode_packet(texts)
                    packet_strs.append(pkt)
                    for t in range(T):
                        packet_of[t * A + a] = (pkt, t)
                packet_bytes = sum(len(p) for p in packet_strs)
                slots_per_packet = T
            else:
                pkt = self.codec.encode_packet(raw_texts)
                packet_of = {s: (pkt, s) for s in raw_texts}
                packet_bytes = len(pkt)
                slots_per_packet = load
            header = json.dumps([
                {"slot": s, "task": s // A, "agent": s % A} for s in raw_texts
            ])
            K = self.codec.codec.num_latents
            D = self.codec.codec.latent_dim
            record = RoundRecord(
                round_num=r,
                num_slots=slots_per_packet,
                packet_bytes=packet_bytes,
                header_bytes=len(header),
                raw_chars=sum(len(v) for v in raw_texts.values()),
                unicast_ae_bytes=load * K * D * 4,
            )

            # 3. Decode each slot once; deliver to that task's other agents.
            for slot, original in raw_texts.items():
                t, a = slot // A, slot % A
                pkt, in_slot = packet_of[slot]
                decoded = self.codec.decode_slot(pkt, in_slot)
                record.slot_fidelity[slot] = reconstruction_fidelity(original, decoded)
                record.slot_decoded[slot] = decoded if decoded is not None else ""
                msg = {
                    "agent_id": a,
                    "round_num": r,
                    "content": decoded if decoded is not None else "",
                }
                for recv in range(A):
                    if recv != a:
                        decoded_views[t][recv].append(msg)
            round_records.append(record)

        # --- final decisions from each agent's decoded view ---
        post_decisions: list[list[AgentDecision]] = []
        prompt_lists = []
        for t in range(T):
            for a in range(A):
                agent = agents_by_task[t][a]
                agent.add_message(Message.user(format_final_decision_prompt(
                    decoded_views[t][a], tasks[t].options,
                )))
                prompt_lists.append(agent.messages)
        responses = self._call_many(prompt_lists)
        for t in range(T):
            decisions = []
            for a in range(A):
                content = responses[t * A + a].content
                agents_by_task[t][a].add_message(Message.assistant(content))
                vote, rationale = parse_decision(content, tasks[t].options)
                decisions.append(AgentDecision(
                    agent_id=a, vote=vote, rationale=rationale,
                    is_correct=(vote == tasks[t].correct_answer),
                    raw_response=content,
                ))
            post_decisions.append(decisions)

        # --- assemble results ---
        all_fid = [f for rec in round_records for f in rec.slot_fidelity.values()]
        per_task = []
        for t, task in enumerate(tasks):
            pre = pre_decisions[t]
            post = post_decisions[t]
            per_task.append({
                "task_id": task.id,
                "task_name": task.name,
                "correct_answer": task.correct_answer,
                "pre_votes": [d.vote for d in pre],
                "post_votes": [d.vote for d in post],
                "pre_accuracy": sum(d.is_correct for d in pre) / len(pre),
                "post_accuracy": sum(d.is_correct for d in post) / len(post),
                "discussion": [m.to_dict() for m in discussion_by_task[t]],
            })

        total_packet_bytes = sum(rec.packet_bytes + rec.header_bytes
                                 for rec in round_records)
        total_raw_chars = sum(rec.raw_chars for rec in round_records)
        total_unicast = sum(rec.unicast_ae_bytes for rec in round_records)

        return {
            "num_tasks": T,
            "num_agents": A,
            "superposition_load": load,
            "num_rounds": self.num_rounds,
            "mean_reconstruction_fidelity": (
                sum(all_fid) / len(all_fid) if all_fid else 0.0
            ),
            "pre_accuracy": sum(x["pre_accuracy"] for x in per_task) / T,
            "post_accuracy": sum(x["post_accuracy"] for x in per_task) / T,
            "bytes": {
                "packet_total": total_packet_bytes,
                "raw_chars_total": total_raw_chars,
                "unicast_ae_total": total_unicast,
                "broadcast_vs_unicast_ratio": (
                    total_packet_bytes / total_unicast if total_unicast else None
                ),
            },
            "rounds": [rec.to_dict() for rec in round_records],
            "tasks": per_task,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_time_seconds": time.time() - t_start,
        }
