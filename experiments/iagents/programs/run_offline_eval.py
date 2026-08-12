#!/usr/bin/env python3
"""Batch offline evaluation of the iAgents two-agent stack on
Needle_in_the_Persona, with a pluggable compression channel from the shared
hiddenbench `channel` library.

No Flask / session / DB dependency: agents are seeded in-memory from each
record (see needle_common.OfflineThinkAgent), and the inter-agent channel is
routed through hiddenbench.channel.build_compressor(...).

Usage:
  python experiments/iagents/programs/run_offline_eval.py --dataset data/.../eval.jsonl \
      --condition identity --out reports/eval_identity.json

  # with compressor params:
  python experiments/iagents/programs/run_offline_eval.py --condition saliency --param rate=0.4 ...
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

from experiments.iagents.programs.needle_common import (  # noqa: E402
    load_records, record_roles, OfflineThinkAgent, grade,
)
from multimodal_comms.apps.iagents.runtime.communication import OfflineCommunication  # noqa: E402
from multimodal_comms.apps.iagents.runtime.util import iAgentsLogger  # noqa: E402
from multimodal_comms.benchmarks.hiddenbench.runtime.channel import build_compressor  # noqa: E402


class ChannelOfflineCommunication(OfflineCommunication):
    """Offline communication whose inter-agent transmitted view is routed
    through a compression channel.

    A canonical transcript of message dicts {agent_id, round_num, content} is
    maintained; before each agent's turn we hand it the *compressed* view of
    that transcript (full-history protocol) as its communication_history. The
    final conclusion is likewise drawn from the instructor's compressed view,
    so compression genuinely affects what information survives to the answer.
    """

    def __init__(self, instructor, assistant, max_round, compressor,
                 is_consensus_conclusion=True):
        super().__init__(instructor, assistant, max_round, is_consensus_conclusion)
        self.compressor = compressor
        self.canonical = []          # [{agent_id, round_num, content}]
        self.raw_chars = 0           # sender content, counted once
        self.transmitted_chars = 0   # content placed into receiver prompts
        self.transmitted_msgs = 0

    def _agent_id(self, agent):
        return 0 if agent is self.instructor else 1

    def _record(self, sender, message, round_num):
        self.canonical.append({
            "agent_id": self._agent_id(sender),
            "round_num": round_num,
            "content": message,
        })
        self.raw_chars += len(message or "")

    def _view_for(self, receiver):
        """Compressed communication_history (list[str]) for `receiver`."""
        rid = self._agent_id(receiver)
        comp = self.compressor.compress(
            [dict(m) for m in self.canonical], rid
        )
        # Accounting happens on the transmitted (compressed) form; decompress
        # then reconstructs the readable view for decompressive codecs
        # (autoencoder / superpose). For compress-only codecs it is a no-op.
        self.transmitted_chars += sum(len(m["content"]) for m in comp)
        self.transmitted_msgs += len(comp)
        comp = self.compressor.decompress(comp, rid)
        hist = ['']
        for m in comp:
            sender_agent = self.instructor if m["agent_id"] == 0 else self.assistant
            recv_agent = self.assistant if m["agent_id"] == 0 else self.instructor
            hist.append(self.format_agent_history(
                sender_agent, recv_agent, m["content"]))
        return hist

    def communicate(self):
        round_index = 0
        while round_index < self.max_round:
            # instructor -> assistant
            view = self._view_for(self.instructor)
            instr_resp = self.instructor.query(self.assistant.master, view)
            self._record(self.instructor, instr_resp, round_index)
            self.communication_history.append(self.format_agent_history(
                self.instructor, self.assistant, instr_resp))
            self.send_message_agent(self.instructor, self.assistant, instr_resp)

            # assistant -> instructor
            view = self._view_for(self.assistant)
            asst_resp = self.assistant.query(self.instructor.master, view)
            self._record(self.assistant, asst_resp, round_index)
            self.communication_history.append(self.format_agent_history(
                self.assistant, self.instructor, asst_resp))
            self.send_message_agent(self.assistant, self.instructor, asst_resp)

            round_index += 1

        # Conclusion from the instructor's *compressed* view of the transcript.
        final_view = self._view_for(self.instructor)
        conclusion = self.consensus_conclusion(
            final_view, self.instructor.infonav_plan, self.assistant.infonav_plan)
        return conclusion


def run_one(rec, compressor, backend, max_round):
    roles = record_roles(rec)
    instructor = OfflineThinkAgent(
        master=roles["instructor_master"], backend=backend, task=roles["task"],
        current_chat=roles["current_chat"], other_chat=roles["instructor_other"],
        is_assistant=False)
    assistant = OfflineThinkAgent(
        master=roles["assistant_master"], backend=backend, task=roles["task"],
        current_chat=roles["current_chat"], other_chat=roles["assistant_other"],
        is_assistant=True)

    comm = ChannelOfflineCommunication(
        instructor=instructor, assistant=assistant, max_round=max_round,
        compressor=compressor, is_consensus_conclusion=True)

    t0 = time.time()
    conclusion = comm.communicate()
    wall = time.time() - t0

    correct = grade(conclusion, roles["answer"])
    return {
        "id": rec.get("id"),
        "task": roles["task"],
        "answer": roles["answer"],
        "conclusion": conclusion,
        "correct": bool(correct),
        "wall_s": round(wall, 2),
        "raw_chars": comm.raw_chars,
        "transmitted_chars": comm.transmitted_chars,
        "transmitted_msgs": comm.transmitted_msgs,
        "transcript": comm.canonical,
    }


def parse_params(pairs):
    params = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        params[k] = v
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--condition", default="identity",
                    help="compressor name (identity/saliency/repmatch_*...)")
    ap.add_argument("--param", action="append", default=[],
                    help="compressor param, e.g. --param rate=0.4")
    ap.add_argument("--backend", default="deepseek")
    ap.add_argument("--max-round", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    iAgentsLogger.set_evaluate_log_path(
        exp_name="Needle", file_prefix=f"offline_{args.condition}")

    records = load_records(args.dataset)
    if args.limit:
        records = records[: args.limit]

    params = parse_params(args.param)
    compressor = build_compressor(args.condition, **params)

    results = []
    t_start = time.time()
    for i, rec in enumerate(records):
        try:
            res = run_one(rec, compressor, args.backend, args.max_round)
        except Exception as e:  # keep the sweep alive on a single-task failure
            import traceback
            res = {
                "id": rec.get("id"),
                "task": rec.get("task_prompt", ""),
                "answer": rec.get("answer", ""),
                "conclusion": "",
                "correct": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:],
                "raw_chars": 0, "transmitted_chars": 0, "transmitted_msgs": 0,
                "wall_s": 0, "transcript": [],
            }
        results.append(res)
        acc = sum(r["correct"] for r in results) / len(results)
        print(f"[{args.condition}] {i+1}/{len(records)} id={res.get('id')} "
              f"correct={res['correct']} acc={acc:.3f} "
              f"tx_chars={res.get('transmitted_chars')} "
              f"wall={res.get('wall_s')}s"
              + (f" ERR={res.get('error')}" if res.get("error") else ""),
              flush=True)

    n = len(results)
    ncorr = sum(r["correct"] for r in results)
    nerr = sum(1 for r in results if r.get("error"))
    summary = {
        "condition": args.condition,
        "params": params,
        "dataset": args.dataset,
        "n": n,
        "n_correct": ncorr,
        "accuracy": ncorr / n if n else 0.0,
        "n_error": nerr,
        "total_raw_chars": sum(r["raw_chars"] for r in results),
        "total_transmitted_chars": sum(r["transmitted_chars"] for r in results),
        "total_transmitted_msgs": sum(r["transmitted_msgs"] for r in results),
        "total_wall_s": round(time.time() - t_start, 1),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== {args.condition}: acc={summary['accuracy']:.3f} "
          f"({ncorr}/{n}, {nerr} errors)  "
          f"tx_chars={summary['total_transmitted_chars']}  "
          f"wall={summary['total_wall_s']}s  -> {args.out}")


if __name__ == "__main__":
    main()
