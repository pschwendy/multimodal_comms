#!/usr/bin/env python3
"""Harvest training data for the GRPO rewriter.

For each training task (eval-16 excluded) and several cut points in its
recorded discussions, builds a transmission view and precomputes the frozen
receiver's vote given the FULL view. The rewriter will be rewarded for
compressed views that keep the receiver's vote unchanged.

Receiver = local Qwen3-4B vLLM server (the same model used as agents).

Output: data/rewriter_train.jsonl
  {"view", "description", "options", "full_vote", "task"}
"""

import json
import random
import sys


from openai import OpenAI

from training.utils import parse_choice

EVAL_REPORT = "reports/sweep_full_identity.json"
FULL_RUN_REPORTS = [
    "reports/hiddenbench_20260712_060139.json",
    "reports/hiddenbench_20260712_062305.json",
    "reports/results-opus-4.5.json",
]
BENCHMARK = "src/multimodal_comms/benchmarks/hiddenbench/data/hiddenbench_official/benchmark.json"
OUT = "outputs/hiddenbench/rewriter_train.jsonl"
CUTS_PER_DISCUSSION = 3
MIN_VIEW_MSGS = 4

# Dedicated server on GPU 1 so harvesting never contends with timed
# benchmark runs on the GPU 0 server
client = OpenAI(api_key="local", base_url="http://127.0.0.1:8001/v1")


def format_view(messages: list[dict]) -> str:
    return "\n".join(
        f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages
    )


def receiver_vote(description: str, options: list[str], view_text: str) -> str:
    options_text = "\n".join(f"- {o}" for o in options)
    prompt = (
        f"## Scenario\n{description}\n\n"
        f"## Group Discussion\n{view_text}\n\n"
        f"## Available Options\n{options_text}\n\n"
        'Based on the discussion, respond with your decision in JSON format:\n'
        '{"vote": "<one of the options above>", "rationale": "<brief>"}'
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0,
    )
    vote, _ = parse_choice(response.choices[0].message.content or "", options)
    return vote


def main() -> None:
    eval_names = {
        r["task"]["name"] for r in json.load(open(EVAL_REPORT))["results"]
    }
    bench = json.load(open(BENCHMARK))
    bench_tasks = bench if isinstance(bench, list) else bench.get("tasks", bench)
    meta_by_name = {t["name"]: t for t in bench_tasks}

    random.seed(29)
    rows = []
    for path in FULL_RUN_REPORTS:
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            continue
        for r in d.get("results", []):
            name = r["task"]["name"]
            if name in eval_names or name not in meta_by_name:
                continue
            meta = meta_by_name[name]
            history = [
                {"agent_id": m["agent_id"], "content": m["content"]}
                for m in r.get("discussion_history", [])
            ]
            if len(history) < MIN_VIEW_MSGS:
                continue
            cuts = random.sample(
                range(MIN_VIEW_MSGS, len(history) + 1),
                min(CUTS_PER_DISCUSSION, len(history) + 1 - MIN_VIEW_MSGS),
            )
            for cut in cuts:
                view = history[:cut]
                rows.append({
                    "view": format_view(view),
                    "description": meta["description"],
                    "options": meta["possible_answers"],
                    "task": name,
                })

    print(f"views harvested: {len(rows)}; computing receiver votes...")
    kept = []
    for i, row in enumerate(rows):
        try:
            vote = receiver_vote(row["description"], row["options"], row["view"])
        except Exception as e:
            print(f"  [{i}] receiver error: {e}", file=sys.stderr)
            continue
        if not vote or vote not in row["options"]:
            continue  # receiver undecided on full view -> useless example
        row["full_vote"] = vote
        kept.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} done, kept {len(kept)}")

    with open(OUT, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(kept)} examples to {OUT} "
          f"({len({r['task'] for r in kept})} tasks)")


if __name__ == "__main__":
    main()
