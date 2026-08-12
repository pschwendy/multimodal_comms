#!/usr/bin/env python3
"""Harvest training data for the GRPO representational-match rewriter.

Same cut-point sampling over the non-eval transcripts as
training/data/harvest_rewriter_data.py (CUTS_PER_DISCUSSION random cuts per
discussion, MIN_VIEW_MSGS minimum), but there is NO receiver-vote field: the
GRPO reward is representational (cosine of the frozen proxy's hidden state on
compressed vs. original text), not behavioral.

For each harvested view we also cache the view's original representation from
the rep server (one /rep_batch entry per row) as "target_rep", so training
does not have to recompute the (fixed) original rep on every step.

Output: data/repmatch_rewriter_train.jsonl
  {"view", "task", "target_rep": [float, ...]}
"""

import json
import random
import sys

import requests

EVAL_REPORT = "reports/sweep_full_identity.json"
FULL_RUN_REPORTS = [
    "reports/hiddenbench_20260712_060139.json",
    "reports/hiddenbench_20260712_062305.json",
    "reports/results-opus-4.5.json",
]
OUT = "data/repmatch_rewriter_train.jsonl"
REPSERVER = "http://127.0.0.1:8100"
BATCH = 64
CUTS_PER_DISCUSSION = 3
MIN_VIEW_MSGS = 4


def format_view(messages: list[dict]) -> str:
    return "\n".join(
        f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages
    )


def rep_batch(texts: list[str]) -> list[list[float]]:
    reps: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        resp = requests.post(
            f"{REPSERVER}/rep_batch",
            json={"items": [{"text": t} for t in chunk]},
            timeout=300,
        )
        resp.raise_for_status()
        reps.extend(resp.json()["reps"])
    return reps


def main() -> None:
    eval_names = {
        r["task"]["name"] for r in json.load(open(EVAL_REPORT))["results"]
    }

    random.seed(29)
    rows = []
    for path in FULL_RUN_REPORTS:
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            print(f"skip missing {path}", file=sys.stderr)
            continue
        for r in d.get("results", []):
            name = r["task"]["name"]
            if name in eval_names:
                continue
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
                rows.append({
                    "view": format_view(history[:cut]),
                    "task": name,
                })

    print(f"views harvested: {len(rows)}; caching original reps...", flush=True)
    reps = rep_batch([r["view"] for r in rows])
    if len(reps) != len(rows):
        raise RuntimeError(f"rep count {len(reps)} != row count {len(rows)}")
    for row, rep in zip(rows, reps):
        row["target_rep"] = rep

    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples to {OUT} "
          f"({len({r['task'] for r in rows})} tasks)")


if __name__ == "__main__":
    main()
