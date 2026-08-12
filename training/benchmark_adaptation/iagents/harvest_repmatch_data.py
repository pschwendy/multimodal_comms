#!/usr/bin/env python3
"""Harvest A1 GRPO rewriter training views from iAgents identity transcripts.

Uses random cut-point
views over each identity transcript; caches each view's original rep (the GRPO
target) from the shared representation server.

Input: offline-eval identity report JSON(s) from run_offline_eval.py.
Output: data/repmatch_rewriter_train.jsonl {"view","task","target_rep"}
"""

import argparse
import json
import random
import time

import requests

REPSERVER = "http://127.0.0.1:8100"
BATCH = 16
CUTS_PER_DISCUSSION = 3
MIN_VIEW_MSGS = 2


def format_view(messages):
    return "\n".join(
        f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages)


def _post_chunk(chunk, tries=8):
    delay = 2.0
    for t in range(tries):
        try:
            resp = requests.post(
                f"{REPSERVER}/rep_batch",
                json={"items": [{"text": x} for x in chunk]}, timeout=300)
            resp.raise_for_status()
            return resp.json()["reps"]
        except Exception as e:
            if t == tries - 1:
                if len(chunk) > 1:
                    mid = len(chunk) // 2
                    return _post_chunk(chunk[:mid]) + _post_chunk(chunk[mid:])
                raise RuntimeError(f"rep server failed on 1 item: {e}")
            time.sleep(delay)
            delay = min(delay * 1.7, 30.0)


def rep_batch(texts):
    reps = []
    for start in range(0, len(texts), BATCH):
        reps.extend(_post_chunk(texts[start:start + BATCH]))
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", default="data/repmatch_rewriter_train.jsonl")
    args = ap.parse_args()

    random.seed(29)
    rows = []
    for path in args.reports:
        d = json.load(open(path))
        for r in d.get("results", []):
            name = str(r.get("id"))
            history = [
                {"agent_id": m["agent_id"], "content": m["content"]}
                for m in r.get("transcript", [])
            ]
            if len(history) < MIN_VIEW_MSGS:
                continue
            n_cuts = min(CUTS_PER_DISCUSSION, len(history) + 1 - MIN_VIEW_MSGS)
            cuts = random.sample(range(MIN_VIEW_MSGS, len(history) + 1), n_cuts)
            for cut in cuts:
                rows.append({"view": format_view(history[:cut]), "task": name})

    print(f"views harvested: {len(rows)}; caching original reps...", flush=True)
    if not rows:
        print("no views; nothing written")
        return
    reps = rep_batch([r["view"] for r in rows])
    for row, rep in zip(rows, reps):
        row["target_rep"] = rep

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples to {args.out} "
          f"({len({r['task'] for r in rows})} tasks)")


if __name__ == "__main__":
    main()
