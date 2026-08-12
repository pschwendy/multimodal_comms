#!/usr/bin/env python3
"""Harvest individual message contents for training the textual autoencoder.

Reads pre-existing full-run report JSONs, extracts individual message
content strings from every discussion turn (not views - single messages),
and outputs one JSONL line per message.

Excludes messages from the 16-task eval-sweep set so training and
evaluation tasks are disjoint.

Output: data/autoencoder_train.jsonl
  {"text": "<message content string>"}
"""

import json
import random
import sys

EVAL_REPORT = "reports/sweep_full_identity.json"
FULL_RUN_REPORTS = [
    "reports/hiddenbench_20260712_060139.json",
    "reports/hiddenbench_20260712_062305.json",
    "reports/results-opus-4.5.json",
]
OUT = "data/autoencoder_train.jsonl"
MIN_CHARS = 20   # skip very short messages
MAX_CHARS = 2000  # skip very long messages
MAX_EXAMPLES = 5000


def main() -> None:
    eval_names = {
        r["task"]["name"] for r in json.load(open(EVAL_REPORT))["results"]
    }

    random.seed(42)
    rows: list[dict[str, str]] = []
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
            for m in r.get("discussion_history", []):
                content = m.get("content", "").strip()
                if MIN_CHARS <= len(content) <= MAX_CHARS:
                    rows.append({"text": content})

    random.shuffle(rows)
    rows = rows[:MAX_EXAMPLES]

    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_tasks = len({r["text"][:40] for r in rows})  # rough estimate
    print(f"wrote {len(rows)} examples to {OUT}")


if __name__ == "__main__":
    main()
