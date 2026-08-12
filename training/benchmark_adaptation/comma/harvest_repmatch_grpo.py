#!/usr/bin/env python3
"""Harvest A1 (repmatch_rewriter GRPO) views from COMMA transcripts.

For each transcript and several cut points, render the channel (Solver/Expert)
messages up to that cut as an "Agent N: ..." view. The GRPO rewriter is
rewarded for compressions whose proxy representation stays close to the full
view's representation, so only the view text is needed here (no receiver LM).

Output: data/repmatch_grpo_train.jsonl of {"view", "source"}.
"""
import glob
import json
import os
import random
import sys

HARVEST_DIRS = sys.argv[1:] or ["/tmp/harvest"]
OUT = "data/repmatch_grpo_train.jsonl"
CUTS_PER = 3
MIN_MSGS = 2
PLACEHOLDERS = ("there was an issue getting a response",)


def is_channel(frm):
    f = frm.lower()
    return f.startswith("solver") or f.startswith("expert")


def load_channel(path):
    msgs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        frm = str(d.get("from", ""))
        val = str(d.get("value", "")).strip()
        if not is_channel(frm) or not val:
            continue
        if any(p in val.lower() for p in PLACEHOLDERS):
            continue
        aid = 0 if frm.lower().startswith("solver") else 1
        msgs.append({"agent_id": aid, "content": val})
    return msgs


def render(msgs):
    return "\n".join(f"Agent {m['agent_id'] + 1}: {m['content']}" for m in msgs)


def main():
    random.seed(29)
    files = []
    for d in HARVEST_DIRS:
        files += glob.glob(os.path.join(d, "**", "conversation.jsonl"), recursive=True)
    rows = []
    for path in files:
        source = "/".join(path.split("/")[-3:-1])
        msgs = load_channel(path)
        if len(msgs) < MIN_MSGS:
            continue
        cut_opts = list(range(MIN_MSGS, len(msgs) + 1))
        cuts = random.sample(cut_opts, min(CUTS_PER, len(cut_opts)))
        for c in cuts:
            view = render(msgs[:c])
            if 20 <= len(view) < 6000:
                rows.append({"view": view, "source": source})

    os.makedirs("data", exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"transcripts: {len(files)}  views: {len(rows)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
