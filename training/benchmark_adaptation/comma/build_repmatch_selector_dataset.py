#!/usr/bin/env python3
"""Build the A2 (repmatch_selector) training dataset from COMMA transcripts.

Label per sentence = representational distance the frozen proxy's hidden state
(rep server, layer 14) undergoes when that sentence is masked out of its
message, in the running channel context:

    label(s_i) = 1 - cos( rep(message | ctx), rep(message_without_s_i | ctx) )

Higher = the sentence matters more to how the message reads. Purely extractive:
the trained regressor scores raw sentences at inference (no rep-server call).

Adapts the hiddenbench build_selector_dataset.py idea to COMMA's short,
imperative Solver/Expert turns. Output: data/repmatch_selector_train.jsonl of
{"sentence", "label", "source"}.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import requests

REP_URL = os.getenv("REP_URL", "http://127.0.0.1:8100")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
HARVEST_DIRS = sys.argv[1:] or ["/tmp/harvest"]
OUT = "data/repmatch_selector_train.jsonl"

PLACEHOLDERS = ("there was an issue getting a response",)


def is_channel(frm: str) -> bool:
    f = frm.lower()
    return f.startswith("solver") or f.startswith("expert")


def rep_batch(items, retries=8):
    import time
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{REP_URL}/rep_batch", json={"items": items}, timeout=120)
            r.raise_for_status()
            return np.array(r.json()["reps"], dtype=np.float64)
        except Exception as e:
            last = e
            # rep server is shared/contended (GPU OOM thrash) -> back off and retry
            time.sleep(2.0 + 1.5 * attempt)
    raise last


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_channel_messages(path):
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
        msgs.append({"from": frm.upper()[:6], "content": val})
    return msgs


def main():
    files = []
    for d in HARVEST_DIRS:
        files += glob.glob(os.path.join(d, "**", "conversation.jsonl"), recursive=True)
    print(f"transcripts: {len(files)}")

    os.makedirs("data", exist_ok=True)
    fout = open(OUT, "w")
    rows = []
    n_msg = 0

    def emit(row):
        rows.append(row)
        fout.write(json.dumps(row) + "\n")
        fout.flush()

    for path in files:
        source = "/".join(path.split("/")[-3:-1])
        msgs = load_channel_messages(path)
        ctx_parts = []
        for m in msgs:
            n_msg += 1
            if n_msg % 25 == 0:
                print(f"  ..{n_msg} msgs, {len(rows)} sentences", flush=True)
            content = m["content"]
            ctx = "\n".join(ctx_parts)
            sentences = [s.strip() for s in SENTENCE_SPLIT.split(content) if s.strip()]
            ctx_parts.append(f"{m['from']}: {content}")
            if not sentences:
                continue
            # A single-sentence message: the sentence carries the whole message,
            # so it is maximally essential (masking it -> empty). Label directly
            # rather than sending an empty string to the rep server.
            if len(sentences) == 1:
                emit({"sentence": sentences[0], "label": 1.0, "source": source})
                continue
            # Build rep queries: full + one per masked sentence. Empty masks are
            # skipped per-item (labeled essential) instead of dropping the message.
            items = [{"text": content, "context": ctx}]
            valid_idx = []
            for i in range(len(sentences)):
                masked = " ".join(sentences[:i] + sentences[i + 1:]).strip()
                if masked:
                    valid_idx.append(i)
                    items.append({"text": masked, "context": ctx})
            try:
                reps = rep_batch(items)
            except Exception as e:
                print(f"  rep error on {source}: {e}", file=sys.stderr)
                continue
            full = reps[0]
            label_by_idx = {}
            for j, i in enumerate(valid_idx):
                label_by_idx[i] = 1.0 - cos(full, reps[j + 1])
            for i, s in enumerate(sentences):
                label = label_by_idx.get(i, 1.0)
                emit({"sentence": s, "label": round(label, 6), "source": source})

    fout.close()
    labels = np.array([r["label"] for r in rows]) if rows else np.array([0.0])
    print(f"sentences: {len(rows)}  label mean={labels.mean():.3f} "
          f"p25={np.percentile(labels,25):.3f} p50={np.percentile(labels,50):.3f} "
          f"p75={np.percentile(labels,75):.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
