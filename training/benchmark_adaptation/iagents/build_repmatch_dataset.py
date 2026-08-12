#!/usr/bin/env python3
"""Build the A2 representational-match selector training set from iAgents
identity-harvest transcripts.

Per-sentence labels measure representation change under sentence deletion:

  label = 1 - cosine( rep(full_message), rep(message with sentence removed) )

computed in-context against the shared representation server (127.0.0.1:8100).

Input: one or more offline-eval report JSONs produced by run_offline_eval.py
(each result carries "transcript" = [{agent_id, round_num, content}] and "id").
Output: data/repmatch_train.jsonl  {"sentence","label","task","source"}
"""

import argparse
import json
import re
import sys
import time

import requests

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
REPSERVER = "http://127.0.0.1:8100"
BATCH = 16


def format_context(messages, upto):
    return "\n".join(
        f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages[:upto]
    )


def _post_chunk(chunk, tries=8):
    """POST one chunk with retry/backoff; the shared rep server on GPU2 can
    transiently 500 under memory contention from the other tracks."""
    delay = 2.0
    for t in range(tries):
        try:
            resp = requests.post(f"{REPSERVER}/rep_batch",
                                 json={"items": chunk}, timeout=300)
            resp.raise_for_status()
            return resp.json()["reps"]
        except Exception as e:
            if t == tries - 1:
                if len(chunk) > 1:  # split and recurse as a last resort
                    mid = len(chunk) // 2
                    return _post_chunk(chunk[:mid]) + _post_chunk(chunk[mid:])
                raise RuntimeError(f"rep server failed on 1 item: {e}")
            time.sleep(delay)
            delay = min(delay * 1.7, 30.0)


def rep_batch(items):
    reps = []
    for start in range(0, len(items), BATCH):
        reps.extend(_post_chunk(items[start:start + BATCH]))
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+",
                    help="offline-eval identity report JSON(s)")
    ap.add_argument("--out", default="data/repmatch_train.jsonl")
    args = ap.parse_args()

    queries = []
    pending = []
    for path in args.reports:
        d = json.load(open(path))
        source = path.split("/")[-1]
        for r in d.get("results", []):
            name = str(r.get("id"))
            history = r.get("transcript", [])
            for i, m in enumerate(history):
                content = (m.get("content", "") or "").strip()
                sentences = [s.strip() for s in SENTENCE_SPLIT.split(content)
                             if s.strip()]
                if len(sentences) < 2 or not content:
                    continue
                context = format_context(history, i)
                full_qi = len(queries)
                queries.append({"text": content, "context": context})
                for j, sent in enumerate(sentences):
                    masked = " ".join(s for k, s in enumerate(sentences) if k != j).strip()
                    if not masked:  # server rejects empty text (500)
                        continue
                    masked_qi = len(queries)
                    queries.append({"text": masked, "context": context})
                    pending.append({
                        "sentence": sent, "task": name, "source": source,
                        "full_qi": full_qi, "masked_qi": masked_qi,
                    })

    print(f"queries: {len(queries)}  sentences: {len(pending)}  "
          f"(rep server, batch {BATCH})", flush=True)
    if not queries:
        print("no data; nothing written", file=sys.stderr)
        return
    reps = rep_batch(queries)
    if len(reps) != len(queries):
        raise RuntimeError(f"rep count {len(reps)} != query count {len(queries)}")

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    rows = []
    for p in pending:
        label = 1.0 - cosine(reps[p["full_qi"]], reps[p["masked_qi"]])
        rows.append({"sentence": p["sentence"], "label": label,
                     "task": p["task"], "source": p["source"]})

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    labels = [r["label"] for r in rows]
    mean = sum(labels) / len(labels) if labels else 0.0
    print(f"sentences: {len(rows)}  mean label: {mean:.4f}  "
          f"min {min(labels):.4f}  max {max(labels):.4f}")
    print(f"unique training tasks: {len({r['task'] for r in rows})}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
