#!/usr/bin/env python3
"""Build the training dataset for the learnable sentence selector.

Mines discussion transcripts from existing full-run reports, splits messages
into sentences, and labels each sentence keep/drop:

  KEEP if the sentence semantically matches one of the task's information
  items (shared or hidden) or mentions an answer option (vote statements are
  needed for consensus).
  DROP otherwise (chit-chat, agreement without content, restatement fluff).

Tasks in the evaluation subset (the 16 sweep tasks) are STRICTLY excluded;
the selector never sees them at training time. Labels use dataset info
annotations of TRAINING tasks only — at inference the model receives raw
sentences with no task annotations, so nothing leaks.

Output: JSONL of {"sentence", "label", "task", "source"}.
"""

import argparse
import json
import re
import sys

import numpy as np

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SIM_THRESHOLD = 0.60  # sentence ~ info-item match threshold

EVAL_REPORT = "reports/sweep_full_identity.json"
FULL_RUN_REPORTS = [
    "reports/hiddenbench_20260712_060139.json",  # Qwen3-4B full run
    "reports/hiddenbench_20260712_062305.json",  # DeepSeek full run
    "reports/results-opus-4.5.json",             # Opus 4.5 reference run
]
BENCHMARK = "src/multimodal_comms/benchmarks/hiddenbench/data/hiddenbench_official/benchmark.json"
OUT = "outputs/hiddenbench/selector_train.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-report", default=EVAL_REPORT)
    parser.add_argument("--reports", nargs="+", default=FULL_RUN_REPORTS)
    parser.add_argument("--benchmark", default=BENCHMARK)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    eval_names = {
        r["task"]["name"] for r in json.load(open(args.eval_report))["results"]
    }

    bench = json.load(open(args.benchmark))
    bench_tasks = bench if isinstance(bench, list) else bench.get("tasks", bench)
    info_by_name = {
        t["name"]: {
            "info": list(t.get("shared_information", []))
                    + list(t.get("hidden_information", [])),
            "options": list(t.get("possible_answers", [])),
        }
        for t in bench_tasks
    }

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    rows = []
    n_reports = 0
    for path in args.reports:
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            print(f"skip missing {path}", file=sys.stderr)
            continue
        n_reports += 1
        source = path.split("/")[-1]
        for r in d.get("results", []):
            name = r["task"]["name"]
            if name in eval_names or name not in info_by_name:
                continue
            meta = info_by_name[name]
            if not meta["info"]:
                continue

            info_emb = model.encode(meta["info"], normalize_embeddings=True)
            # Distinctive option words (e.g. "Gamma" from "Data Center Gamma")
            # so vote statements like "Vote confirmed for Gamma" are kept
            stop = {"the", "a", "an", "of", "and", "or", "for", "to", "in",
                    "on", "at", "site", "option", "center", "camp", "new"}
            option_words = set()
            for o in meta["options"]:
                for w in re.findall(r"[A-Za-z]+", o.lower()):
                    if len(w) > 2 and w not in stop:
                        option_words.add(w)

            for m in r.get("discussion_history", []):
                content = m.get("content", "")
                sentences = [s.strip() for s in SENTENCE_SPLIT.split(content) if s.strip()]
                if not sentences:
                    continue
                sent_emb = model.encode(sentences, normalize_embeddings=True)
                sims = sent_emb @ info_emb.T  # (n_sent, n_info)
                for sent, sim_row in zip(sentences, sims):
                    sent_words = set(re.findall(r"[A-Za-z]+", sent.lower()))
                    has_option = bool(option_words & sent_words)
                    label = int(float(sim_row.max()) >= SIM_THRESHOLD or has_option)
                    rows.append({
                        "sentence": sent,
                        "label": label,
                        "task": name,
                        "source": source,
                    })

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_keep = sum(r["label"] for r in rows)
    print(f"reports used: {n_reports}")
    print(f"sentences: {len(rows)}  keep: {n_keep} ({n_keep/len(rows):.1%})  "
          f"drop: {len(rows)-n_keep}")
    print(f"unique training tasks: {len({r['task'] for r in rows})}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
