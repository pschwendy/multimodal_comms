#!/usr/bin/env python3
"""Harvest training data for Method 2 (Variational Information-Bottleneck RL
sender). Unlike the earlier vote-matching rewriter, Y here is sampled from a
DISTRIBUTION of downstream true/false probes about the task's actual info
items, not the final vote -- this forces content-preserving rather than
conclusion-preserving codes (a message that happens to nudge the receiver to
the right vote without conveying WHY should not score well).

Probe construction: for each message, find the info item it most plausibly
conveys (embedding match to the task's shared/hidden info); the TRUE probe is
that fact restated as a statement. The FALSE probe is a random info item from
an UNRELATED task (guaranteed false in this scenario, no negation-NLG needed).

Output: data/vib_train.jsonl
  {"view", "description", "probes": [{"statement", "label": true/false}, ...], "task"}
"""
import json
import random
import re
import sys


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
EVAL_REPORT = "reports/sweep_full_identity.json"
FULL_RUN_REPORTS = [
    "reports/hiddenbench_20260712_060139.json",
    "reports/hiddenbench_20260712_062305.json",
    "reports/results-opus-4.5.json",
]
BENCHMARK = "src/multimodal_comms/benchmarks/hiddenbench/data/hiddenbench_official/benchmark.json"
OUT = "outputs/hiddenbench/vib_train.jsonl"
MIN_VIEW_MSGS = 3
PROBES_PER_VIEW = 3


def format_view(messages: list[dict]) -> str:
    return "\n".join(f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages)


def main() -> None:
    eval_names = {r["task"]["name"] for r in json.load(open(EVAL_REPORT))["results"]}
    bench = json.load(open(BENCHMARK))
    bench_tasks = bench if isinstance(bench, list) else bench.get("tasks", bench)
    meta_by_name = {t["name"]: t for t in bench_tasks}

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Pool of all info items across all (non-eval) tasks, for false-probe sampling
    all_info: list[tuple[str, str]] = []  # (task_name, statement)
    for name, meta in meta_by_name.items():
        if name in eval_names:
            continue
        for item in list(meta.get("shared_information", [])) + list(meta.get("hidden_information", [])):
            content = item.get("content") if isinstance(item, dict) else item
            if content:
                all_info.append((name, content))

    random.seed(31)
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
            own_info = [
                (item.get("content") if isinstance(item, dict) else item)
                for item in list(meta.get("shared_information", [])) + list(meta.get("hidden_information", []))
            ]
            own_info = [c for c in own_info if c]
            if not own_info:
                continue
            history = [
                {"agent_id": m["agent_id"], "content": m["content"]}
                for m in r.get("discussion_history", [])
            ]
            if len(history) < MIN_VIEW_MSGS:
                continue

            # A handful of cut points per discussion (same spirit as the rewriter harvest)
            cuts = random.sample(
                range(MIN_VIEW_MSGS, len(history) + 1),
                min(3, len(history) + 1 - MIN_VIEW_MSGS),
            )
            for cut in cuts:
                view = history[:cut]
                view_text = format_view(view)

                # true probes: info items whose content is semantically close to
                # something actually said in this view (so the probe is answerable
                # from the transcript, not from background world knowledge)
                view_sentences = [s for m in view for s in SENTENCE_SPLIT.split(m["content"]) if s.strip()]
                if not view_sentences:
                    continue
                view_emb = encoder.encode(view_sentences, normalize_embeddings=True)
                info_emb = encoder.encode(own_info, normalize_embeddings=True)
                sims = info_emb @ view_emb.T
                best_sim = sims.max(axis=1)
                true_candidates = [own_info[i] for i in range(len(own_info)) if best_sim[i] >= 0.55]
                if not true_candidates:
                    continue

                n_true = min(PROBES_PER_VIEW // 2 + 1, len(true_candidates))
                true_probes = random.sample(true_candidates, n_true)
                n_false = PROBES_PER_VIEW - n_true
                false_pool = [s for t, s in all_info if t != name]
                false_probes = random.sample(false_pool, min(n_false, len(false_pool)))

                probes = (
                    [{"statement": s, "label": True} for s in true_probes] +
                    [{"statement": s, "label": False} for s in false_probes]
                )
                random.shuffle(probes)
                rows.append({
                    "view": view_text,
                    "description": meta["description"],
                    "probes": probes,
                    "task": name,
                })

    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} views ({len({r['task'] for r in rows})} tasks) to {OUT}")


if __name__ == "__main__":
    main()
