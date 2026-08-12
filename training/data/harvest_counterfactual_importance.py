#!/usr/bin/env python3
"""Harvest counterfactual token-importance labels for Method 1 (see handoff.md
'Counterfactual Token-Importance Filtering'). For each sentence span s in a
message, importance is the KL divergence the frozen receiver's belief
distribution over answer options undergoes when s is removed, averaged over
random maskings of the message's OTHER sentences (Shapley-style redundancy
correction). Labels are mined ONLY from held-out (non-eval) tasks.

Receiver belief q_phi(.|c,x) is extracted via forced-choice letter-label
logprobs (validated more reliable than raw continuation-likelihood scoring):
present the options as A)/B)/C).../ and read top_logprobs of the single
generated token.

Output: JSONL of {"sentence", "importance", "task", "source"}.
"""
import concurrent.futures
import json
import random
import re
import sys

import numpy as np
import requests

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
URL = "http://localhost:8001/v1/completions"
MODEL = "Qwen/Qwen3-4B"
N_MASKS = 3
MASK_KEEP_PROB = 0.5
MAX_WORKERS = 32

EVAL_REPORT = "reports/sweep_full_identity.json"
FULL_RUN_REPORTS = [
    "reports/hiddenbench_20260712_060139.json",
    "reports/hiddenbench_20260712_062305.json",
    "reports/results-opus-4.5.json",
]
BENCHMARK = "src/multimodal_comms/benchmarks/hiddenbench/data/hiddenbench_official/benchmark.json"
OUT = "outputs/hiddenbench/counterfactual_train.jsonl"


def option_distribution(context: str, options: list[str]) -> np.ndarray:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    opts_block = "\n".join(f"{l}) {o}" for l, o in zip(letters, options))
    prompt = (
        context + "\n" + opts_block +
        f"\nBased on the discussion, respond with exactly one letter "
        f"({', '.join(letters)}) and nothing else. Answer:"
    )
    r = requests.post(URL, json={
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "logprobs": 20, "temperature": 0,
    }, timeout=30)
    r.raise_for_status()
    top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    scores = []
    for l in letters:
        best = -1e9
        for tok, lp in top.items():
            if tok.strip() == l:
                best = max(best, lp)
        scores.append(best)
    scores = np.array(scores)
    probs = np.exp(scores - scores.max())
    return probs / probs.sum()


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def render_message(agent_name: str, sentences: list[str]) -> str:
    return f"{agent_name}: " + " ".join(sentences)


def score_sentence(scenario: str, options: list[str], agent_name: str,
                    sentences: list[str], target_idx: int, rng: random.Random) -> float:
    kls = []
    for _ in range(N_MASKS):
        others_kept = [
            i for i in range(len(sentences))
            if i == target_idx or rng.random() < MASK_KEEP_PROB
        ]
        with_span = [sentences[i] for i in others_kept]
        without_span = [sentences[i] for i in others_kept if i != target_idx]
        ctx_with = scenario + "\n" + render_message(agent_name, with_span)
        ctx_without = scenario + "\n" + render_message(agent_name, without_span) if without_span else scenario
        p_with = option_distribution(ctx_with, options)
        p_without = option_distribution(ctx_without, options)
        kls.append(kl(p_with, p_without))
    return float(np.mean(kls))


def main() -> None:
    eval_names = {r["task"]["name"] for r in json.load(open(EVAL_REPORT))["results"]}
    bench = json.load(open(BENCHMARK))
    bench_tasks = bench if isinstance(bench, list) else bench.get("tasks", bench)
    info_by_name = {
        t["name"]: {"options": list(t.get("possible_answers", [])),
                    "description": t.get("description", "")}
        for t in bench_tasks
    }

    jobs = []  # (scenario, options, agent_name, sentences, target_idx, seed) -> row meta
    rows_meta = []
    n_reports = 0
    for path in FULL_RUN_REPORTS:
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
            if len(meta["options"]) < 2:
                continue
            scenario = f"Scenario: {meta['description']}"
            for m in r.get("discussion_history", []):
                content = m.get("content", "")
                sentences = [s.strip() for s in SENTENCE_SPLIT.split(content) if s.strip()]
                if len(sentences) < 1:
                    continue
                agent_name = f"Agent {m['agent_id'] + 1}"
                for idx in range(len(sentences)):
                    rows_meta.append({
                        "sentence": sentences[idx], "task": name, "source": source,
                    })
                    jobs.append((scenario, meta["options"], agent_name, sentences, idx))

    print(f"reports used: {n_reports}, spans to score: {len(jobs)}")

    def run_job(i):
        scenario, options, agent_name, sentences, idx = jobs[i]
        rng = random.Random(1000 + i)
        try:
            return score_sentence(scenario, options, agent_name, sentences, idx, rng)
        except Exception as e:
            print(f"job {i} failed: {e}", file=sys.stderr)
            return None

    results = [None] * len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_job, i): i for i in range(len(jobs))}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} spans scored", file=sys.stderr)

    with open(OUT, "w") as f:
        for meta, imp in zip(rows_meta, results):
            if imp is None:
                continue
            meta["importance"] = imp
            f.write(json.dumps(meta) + "\n")

    n_ok = sum(1 for r in results if r is not None)
    print(f"scored {n_ok}/{len(jobs)} spans successfully")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
