#!/usr/bin/env python3
"""GRPO training of the A1 abstractive rewriter (rep-match reward).

Trains the iAgents representation-match rewriting policy.
Reward = cosine(rep(compressed), rep(original)); HARD max_completion_length is
the only length control (no codelength term -> no reward-hacking collapse).

Run on GPU 5:
  CUDA_VISIBLE_DEVICES=5 python -m training.benchmark_adaptation.iagents.train_repmatch_grpo
Saves to data/repmatch_grpo/final.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import requests
from datasets import Dataset

POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = "data/repmatch_rewriter_train.jsonl"
OUT_DIR = "data/repmatch_grpo"
REPSERVER = "http://127.0.0.1:8100"
TARGET_RATE = 0.4

_pool = ThreadPoolExecutor(max_workers=16)


def make_prompt(view):
    target = max(200, int(len(view) * TARGET_RATE))
    return (
        "Compress the following group-discussion transcript to at most "
        f"{target} characters so that a teammate reading only the "
        "compressed version would form the same understanding as reading "
        "the original. Use ONLY information already present in the "
        "transcript - do not add reasoning or conclusions of your own. "
        "Keep the 'Agent N: ...' format.\n\n"
        f"Transcript:\n{view}\n\nCompressed transcript:"
    )


def _rep(text):
    try:
        resp = requests.post(f"{REPSERVER}/rep", json={"text": text}, timeout=60)
        resp.raise_for_status()
        return resp.json()["rep"]
    except Exception:
        return None


def _cosine(a, b):
    return float(sum(x * y for x, y in zip(a, b)))


def repmatch_reward(prompts, completions, view, target_rep=None, **kwargs):
    texts = [c.strip() for c in completions]
    n = len(texts)
    comp_reps = list(_pool.map(lambda t: _rep(t) if t else None, texts))
    if target_rep is not None:
        targets = list(target_rep)
    else:
        targets = list(_pool.map(_rep, view))
    rewards = []
    for i in range(n):
        cr, tg = comp_reps[i], targets[i]
        if not texts[i] or cr is None or tg is None:
            rewards.append(-1.0)
        else:
            rewards.append(_cosine(cr, tg))
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max-examples", type=int, default=1200)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)][: args.max_examples]
    dataset = Dataset.from_list([
        {"prompt": make_prompt(r["view"]), "view": r["view"],
         "target_rep": r.get("target_rep")}
        for r in rows if len(r["view"]) < 6000
    ])
    print(f"training examples: {len(dataset)}")

    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir=OUT_DIR,
        max_steps=args.steps,
        per_device_train_batch_size=2,
        num_generations=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        max_prompt_length=2048,
        max_completion_length=400,
        temperature=0.9,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=1,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=POLICY_MODEL,
        reward_funcs=repmatch_reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(OUT_DIR + "/final")
    print("saved", OUT_DIR + "/final")


if __name__ == "__main__":
    main()
