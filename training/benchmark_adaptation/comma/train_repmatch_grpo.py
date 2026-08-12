#!/usr/bin/env python3
"""GRPO training of the A1 repmatch_rewriter (representational-reconstruction reward).

Policy: small instruct LM that compresses a channel view.
Reward per completion (NLAE recipe port):
  + cos( rep(compressed), rep(original_view) )   representational fidelity (rep server)
  - 1.0 * max(0, novel_4gram_rate - 0.2)         anti-injection guard

Length is controlled ONLY by a hard max_completion_length cap (not a soft
codelength/rate term) - the named fix to the reward-hacking collapse.

Usage:
  CUDA_VISIBLE_DEVICES=4 python -m training.benchmark_adaptation.comma.train_repmatch_grpo [--steps 80]
"""
import argparse
import json
import os
import re

import numpy as np
import requests
from datasets import Dataset

REP_URL = os.getenv("REP_URL", "http://127.0.0.1:8100")
POLICY_MODEL = os.getenv("POLICY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA = "data/repmatch_grpo_train.jsonl"
OUT_DIR = "data/repmatch_grpo"
TARGET_RATE = 0.4


def make_prompt(view: str) -> str:
    target = max(200, int(len(view) * TARGET_RATE))
    return (
        "Compress the following group-discussion transcript to at most "
        f"{target} characters so that a teammate reading only the compressed "
        "version would form the same understanding as reading the original. "
        "Use ONLY information already present in the transcript - do not add "
        "reasoning or conclusions of your own. Keep the 'Agent N: ...' format.\n\n"
        f"Transcript:\n{view}\n\nCompressed transcript:"
    )


def rep_batch(texts):
    r = requests.post(f"{REP_URL}/rep_batch",
                      json={"items": [{"text": t, "context": ""} for t in texts]},
                      timeout=180)
    r.raise_for_status()
    return np.array(r.json()["reps"], dtype=np.float64)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ngram_set(text, n=4):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def repmatch_reward(prompts, completions, view, **kwargs):
    rewards = []
    # Batch rep server: for each completion pair (comp, its view)
    for comp, v in zip(completions, view):
        comp = (comp or "").strip()
        if not comp:
            rewards.append(-1.0)
            continue
        try:
            reps = rep_batch([comp, v])
            r = cos(reps[0], reps[1])
        except Exception:
            r = 0.0
        cg = ngram_set(comp)
        if cg:
            sg = ngram_set(v)
            novel = 1.0 - len(cg & sg) / len(cg)
            r -= 1.0 * max(0.0, novel - 0.2)
        rewards.append(float(r))
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max-examples", type=int, default=1000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)][: args.max_examples]
    dataset = Dataset.from_list([
        {"prompt": make_prompt(r["view"]), "view": r["view"]} for r in rows
    ])
    print(f"training examples: {len(dataset)}")

    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir=OUT_DIR,
        max_steps=args.steps,
        per_device_train_batch_size=4,
        num_generations=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        max_prompt_length=1536,
        max_completion_length=400,   # HARD length cap = the only length control
        temperature=0.9,
        bf16=True,
        logging_steps=2,
        save_steps=40,
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
