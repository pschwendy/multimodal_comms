#!/usr/bin/env python3
"""GRPO training of the abstractive rewriter (behavior-matching reward).

Policy: small instruct LM that compresses a transmission view.
Frozen receiver: Qwen3-4B on the dedicated GPU-1 vLLM server (port 8001).

Reward per completion:
  +1.0  if the receiver's vote given the COMPRESSED view matches its vote
        given the FULL view (behavior preservation, precomputed at harvest)
  +0.5 * (1 - len_ratio)          shorter is better
  -1.0 * max(0, novel_rate - 0.2) anti-injection: penalize content whose
        word 4-grams don't appear in the source view

The policy can only be rewarded for keeping the receiver's behavior with
fewer characters; injecting external reasoning is penalized directly.

Usage:
  python training/programs/train_rewriter_grpo.py [--steps 300] [--device cuda:2]
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset
from openai import OpenAI

import sys
from training.utils import parse_choice

POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = "data/rewriter_train.jsonl"
OUT_DIR = "data/rewriter_grpo"
TARGET_RATE = 0.4

receiver = OpenAI(api_key="local", base_url="http://127.0.0.1:8001/v1")
_pool = ThreadPoolExecutor(max_workers=16)


def make_prompt(view: str) -> str:
    target = max(200, int(len(view) * TARGET_RATE))
    return (
        "Compress the following group-discussion transcript to at most "
        f"{target} characters. Keep only decision-relevant facts and each "
        "agent's stated preference. Use ONLY information already present in "
        "the transcript - do not add reasoning or conclusions of your own. "
        "Keep the 'Agent N: ...' format.\n\n"
        f"Transcript:\n{view}\n\nCompressed transcript:"
    )


def receiver_vote(description: str, options: list[str], view_text: str) -> str:
    options_text = "\n".join(f"- {o}" for o in options)
    prompt = (
        f"## Scenario\n{description}\n\n"
        f"## Group Discussion\n{view_text}\n\n"
        f"## Available Options\n{options_text}\n\n"
        'Based on the discussion, respond with your decision in JSON format:\n'
        '{"vote": "<one of the options above>", "rationale": "<brief>"}'
    )
    try:
        response = receiver.chat.completions.create(
            model="Qwen/Qwen3-4B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
            timeout=60,
        )
        vote, _ = parse_choice(response.choices[0].message.content or "", options)
        return vote
    except Exception:
        return ""


def ngram_set(text: str, n: int = 4) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def behavior_reward(prompts, completions, view, description, options, full_vote, **kwargs):
    def score_one(args):
        comp, v, desc, opts, fvote = args
        comp = comp.strip()
        if not comp:
            return -1.0
        # Behavior preservation
        vote = receiver_vote(desc, opts, comp)
        r = 1.0 if (vote and vote == fvote) else 0.0
        # Brevity
        ratio = min(len(comp) / max(len(v), 1), 1.0)
        r += 0.5 * (1.0 - ratio)
        # Anti-injection: novel 4-grams not present in the source view
        comp_grams = ngram_set(comp)
        if comp_grams:
            src_grams = ngram_set(v)
            novel_rate = 1.0 - len(comp_grams & src_grams) / len(comp_grams)
            r -= 1.0 * max(0.0, novel_rate - 0.2)
        return r

    jobs = list(zip(completions, view, description, options, full_vote))
    return list(_pool.map(score_one, jobs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-examples", type=int, default=1200)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)][: args.max_examples]
    dataset = Dataset.from_list([
        {
            "prompt": make_prompt(r["view"]),
            "view": r["view"],
            "description": r["description"],
            "options": r["options"],
            "full_vote": r["full_vote"],
        }
        for r in rows
        if len(r["view"]) < 6000
    ])
    print(f"training examples: {len(dataset)}")

    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir=OUT_DIR,
        max_steps=args.steps,
        # Micro-batch 2 x accum 8: full-vocab logits for a 3.5K-token
        # sequence are ~1.1GB each; batch 8 OOMs a 44GB L40S.
        per_device_train_batch_size=2,
        num_generations=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        max_prompt_length=2048,
        max_completion_length=512,
        temperature=0.9,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=POLICY_MODEL,
        reward_funcs=behavior_reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(OUT_DIR + "/final")
    print("saved", OUT_DIR + "/final")


if __name__ == "__main__":
    main()
