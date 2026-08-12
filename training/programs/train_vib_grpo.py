#!/usr/bin/env python3
"""GRPO training of Method 2: a Variational Information-Bottleneck sender.

Policy: small instruct LM that re-encodes a transmission view into a message M.
Frozen receiver: Qwen3-4B on the dedicated harvesting server (port 8001).

Reward per completion (Eq. 6 in the method spec):
    R(M) = E_probes[ log q_phi(Y | primer, M) ] + beta * log p_prior(M)

- log q_phi(Y|primer,M): each harvested example carries several true/false
  probe statements about the task's actual info items (see
  harvest_vib_data.py); q_phi(Y|.) is the receiver's forced-choice
  True/False logprob for the CORRECT label, given ONLY the primer + M as
  context (not the original view) -- this is the same forced-choice-logprob
  extraction validated for Method 1, generalized to a binary label. Using
  probes sampled from a *distribution* of facts (not "what's your final
  vote") is what forces content-preserving rather than conclusion-preserving
  codes: a message that nudges the receiver to the right vote via spurious
  correlation without conveying the actual facts scores poorly here.
- log p_prior(M): codelength of M under the SAME frozen model used as a
  generic (unconditional) prior -- teacher-forced total logprob of M via
  echo+logprobs, tokenizer-independent unlike a char/token ratio.

beta is the rate-distortion knob (sweep it, same role as tau in Method 1).

Usage:
  python training/programs/train_vib_grpo.py [--steps 300] [--beta 0.1]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from datasets import Dataset
from openai import OpenAI

POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = "data/vib_train.jsonl"
OUT_DIR = "data/vib_grpo"
TARGET_RATE = 0.4
PRIMER = "A teammate sent you this note about the current situation:"

receiver = OpenAI(api_key="local", base_url="http://127.0.0.1:8001/v1")
_pool = ThreadPoolExecutor(max_workers=16)


def make_prompt(view: str) -> str:
    target = max(150, int(len(view) * TARGET_RATE))
    return (
        "Compress the following group-discussion transcript into a short note "
        f"of at most {target} characters that preserves the key facts a "
        "teammate would need. Use ONLY information already present in the "
        "transcript - do not add reasoning or conclusions of your own.\n\n"
        f"Transcript:\n{view}\n\nCompressed note:"
    )


def _forced_binary_logprob(context: str, correct_label: bool) -> float:
    """log q_phi(Y=correct_label | context), via forced-choice True/False."""
    prompt = (
        context +
        "\n\nIs the above statement true or false, based only on the note? "
        "Respond with exactly one word: True or False. Answer:"
    )
    try:
        r = receiver.completions.create(
            model="Qwen/Qwen3-4B", prompt=prompt, max_tokens=1,
            logprobs=20, temperature=0.0, timeout=30,
        )
        top = r.choices[0].logprobs.top_logprobs[0]
    except Exception:
        return float(np.log(0.5))  # uninformative fallback, not a crash

    def best(word):
        b = -1e9
        for tok, lp in top.items():
            if tok.strip() == word:
                b = max(b, lp)
        return b

    lp_true, lp_false = best("True"), best("False")
    m = max(lp_true, lp_false)
    p_true = np.exp(lp_true - m) / (np.exp(lp_true - m) + np.exp(lp_false - m))
    p_true = float(np.clip(p_true, 1e-4, 1 - 1e-4))
    return float(np.log(p_true if correct_label else 1 - p_true))


def _prior_codelength(message: str) -> float:
    """log p_prior(M): teacher-forced logprob of M under the frozen model,
    unconditional (minimal generic prefix), summed over tokens."""
    if not message.strip():
        return -50.0  # empty message: heavily penalized, degenerate
    try:
        r = receiver.completions.create(
            model="Qwen/Qwen3-4B", prompt=message, max_tokens=0,
            echo=True, logprobs=1, temperature=0.0, timeout=30,
        )
        lps = r.choices[0].logprobs.token_logprobs
        lps = [lp for lp in lps if lp is not None]
        return float(sum(lps)) if lps else -50.0
    except Exception:
        return -50.0


def vib_reward(prompts, completions, view, probes, beta, **kwargs):
    def score_one(args):
        comp, probe_list, b = args
        comp = comp.strip()
        if not comp:
            return -5.0
        ctx = f"{PRIMER}\n{comp}"
        relevance = np.mean([
            _forced_binary_logprob(ctx + f"\n\nStatement: {p['statement']}", p["label"])
            for p in probe_list
        ])
        rate = _prior_codelength(comp)
        return float(relevance + b * rate)

    b = beta[0] if isinstance(beta, list) else beta
    jobs = [(c, p, b) for c, p in zip(completions, probes)]
    return list(_pool.map(score_one, jobs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--beta", type=float, default=0.05,
                    help="Rate weight in the IB Lagrangian (higher = more compression pressure)")
    ap.add_argument("--max-examples", type=int, default=1200)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)][: args.max_examples]
    dataset = Dataset.from_list([
        {
            "prompt": make_prompt(r["view"]),
            "view": r["view"],
            "probes": r["probes"],
            "beta": args.beta,
        }
        for r in rows
        if len(r["view"]) < 6000
    ])
    print(f"training examples: {len(dataset)}, beta={args.beta}")

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
        save_total_limit=2,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=POLICY_MODEL,
        reward_funcs=vib_reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(OUT_DIR + "/final")
    print("saved", OUT_DIR + "/final")


if __name__ == "__main__":
    main()
