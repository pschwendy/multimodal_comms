#!/usr/bin/env python3
"""GRPO training of the abstractive rewriter (representational-match reward).

Same TRL GRPOTrainer setup as training/programs/train_rewriter_grpo.py (policy
Qwen2.5-0.5B-Instruct, per_device_train_batch_size=2, num_generations=8,
gradient_accumulation_steps=8, lr=1e-5, max_prompt_length=2048,
temperature=0.9, bf16), but:

  * max_completion_length = 400 (HARD cap) is the ONLY length control. There
    is deliberately NO codelength / log-prob rate reward term. Part 5's VIB
    sender added `beta * log p_prior(M)` "to nudge brevity" and collapsed into
    degenerate low-surprisal repetition; the hard generation cap replaces it,
    exactly as in agentic_learning_algorithms/trainers/nlae_stream_engine.py.

  * reward = cosine( rep_server(compressed_completion), rep_server(original) )
    where the original's rep is the precomputed target_rep from harvest
    (else computed live from `view`). Cosine to the original rep already
    punishes degenerate/repetitive output (it drifts far from the original
    representation), so no separate brevity or anti-repetition term is needed.

Run on GPU 3:
  CUDA_VISIBLE_DEVICES=3 python training/programs/train_repmatch_grpo.py

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

# The shared rep server serializes forward passes behind a global lock, so
# heavy client concurrency only queues; 8 threads is plenty to hide HTTP
# latency without hammering the server the other tracks also use.
_pool = ThreadPoolExecutor(max_workers=8)


def make_prompt(view: str) -> str:
    # Mirror RepMatchRewriterCompressor.compress's deployment prompt exactly.
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


def _rep(text: str) -> list[float] | None:
    try:
        resp = requests.post(
            f"{REPSERVER}/rep", json={"text": text}, timeout=60
        )
        resp.raise_for_status()
        return resp.json()["rep"]
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    # Rep server returns L2-normalized vectors, so dot == cosine.
    return float(sum(x * y for x, y in zip(a, b)))


def _repetition_penalty(text: str) -> float:
    """Fraction-based degeneracy penalty. NOT a codelength/rate term: it does
    NOT reward brevity or low surprisal (the VIB mistake that made repetitive
    text 'cheap'); it does the OPPOSITE, penalizing token repetition directly.

    A first GRPO run with the pure-cosine reward collapsed into degenerate
    repetition ('... that that that that ...' padded to the 400-token cap):
    cosine to the original rep stayed a mediocre ~0.48 for such blobs, a flat
    local optimum GRPO could not climb out of. This term makes that blob score
    strongly negative so the policy is pushed toward genuinely distinct text,
    while faithful compressions (high trigram diversity) are unaffected.
    """
    import re
    words = re.findall(r"\w+", text.lower())
    tris = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if not tris:
        return 1.0
    distinct = len(set(tris)) / len(tris)
    return max(0.0, 0.6 - distinct)  # 0 when diverse, up to 0.6 when repetitive


def repmatch_reward(prompts, completions, view, target_rep=None, **kwargs):
    texts = [c.strip() for c in completions]
    n = len(texts)
    # Parallelize rep-server calls the same way the behavior rewriter
    # parallelizes receiver_vote calls (ThreadPoolExecutor over the batch).
    comp_reps = list(_pool.map(lambda t: _rep(t) if t else None, texts))

    # Targets: precomputed at harvest (target_rep column), else compute live.
    if target_rep is not None:
        targets = list(target_rep)
    else:
        targets = list(_pool.map(_rep, view))

    rewards = []
    for i in range(n):
        cr, tg = comp_reps[i], targets[i]
        if not texts[i] or cr is None or tg is None:
            rewards.append(-1.0)  # empty / unreachable -> worst reward
        else:
            rewards.append(_cosine(cr, tg) - 2.0 * _repetition_penalty(texts[i]))
    return rewards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-examples", type=int, default=1200)
    ap.add_argument("--resume", action="store_true",
                     help="Resume from the latest checkpoint in OUT_DIR if one exists.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)][: args.max_examples]
    dataset = Dataset.from_list([
        {
            "prompt": make_prompt(r["view"]),
            "view": r["view"],
            "target_rep": r.get("target_rep"),
        }
        for r in rows
        if len(r["view"]) < 6000
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
        max_completion_length=400,   # HARD cap; the ONLY length control.
        temperature=0.9,
        # Generation-time repetition penalty: suppresses degenerate repeated
        # rollouts at the source so the group of 8 samples carries real
        # variation for GRPO's advantage to act on. This shapes SAMPLING, not
        # the reward, and is unrelated to the forbidden codelength/rate term.
        repetition_penalty=1.3,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=POLICY_MODEL,
        reward_funcs=repmatch_reward,
        args=config,
        train_dataset=dataset,
    )

    resume_path = None
    if args.resume:
        import glob
        ckpts = sorted(
            glob.glob(f"{OUT_DIR}/checkpoint-*"),
            key=lambda p: int(p.rsplit("-", 1)[-1]),
        )
        if ckpts:
            resume_path = ckpts[-1]
            print(f"resuming from {resume_path}")

    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(OUT_DIR + "/final")
    print("saved", OUT_DIR + "/final")


if __name__ == "__main__":
    main()
