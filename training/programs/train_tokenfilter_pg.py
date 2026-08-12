#!/usr/bin/env python3
"""Train the token-filtering policy via group-normalized REINFORCE with top-k compression.

Policy: lightweight MLP head on frozen Qwen3-4B that scores each token for
importance. Exactly k = rate * N tokens are kept (top-k). Multiple rollouts per
view (Gumbel-noise exploration) enable group-normalized REINFORCE: rewards are
normalized within the batch, so the policy learns which tokens matter MOST
relative to other possible selections of the same text.

Reward: sim_reward = max(0, min(-log(||h_comp - h_orig||^2 + 1e-4), 5.0))
No compression penalty — the fixed rate handles the trade-off.

Multiple keep_rates are used during training (cycling through 0.3, 0.5, 0.7)
so the policy learns to prioritize at different compression levels.

Usage:
  CUDA_VISIBLE_DEVICES=3 python training/programs/train_tokenfilter_pg.py
Saves to data/tokenfilter_pg/final.
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

from multimodal_comms.methods.text.token_filter_model import TokenFilterModel

DATA = "data/repmatch_rewriter_train.jsonl"
OUT_DIR = "data/tokenfilter_pg"
MODEL_NAME = "Qwen/Qwen3-4B"
LAYER = 14
MAX_CHARS = 6000
KEEP_RATES = [0.3, 0.5, 0.7]
NUM_SAMPLES = 8


def load_dataset(path: str, max_examples: int) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if len(r.get("view", "")) < MAX_CHARS]
    random.shuffle(rows)
    return rows[:max_examples]


def save_checkpoint(tf: TokenFilterModel, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save({"head": tf.head.state_dict()}, os.path.join(path, "head_weights.pt"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max-examples", type=int, default=418)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=200)
    args = ap.parse_args()

    print(f"[tokenfilter] loading {MODEL_NAME} on {args.device}, layer={LAYER}")
    tf = TokenFilterModel(model_name=MODEL_NAME, layer=LAYER, device=args.device)
    tf.load()
    print(f"[tokenfilter] head params: {sum(p.numel() for p in tf.head.parameters()):,}")

    dataset = load_dataset(DATA, args.max_examples)
    print(f"[tokenfilter] training examples: {len(dataset)}")

    optimizer = torch.optim.AdamW(tf.head.parameters(), lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        row = random.choice(dataset)
        keep_rate = KEEP_RATES[step % len(KEEP_RATES)]

        result = tf.rollout(row["view"], keep_rate=keep_rate, num_samples=NUM_SAMPLES, noise_scale=1.0)

        rewards = result["rewards"]       # (N,)
        log_probs = result["log_probs"]   # (N, seq_len)

        # Group normalization: standardize within the batch
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r  # (N,)

        # Per-token REINFORCE loss, normalized by sequence length
        mean_log_prob = log_probs.sum(dim=-1) / log_probs.shape[-1]  # (N,)
        loss = -(mean_log_prob * advantages.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(tf.head.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps - 1:
            print(
                f"step {step:4d}/{args.steps}  "
                f"rate: {keep_rate:.1f}  "
                f"loss: {loss.item():.4f}  "
                f"reward: {mean_r.item():.3f}±{std_r.item():.3f}  "
                f"grad: {grad_norm.item():.3f}",
                flush=True,
            )

        if (step + 1) % args.save_interval == 0 and step > 0:
            save_checkpoint(tf, OUT_DIR + f"/checkpoint-{step + 1}")

    final_dir = os.path.join(OUT_DIR, "final")
    save_checkpoint(tf, final_dir)
    print(f"[tokenfilter] saved {final_dir}")

    # Sanity check: example decisions at different rates
    print("\n[tokenfilter] --- example decisions ---")
    for keep_rate in [0.3, 0.5, 0.7]:
        row = random.choice(dataset)
        result = tf.rollout(row["view"], keep_rate=keep_rate, num_samples=1, noise_scale=0.0)
        keep_mask = result["keep_masks"][0]
        ids, _ = tf.get_token_hidden_states(row["view"])
        from multimodal_comms.methods.text.token_filter_model import build_compressed_text
        comp, kr = build_compressed_text(ids, tf.tokenizer, keep_mask)
        print(f"  keep_rate={keep_rate:.1f}: {len(row['view'])} chars -> "
              f"{len(comp) if comp else 0} chars "
              f"({kr:.1%} kept)  reward: {result['rewards'][0].item():.3f}")
        if comp:
            print(f"    preview: {comp[:150]}{'...' if len(comp) > 150 else ''}")

    tf.unload_model()


if __name__ == "__main__":
    main()
