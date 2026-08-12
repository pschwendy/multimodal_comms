#!/usr/bin/env python3
"""Harvest much longer (thousands-of-tokens) chunks from FineWeb-Edu, for an
autoencoder pilot that tests compression of long inputs into the same small
number of latents used for the short-message pilots.

Most FineWeb-Edu docs are too short for this (median ~535 tokens per a 300-doc
sample), so this filters to documents whose *token* length falls in
[MIN_TOKENS, MAX_TOKENS] and takes one chunk per qualifying document (a
random-length token window, decoded back to text), rather than aggressively
sub-chunking every document like harvest_fineweb_data.py does.

Output:
  data/fineweb_ae_longctx/train.jsonl  {"text": "<chunk>", "num_tokens": N}
  data/fineweb_ae_longctx/dev.jsonl
"""

import argparse
import json
import os
import random
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
BASE_MODEL = "Qwen/Qwen3-4B"
OUT_DIR = "data/fineweb_ae_longctx"
MIN_TOKENS = 1000
MAX_TOKENS = 4000
CHAR_PREFILTER = MIN_TOKENS * 3  # cheap skip before tokenizing (~3 chars/token floor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-examples", type=int, default=15000)
    ap.add_argument("--dev-examples", type=int, default=500)
    ap.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total_target = args.train_examples + args.dev_examples

    print(f"Loading tokenizer ({BASE_MODEL})...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Streaming {DATASET} ({DATASET_CONFIG}), targeting {total_target} "
          f"long chunks ({args.min_tokens}-{args.max_tokens} tokens each)...")
    ds = load_dataset(DATASET, name=DATASET_CONFIG, split="train", streaming=True)

    rows: list[dict] = []
    n_docs = 0
    n_candidates = 0
    for doc in ds:
        n_docs += 1
        text = doc.get("text", "")
        if len(text) < CHAR_PREFILTER:
            continue
        n_candidates += 1

        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < args.min_tokens:
            continue

        target_len = rng.randint(args.min_tokens, min(args.max_tokens, len(ids)))
        chunk_ids = ids[:target_len]
        chunk_text = tok.decode(chunk_ids, skip_special_tokens=True)
        rows.append({"text": chunk_text, "num_tokens": len(chunk_ids)})

        if len(rows) >= total_target:
            break
        if n_docs % 5000 == 0:
            print(f"  ...{n_docs} docs scanned, {n_candidates} candidates, "
                  f"{len(rows)} chunks so far", file=sys.stderr)

    print(f"Collected {len(rows)} chunks from {n_docs} documents "
          f"({n_candidates} passed the char prefilter)")

    rng.shuffle(rows)
    dev_rows = rows[: args.dev_examples]
    train_rows = rows[args.dev_examples: args.dev_examples + args.train_examples]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split_rows in [("train", train_rows), ("dev", dev_rows)]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")
        lens = [r["num_tokens"] for r in split_rows]
        avg_len = sum(lens) / len(lens) if lens else 0
        print(f"wrote {len(split_rows)} examples to {path} "
              f"(avg {avg_len:.0f} tokens, min {min(lens) if lens else 0}, "
              f"max {max(lens) if lens else 0})")


if __name__ == "__main__":
    main()
