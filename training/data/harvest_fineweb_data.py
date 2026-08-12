#!/usr/bin/env python3
"""Harvest message-length text chunks from FineWeb-Edu for large-scale
autoencoder pretraining.

Streams HuggingFaceFW/fineweb-edu (sample-10BT split) so nothing has to be
downloaded in full, splits each document into sentence-grouped chunks whose
length is randomly targeted within [MIN_CHARS, MAX_CHARS] (matching the
short-message regime used by the communication experiments), caps
chunks-per-doc so no single page dominates the corpus,
and writes disjoint train/dev JSONL files.

Output:
  data/fineweb_ae/train.jsonl  {"text": "<chunk>"}
  data/fineweb_ae/dev.jsonl    {"text": "<chunk>"}
"""

import argparse
import json
import random
import re
import sys

from datasets import load_dataset

DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
OUT_DIR = "data/fineweb_ae"
MIN_CHARS = 40
MAX_CHARS = 1200
MAX_CHUNKS_PER_DOC = 3

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_document(text: str, rng: random.Random) -> list[str]:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    target = rng.randint(MIN_CHARS, MAX_CHARS)

    def flush():
        if buf:
            chunk = " ".join(buf).strip()
            if len(chunk) >= MIN_CHARS:
                chunks.append(chunk[:MAX_CHARS])

    for sent in sentences:
        buf.append(sent)
        cur_len = sum(len(s) for s in buf) + len(buf) - 1
        if cur_len >= target:
            flush()
            buf = []
            target = rng.randint(MIN_CHARS, MAX_CHARS)
            if len(chunks) >= MAX_CHUNKS_PER_DOC:
                break
    else:
        flush()

    return chunks[:MAX_CHUNKS_PER_DOC]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-examples", type=int, default=60000)
    ap.add_argument("--dev-examples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total_target = args.train_examples + args.dev_examples

    print(f"Streaming {DATASET} ({DATASET_CONFIG}), targeting {total_target} chunks...")
    ds = load_dataset(DATASET, name=DATASET_CONFIG, split="train", streaming=True)

    rows: list[dict[str, str]] = []
    n_docs = 0
    for doc in ds:
        n_docs += 1
        text = doc.get("text", "")
        if not text:
            continue
        for chunk in chunk_document(text, rng):
            rows.append({"text": chunk})
        if len(rows) >= total_target:
            break
        if n_docs % 5000 == 0:
            print(f"  ...{n_docs} docs scanned, {len(rows)} chunks so far", file=sys.stderr)

    print(f"Collected {len(rows)} chunks from {n_docs} documents")

    rng.shuffle(rows)
    dev_rows = rows[: args.dev_examples]
    train_rows = rows[args.dev_examples : args.dev_examples + args.train_examples]

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    for name, split_rows in [("train", train_rows), ("dev", dev_rows)]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")
        lens = [len(r["text"]) for r in split_rows]
        avg_len = sum(lens) / len(lens) if lens else 0
        print(f"wrote {len(split_rows)} examples to {path} (avg {avg_len:.0f} chars)")


if __name__ == "__main__":
    main()
