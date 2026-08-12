#!/usr/bin/env python3
"""Harvest real token-embedding packets for compressed-sensing eval.

Packets here are raw token embeddings -- the model's embed_tokens lookup,
not processed hidden states from a forward pass. Each packet row is
therefore *exactly* one row of the model's embedding table, i.e. exactly
1-sparse in that table (see CodebookCS in
the reusable sensing codecs). Concretely: we save the embedding
table once (the shared "dictionary", already known to both endpoints as
part of the model weights -- nothing to fit or transmit) plus a stream of
real token ids from FineWeb-Edu text (the actual traffic distribution over
which atom of the table gets used). No forward pass through the transformer
is needed -- just tokenization and an embedding-table lookup.

Output, under --out-dir:
  embedding_table.pt  (vocab_size, hidden_size) float32
  eval_token_ids.pt    (N,) int64 -- real token ids, reshape by the eval
                        script into (n_packets, packet_size) and look up
                        rows of embedding_table.pt to get packets
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
MODEL = "Qwen/Qwen2.5-3B"


def collect_token_ids(tokenizer, doc_iter, target_tokens, max_len):
    ids: list[int] = []
    for text in doc_iter:
        if len(ids) >= target_tokens:
            break
        enc = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
        if len(enc) < 8:
            continue
        ids.extend(enc)
    return torch.tensor(ids[:target_tokens], dtype=torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-dir", default="data/cs_packets")
    ap.add_argument("--eval-tokens", type=int, default=10000)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    embedding_table = model.get_input_embeddings().weight.detach().to(torch.float32)
    print(f"  embedding table: {tuple(embedding_table.shape)}")
    del model  # only needed the embedding weights, no forward pass required

    ds = load_dataset(DATASET, name=DATASET_CONFIG, split="train", streaming=True)
    doc_iter = (row["text"] for row in ds)

    print(f"Collecting {args.eval_tokens} real token ids ...")
    token_ids = collect_token_ids(tokenizer, doc_iter, args.eval_tokens, args.max_len)
    print(f"  got {token_ids.shape}, {token_ids.unique().numel()} distinct tokens")

    torch.save(embedding_table, os.path.join(args.out_dir, "embedding_table.pt"))
    torch.save(token_ids, os.path.join(args.out_dir, "eval_token_ids.pt"))
    meta = {
        "model": args.model,
        "vocab_size": embedding_table.shape[0],
        "hidden_size": embedding_table.shape[1],
        "eval_tokens": token_ids.shape[0],
        "distinct_tokens": int(token_ids.unique().numel()),
    }
    print(f"Wrote {args.out_dir}/{{embedding_table,eval_token_ids}}.pt -- {meta}")


if __name__ == "__main__":
    main()
