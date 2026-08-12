#!/usr/bin/env python3
"""GKD phase 2: teacher-score student rollouts with vLLM prompt_logprobs.

The frozen base teacher sees [full article ceiling prompt + student trace]
as token ids and returns per-position top-K logprobs over the trace span.
Processes prompts in small chunks because vLLM materializes prompt_logprobs
for every position (article included), which is RAM-heavy.

Output: <out>/scored_shard<K>.pt  list of
  {qidx, trace_ids: int32 (L,), topk_ids: int32 (L, K+1), topk_lp: fp16 (L, K+1)}
(K+1 because the realized token is appended when outside top-K; rows padded
with id=-1, lp=-inf.)
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL = "Qwen/Qwen3-4B"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def question_block(item):
    o = item["options"]
    return (f"Question: {item['question']}\n"
            f"A. {o[0]}\nB. {o[1]}\nC. {o[2]}\nD. {o[3]}\n\n"
            "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
            "Your final answer must be one of A,B,C,D. Do not add any other contents inside the box.")


def teacher_prompt_ids(tok, item):
    user = (f"Read the following document and answer the question.\n\n"
            f"Document:\n{item['article']}\n\n{question_block(item)}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="rollouts_shard<K>.jsonl")
    ap.add_argument("--out", default="data/gkd/scored")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    rolls = [json.loads(l) for l in open(args.rollouts)]
    ds = load_dataset("emozilla/quality", split="train")
    tok = AutoTokenizer.from_pretrained(MODEL)
    base_vocab = len(tok)  # <|Li|> ids from the student are >= this
    prompts_cache = {}

    # prompt_logprobs materializes [prefill_chunk, vocab] logits: keep the
    # prefill chunk small and leave headroom outside the KV preallocation.
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.7, enable_prefix_caching=True,
              max_num_batched_tokens=2048, enable_chunked_prefill=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=args.topk,
                       temperature=0.0, detokenize=False)

    out_rows = []
    skipped = 0
    for c0 in range(0, len(rolls), args.chunk):
        chunk = rolls[c0:c0 + args.chunk]
        reqs, keep = [], []
        for r in chunk:
            if any(t >= base_vocab for t in r["trace_ids"]) or not r["trace_ids"]:
                skipped += 1
                continue
            qidx = r["qidx"]
            if qidx not in prompts_cache:
                prompts_cache[qidx] = teacher_prompt_ids(tok, ds[qidx])
            pids = prompts_cache[qidx]
            if len(pids) + len(r["trace_ids"]) > args.max_model_len:
                skipped += 1
                continue
            reqs.append(TokensPrompt(prompt_token_ids=pids + r["trace_ids"]))
            keep.append((r, len(pids)))
        if not reqs:
            continue
        outs = llm.generate(reqs, sp, use_tqdm=False)
        for (r, plen), o in zip(keep, outs):
            plps = o.prompt_logprobs  # list, len == prompt tokens, [0] is None
            L = len(r["trace_ids"])
            K = args.topk + 1
            ids_m = torch.full((L, K), -1, dtype=torch.int32)
            lp_m = torch.full((L, K), float("-inf"), dtype=torch.float32)
            for j in range(L):
                entry = plps[plen + j]
                items = list(entry.items())[:K]
                for k, (tid, lp) in enumerate(items):
                    ids_m[j, k] = tid
                    lp_m[j, k] = lp.logprob
            out_rows.append({"qidx": r["qidx"],
                             "trace_ids": torch.tensor(r["trace_ids"], dtype=torch.int32),
                             "topk_ids": ids_m, "topk_lp": lp_m.to(torch.float16)})
        if (c0 // args.chunk) % 5 == 0:
            print(f"[score shard {args.shard_id}] {c0 + len(chunk)}/{len(rolls)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"scored_shard{args.shard_id}.pt")
    torch.save(out_rows, path)
    print(f"[score shard {args.shard_id}] wrote {len(out_rows)} scored "
          f"({skipped} skipped) -> {path}", flush=True)


if __name__ == "__main__":
    main()
