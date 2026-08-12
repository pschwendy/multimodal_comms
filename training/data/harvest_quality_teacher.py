#!/usr/bin/env python3
"""Teacher traces / baseline arms for the latent-reader experiment (vLLM).

Modes (--arm):
  teacher   - QuALITY train, full-article prompt, N samples/question at T=0.6;
              keeps up to --max-keep traces whose \\boxed answer is correct.
              Output: <out>/traces_shard<K>.jsonl
  ceiling   - full article + question, 1 sample (accuracy anchor)
  floor     - question only
  truncate  - last --trunc-tokens article tokens + question

Sharding: --shard-id/--num-shards slice questions[shard::num_shards].
"""

import argparse
import hashlib
import json
import os
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
LETTERS = ["a", "b", "c", "d"]


def question_block(item) -> str:
    o = item["options"]
    return (f"Question: {item['question']}\n"
            f"A. {o[0]}\nB. {o[1]}\nC. {o[2]}\nD. {o[3]}\n\n"
            "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
            "Your final answer must be one of A,B,C,D. Do not add any other contents inside the box.")


def extract_boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not m:
        return ""
    return m[-1].strip().strip("$").lower()[:1]


def article_sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_prompt(tok, item, arm, trunc_tokens):
    if arm in ("teacher", "ceiling"):
        user = (f"Read the following document and answer the question.\n\n"
                f"Document:\n{item['article']}\n\n{question_block(item)}")
    elif arm == "floor":
        user = ("Answer the question about a story you have not been shown — "
                "give your best guess.\n\n" + question_block(item))
    elif arm == "truncate":
        ids = tok(item["article"], add_special_tokens=False)["input_ids"]
        tail = tok.decode(ids[-trunc_tokens:], skip_special_tokens=True)
        user = (f"Read the following excerpt (the end of a longer document) and answer "
                f"the question about the full document — give your best supported answer."
                f"\n\nExcerpt:\n{tail}\n\n{question_block(item)}")
    else:
        raise ValueError(arm)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["teacher", "ceiling", "floor", "truncate"], required=True)
    ap.add_argument("--split", default=None, help="default: train for teacher, validation otherwise")
    ap.add_argument("--out", default="data/quality_teacher")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-keep", type=int, default=3)
    ap.add_argument("--trunc-tokens", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()

    split = args.split or ("train" if args.arm == "teacher" else "validation")
    ds = load_dataset("emozilla/quality", split=split)
    items = [(i, ds[i]) for i in range(len(ds))]
    if args.max_samples:
        items = items[:args.max_samples]
    items = items[args.shard_id::args.num_shards]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(items)} questions, arm={args.arm}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [build_prompt(tok, it, args.arm, args.trunc_tokens) for _, it in items]

    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.9)
    n = args.n_samples if args.arm == "teacher" else 1
    sp = SamplingParams(n=n, temperature=0.6, top_p=0.95,
                        max_tokens=args.max_new_tokens, seed=42 + args.shard_id)
    outs = llm.generate(prompts, sp)

    os.makedirs(args.out, exist_ok=True)
    if args.arm == "teacher":
        path = os.path.join(args.out, f"traces_shard{args.shard_id}.jsonl")
        kept = total_correct = 0
        with open(path, "w") as f:
            for (qidx, item), out in zip(items, outs):
                gold = LETTERS[item["answer"]]
                traces = [c.text for c in out.outputs
                          if extract_boxed(c.text) == gold]
                total_correct += len(traces)
                for t in traces[:args.max_keep]:
                    f.write(json.dumps({
                        "qidx": qidx, "split": split,
                        "article_sha": article_sha(item["article"]),
                        "gold": gold, "trace": t}) + "\n")
                    kept += 1
        print(json.dumps({"shard": args.shard_id, "questions": len(items),
                          "correct_traces": total_correct, "kept": kept,
                          "pass_at_4": sum(1 for (qi, it), o in zip(items, outs)
                                           if any(extract_boxed(c.text) == LETTERS[it["answer"]]
                                                  for c in o.outputs)) / max(1, len(items))}))
    else:
        correct = 0
        for (qidx, item), out in zip(items, outs):
            pred = extract_boxed(out.outputs[0].text)
            correct += int(pred == LETTERS[item["answer"]])
        tag = args.arm + (f"_{args.trunc_tokens}" if args.arm == "truncate" else "")
        res = {"method": f"quality4b_{tag}", "split": split, "n": len(items),
               "accuracy": correct / len(items), "correct": correct,
               "shard": args.shard_id}
        print(json.dumps(res))
        with open(os.path.join(args.out, f"baseline_{tag}_shard{args.shard_id}.json"), "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
