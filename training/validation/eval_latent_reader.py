#!/usr/bin/env python3
"""Evaluate a latent-reader checkpoint on QuALITY validation.

Same prompt assembly as train_latent_reader.py; generates with sampling and
scores the \\boxed answer. Shardable across GPUs via --shard-id/--num-shards.
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.programs.train_latent_reader import SYSTEM, HEADER, NUM_LATENTS, question_block

LETTERS = ["a", "b", "c", "d"]


def extract_boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1].strip().strip("$").lower()[:1] if m else ""


@torch.no_grad()
def run_item(model, tok, embed_layer, meta, latents, device, args):
    lat_block = "".join(f"<|L{i}|>" for i in range(NUM_LATENTS))
    user = (HEADER + "\n" + lat_block * meta["n_chunks"] + "\n\n"
            + question_block(meta["question"], meta["options"]))
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    li_ids = {tok.convert_tokens_to_ids(f"<|L{i}|>") for i in range(NUM_LATENTS)}
    pos = [p for p, t in enumerate(ids[0].tolist()) if t in li_ids]
    assert len(pos) == NUM_LATENTS * meta["n_chunks"]
    embeds = embed_layer(ids)
    flat = latents.reshape(-1, latents.shape[-1]).to(device=device, dtype=embeds.dtype)
    embeds[0, torch.tensor(pos, device=device), :] = flat
    out = model.generate(
        inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
        max_new_tokens=args.max_new_tokens, do_sample=True,
        temperature=0.6, top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latent-dir", default="data/quality_latents")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--wrong-latents", action="store_true",
                    help="control: feed latents from a DIFFERENT article "
                         "(rolled by one question), tiled/trimmed to n_chunks")
    args = ap.parse_args()

    torch.manual_seed(args.seed + args.shard_id)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()
    embed_layer = model.get_input_embeddings()

    rows = [json.loads(l) for l in
            open(os.path.join(args.latent_dir, f"index_{args.split}.jsonl"))]
    rows = rows[args.offset:args.offset + args.max_samples]
    rows = rows[args.shard_id::args.num_shards]

    lat_dir = os.path.join(args.latent_dir, "latents")
    correct = 0
    results = []
    shas = [r["article_sha"] for r in rows]
    for i, meta in enumerate(rows):
        sha = meta["article_sha"]
        if args.wrong_latents:
            others = [s for s in shas if s != meta["article_sha"]]
            sha = others[i % len(others)]
        lats = torch.load(os.path.join(lat_dir, f"{sha}.pt"),
                          map_location="cpu", weights_only=True)["latents"]
        if args.wrong_latents:
            n = meta["n_chunks"]
            reps = (n + lats.shape[0] - 1) // lats.shape[0]
            lats = lats.repeat(reps, 1, 1)[:n]
        text = run_item(model, tok, embed_layer, meta, lats, args.device, args)
        pred = extract_boxed(text)
        gold = LETTERS[meta["answer"]]
        ok = pred == gold
        correct += int(ok)
        results.append({"qidx": meta["qidx"], "pred": pred, "gold": gold, "ok": ok})
        print(f"[{i + 1}/{len(rows)}] pred={pred} gold={gold} ok={ok} "
              f"acc={correct / (i + 1):.3f}", flush=True)

    out = {"method": f"latent_reader_{args.tag}", "checkpoint": args.checkpoint,
           "split": args.split, "shard": args.shard_id, "n": len(rows),
           "accuracy": correct / max(1, len(rows)), "correct": correct}
    print(json.dumps(out))
    os.makedirs("data/latent_reader_eval", exist_ok=True)
    with open(f"data/latent_reader_eval/{args.tag}_shard{args.shard_id}.json", "w") as f:
        json.dump({**out, "results": results}, f)


if __name__ == "__main__":
    main()
