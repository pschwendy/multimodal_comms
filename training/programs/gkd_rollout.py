#!/usr/bin/env python3
"""GKD phase 1: sample on-policy traces from the latent-reader student.

For every QuALITY train question, builds the same latent prompt as
train_latent_reader.py and samples one trace (T=1.0 by default, proper
on-policy distribution). <|Li|> tokens are banned from generation so the
trace stays inside the base-model vocab for teacher scoring.

Output: <out>/rollouts_shard<K>.jsonl  {qidx, article_sha, n_chunks,
                                        trace_ids: [int], trace_text}
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.programs.train_latent_reader import SYSTEM, HEADER, NUM_LATENTS, question_block


def build_prompt_ids(tok, meta):
    lat_block = "".join(f"<|L{i}|>" for i in range(NUM_LATENTS))
    user = (HEADER + "\n" + lat_block * meta["n_chunks"] + "\n\n"
            + question_block(meta["question"], meta["options"]))
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    return tok(prompt, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def rollout_batch(model, tok, embed_layer, metas, latents_list, device, args, li_ids):
    """Left-padded batched generation with injected latent embeddings."""
    prompts = [build_prompt_ids(tok, m) for m in metas]
    maxlen = max(len(p) for p in prompts)
    pad_id = tok.pad_token_id or tok.eos_token_id
    B = len(prompts)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    for b, p in enumerate(prompts):
        ids[b, maxlen - len(p):] = torch.tensor(p)
        attn[b, maxlen - len(p):] = 1
    ids, attn = ids.to(device), attn.to(device)
    embeds = embed_layer(ids).clone()
    li_set = set(li_ids)
    for b, (p, lats) in enumerate(zip(prompts, latents_list)):
        off = maxlen - len(p)
        pos = [off + i for i, t in enumerate(p) if t in li_set]
        flat = lats.reshape(-1, lats.shape[-1]).to(device=device, dtype=embeds.dtype)
        assert len(pos) == flat.shape[0]
        embeds[b, torch.tensor(pos, device=device), :] = flat
    out = model.generate(
        inputs_embeds=embeds, attention_mask=attn,
        max_new_tokens=args.max_new_tokens, do_sample=True,
        temperature=args.temperature, top_p=args.top_p,
        bad_words_ids=[[i] for i in li_ids],
        pad_token_id=pad_id)
    results = []
    eos = tok.eos_token_id
    for b in range(B):
        seq = out[b].tolist()
        if eos in seq:
            seq = seq[:seq.index(eos) + 1]  # keep <|im_end|>
        results.append(seq)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latent-dir", default="data/quality_latents")
    ap.add_argument("--out", default="data/gkd/rollouts")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed * 1000 + args.shard_id)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()
    embed_layer = model.get_input_embeddings()
    li_ids = [tok.convert_tokens_to_ids(f"<|L{i}|>") for i in range(NUM_LATENTS)]

    rows = [json.loads(l) for l in
            open(os.path.join(args.latent_dir, "index_train.jsonl"))]
    rows = rows[args.shard_id::args.num_shards]
    lat_dir = os.path.join(args.latent_dir, "latents")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"rollouts_shard{args.shard_id}.jsonl")
    with open(path, "w") as f:
        for i in range(0, len(rows), args.batch_size):
            metas = rows[i:i + args.batch_size]
            lats = [torch.load(os.path.join(lat_dir, f"{m['article_sha']}.pt"),
                               map_location="cpu", weights_only=True)["latents"]
                    for m in metas]
            seqs = rollout_batch(model, tok, embed_layer, metas, lats,
                                 args.device, args, li_ids)
            for m, s in zip(metas, seqs):
                f.write(json.dumps({
                    "qidx": m["qidx"], "article_sha": m["article_sha"],
                    "n_chunks": m["n_chunks"], "trace_ids": s,
                    "trace_text": tok.decode(s, skip_special_tokens=True)}) + "\n")
            if (i // args.batch_size) % 5 == 0:
                print(f"[shard {args.shard_id}] {i + len(metas)}/{len(rows)}", flush=True)
    print(f"[shard {args.shard_id}] wrote {len(rows)} rollouts -> {path}", flush=True)


if __name__ == "__main__":
    main()
