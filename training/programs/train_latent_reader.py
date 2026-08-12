#!/usr/bin/env python3
"""Off-policy distillation: teach a base LM to reason over AE chunk latents.

Student init: the longctx AE checkpoint (it already reads <|Li|> latent slots).
Input per example:
  <|im_start|>system\n{SYSTEM}<|im_end|>\n
  <|im_start|>user\n{HEADER}\n[<|L0|>..<|L15|>]*n_chunks\n\n{question_block}<|im_end|>\n
  <|im_start|>assistant\n{teacher trace}<|im_end|>
with the cached chunk latents scattered into the input embeddings at the
<|Li|> positions (16 per chunk, same ids repeated; RoPE disambiguates).
Loss: CE on the assistant span only (hard-token sequence-level KD).

Run (DDP):
  torchrun --standalone --nproc_per_node=9 training/programs/train_latent_reader.py
"""

import argparse
import contextlib
import glob
import json
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
HEADER = ("You have read a document that is stored in your memory as the following "
          "sequence of latent memory tokens (16 per document segment, in order):")
NUM_LATENTS = 16


def question_block(question, options):
    return (f"Question: {question}\n"
            f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n\n"
            "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
            "Your final answer must be one of A,B,C,D. Do not add any other contents inside the box.")


def build_example(tok, meta, trace, max_trace_tokens):
    """Returns (ids, latent_positions, prompt_len). latent_positions: flat list,
    chunk-major, matching latents.reshape(-1, H) order."""
    lat_block = "".join(f"<|L{i}|>" for i in range(NUM_LATENTS))
    user = (HEADER + "\n" + lat_block * meta["n_chunks"] + "\n\n"
            + question_block(meta["question"], meta["options"]))
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    trace_ids = tok(trace + "<|im_end|>", add_special_tokens=False,
                    truncation=True, max_length=max_trace_tokens)["input_ids"]
    li_ids = {tok.convert_tokens_to_ids(f"<|L{i}|>") for i in range(NUM_LATENTS)}
    latent_pos = [p for p, t in enumerate(prompt_ids) if t in li_ids]
    assert len(latent_pos) == NUM_LATENTS * meta["n_chunks"]
    return prompt_ids + trace_ids, latent_pos, len(prompt_ids)


def load_data(latent_dir, trace_glob):
    index = {}
    for row in map(json.loads, open(os.path.join(latent_dir, "index_train.jsonl"))):
        index[row["qidx"]] = row
    examples = []
    for path in sorted(glob.glob(trace_glob)):
        for row in map(json.loads, open(path)):
            meta = index[row["qidx"]]
            examples.append({"meta": meta, "trace": row["trace"]})
    return examples


class LatentCache:
    def __init__(self, latent_dir):
        self.dir = os.path.join(latent_dir, "latents")
        self.cache = {}

    def get(self, sha):
        if sha not in self.cache:
            self.cache[sha] = torch.load(
                os.path.join(self.dir, f"{sha}.pt"), map_location="cpu",
                weights_only=True)["latents"]
        return self.cache[sha]


def step_loss(model, embed_layer, tok, ex, latents, device):
    ids, latent_pos, prompt_len = ex
    ids_t = torch.tensor([ids], device=device)
    labels = ids_t.clone()
    labels[:, :prompt_len] = -100
    embeds = embed_layer(ids_t).clone()
    flat = latents.reshape(-1, latents.shape[-1]).to(device=device, dtype=embeds.dtype)
    embeds[0, torch.tensor(latent_pos, device=device), :] = flat
    out = model(inputs_embeds=embeds,
                attention_mask=torch.ones_like(ids_t), labels=labels)
    return out.loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-checkpoint", default="data/autoencoder_pretrain_longctx_big/final")
    ap.add_argument("--latent-dir", default="data/quality_latents")
    ap.add_argument("--traces", default="data/quality_teacher/traces_shard*.jsonl")
    ap.add_argument("--out-dir", default="data/latent_reader")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-trace-tokens", type=int, default=2048,
                    help="traces LONGER than this are dropped (not truncated, "
                         "which would cut off the \\boxed answer)")
    ap.add_argument("--dev-frac", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world, local_rank, device = 0, 1, 0, "cuda:0"
    is_main = rank == 0

    tok = AutoTokenizer.from_pretrained(args.ae_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.ae_checkpoint, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    embed_layer = model.get_input_embeddings()

    examples = load_data(args.latent_dir, args.traces)
    n_all = len(examples)
    examples = [ex for ex in examples
                if len(tok(ex["trace"], add_special_tokens=False)["input_ids"])
                <= args.max_trace_tokens]
    if is_main:
        print(f"dropped {n_all - len(examples)}/{n_all} over-length traces "
              f"(cap {args.max_trace_tokens})", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_dev = max(8, int(len(examples) * args.dev_frac))
    dev, train = examples[:n_dev], examples[n_dev:]
    if is_main:
        print(f"{len(train)} train / {len(dev)} dev traces, world={world}", flush=True)

    cache = LatentCache(args.latent_dir)
    raw_model = model
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    gradient_as_bucket_view=True)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)
    micro_per_rank = len(train) * args.epochs / world
    total_steps = int(micro_per_rank / args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        0.5 * (1 + torch.cos(torch.tensor(min(1.0, s / max(1, total_steps)) * 3.14159)).item()))
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"total optimizer steps ~{total_steps} "
              f"(eff batch {args.grad_accum * world})", flush=True)

    def prep(ex):
        built = build_example(tok, ex["meta"], ex["trace"], args.max_trace_tokens)
        return built, cache.get(ex["meta"]["article_sha"])

    def run_dev():
        raw_model.eval()
        tot = cnt = 0.0
        with torch.no_grad():
            for ex in dev[:64]:
                built, lats = prep(ex)
                tot += step_loss(raw_model, embed_layer, tok, built, lats, device).item()
                cnt += 1
        raw_model.train()
        return tot / cnt

    step = micro = 0
    order = list(range(rank, len(train), world))
    t0 = time.time()
    running = 0.0
    opt.zero_grad()
    for epoch in range(int(args.epochs + 0.999)):
        rng2 = random.Random(args.seed + epoch)
        rng2.shuffle(order)
        for oi in order:
            if step >= total_steps:
                break
            built, lats = prep(train[oi])
            sync = (micro + 1) % args.grad_accum == 0
            ctx = (model.no_sync() if (world > 1 and not sync)
                   else contextlib.nullcontext())
            with ctx:
                loss = step_loss(model, embed_layer, tok, built, lats, device)
                (loss / args.grad_accum).backward()
            running += loss.item()
            micro += 1
            if sync:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if is_main and step % args.log_every == 0:
                    dt = (time.time() - t0) / args.log_every
                    print(f"step {step}/{total_steps} loss={running / (args.log_every * args.grad_accum):.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} ({dt:.1f}s/step)", flush=True)
                    running = 0.0
                    t0 = time.time()
                if is_main and step % args.eval_every == 0:
                    print(f"  [dev] step {step} loss={run_dev():.4f}", flush=True)
                if is_main and step % args.save_every == 0:
                    p = os.path.join(args.out_dir, f"checkpoint-{step}")
                    raw_model.save_pretrained(p)
                    tok.save_pretrained(p)
                    print(f"  saved {p}", flush=True)

    if is_main:
        print(f"  [dev] final loss={run_dev():.4f}", flush=True)
        p = os.path.join(args.out_dir, "final")
        raw_model.save_pretrained(p)
        tok.save_pretrained(p)
        print(f"Saved {p}", flush=True)
    if world > 1:
        dist.barrier()
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"(non-fatal) destroy_process_group: {e}", flush=True)


if __name__ == "__main__":
    main()
