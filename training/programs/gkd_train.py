#!/usr/bin/env python3
"""GKD phase 3: update the student toward the teacher's distribution on the
student's OWN rollouts (forward KL / cross-entropy against the teacher's
top-K, which the teacher computed with the full article in context).

Loss per trace position j:
  q = softmax(topk_lp / teacher_temp)          (renormalized over stored K)
  loss_j = - sum_k q_k * log p_student(topk_ids_k)
averaged over trace positions; prompt positions contribute nothing.

Run: torchrun --standalone --nproc_per_node=8 training/programs/gkd_train.py \
       --init <ckpt> --scored "data/gkd/scored/scored_shard*.pt"
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

from training.programs.train_latent_reader import LatentCache, build_example  # noqa: F401 (SYSTEM/HEADER via build path)
from training.programs.gkd_rollout import build_prompt_ids


def gkd_loss(model, embed_layer, prompt_ids, latent_pos, latents, row,
             device, teacher_temp):
    trace_ids = row["trace_ids"].tolist()
    ids = torch.tensor([prompt_ids + trace_ids], device=device)
    embeds = embed_layer(ids).clone()
    flat = latents.reshape(-1, latents.shape[-1]).to(device=device, dtype=embeds.dtype)
    embeds[0, torch.tensor(latent_pos, device=device), :] = flat
    out = model(inputs_embeds=embeds, attention_mask=torch.ones_like(ids))
    plen = len(prompt_ids)
    L = len(trace_ids)
    # logits at index plen+j-1 predict trace token j
    logits = out.logits[0, plen - 1:plen + L - 1, :].float()
    logp = F.log_softmax(logits, dim=-1)                      # (L, V)
    topk_ids = row["topk_ids"].to(device).long()              # (L, K)
    topk_lp = row["topk_lp"].to(device).float()               # (L, K)
    q = F.softmax(topk_lp / teacher_temp, dim=-1)             # -inf pads -> 0
    gather_ids = topk_ids.clamp(min=0)
    s_lp = logp.gather(1, gather_ids)                         # (L, K)
    return -(q * s_lp).sum(dim=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--scored", default="data/gkd/scored/scored_shard*.pt")
    ap.add_argument("--latent-dir", default="data/quality_latents")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--teacher-temp", type=float, default=1.0)
    ap.add_argument("--max-total-tokens", type=int, default=2600)
    ap.add_argument("--log-every", type=int, default=10)
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

    tok = AutoTokenizer.from_pretrained(args.init)
    model = AutoModelForCausalLM.from_pretrained(
        args.init, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    embed_layer = model.get_input_embeddings()

    index = {r["qidx"]: r for r in
             map(json.loads, open(os.path.join(args.latent_dir, "index_train.jsonl")))}
    rows = []
    for p in sorted(glob.glob(args.scored)):
        rows.extend(torch.load(p, weights_only=True))
    li_ids = [tok.convert_tokens_to_ids(f"<|L{i}|>") for i in range(16)]
    li_set = set(li_ids)

    prompts = {}
    def get_prompt(qidx):
        if qidx not in prompts:
            pids = build_prompt_ids(tok, index[qidx])
            lpos = [i for i, t in enumerate(pids) if t in li_set]
            prompts[qidx] = (pids, lpos)
        return prompts[qidx]

    kept = [r for r in rows
            if len(get_prompt(r["qidx"])[0]) + len(r["trace_ids"]) <= args.max_total_tokens]
    if is_main:
        print(f"{len(kept)}/{len(rows)} scored rollouts within token budget", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    cache = LatentCache(args.latent_dir)
    raw_model = model
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    gradient_as_bucket_view=True)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)
    total_steps = len(kept) // (world * args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup))
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"total steps ~{total_steps} (eff batch {world * args.grad_accum})", flush=True)

    order = list(range(rank, len(kept), world))
    step = micro = 0
    running, t0 = 0.0, time.time()
    opt.zero_grad()
    for oi in order:
        if step >= total_steps:
            break
        r = kept[oi]
        pids, lpos = get_prompt(r["qidx"])
        lats = cache.get(index[r["qidx"]]["article_sha"])
        sync = (micro + 1) % args.grad_accum == 0
        ctx = (model.no_sync() if (world > 1 and not sync) else contextlib.nullcontext())
        with ctx:
            loss = gkd_loss(model, embed_layer, pids, lpos, lats, r, device,
                            args.teacher_temp)
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
                print(f"step {step}/{total_steps} gkd_loss="
                      f"{running / (args.log_every * args.grad_accum):.4f} "
                      f"({dt:.1f}s/step)", flush=True)
                running, t0 = 0.0, time.time()

    # rank 0 saves FIRST, others wait at the barrier — if the barrier preceded
    # the save, a non-main rank could reach process-group teardown (which can
    # throw on this cluster's socket transport) and SIGTERM rank 0 mid-write.
    if is_main:
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
