#!/usr/bin/env python3
"""Superposition fine-tuning: many messages -> one latent packet.

Extends the single-message autoencoder (pretrain_autoencoder.py) to
superposed packets. Each step:

  1. Sample N (the superposition load) from the curriculum.
  2. Encode B*N texts -> latents (B*N, K, H).
  3. Group into B packets of N slots; bind slot j's latents with its
     orthogonal key Q_j (float32) and sum:  Z_b = sum_j z_bj @ Q_j.
  4. Unbind every slot:  Z_b @ Q_j^T = z_bj + crosstalk.
  5. Decode all B*N unbound latents with the standard RECONSTRUCT prompt;
     mean CE loss. Gradients flow through unbind/bind into the encoder,
     teaching it to place content where crosstalk hurts least, and the
     decoder to denoise the crosstalk.

Keys come from the reusable superposition method module used by every
benchmark adapter, and binding is float32 in both places, so the
train and inference computation paths cannot drift apart.

Warm start from a single-message checkpoint (curriculum starts at N=1,
which reproduces plain autoencoding):

  CUDA_VISIBLE_DEVICES=0 python training/programs/pretrain_superpose.py \
      --init-from data/autoencoder_pretrain_large/final \
      --train-data data/fineweb_ae_large/train.jsonl \
      --dev-data data/fineweb_ae_large/dev.jsonl \
      --out-dir data/superpose_pretrain --max-slots 8

Multi-GPU (DDP):

  torchrun --standalone --nproc_per_node=8 training/programs/pretrain_superpose.py ...
"""

import argparse
import contextlib
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))

from training.programs.pretrain_autoencoder import (  # noqa: E402
    setup_distributed,
    load_jsonl,
    encode_batch,
    decode_prompt_ids,
    latent_token_positions,
    decode_batch_loss,
)
from multimodal_comms.methods.superposition.latent import build_keyring  # noqa: E402


def allowed_slots(step: int, total_steps: int, max_slots: int) -> int:
    """Curriculum: grow the maximum superposition load over training."""
    frac = step / max(total_steps, 1)
    if frac < 0.15:
        cap = 1
    elif frac < 0.35:
        cap = 2
    elif frac < 0.60:
        cap = 4
    else:
        cap = max_slots
    return min(cap, max_slots)


class DeviceKeyring:
    """Keyring wrapper that caches key/basis tensors on the training device."""

    def __init__(self, keyring, device: str):
        self.keyring = keyring
        self.device = device
        self._cache: dict[tuple, torch.Tensor] = {}

    def key(self, slot: int) -> torch.Tensor:
        if ("k", slot) not in self._cache:
            self._cache[("k", slot)] = self.keyring.key(slot).to(self.device)
        return self._cache[("k", slot)]

    def basis(self, slot: int, n_slots: int) -> torch.Tensor:
        if ("b", slot, n_slots) not in self._cache:
            self._cache[("b", slot, n_slots)] = (
                self.keyring.basis(slot, n_slots).to(self.device))
        return self._cache[("b", slot, n_slots)]


def superpose_unbind(latents: torch.Tensor, n_slots: int,
                     dkr: DeviceKeyring, key_mode: str) -> torch.Tensor:
    """latents: (B*N, K, H) -> unbound noisy latents (B*N, K, H).

    Binding/unbinding runs in float32 (matching runtime); output is cast
    back to the input dtype for the decode pass.
    """
    in_dtype = latents.dtype
    bn, K, H = latents.shape
    z = latents.to(torch.float32).view(-1, n_slots, K, H)  # (B, N, K, H)

    bound = []
    for j in range(n_slots):
        q = dkr.key(j)
        if key_mode == "sign":
            bound.append(z[:, j] * q)
        else:
            bound.append(z[:, j] @ q)
    packet = torch.stack(bound, dim=1).sum(dim=1)  # (B, K, H)

    unbound = []
    for j in range(n_slots):
        q = dkr.key(j)
        if key_mode == "sign":
            unbound.append(packet * q)
        else:
            unbound.append(packet @ q.T)
    out = torch.stack(unbound, dim=1).view(bn, K, H)  # (B*N, K, H)
    return out.to(in_dtype)


def project_subspace(latents: torch.Tensor, n_slots: int,
                     dkr: DeviceKeyring, slots: list[int]) -> torch.Tensor:
    """Subspace mode: latents (B, K, H) -> per-example rank-H/N projections.

    Zero-crosstalk superposition means decoding slot j from a packet is
    mathematically identical to decoding from P_j(z_j) alone, so training
    never needs to assemble packets (and per-step memory is independent of
    the load). slots[i] picks example i's subspace.
    """
    in_dtype = latents.dtype
    z = latents.to(torch.float32)
    out = torch.empty_like(z)
    for i, j in enumerate(slots):
        b = dkr.basis(j, n_slots)
        out[i] = (z[i] @ b) @ b.T
    return out.to(in_dtype)


def save_checkpoint(raw_model, tokenizer, args, path):
    os.makedirs(path, exist_ok=True)
    raw_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "ae_config.json"), "w") as f:
        json.dump({"num_latents": args.num_latents, "bottleneck_dim": None}, f)
    with open(os.path.join(path, "superpose_config.json"), "w") as f:
        json.dump({
            "num_latents": args.num_latents,
            "max_slots": args.max_slots,
            "key_seed": args.key_seed,
            "key_mode": args.key_mode,
        }, f)


@torch.no_grad()
def eval_dev(model, tokenizer, dev_texts, args, prompt_ids, li_positions,
             device, dkr, last_layer, embed_layer, loads=(1, 2, 4, 8)):
    """Reconstruction token-accuracy at several superposition loads."""
    model.eval()
    results = {}
    loads = sorted({min(n, args.max_slots) for n in (*loads, args.max_slots)})
    for n in loads:
        accs = []
        per_batch = max(1, args.batch_size)
        for i in range(0, min(len(dev_texts), per_batch * n * 4), per_batch * n):
            batch = dev_texts[i:i + per_batch * n]
            if len(batch) < n:
                break
            batch = batch[:(len(batch) // n) * n]
            latents = encode_batch(model, tokenizer, batch, args.num_latents,
                                   args.max_len, device, last_layer)
            if args.key_mode == "subspace":
                noisy = project_subspace(
                    latents, n, dkr, [k % n for k in range(len(batch))])
            else:
                noisy = superpose_unbind(latents, n, dkr, args.key_mode)
            _, acc = decode_batch_loss(model, tokenizer, batch, noisy,
                                       prompt_ids, li_positions, args.max_len,
                                       device, embed_layer)
            accs.append(acc)
        results[n] = sum(accs) / len(accs) if accs else float("nan")
    model.train()
    return results


def maybe_shorten(text: str, frac: float, lo: int, hi: int) -> str:
    """Chat-length augmentation: with prob `frac`, truncate to a random
    length in [lo, hi] chars so the codec's prior covers discussion-message
    lengths, not just full FineWeb chunks."""
    if frac > 0 and random.random() < frac:
        return text[:random.randint(lo, hi)]
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=str, default="data/autoencoder_pretrain_large/final",
                    help="Single-message AE checkpoint to warm-start from.")
    ap.add_argument("--train-data", type=str, default="data/fineweb_ae_large/train.jsonl")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--out-dir", type=str, default="data/superpose_pretrain")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Packets per micro-step (texts per step = batch * N).")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-latents", type=int, default=4)
    ap.add_argument("--max-slots", type=int, default=8,
                    help="Maximum superposition load N reached by the curriculum.")
    ap.add_argument("--key-seed", type=int, default=1234)
    ap.add_argument("--key-mode", type=str, default="qr",
                    choices=["qr", "sign", "subspace"])
    ap.add_argument("--short-frac", type=float, default=0.0,
                    help="Prob of truncating a training text to chat length.")
    ap.add_argument("--short-min", type=int, default=150)
    ap.add_argument("--short-max", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rank, local_rank, world_size, ddp_device = setup_distributed()
    is_main = rank == 0
    device = ddp_device if ddp_device is not None else args.device

    def log(msg: str):
        if is_main:
            print(msg)

    for path in (args.train_data, args.dev_data, args.init_from):
        if not os.path.exists(path):
            if is_main:
                print(f"Path {path} not found.", file=sys.stderr)
            sys.exit(1)

    random.seed(args.seed + rank)
    train_texts = load_jsonl(args.train_data)[rank::max(world_size, 1)]
    random.shuffle(train_texts)
    dev_texts = load_jsonl(args.dev_data) if is_main else []
    log(f"{len(train_texts)} train texts (rank shard) / {len(dev_texts)} dev texts")

    log(f"Loading warm-start checkpoint {args.init_from}...")
    tok = AutoTokenizer.from_pretrained(args.init_from)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    raw_model = AutoModelForCausalLM.from_pretrained(
        args.init_from, torch_dtype=torch.bfloat16,
    ).to(device)
    raw_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    raw_model.train()
    H = raw_model.config.hidden_size
    last_layer = raw_model.config.num_hidden_layers - 1
    embed_layer = raw_model.get_input_embeddings()

    dkr = DeviceKeyring(build_keyring(H, seed=args.key_seed, mode=args.key_mode),
                        device)
    # Materialize keys/bases up front (also validates determinism cheaply).
    if args.key_mode == "subspace":
        for n in (1, 2, 4, 8, 16):
            if n <= args.max_slots:
                for j in range(n):
                    dkr.basis(j, n)
    else:
        for j in range(args.max_slots):
            dkr.key(j)

    # No DDP wrapper: on 44 GB cards the 4B model + AdamW leaves no room for
    # DDP's gradient buckets alongside param.grad (an extra ~8 GB that OOMs
    # every config). Data parallelism is instead a manual grad all-reduce at
    # each sync step -- one 8 GB transfer per optimizer step, amortized over
    # grad_accum micro-steps.
    model = raw_model

    prompt_ids = decode_prompt_ids(tok, args.num_latents)
    li_positions = latent_token_positions(tok, prompt_ids, args.num_latents)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log(f"Training {args.steps} steps, max_slots={args.max_slots}, "
        f"key_mode={args.key_mode}, packet={args.num_latents}x{H} fp32 "
        f"({args.num_latents * H * 4} bytes regardless of load)")

    n_texts = len(train_texts)
    idx = 0
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    t0 = time.time()
    optimizer.zero_grad()

    subspace_loads = [n for n in (1, 2, 4, 8, 16) if n <= args.max_slots]

    for step in range(args.steps):
        cap = allowed_slots(step, args.steps, args.max_slots)
        if args.key_mode == "subspace":
            # Zero-crosstalk: no packets to build, so batch size is
            # independent of the load. Sample a width, project each example
            # onto a random slot's subspace.
            n_slots = random.choice([n for n in subspace_loads if n <= cap])
            n_batch_texts = args.batch_size
        else:
            n_slots = random.randint(1, cap)
            n_batch_texts = args.batch_size * n_slots
        batch = []
        for _ in range(n_batch_texts):
            batch.append(maybe_shorten(train_texts[idx % n_texts],
                                       args.short_frac, args.short_min,
                                       args.short_max))
            idx += 1
            if idx % n_texts == 0:
                random.shuffle(train_texts)

        is_sync_step = (step + 1) % args.grad_accum == 0
        latents = encode_batch(model, tok, batch, args.num_latents,
                               args.max_len, device, last_layer)
        if args.key_mode == "subspace":
            slots = [random.randrange(n_slots) for _ in batch]
            noisy = project_subspace(latents, n_slots, dkr, slots)
        else:
            noisy = superpose_unbind(latents, n_slots, dkr, args.key_mode)
        loss, acc = decode_batch_loss(model, tok, batch, noisy, prompt_ids,
                                      li_positions, args.max_len, device,
                                      embed_layer)
        (loss / args.grad_accum).backward()

        total_loss += loss.item()
        total_acc += acc
        total_n += n_slots

        if is_sync_step:
            if world_size > 1:
                for p in raw_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if is_main and (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step+1:5d}/{args.steps}  loss={total_loss/args.log_every:.4f}  "
                  f"tok_acc={total_acc/args.log_every:.3f}  "
                  f"avg_N={total_n/args.log_every:.1f} (cap {cap})  "
                  f"({elapsed/args.log_every:.2f}s/step)")
            total_loss, total_acc, total_n = 0.0, 0.0, 0
            t0 = time.time()

        if is_main and (step + 1) % args.eval_every == 0:
            results = eval_dev(raw_model, tok, dev_texts, args, prompt_ids,
                               li_positions, device, dkr, last_layer, embed_layer)
            summary = "  ".join(f"N={n}: {a:.3f}" for n, a in results.items())
            print(f"  [dev tok_acc by load] {summary}")

        if is_main and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, f"checkpoint-{step+1}")
            save_checkpoint(raw_model, tok, args, ckpt)
            print(f"  saved {ckpt}")

    if is_main:
        final_path = os.path.join(args.out_dir, "final")
        save_checkpoint(raw_model, tok, args, final_path)
        print(f"Saved to {final_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
