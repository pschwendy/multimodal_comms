#!/usr/bin/env python3
"""Large-scale pretraining for a neural-operator ("generator") continuous-
latent textual autoencoder.

This is pretrain_autoencoder.py with one change: instead of collapsing a
text's hidden states (S x H) into K latents by *sampling* K positions
(discarding the rest), we run the *entire* hidden-state matrix through a
small multiwavelet neural-operator encoder from the method-level MWNOT module
that multiscale-decomposes it and
cross-attention-pools it into K fixed-size "generator" vectors. All S
positions inform the generator, not just K sampled ones.

  Encode:  text -> model -> last-layer hidden states (S x H)
           -> SequenceGeneratorEncoder(hidden, mask) -> generator (K x H)
  Decode:  generator (K x H) injected as prefix token embeddings via K
           reserved <|Li|> tokens -> model -> reconstructed text

Gradients flow end-to-end through a single shared model plus the small
generator encoder: reconstruction loss -> decode prefix embeddings ->
generator -> generator encoder params, and separately -> encode hidden
states -> encoder (=same model) params. Both the encode and decode forward
passes run through the same base model instance every step, exactly as in
pretrain_autoencoder.py (this is a self-distillation-style autoencoder, the
generator encoder is the only new trainable piece).

The base model and the generator encoder are wrapped in a single
AEWithGenerator container so a single DDP instance (or none, on 1 GPU)
covers gradient sync for both -- see AEWithGenerator.forward, which is the
one call per step that should go through the (possibly DDP-wrapped)
container.

Run (pilot, single GPU):
  CUDA_VISIBLE_DEVICES=0 python training/programs/pretrain_mwnot_autoencoder.py \
      --train-data data/fineweb_ae/train.jsonl \
      --dev-data data/fineweb_ae/dev.jsonl \
      --out-dir data/mwnot_autoencoder_pilot

Run (multi-GPU DDP):
  torchrun --standalone --nproc_per_node=<N> \
      training/programs/pretrain_mwnot_autoencoder.py \
      --train-data data/fineweb_ae_large/train.jsonl \
      --dev-data data/fineweb_ae_large/dev.jsonl \
      --out-dir data/mwnot_autoencoder_large

Saves to <out-dir>/checkpoint-<step> and <out-dir>/final: the base model +
tokenizer (as in pretrain_autoencoder.py) plus generator.pt (the
SequenceGeneratorEncoder state_dict + its constructor config).
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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from multimodal_comms.methods.autoencoders.mwnot_generator import SequenceGeneratorEncoder  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-4B"


class AEWithGenerator(nn.Module):
    """Base LM + SequenceGeneratorEncoder, called together so a single DDP
    wrapper synchronizes gradients for both."""

    def __init__(self, lm, generator: SequenceGeneratorEncoder, last_layer: int):
        super().__init__()
        self.lm = lm
        self.generator = generator
        self.last_layer = last_layer

    def forward(self, enc_ids, enc_mask, decode_ids, decode_attn, labels, li_positions):
        enc_out = self.lm(enc_ids, attention_mask=enc_mask, output_hidden_states=True)
        hidden = enc_out.hidden_states[self.last_layer]  # (B, S, H)
        gen = self.generator(hidden.float(), enc_mask.bool()).to(hidden.dtype)  # (B, K, H)

        embed_layer = self.lm.get_input_embeddings()
        embeds = embed_layer(decode_ids).clone()
        for i, pos in enumerate(li_positions):
            embeds[:, pos, :] = gen[:, i, :]

        out = self.lm(inputs_embeds=embeds, attention_mask=decode_attn, labels=labels)
        return out.loss, out.logits


def setup_distributed():
    if "WORLD_SIZE" not in os.environ:
        return 0, 0, 1, None
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, f"cuda:{local_rank}"


def build_tokenizer(num_latents: int):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    for i in range(num_latents):
        tok.add_tokens([f"<|L{i}|>"], special_tokens=True)
    return tok


def build_model(device: str, vocab_size: int):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    ).to(device)
    model.resize_token_embeddings(vocab_size)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def load_jsonl(path: str) -> list[str]:
    return [json.loads(l)["text"] for l in open(path)]


def decode_prompt_ids(tokenizer, num_latents: int) -> list[int]:
    dec_prompt = "<|im_start|>user\n"
    for i in range(num_latents):
        dec_prompt += f"<|L{i}|>"
    dec_prompt += "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(dec_prompt, add_special_tokens=False)


def latent_token_positions(tokenizer, prompt_ids: list[int], num_latents: int) -> list[int]:
    positions = []
    for i in range(num_latents):
        li_id = tokenizer.convert_tokens_to_ids(f"<|L{i}|>")
        positions.append(prompt_ids.index(li_id))
    return positions


def build_decode_batch(tokenizer, texts: list[str], prompt_ids: list[int], max_len: int, device: str):
    pad_id = tokenizer.pad_token_id
    prompt_len = len(prompt_ids)
    target_ids = [tokenizer.encode(f"{t}<|im_end|>", add_special_tokens=False,
                                    truncation=True, max_length=max_len)
                  for t in texts]
    max_target_len = max(len(t) for t in target_ids)
    total_len = prompt_len + max_target_len

    B = len(texts)
    decode_ids = torch.full((B, total_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, total_len), dtype=torch.long)
    labels = torch.full((B, total_len), -100, dtype=torch.long)

    for b, tgt in enumerate(target_ids):
        decode_ids[b, :prompt_len] = torch.tensor(prompt_ids)
        decode_ids[b, prompt_len:prompt_len + len(tgt)] = torch.tensor(tgt)
        attn[b, :prompt_len + len(tgt)] = 1
        labels[b, prompt_len:prompt_len + len(tgt)] = torch.tensor(tgt)

    return decode_ids.to(device), attn.to(device), labels.to(device)


def train_step(model, tokenizer, batch_texts, prompt_ids, li_positions, max_len, enc_max_len, device):
    enc_texts = [f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in batch_texts]
    enc = tokenizer(enc_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=enc_max_len).to(device)
    decode_ids, decode_attn, labels = build_decode_batch(tokenizer, batch_texts, prompt_ids, max_len, device)

    loss, logits = model(enc["input_ids"], enc["attention_mask"], decode_ids, decode_attn, labels, li_positions)

    with torch.no_grad():
        preds = logits[:, :-1].argmax(dim=-1)
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100
        correct = ((preds == shifted_labels) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0
    return loss, acc


@torch.no_grad()
def eval_dev(model, tokenizer, dev_texts, prompt_ids, li_positions, max_len, enc_max_len,
             device, batch_size, n_batches=8):
    model.eval()
    losses, accs = [], []
    for i in range(0, min(len(dev_texts), batch_size * n_batches), batch_size):
        batch = dev_texts[i:i + batch_size]
        if not batch:
            continue
        loss, acc = train_step(model, tokenizer, batch, prompt_ids, li_positions, max_len, enc_max_len, device)
        losses.append(loss.item())
        accs.append(acc)
    model.train()
    return sum(losses) / len(losses), sum(accs) / len(accs)


def save_checkpoint(raw_lm, raw_generator, tokenizer, num_latents, path):
    os.makedirs(path, exist_ok=True)
    raw_lm.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(
        {"state_dict": raw_generator.state_dict(), "config": raw_generator.config_dict()},
        os.path.join(path, "generator.pt"),
    )
    with open(os.path.join(path, "ae_config.json"), "w") as f:
        json.dump({"num_latents": num_latents, "bottleneck_dim": None, "generator": "mwnot"}, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=str, default="data/fineweb_ae/train.jsonl")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae/dev.jsonl")
    ap.add_argument("--out-dir", type=str, default="data/mwnot_autoencoder_pretrain")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-latents", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--generator-lr-mult", type=float, default=10.0,
                     help="Generator encoder is randomly initialized and small; "
                          "give it a higher LR than the pretrained base model.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    # SequenceGeneratorEncoder hyperparameters
    ap.add_argument("--lift-channels", type=int, default=32)
    ap.add_argument("--gen-embed-dim", type=int, default=256)
    ap.add_argument("--wavelet-levels", type=int, default=3)
    ap.add_argument("--gen-heads", type=int, default=4)
    ap.add_argument("--gen-layers", type=int, default=2)
    args = ap.parse_args()

    rank, local_rank, world_size, ddp_device = setup_distributed()
    is_main = rank == 0
    device = ddp_device if ddp_device is not None else args.device

    shard_id = rank if world_size > 1 else args.shard_id
    num_shards = world_size if world_size > 1 else args.num_shards

    def log(msg: str):
        if is_main:
            print(msg)

    for path in (args.train_data, args.dev_data):
        if not os.path.exists(path):
            if is_main:
                print(f"Data file {path} not found.", file=sys.stderr)
            sys.exit(1)

    random.seed(args.seed + shard_id)
    full_train = load_jsonl(args.train_data)
    train_texts = full_train[shard_id::num_shards]
    random.shuffle(train_texts)
    dev_texts = load_jsonl(args.dev_data) if is_main else []
    log(f"Loaded {len(full_train)} train examples "
        f"({len(train_texts)} in shard {shard_id}/{num_shards}) / {len(dev_texts)} dev examples")

    log("Building tokenizer...")
    tok = build_tokenizer(args.num_latents)
    log(f"Vocab size: {len(tok)}")

    log(f"Loading {BASE_MODEL}...")
    raw_lm = build_model(device, len(tok))
    raw_lm.train()
    H = raw_lm.config.hidden_size
    last_layer = raw_lm.config.num_hidden_layers - 1

    raw_generator = SequenceGeneratorEncoder(
        hidden_size=H,
        num_latents=args.num_latents,
        lift_channels=args.lift_channels,
        embed_dim=args.gen_embed_dim,
        wavelet_levels=args.wavelet_levels,
        num_heads=args.gen_heads,
        num_layers=args.gen_layers,
    ).to(device)
    log(f"Generator encoder params: {sum(p.numel() for p in raw_generator.parameters()):,} "
        f"(hidden dim {H}, {args.num_latents} latents pooled from the full sequence)")

    raw_container = AEWithGenerator(raw_lm, raw_generator, last_layer)
    raw_container.train()

    if world_size > 1:
        container = DDP(raw_container, device_ids=[local_rank], output_device=local_rank,
                         gradient_as_bucket_view=True)
    else:
        container = raw_container

    prompt_ids = decode_prompt_ids(tok, args.num_latents)
    li_positions = latent_token_positions(tok, prompt_ids, args.num_latents)

    optimizer = torch.optim.AdamW([
        {"params": raw_lm.parameters(), "lr": args.lr},
        {"params": raw_generator.parameters(), "lr": args.lr * args.generator_lr_mult},
    ])

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    n = len(train_texts)
    bs = args.batch_size
    global_batch = bs * args.grad_accum * world_size

    log(f"Training ({args.steps} steps, {world_size} GPU(s), batch {bs} x "
        f"grad_accum {args.grad_accum} x world_size {world_size} = effective batch "
        f"{global_batch}, {args.num_latents} latents, "
        f"{H * args.num_latents * 2} bytes/example transmitted at fp16)")

    total_loss, total_acc = 0.0, 0.0
    t0 = time.time()
    optimizer.zero_grad()
    idx = 0
    for step in range(args.steps):
        batch = []
        for _ in range(bs):
            batch.append(train_texts[idx % n])
            idx += 1
            if idx % n == 0:
                random.shuffle(train_texts)

        is_sync_step = (step + 1) % args.grad_accum == 0
        sync_ctx = (container.no_sync() if (world_size > 1 and not is_sync_step)
                    else contextlib.nullcontext())
        with sync_ctx:
            loss, acc = train_step(container, tok, batch, prompt_ids, li_positions,
                                    args.max_len, args.max_len, device)
            (loss / args.grad_accum).backward()
        total_loss += loss.item()
        total_acc += acc

        if is_sync_step:
            torch.nn.utils.clip_grad_norm_(raw_container.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if is_main and (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / args.log_every
            avg_acc = total_acc / args.log_every
            print(f"step {step+1:5d}/{args.steps}  loss={avg_loss:.4f}  "
                  f"tok_acc={avg_acc:.3f}  ({elapsed / args.log_every:.2f}s/step)")
            total_loss, total_acc = 0.0, 0.0
            t0 = time.time()

        if is_main and (step + 1) % args.eval_every == 0:
            dev_loss, dev_acc = eval_dev(raw_container, tok, dev_texts, prompt_ids, li_positions,
                                          args.max_len, args.max_len, device, bs)
            print(f"  [dev] step {step+1}  loss={dev_loss:.4f}  tok_acc={dev_acc:.3f}")

        if is_main and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, f"checkpoint-{step+1}")
            save_checkpoint(raw_lm, raw_generator, tok, args.num_latents, ckpt)
            print(f"  saved {ckpt}")

    if is_main:
        final_path = os.path.join(args.out_dir, "final")
        save_checkpoint(raw_lm, raw_generator, tok, args.num_latents, final_path)
        print(f"Saved to {final_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
