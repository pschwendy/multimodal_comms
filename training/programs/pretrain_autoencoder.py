#!/usr/bin/env python3
"""Large-scale pretraining for the continuous-latent textual autoencoder.

Architecture (no dimensionality bottleneck -- compression comes purely from
collapsing N input tokens into K latent vectors, each the model's own
hidden_size wide):

  Encode:  text -> model -> last-layer hidden states (seq_len x H)
           -> sample K positions -> latents (K x H)
  Decode:  latents (K x H) injected as prefix token embeddings via K
           reserved <|Li|> tokens -> model -> reconstructed text

Gradients flow end-to-end through a single shared model: reconstruction
loss -> decode prefix embeddings -> latents -> encode hidden states ->
encoder params. Both the encode and decode forward passes run through the
same model instance every step (this is a self-distillation-style
autoencoder, not a separate encoder/decoder).

Unlike the original single-example hiddenbench-message trainer
(train_autoencoder.py), this script batches examples (padded + masked) so
it can absorb a much larger, more diverse corpus (FineWeb-Edu chunks from
harvest_fineweb_data.py) at reasonable throughput.

Run (pilot, single GPU):
  CUDA_VISIBLE_DEVICES=0 python training/programs/pretrain_autoencoder.py \
      --train-data data/fineweb_ae/train.jsonl \
      --dev-data data/fineweb_ae/dev.jsonl

Run (multi-GPU, data-parallel via DDP -- each rank trains the same model on
its own data shard, gradients averaged across ranks every step):
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 torchrun --standalone --nproc_per_node=8 \
      training/programs/pretrain_autoencoder.py \
      --train-data data/fineweb_ae_large/train.jsonl \
      --dev-data data/fineweb_ae_large/dev.jsonl \
      --out-dir data/autoencoder_pretrain_large

Saves to <out-dir>/checkpoint-<step> and <out-dir>/final (model + tokenizer;
no projection.pt, so AutoencoderCompressor in channel.py loads it and skips
the (now nonexistent) bottleneck automatically).
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

BASE_MODEL = "Qwen/Qwen3-4B"


def setup_distributed():
    """Returns (rank, local_rank, world_size, device). world_size==1 if not
    launched under torchrun (RANK/WORLD_SIZE unset)."""
    if "WORLD_SIZE" not in os.environ:
        return 0, 0, 1, None
    # 40-min collective timeout (default 10). Ranks that finish a step block
    # at the gradient all-reduce until the slowest rank arrives; when rank 0
    # runs a periodic eval or writes a checkpoint the others wait there, and
    # the default watchdog would abort the whole job mid-eval. Sharded eval
    # keeps that wait short, but a generous timeout removes the failure class.
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=40))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, f"cuda:{local_rank}"


def build_tokenizer(num_latents: int, init_from: str | None = None):
    """Tokenizer with <|L0|>..<|L{num_latents-1}|>.

    When `init_from` is a checkpoint that already has FEWER latent tokens,
    add_tokens is idempotent for the existing ones and appends only the new
    ones -- which is what makes K=16 -> K=32 a continuation rather than a
    restart. The freshly-added embeddings start random, but that costs
    nothing here: the <|Li|> positions are overwritten with the actual latent
    vectors on the decode side and are never consulted on the encode side.
    """
    tok = AutoTokenizer.from_pretrained(init_from or BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    for i in range(num_latents):
        tok.add_tokens([f"<|L{i}|>"], special_tokens=True)
    return tok


def build_model(device: str, vocab_size: int, init_from: str | None = None,
                lora_r: int = 0):
    model = AutoModelForCausalLM.from_pretrained(
        init_from or BASE_MODEL, torch_dtype=torch.bfloat16,
    ).to(device)
    if model.get_input_embeddings().weight.shape[0] != vocab_size:
        model.resize_token_embeddings(vocab_size)
    if lora_r > 0:
        # LoRA before checkpointing: peft installs a make_inputs_require_grad
        # hook if checkpointing is already on, which makes the decode path's
        # in-place latent write illegal.
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def load_jsonl(path: str) -> list[str]:
    return [json.loads(l)["text"] for l in open(path)]


def latent_indices(seq_len: int, num_latents: int) -> list[int]:
    if num_latents == 1:
        return [seq_len - 1]
    step = seq_len / num_latents
    return [min(int(i * step + step / 2), seq_len - 1) for i in range(num_latents)]


def encode_batch(model, tokenizer, texts: list[str], num_latents: int,
                  max_len: int, device: str, last_layer: int) -> torch.Tensor:
    """Returns latents: (B, num_latents, H)."""
    enc_texts = [f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in texts]
    enc = tokenizer(enc_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len).to(device)

    out = model(enc["input_ids"], attention_mask=enc["attention_mask"],
                output_hidden_states=True)
    hidden = out.hidden_states[last_layer]  # (B, S, H)

    seq_lens = enc["attention_mask"].sum(dim=1).tolist()
    latents = []
    for b, seq_len in enumerate(seq_lens):
        idx = latent_indices(seq_len, num_latents)
        latents.append(hidden[b, idx, :])
    return torch.stack(latents)  # (B, num_latents, H)


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


def decode_batch_loss(model, tokenizer, texts: list[str], latents: torch.Tensor,
                       prompt_ids: list[int], li_positions: list[int],
                       max_len: int, device: str, embed_layer):
    """latents: (B, num_latents, H). Returns (loss, token_accuracy)."""
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

    decode_ids = decode_ids.to(device)
    attn = attn.to(device)
    labels = labels.to(device)

    # .clone(): current peft/transformers marks a frozen embedding layer's
    # output as a requires_grad leaf when grad checkpointing is on (so grad
    # can flow into the first checkpointed layer), and in-place writes into
    # a leaf that requires grad are illegal. Cloning gives a non-leaf tensor
    # the in-place scatter below is safe to write into.
    embeds = embed_layer(decode_ids).clone()
    for i, pos in enumerate(li_positions):
        embeds[:, pos, :] = latents[:, i, :].to(embeds.dtype)

    out = model(inputs_embeds=embeds, attention_mask=attn, labels=labels)

    with torch.no_grad():
        # logits at position p predict the token at p+1: shift before comparing
        preds = out.logits[:, :-1].argmax(dim=-1)
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100
        correct = ((preds == shifted_labels) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0

    return out.loss, acc


def train_step(model, tokenizer, batch_texts, num_latents, prompt_ids,
                li_positions, max_len, device, optimizer, grad_accum,
                last_layer, embed_layer):
    latents = encode_batch(model, tokenizer, batch_texts, num_latents, max_len,
                            device, last_layer)
    loss, acc = decode_batch_loss(model, tokenizer, batch_texts, latents,
                                   prompt_ids, li_positions, max_len, device,
                                   embed_layer)
    (loss / grad_accum).backward()
    return loss.item(), acc


@torch.no_grad()
def eval_dev(model, tokenizer, dev_texts, num_latents, prompt_ids, li_positions,
             max_len, device, batch_size, last_layer, embed_layer, n_batches=8):
    model.eval()
    losses, accs = [], []
    for i in range(0, min(len(dev_texts), batch_size * n_batches), batch_size):
        batch = dev_texts[i:i + batch_size]
        if not batch:
            continue
        latents = encode_batch(model, tokenizer, batch, num_latents, max_len,
                                device, last_layer)
        loss, acc = decode_batch_loss(model, tokenizer, batch, latents,
                                       prompt_ids, li_positions, max_len, device,
                                       embed_layer)
        losses.append(loss.item())
        accs.append(acc)
    model.train()
    return sum(losses) / len(losses), sum(accs) / len(accs)


def save_checkpoint(raw_model, tokenizer, num_latents, path):
    os.makedirs(path, exist_ok=True)
    # For a LoRA run this writes ONLY the adapter (peft's save_pretrained), NOT
    # a standalone model. That is correct and cheap -- merging inline would need
    # a deepcopy of the base, which at 8B (~16GB) on top of the live training
    # state (~40GB) OOMs the 46GB card. Downstream consumers that need a plain
    # AutoModelForCausalLM (pretrain_packed --init-from, the eval scripts) must
    # merge first: training/programs/merge_adapter.py base=<...> adapter=<this path>.
    # The full-FT path (lora_r=0) still writes a full model here, unchanged.
    raw_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "ae_config.json"), "w") as f:
        json.dump({"num_latents": num_latents, "bottleneck_dim": None,
                   "lora_base": getattr(raw_model, "_lora_base", None)}, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=str, default="data/fineweb_ae/train.jsonl")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae/dev.jsonl")
    ap.add_argument("--out-dir", type=str, default="data/autoencoder_pretrain")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-latents", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", type=str, default="cuda:0",
                     help="Ignored under torchrun (each rank uses its assigned local GPU).")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-id", type=int, default=0,
                     help="For independent (non-torchrun) parallel jobs: train on "
                          "train_texts[shard_id::num_shards]. Ignored under torchrun, "
                          "where rank/world_size are used instead.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--init-from", type=str, default=None,
                    help="Continue from an AE checkpoint instead of the base "
                         "model; may raise --num-latents (new latent tokens are "
                         "appended and embeddings resized).")
    ap.add_argument("--lora-r", type=int, default=0,
                    help="LoRA rank (0 = full fine-tune).")
    args = ap.parse_args()

    rank, local_rank, world_size, ddp_device = setup_distributed()
    is_main = rank == 0
    device = ddp_device if ddp_device is not None else args.device

    # Under torchrun, rank/world_size (synchronized DDP) take precedence.
    # Standalone, --shard-id/--num-shards let several independent single-GPU
    # processes each train their own model on a disjoint data slice with no
    # gradient sync between them (see pretrain_autoencoder_parallel.sh).
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
    tok = build_tokenizer(args.num_latents, args.init_from)
    log(f"Vocab size: {len(tok)}")

    log(f"Loading {args.init_from or BASE_MODEL}...")
    raw_model = build_model(device, len(tok), args.init_from, args.lora_r)
    if is_main and args.lora_r:
        tr = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"LoRA r={args.lora_r}: {tr/1e6:.1f}M trainable", flush=True)
    raw_model.train()
    _cfg = raw_model.get_base_model() if args.lora_r else raw_model
    H = _cfg.config.hidden_size
    last_layer = _cfg.config.num_hidden_layers - 1
    embed_layer = _cfg.get_input_embeddings()
    log(f"Hidden dim: {H} (no bottleneck -- latents are raw {H}-dim vectors)")

    if world_size > 1:
        # static_graph=True would conflict with no_sync()-based grad
        # accumulation below (DDP's reducer requires the very first backward
        # to be a synced one under static_graph). Non-reentrant gradient
        # checkpointing (set above) is what actually fixes the "double
        # forward per step" DDP issue, so static_graph isn't needed here.
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank,
                    gradient_as_bucket_view=True)
    else:
        model = raw_model

    prompt_ids = decode_prompt_ids(tok, args.num_latents)
    li_positions = latent_token_positions(tok, prompt_ids, args.num_latents)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)

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
        # Gradient allreduce is expensive here (NCCL falls back to sockets on
        # this cluster, no P2P/shared-memory), so skip it on all but the last
        # micro-step of each accumulation window via no_sync() -- otherwise
        # DDP would allreduce every micro-step and communication would
        # dominate wall-clock time.
        sync_ctx = (model.no_sync() if (world_size > 1 and not is_sync_step)
                    else contextlib.nullcontext())
        with sync_ctx:
            loss_val, acc = train_step(model, tok, batch, args.num_latents, prompt_ids,
                                        li_positions, args.max_len, device,
                                        optimizer, args.grad_accum, last_layer, embed_layer)
        total_loss += loss_val
        total_acc += acc

        if is_sync_step:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
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
            dev_loss, dev_acc = eval_dev(raw_model, tok, dev_texts, args.num_latents,
                                          prompt_ids, li_positions, args.max_len,
                                          device, bs, last_layer, embed_layer)
            print(f"  [dev] step {step+1}  loss={dev_loss:.4f}  tok_acc={dev_acc:.3f}")

        if is_main and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, f"checkpoint-{step+1}")
            save_checkpoint(raw_model, tok, args.num_latents, ckpt)
            print(f"  saved {ckpt}")

    if is_main:
        final_path = os.path.join(args.out_dir, "final")
        save_checkpoint(raw_model, tok, args.num_latents, final_path)
        print(f"Saved to {final_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
