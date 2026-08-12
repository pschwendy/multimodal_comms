#!/usr/bin/env python3
"""Train a continuous-latent textual autoencoder with a projection bottleneck.

Architecture:
  The pre-token-layer hidden states from the LAST layer of Qwen3-4B
  are used as a continuous latent representation of the input text.

  A learnable projection bottleneck compresses the latent for transmission:

    Encode:  text -> model -> hidden states (N×2560)
             -> proj_down -> bottleneck (N×bottleneck_dim)
    Decode:  bottleneck (N×bottleneck_dim) -> proj_up -> (N×2560)
             injected as prefix token embeddings -> model -> text

  Transmitted size: N × bottleneck_dim × fp16 bytes.
  e.g. N=4, bottleneck=32 -> 256 bytes (~350 base64 chars).

  Gradients flow end-to-end: reconstruction loss -> proj_up -> bottleneck
  -> proj_down -> hidden states -> encoder parameters.

Run on GPU 3:
  CUDA_VISIBLE_DEVICES=3 python training/programs/train_autoencoder.py

Saves to data/autoencoder/final (model + projection weights).
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B"
DATA = "data/autoencoder_train.jsonl"
OUT_DIR = "data/autoencoder"
NUM_LATENTS = 4
BOTTLENECK_DIM = 32


def build_tokenizer(num_latents: int):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(num_latents):
        tok.add_tokens([f"<|L{i}|>"], special_tokens=True)
    return tok


def build_model(device: str, vocab_size: int):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    ).to(device)
    model.resize_token_embeddings(vocab_size)
    model.gradient_checkpointing_enable()
    return model


class LatentProjection(nn.Module):
    """Projection bottleneck: down-project hidden states for compression,
    up-project for reconstruction."""

    def __init__(self, hidden_dim: int = 2560, bottleneck_dim: int = 32):
        super().__init__()
        self.proj_down = nn.Linear(hidden_dim, bottleneck_dim, dtype=torch.bfloat16)
        self.proj_up = nn.Linear(bottleneck_dim, hidden_dim, dtype=torch.bfloat16)

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (num_latents, hidden_dim) -> (num_latents, bottleneck_dim)"""
        return self.proj_down(latents)

    def decode(self, compressed: torch.Tensor) -> torch.Tensor:
        """compressed: (num_latents, bottleneck_dim) -> (num_latents, hidden_dim)"""
        return self.proj_up(compressed)


def autoencoder_loss(model, proj, tokenizer, text: str,
                     num_latents: int) -> torch.Tensor:
    device = next(model.parameters()).device
    last_layer = model.config.num_hidden_layers - 1

    # ---- Pass 1: Encode ----
    enc_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    enc_ids = tokenizer.encode(enc_text, return_tensors="pt").to(device)
    seq_len = enc_ids.shape[1]

    enc_out = model(enc_ids, output_hidden_states=True)
    hidden = enc_out.hidden_states[last_layer][0]  # (seq_len, H)

    if num_latents == 1:
        indices = torch.tensor([seq_len - 1], device=device)
    else:
        step = seq_len / num_latents
        indices = (torch.arange(num_latents, device=device).float() * step + step / 2)
        indices = indices.long().clamp(0, seq_len - 1)
    latents = hidden[indices]  # (num_latents, H)

    # ---- Bottleneck ----
    compressed = proj.encode(latents)       # (num_latents, bottleneck_dim)
    reconstructed = proj.decode(compressed)  # (num_latents, H)

    # ---- Pass 2: Decode ----
    dec_prompt = "<|im_start|>user\n"
    for i in range(num_latents):
        dec_prompt += f"<|L{i}|>"
    dec_prompt += "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n"
    dec_target = f"{text}<|im_end|>"

    prompt_ids = tokenizer.encode(dec_prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(dec_target, add_special_tokens=False)
    decode_ids = torch.tensor([prompt_ids + target_ids], device=device)

    labels = torch.full_like(decode_ids, -100)
    labels[0, len(prompt_ids):] = torch.tensor(target_ids, device=device)

    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(decode_ids).clone()

    for i in range(num_latents):
        li_id = tokenizer.convert_tokens_to_ids(f"<|L{i}|>")
        positions = (decode_ids[0] == li_id).nonzero(as_tuple=True)[0]
        for p in positions:
            embeds[0, p] = reconstructed[i]

    dec_out = model(inputs_embeds=embeds, labels=labels)
    return dec_out.loss


def train_step(model, proj, tokenizer, batch_texts, num_latents, optimizer):
    optimizer.zero_grad()
    losses = []
    for text in batch_texts:
        loss = autoencoder_loss(model, proj, tokenizer, text, num_latents)
        losses.append(loss)
    total = torch.stack(losses).mean()
    total.backward()
    optimizer.step()
    return total.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--max-examples", type=int, default=2000)
    ap.add_argument("--num-latents", type=int, default=NUM_LATENTS)
    ap.add_argument("--bottleneck-dim", type=int, default=BOTTLENECK_DIM)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DATA):
        print(f"Data file {DATA} not found.")
        sys.exit(1)

    rows = [json.loads(l) for l in open(DATA)][:args.max_examples]
    print(f"Loaded {len(rows)} training examples")

    print("Building tokenizer...")
    tok = build_tokenizer(args.num_latents)
    print(f"Vocab size: {len(tok)}")

    print(f"Loading {BASE_MODEL}...")
    model = build_model(args.device, len(tok))
    H = model.config.hidden_size
    L = model.config.num_hidden_layers
    print(f"Hidden dim: {H}, layers: {L}")

    proj = LatentProjection(hidden_dim=H, bottleneck_dim=args.bottleneck_dim).to(args.device)
    print(f"Bottleneck: {H} -> {args.bottleneck_dim} -> {H}")

    all_params = list(model.parameters()) + list(proj.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    import random
    random.seed(42)
    texts = [r["text"] for r in rows]
    random.shuffle(texts)
    n = len(texts)

    os.makedirs(OUT_DIR, exist_ok=True)

    transmitted_bytes = args.num_latents * args.bottleneck_dim * 2  # fp16
    print(f"Training ({args.steps} steps, {args.num_latents} latents, "
          f"bottleneck {args.bottleneck_dim} dims, {transmitted_bytes} bytes transmitted)")

    total_loss = 0.0
    for step in range(args.steps):
        idx = step % n
        batch = [texts[idx]]
        loss_val = train_step(model, proj, tok, batch, args.num_latents, optimizer)
        total_loss += loss_val

        if (step + 1) % args.log_every == 0:
            avg = total_loss / args.log_every
            print(f"step {step+1:4d}/{args.steps}  loss={avg:.4f}")
            total_loss = 0.0

        if (step + 1) % args.save_every == 0:
            ckpt = os.path.join(OUT_DIR, f"checkpoint-{step+1}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            torch.save({
                "proj_down": proj.proj_down.state_dict(),
                "proj_up": proj.proj_up.state_dict(),
                "num_latents": args.num_latents,
                "bottleneck_dim": args.bottleneck_dim,
            }, os.path.join(ckpt, "projection.pt"))
            print(f"  saved {ckpt}")

    final_path = os.path.join(OUT_DIR, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tok.save_pretrained(final_path)
    torch.save({
        "proj_down": proj.proj_down.state_dict(),
        "proj_up": proj.proj_up.state_dict(),
        "num_latents": args.num_latents,
        "bottleneck_dim": args.bottleneck_dim,
    }, os.path.join(final_path, "projection.pt"))
    print(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
