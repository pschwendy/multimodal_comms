#!/usr/bin/env python3
"""Representation server: exposes a frozen local model's hidden states over HTTP.

New infrastructure (nothing in this codebase previously extracted raw
activations from a local model — prior local-model use was either
`.generate()` or vLLM's OpenAI-compatible logprobs API, neither of which
exposes intermediate hidden states). Backs the representational-match family
of compressors (A2 selector labels, A4 best-of-k, B1 saliency, A1 GRPO
reward) across all three benchmarks in this pass.

The proxy model plays no role as an actual sender/receiver (those are
deepseek-v4-flash via API in every benchmark this pass) — it is purely an
offline/auxiliary stand-in whose hidden states define "does the receiver's
understanding change," exactly the role the dedicated Qwen3-4B receiver
already played in Parts 2/4/5's harvesting scripts, just now exposing
activations instead of text/logprobs.

Endpoints:
  POST /rep        {text, context=""}                    -> {rep: [float], dim: int}
  POST /rep_batch   {items: [{text, context}, ...]}        -> {reps: [[float]], dim: int}
  POST /saliency    {text, context=""}                     -> {tokens: [str], scores: [float]}
  GET  /health

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m training.services.repserver --port 8100 --layer 14
"""

import argparse
import threading
from typing import Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
_STATE: dict = {}
# One model, one GPU: all forward/backward access must be serialized.
# FastAPI runs sync endpoints in a threadpool, and concurrent requests from
# multiple clients corrupted the shared HF model's forward dispatch in
# production (RecursionError after ~1.5K calls, then every request 500s).
_MODEL_LOCK = threading.Lock()


class RepRequest(BaseModel):
    text: str
    context: str = ""


class RepBatchRequest(BaseModel):
    items: list[RepRequest]


class SaliencyRequest(BaseModel):
    text: str
    context: str = ""


def _build_input(tokenizer, text: str, context: str) -> str:
    # The message is embedded in its natural surrounding context (if any),
    # matching how every existing compressor receives (messages, receiver_id)
    # in-context rather than scoring a message in isolation.
    return f"{context}\n\n{text}" if context else text


def _last_token_hidden(text: str, context: str) -> torch.Tensor:
    model, tokenizer, layer, device = (
        _STATE["model"], _STATE["tokenizer"], _STATE["layer"], _STATE["device"]
    )
    full = _build_input(tokenizer, text, context)
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with _MODEL_LOCK, torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple(num_layers+1) of (1, seq, dim); index 0 = embeddings
        hs = out.hidden_states[layer]
        return hs[0, -1, :].float()


@app.get("/health")
def health():
    return {"status": "ok", "model": _STATE.get("model_name"), "layer": _STATE.get("layer")}


@app.post("/rep")
def rep(req: RepRequest):
    vec = _last_token_hidden(req.text, req.context)
    vec = F.normalize(vec, dim=0)
    return {"rep": vec.cpu().tolist(), "dim": vec.shape[0]}


@app.post("/rep_batch")
def rep_batch(req: RepBatchRequest):
    reps = []
    for item in req.items:
        vec = _last_token_hidden(item.text, item.context)
        vec = F.normalize(vec, dim=0)
        reps.append(vec.cpu().tolist())
    dim = len(reps[0]) if reps else 0
    return {"reps": reps, "dim": dim}


@app.post("/saliency")
def saliency(req: SaliencyRequest):
    """Gradient-magnitude saliency of each input token toward preserving the
    message's own representation (used to decide what a pruned version of
    THIS text must keep). We backprop d(||h(text)||-direction self-similarity
    reward)/d(input embeddings) — i.e. how much each token's embedding
    contributes to the last-token hidden state at the target layer — and
    return per-token L2 gradient norms as the saliency score.
    """
    model, tokenizer, layer, device = (
        _STATE["model"], _STATE["tokenizer"], _STATE["layer"], _STATE["device"]
    )
    full = _build_input(tokenizer, req.text, req.context)
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with _MODEL_LOCK:
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(enc["input_ids"]).detach().clone().requires_grad_(True)

        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=enc.get("attention_mask"),
            output_hidden_states=True,
        )
        hs = out.hidden_states[layer][0, -1, :]  # (dim,) last-token rep at target layer
        # Objective: the squared norm of the representation itself (a fixed,
        # target-free scalar whose gradient w.r.t. each input embedding measures
        # how much that token contributes to shaping this representation - a
        # single fixed forward+backward pass per message, no growing state).
        objective = (hs.float() ** 2).sum()
        model.zero_grad(set_to_none=True)
        objective.backward()

        grad = inputs_embeds.grad[0]  # (seq, embed_dim)
        token_scores = grad.norm(dim=-1).detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())

    # Only report saliency for the `text` portion, not the prepended context:
    # re-tokenize context alone to find the split point.
    if req.context:
        ctx_len = len(tokenizer(req.context + "\n\n")["input_ids"])
    else:
        ctx_len = 0
    return {
        "tokens": tokens[ctx_len:],
        "scores": token_scores[ctx_len:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[repserver] loading {args.model} on {device}, layer={args.layer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _STATE["model"] = model
    _STATE["tokenizer"] = tokenizer
    _STATE["layer"] = args.layer
    _STATE["device"] = device
    _STATE["model_name"] = args.model
    print("[repserver] ready", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
