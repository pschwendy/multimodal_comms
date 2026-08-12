#!/usr/bin/env python3
"""Encode QuALITY articles into cached chunk latents with the frozen longctx AE.

Each unique article is split into n = max(1, len_tokens // 1000) equal token
chunks (sizes land in [1000, 2000), matching the encoder's 1000-4000-token
training band), and each chunk is encoded into 16 latents via the exact
encode path of pretrain_autoencoder.py (user-turn wrapper, last-layer hidden
states, midpoint position sampling).

Outputs, under --out-dir:
  latents/<article_sha>.pt   {"latents": (n_chunks, 16, H) bf16 tensor,
                              "chunk_lens": [int]}
  index_<split>.jsonl        one row per QUESTION: {qidx, split, article_sha,
                              question, options, answer, n_chunks, article_tokens}
"""

import argparse
import hashlib
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CHUNK_TOKENS = 1000


def latent_indices(seq_len: int, num_latents: int) -> list[int]:
    if num_latents == 1:
        return [seq_len - 1]
    step = seq_len / num_latents
    return [min(int(i * step + step / 2), seq_len - 1) for i in range(num_latents)]


@torch.no_grad()
def encode_chunks(model, tokenizer, texts, num_latents, max_len, device, last_layer):
    enc_texts = [f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in texts]
    enc = tokenizer(enc_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len).to(device)
    out = model(enc["input_ids"], attention_mask=enc["attention_mask"],
                output_hidden_states=True)
    hidden = out.hidden_states[last_layer]
    seq_lens = enc["attention_mask"].sum(dim=1).tolist()
    latents = []
    for b, seq_len in enumerate(seq_lens):
        idx = latent_indices(seq_len, num_latents)
        latents.append(hidden[b, idx, :])
    return torch.stack(latents), seq_lens  # (B, num_latents, H)


def article_sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-checkpoint", default="data/autoencoder_pretrain_longctx_big/final")
    ap.add_argument("--out-dir", default="data/quality_latents")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-chunks", type=int, default=4)
    args = ap.parse_args()

    ae_cfg = json.load(open(os.path.join(args.ae_checkpoint, "ae_config.json")))
    num_latents = ae_cfg["num_latents"]

    tok = AutoTokenizer.from_pretrained(args.ae_checkpoint)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.ae_checkpoint, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()
    last_layer = model.config.num_hidden_layers - 1

    lat_dir = os.path.join(args.out_dir, "latents")
    os.makedirs(lat_dir, exist_ok=True)

    ds = load_dataset("emozilla/quality")
    done = set()
    for split in ("train", "validation"):
        index_rows = []
        for qidx, item in enumerate(ds[split]):
            sha = article_sha(item["article"])
            ids = tok(item["article"], add_special_tokens=False)["input_ids"]
            n_chunks = max(1, len(ids) // CHUNK_TOKENS)
            index_rows.append({
                "qidx": qidx, "split": split, "article_sha": sha,
                "question": item["question"], "options": item["options"],
                "answer": item["answer"], "n_chunks": n_chunks,
                "article_tokens": len(ids),
            })
            if sha in done:
                continue
            done.add(sha)
            step = (len(ids) + n_chunks - 1) // n_chunks
            chunks = [tok.decode(ids[i * step:(i + 1) * step], skip_special_tokens=True)
                      for i in range(n_chunks)]
            lats, chunk_lens = [], []
            for i in range(0, len(chunks), args.batch_chunks):
                batch = chunks[i:i + args.batch_chunks]
                l, lens = encode_chunks(model, tok, batch, num_latents,
                                        args.max_len, args.device, last_layer)
                lats.append(l.cpu())
                chunk_lens.extend(lens)
            torch.save({"latents": torch.cat(lats).to(torch.bfloat16),
                        "chunk_lens": chunk_lens},
                       os.path.join(lat_dir, f"{sha}.pt"))
            if len(done) % 25 == 0:
                print(f"[{split}] encoded {len(done)} articles", flush=True)
        with open(os.path.join(args.out_dir, f"index_{split}.jsonl"), "w") as f:
            for row in index_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[{split}] wrote {len(index_rows)} question rows", flush=True)

    print(f"Done: {len(done)} unique articles -> {lat_dir}")


if __name__ == "__main__":
    main()
