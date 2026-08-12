#!/usr/bin/env python3
"""Qualitative check: encode real dev examples -> latents -> greedy-decode
back, and print original vs reconstruction side by side. Also reports a
rough semantic-similarity score (cosine similarity of mean-pooled last-layer
hidden states from the same frozen base model, before/after reconstruction)
as a cheap proxy for "did the meaning survive even where tokens don't match".
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_checkpoint(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    with open(os.path.join(path, "ae_config.json")) as f:
        cfg = json.load(f)
    generator = None
    generator_path = os.path.join(path, "generator.pt")
    if cfg.get("generator") == "mwnot" or os.path.exists(generator_path):
        from multimodal_comms.methods.autoencoders.mwnot_generator import (
            SequenceGeneratorEncoder,
        )

        checkpoint = torch.load(generator_path, map_location=device, weights_only=True)
        generator = SequenceGeneratorEncoder(**checkpoint["config"]).to(device)
        generator.load_state_dict(checkpoint["state_dict"])
        generator.eval()
    return model, tok, cfg["num_latents"], generator


def encode(model, tok, text, num_latents, device, max_len, generator=None):
    enc_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    encoded = tok(enc_text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    enc_ids = encoded.input_ids
    seq_len = enc_ids.shape[1]
    last_layer = model.config.num_hidden_layers - 1
    with torch.no_grad():
        out = model(enc_ids, output_hidden_states=True)
        hidden_batch = out.hidden_states[last_layer]
        if generator is not None:
            mask = encoded.attention_mask.bool()
            return generator(hidden_batch.float(), mask)[0].to(hidden_batch.dtype)
        hidden = hidden_batch[0]
    if num_latents == 1:
        idx = [seq_len - 1]
    else:
        step = seq_len / num_latents
        idx = [min(int(i * step + step / 2), seq_len - 1) for i in range(num_latents)]
    return hidden[idx]  # (num_latents, H)


def decode(model, tok, latents, num_latents, device, max_new_tokens):
    dec_prompt = "<|im_start|>user\n" + "".join(f"<|L{i}|>" for i in range(num_latents))
    dec_prompt += "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tok(dec_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(prompt_ids)
    for i in range(num_latents):
        li_id = tok.convert_tokens_to_ids(f"<|L{i}|>")
        pos = (prompt_ids[0] == li_id).nonzero(as_tuple=True)[0]
        embeds[0, pos] = latents[i].to(embeds.dtype)

    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    generated = []
    past_kv = None
    current = embeds
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(inputs_embeds=current, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = out.logits[0, -1, :].argmax().item()
            if next_id == eos_id:
                break
            generated.append(next_id)
            current = embed_layer(torch.tensor([[next_id]], device=device))
    return tok.decode(generated, skip_special_tokens=True).strip()


def pooled_embedding(model, tok, text, device, max_len):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0]  # (S, H)
    return h.mean(dim=0)  # mean pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-data", required=True)
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    model, tok, num_latents, generator = load_checkpoint(args.checkpoint, args.device)
    encoder_name = "MWNOT generator" if generator is not None else "sampled positions"
    print(f"Loaded {args.checkpoint} (num_latents={num_latents}, encoder={encoder_name})\n")

    import random
    rows = [json.loads(l)["text"] for l in open(args.dev_data)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_examples]

    sims = []
    for i, text in enumerate(rows):
        latents = encode(
            model, tok, text, num_latents, args.device, args.max_len, generator
        )
        recon = decode(model, tok, latents, num_latents, args.device, args.max_new_tokens)

        orig_emb = pooled_embedding(model, tok, text, args.device, args.max_len)
        recon_emb = pooled_embedding(model, tok, recon, args.device, args.max_len)
        sim = F.cosine_similarity(orig_emb.unsqueeze(0).float(), recon_emb.unsqueeze(0).float()).item()
        sims.append(sim)

        print(f"=== Example {i+1} (orig {len(tok.encode(text, add_special_tokens=False))} tok, "
              f"semantic cos-sim={sim:.3f}) ===")
        print(f"ORIGINAL:      {text[:600]}{'...' if len(text) > 600 else ''}")
        print(f"RECONSTRUCTED: {recon[:600]}{'...' if len(recon) > 600 else ''}")
        print()

    print(f"Mean semantic cosine similarity over {len(sims)} examples: {sum(sims)/len(sims):.3f}")


if __name__ == "__main__":
    main()
