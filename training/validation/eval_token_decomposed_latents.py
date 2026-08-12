#!/usr/bin/env python3
"""Quick experiment: can the AE's continuous latents be replaced by a sparse
*token-space* decomposition and still decode?

Idea under test (user's sketch): a latent h (H floats = 5-10 KB) is projected
through the LM head into vocab space; keep only the top-k tokens with
1-digit-quantised weights ("a3b6c1"), transmit those, and rebuild an
approximate latent on the receiver as a weighted sum of unembedding rows
before handing it to the AE decoder.

Reconstruction detail: tokens are *selected* in logit space (top-k of
U @ norm(x), i.e. the actual top tokens a logit lens would name), but the
latent is *rebuilt* in raw hidden space:
    x_hat = (sum_k w_k * U[t_k]) rescaled to ||x||
Rebuilding via the normed space and dividing out the RMSNorm gain g does not
work -- g has near-zero and negative entries, so 1/g amplifies noise and the
result ends up anti-correlated with the true latent (cos ~ -0.6).
The scale ||x|| is one extra fp16 number per latent.

Variants:
  full      -- baseline, exact latent round-trip (upper bound)
  softmax-k -- top-k tokens, softmax weights quantised to one decimal digit
  lstsq-k   -- top-k tokens, least-squares coefficients quantised to 8 bits
  rand-k    -- control: k random tokens with lstsq coefficients
  noise-c   -- control: exact latent corrupted to cosine c, to calibrate how
               much latent fidelity the decoder actually needs
"""

import argparse
import difflib
import json
import math
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "data/autoencoder_pretrain_large/final"


# ---------------------------------------------------------------- metrics
def unigram_f1(ref: str, hyp: str) -> float:
    r = re.findall(r"\w+", ref.lower())
    h = re.findall(r"\w+", hyp.lower())
    if not r or not h:
        return 0.0
    from collections import Counter
    common = Counter(r) & Counter(h)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, rc = n / len(h), n / len(r)
    return 2 * p * rc / (p + rc)


def char_sim(ref: str, hyp: str) -> float:
    return difflib.SequenceMatcher(None, ref, hyp).ratio()


# ---------------------------------------------------------------- AE pieces
def latent_indices(seq_len: int, k: int) -> list[int]:
    if k == 1:
        return [seq_len - 1]
    step = seq_len / k
    return [min(int(i * step + step / 2), seq_len - 1) for i in range(k)]


@torch.no_grad()
def encode(model, tok, text: str, num_latents: int, device: str) -> torch.Tensor:
    enc_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(enc_text, return_tensors="pt").to(device)
    last_layer = model.config.num_hidden_layers - 1
    out = model(ids, output_hidden_states=True)
    hidden = out.hidden_states[last_layer][0]
    return hidden[latent_indices(ids.shape[1], num_latents)]  # (K, H)


@torch.no_grad()
def decode(model, tok, latents: torch.Tensor, device: str, max_new_tokens: int) -> str:
    num_latents = latents.shape[0]
    prompt = "<|im_start|>user\n" + "".join(
        f"<|L{i}|>" for i in range(num_latents)
    ) + "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tok.encode(prompt, return_tensors="pt").to(device)
    embed = model.get_input_embeddings()
    embeds = embed(prompt_ids)
    for i in range(num_latents):
        li = tok.convert_tokens_to_ids(f"<|L{i}|>")
        for p in (prompt_ids[0] == li).nonzero(as_tuple=True)[0]:
            embeds[0, p] = latents[i].to(embeds.dtype)

    eos = tok.convert_tokens_to_ids("<|im_end|>")
    past, cur, gen = None, embeds, []
    for _ in range(max_new_tokens):
        out = model(inputs_embeds=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1, :].argmax().item()
        if nxt == eos:
            break
        gen.append(nxt)
        cur = embed(torch.tensor([[nxt]], device=device))
    return tok.decode(gen, skip_special_tokens=True).strip()


# ------------------------------------------------- token-space decomposition
def normed(model, x: torch.Tensor) -> torch.Tensor:
    """Apply the model's final RMSNorm the way the LM head sees it."""
    return model.model.norm(x)


def decompose(model, x: torch.Tensor, U: torch.Tensor, g: torch.Tensor,
              k: int, mode: str, digits: int = 1):
    """x: (H,) raw latent. Returns (x_hat, token_ids, quantised weights, cos)."""
    xf = x.float()
    y = normed(model, x.unsqueeze(0))[0].float()          # normed space
    logits = U.float() @ y                                 # (V,) logit lens

    if mode == "noise":
        # k is the target cosine in percent; corrupt the exact latent to it
        target = k / 100.0
        n = torch.randn_like(xf)
        n = n - n.dot(xf) / xf.dot(xf) * xf               # orthogonalise
        n = n / n.norm() * xf.norm()
        x_hat = target * xf + math.sqrt(1 - target ** 2) * n
        x_hat = x_hat / x_hat.norm() * xf.norm()
        return x_hat.to(x.dtype), None, None, torch.nn.functional.cosine_similarity(
            x_hat, xf, dim=0).item()

    if mode == "rand":
        idx = torch.randint(0, U.shape[0], (k,), device=U.device)
    else:
        idx = logits.topk(k).indices

    B = U[idx].float()                                     # (k, H)

    if mode == "softmax":
        w = torch.softmax(logits[idx], dim=0)
        q = torch.round(w * (10 ** digits - 1)).clamp(min=0)   # e.g. a3b6c1
        w_hat = q / q.sum() if q.sum() > 0 else w
    else:  # lstsq / rand -> least-squares fit in raw space, 8-bit quantised
        coef = torch.linalg.lstsq(B.T, xf).solution          # (k,)
        s = coef.abs().max()
        w_hat = torch.round(coef / s * 127) / 127 * s if s > 0 else coef

    x_hat = w_hat @ B                                      # raw hidden space
    x_hat = x_hat / x_hat.norm() * xf.norm()               # match latent scale
    cos = torch.nn.functional.cosine_similarity(x_hat, xf, dim=0).item()
    return x_hat.to(x.dtype), idx, w_hat, cos


def payload_bytes(k: int, num_latents: int, mode: str, vocab: int, digits: int) -> int:
    """Rough transmitted size: token ids + weights + one fp16 scale per latent."""
    id_bits = math.ceil(math.log2(vocab))
    w_bits = math.ceil(math.log2(10 ** digits)) if mode == "softmax" else 8
    return math.ceil(num_latents * (k * (id_bits + w_bits) + 16) / 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--data", default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--num-latents", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-chars", type=int, default=700)
    ap.add_argument("--out", default="data/token_decomposed_latents.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16).to(args.device).eval()
    H = model.config.hidden_size
    U = model.get_output_embeddings().weight.detach()      # (V, H)
    g = model.model.norm.weight.detach()                   # (H,)
    V = U.shape[0]
    print(f"hidden={H} vocab={V} latents={args.num_latents} "
          f"full payload = {args.num_latents * H * 2} B (fp16)")

    texts = [json.loads(l)["text"][:args.max_chars]
             for l in open(args.data)][:args.n]

    variants = [("full", 0)] + [
        (m, k) for m in ("softmax", "lstsq") for k in (3, 8, 32, 128, 512)
    ] + [("rand", 128)] + [("noise", c) for c in (99, 90, 70, 50, 30)]

    results = {f"{m}-{k}" if k else m: {"f1": [], "chars": [], "cos": []}
               for m, k in variants}
    samples = []

    for i, text in enumerate(texts):
        lat = encode(model, tok, text, args.num_latents, args.device)
        rec = {"text": text}
        for mode, k in variants:
            name = f"{mode}-{k}" if k else mode
            if mode == "full":
                lat_hat, coss = lat, [1.0]
                demo = None
            else:
                rows, coss = [], []
                demo = None
                for j in range(lat.shape[0]):
                    xh, idx, w, c = decompose(model, lat[j], U, g, k, mode)
                    rows.append(xh)
                    coss.append(c)
                    if j == lat.shape[0] - 1 and k <= 8:
                        demo = "".join(
                            f"{tok.decode([t])!r}:{v:.2f} "
                            for t, v in zip(idx.tolist(), w.tolist()))
                lat_hat = torch.stack(rows)
            out = decode(model, tok, lat_hat, args.device, args.max_new_tokens)
            results[name]["f1"].append(unigram_f1(text, out))
            results[name]["chars"].append(char_sim(text, out))
            results[name]["cos"].append(sum(coss) / len(coss))
            rec[name] = {"out": out, "demo": demo}
        samples.append(rec)
        print(f"[{i+1}/{len(texts)}] done")

    print("\n" + "=" * 88)
    print(f"{'variant':<14}{'cos(latent)':>12}{'unigram F1':>12}"
          f"{'char sim':>10}{'bytes':>10}{'vs full':>10}")
    print("-" * 88)
    full_b = args.num_latents * H * 2
    for mode, k in variants:
        name = f"{mode}-{k}" if k else mode
        r = results[name]
        b = full_b if mode in ("full", "noise") else payload_bytes(
            k, args.num_latents, mode, V, 1)
        size = "  (n/a)" if mode == "noise" else f"{b:>10d}{full_b/b:>9.0f}x"
        print(f"{name:<14}{sum(r['cos'])/len(r['cos']):>12.3f}"
              f"{sum(r['f1'])/len(r['f1']):>12.3f}"
              f"{sum(r['chars'])/len(r['chars']):>10.3f}{size}")
    print("=" * 88)

    json.dump({"results": results, "samples": samples},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
