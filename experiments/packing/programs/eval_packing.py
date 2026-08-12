#!/usr/bin/env python3
"""Reconstruction quality vs. messages-per-packet, across fusion methods.

Builds REAL packets from REAL dev text at each (fusion, message count)
operating point, has each receiver recover its own message, and scores the
reconstruction. The question this answers is the headline one: how many
messages fit in one fixed-size packet before per-message reconstruction
falls apart, and which fusion method gets furthest.

Metrics, and why these ones (same reasoning as
The semantic-fidelity evaluator showed that full-string difflib badly understates
paraphrase-faithful reconstruction, so it is reported but never led with):

  code_cos     recovered code vs. true code. Pure channel term: what the
               fusion did, before the decoder is involved at all.
  latent_cos   bottleneck-reconstructed (K,H) latents vs. the true ones.
               Adds the bottleneck's own loss to the channel's.
  semantic_cos re-encode the DECODED TEXT through the model's own encoder
               and compare to the true latent -- end-to-end content
               preservation. MUST be read against `floor_cos`, the same
               metric on UNRELATED dev texts (~0.3-0.4 in this embedding
               space, not 0), which is reported alongside.
  difflib      secondary literal-overlap cross-reference.

Example:
  python experiments/packing/programs/eval_packing.py --model-path data/packed_matryoshka/final \
      --fusion rotor frame block --n-packets 8 --device cuda:8
"""

import argparse
import difflib
import json
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from multimodal_comms.benchmarks.hiddenbench.runtime.packing import (  # noqa: E402
    PackedCodec,
    build_packer,
    frame_leakage_bound,
    quantize_packet,
)


def load_texts(path, n, seed=0):
    texts = [json.loads(l)["text"] for l in open(path)]
    random.Random(seed).shuffle(texts)
    return texts[:n]


def cos(a, b):
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


_STOP = set("the a an and or but if of to in on at for with by from as is are was "
            "were be been being it its this that these those he she they we you i "
            "his her their our your my not no so than then there here have has had "
            "do does did will would can could should may might must about into over "
            "after before under above s t re ve ll d m".split())


def content_words(text: str) -> list[str]:
    import re
    return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
            if w not in _STOP and len(w) > 2]


def content_f1(ref: str, hyp: str) -> float:
    """Bag-of-content-words F1 between source and reconstruction.

    Needed because semantic_cos loses all resolution on short messages: the
    unrelated-pair floor for 200-char FineWeb snippets is ~0.84 (they all
    share generic-web-prose structure), so a channel carrying nothing and a
    channel carrying everything score within 0.04 of each other. Content-word
    overlap has a chance floor near 0.02 on the same data, which leaves room
    to actually measure something. Reported against `f1_floor`, the same
    statistic on unrelated pairs, exactly as semantic_cos is.
    """
    from collections import Counter
    r, h = Counter(content_words(ref)), Counter(content_words(hyp))
    if not r or not h:
        return 0.0
    overlap = sum((r & h).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(h.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="data/packed_matryoshka/final")
    ap.add_argument("--dev-data", default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--fusion", nargs="+", default=["rotor", "frame", "block"])
    ap.add_argument("--messages", type=int, nargs="+", default=None,
                    help="Message counts M to test; default = the ladder's own "
                         "P//d points (4,8,...,256).")
    ap.add_argument("--n-packets", type=int, default=6,
                    help="Packets per operating point (each contributes "
                         "`--slots-scored` decoded messages).")
    ap.add_argument("--slots-scored", type=int, default=4,
                    help="Receivers actually decoded per packet (decoding all "
                         "M would be 256 generations for one data point).")
    ap.add_argument("--short-chars", type=int, default=400,
                    help="Truncate dev texts to chat-message length; 0 = full.")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="Drop dev texts shorter than this after truncation.")
    ap.add_argument("--device", default="cuda:8")
    ap.add_argument("--out", default="reports/packing_sweep.json")
    ap.add_argument("--bits", type=int, default=32,
                    help="Quantise the packet to this many bits/dim before "
                         "unpacking (the wire format). 32 = raw fp32.")
    ap.add_argument("--overload", type=float, nargs="*", default=[2.0],
                    help="Extra frame-only points at these multiples of the "
                         "zero-crosstalk capacity P//d.")
    ap.add_argument("--max-len", type=int, default=384,
                    help="Encoder truncation. MUST cover the message length or "
                         "the encoder silently never sees the tail.")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="Decode budget; 0 = 1.2*max_len+20. Too small scores a "
                         "long reference against a truncated generation, which "
                         "reads as channel loss but is a measurement artefact.")
    ap.add_argument("--num-latents", type=int, default=0,
                    help="Override K (default: from the checkpoint config).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    codec = PackedCodec(
        args.model_path, device=args.device, max_len=args.max_len,
        max_new_tokens=args.max_new_tokens or int(args.max_len * 1.2) + 20)
    codec._load()
    if args.num_latents:
        codec.num_latents = args.num_latents
    P, ladder = codec.packet_dim, sorted(codec.ladder, reverse=True)
    print(f"packet P={P} floats ({P*4/1024:.0f} KB fp32), ladder={ladder}")

    if args.messages:
        points = [(P // m, m) for m in args.messages]
    else:
        points = [(d, P // d) for d in ladder]
        # Overload points: M beyond P/d, which block/rotor structurally
        # cannot reach (no slots left) but frame degrades into gracefully.
        # This is where "how many messages can we possibly cram in" stops
        # being a capacity question and becomes an SNR one.
        for mult in args.overload:
            points += [(d, int(P // d * mult)) for d in ladder[-3:]]

    n_texts = args.n_packets * max(m for _, m in points) + 64
    texts = load_texts(args.dev_data, n_texts, args.seed)
    if args.short_chars:
        texts = [t[:args.short_chars] for t in texts]
    texts = [t for t in texts if len(t) >= args.min_chars]
    # Truncate to the encoder's own window so every metric compares against
    # the text the channel was actually given, not the part it never saw.
    _tk = codec._tok
    texts = [_tk.decode(_tk.encode(t, add_special_tokens=False)[:args.max_len])
             for t in texts]

    # Unrelated-pair floor: semantic_cos of independent real texts. Every
    # semantic number below is meaningless without it (LLM hidden states
    # share a large generic-language component, so unrelated text scores
    # well above zero).
    print("computing unrelated-pair floor...")
    with torch.no_grad():
        fl = codec.latents(texts[:32]).float()
    floor = statistics.mean(cos(fl[i], fl[j]) for i in range(0, 16)
                            for j in range(16, 32))
    f1_floor = statistics.mean(content_f1(texts[i], texts[j])
                               for i in range(0, 16) for j in range(16, 32))
    print(f"  floor_cos = {floor:.3f}   f1_floor = {f1_floor:.3f}\n")

    rows = []
    for kind in args.fusion:
        for d, M in points:
            if kind in ("block", "rotor") and M > P // d:
                continue
            codec.code_dim = d
            per = {"code_cos": [], "latent_cos": [], "semantic_cos": [],
                   "content_f1": [], "difflib": []}
            for p in range(args.n_packets):
                batch = [texts[(p * M + i) % len(texts)] for i in range(M)]
                with torch.no_grad():
                    true_lat = codec.latents(batch).float()
                    codes = codec._bn.encode(true_lat, d)[:, :d].cpu()
                packer = build_packer(kind, P, d, seed=1234 + p, nonce=p)
                packet = packer.pack({i: codes[i] for i in range(M)})
                if args.bits < 32:
                    packet, qmeta = quantize_packet(packet, bits=args.bits)
                    wire_bytes = qmeta["bytes"]
                else:
                    wire_bytes = P * 4

                scored = random.Random(p).sample(range(M),
                                                 min(args.slots_scored, M))
                for slot in scored:
                    rec = packer.unpack(packet, slot)
                    per["code_cos"].append(cos(rec, codes[slot]))
                    padded = torch.zeros(codec._bn.code_dim)
                    padded[:d] = rec
                    with torch.no_grad():
                        z = codec._bn.decode(padded.unsqueeze(0).to(args.device))[0]
                    per["latent_cos"].append(cos(z, true_lat[slot]))
                    txt = codec.decode(padded.to(args.device))
                    if txt:
                        with torch.no_grad():
                            re_lat = codec.latents([txt]).float()[0]
                        per["semantic_cos"].append(cos(re_lat, true_lat[slot]))
                        per["content_f1"].append(content_f1(batch[slot], txt))
                        per["difflib"].append(difflib.SequenceMatcher(
                            None, batch[slot][:150], txt[:150]).ratio())
                    else:
                        per["semantic_cos"].append(0.0)
                        per["content_f1"].append(0.0)
                        per["difflib"].append(0.0)

            row = {"fusion": kind, "code_dim": d, "messages": M,
                   "rho": M * d / P, "bits": args.bits,
                   "packet_kb": wire_bytes / 1024,
                   "bytes_per_msg": wire_bytes / M, "floor_cos": floor,
                   "f1_floor": f1_floor}
            row.update({k: statistics.mean(v) if v else float("nan")
                        for k, v in per.items()})
            row["snr_db"] = frame_leakage_bound(M, d, P)["snr_db"]
            rows.append(row)
            print(f"{kind:6s} M={M:4d} d={d:5d} rho={row['rho']:.2f} "
                  f"| code_cos={row['code_cos']:.3f} latent_cos={row['latent_cos']:.3f} "
                  f"F1={row['content_f1']:.3f} (floor {f1_floor:.3f}) "
                  f"sem={row['semantic_cos']:.3f} (floor {floor:.3f}) "
                  f"| {row['bytes_per_msg']:.0f} B/msg", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model_path, "packet_dim": P, "floor_cos": floor,
               "f1_floor": f1_floor, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
