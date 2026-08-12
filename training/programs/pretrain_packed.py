#!/usr/bin/env python3
"""Train a matryoshka-nested bottleneck so ONE packet holds MANY messages.

The capacity argument: a packet is P real
numbers, so fitting M messages means each message must live in ~P/M dims.
This script trains the head that makes a P/M-dim code actually decodable:

  message -> LM encoder -> (K, H) latents -> down -> d-dim code
          -> [ packet: crosstalk noise calibrated to the load ]
          -> up -> (K, H) latents -> LM decoder -> message

Each step samples a width d from the matryoshka ladder and truncates the
code to its first d dims, so ONE run yields the entire quality-vs-message-
count frontier (d=2560 -> 4 messages/packet ... d=40 -> 256 messages/packet)
rather than one training run per operating point. Each step also samples a
load factor rho and adds Gaussian noise of std sqrt(rho), which is exactly
FramePacker's crosstalk law -- so the same checkpoint serves the exact
BlockPacker and the overloadable FramePacker without ever building a packet
during training (and therefore with per-step memory independent of M).

Warm start from a trained single-message autoencoder; the LM already knows
the encode/decode protocol, this run only has to learn the bottleneck:

  torchrun --standalone --nproc_per_node=8 training/programs/pretrain_packed.py \
      --init-from data/autoencoder_pretrain_large/final \
      --out-dir data/packed_matryoshka
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist
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
from multimodal_comms.methods.packing.learned import (  # noqa: E402
    DEFAULT_LADDER,
    PackedBottleneck,
    crosstalk_std,
)


def apply_lora(model, r: int, alpha: int = 0, dropout: float = 0.0):
    """Wrap the LM in LoRA adapters and return the wrapped model.

    Why LoRA rather than a full fine-tune, for THIS job specifically: the
    decoder does not need new knowledge, it needs to change what it reads --
    a narrow code instead of a raw hidden state. That is a re-parameterisation
    of its input interface, which is exactly what low-rank adapters are good
    at. The decisive practical reason is bandwidth: a full 4B all-reduce costs
    ~9 s per optimizer step on this cluster (NCCL_P2P_DISABLE is required
    here), so a 3-hour run buys only ~750 optimizer steps -- far too few to
    retrain an input interface. Adapters cut the synchronised parameter count
    ~20x, which converts the same wall clock into thousands of steps.

    The adapters ride BOTH passes, which is wanted: the encoder must be free
    to move its latents somewhere compressible (the whole point -- see the
    near-isotropy finding in reports/packed_fusion_20260722.md), and the
    decoder must learn to read the code.
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    # Deliberately NOT calling enable_input_require_grads(): it marks the
    # embedding output as a leaf requiring grad, and the decode path writes
    # the latents into that tensor in place, which autograd then rejects
    # ("a view of a leaf Variable ... used in an in-place operation"). It is
    # only needed for REENTRANT gradient checkpointing; this script uses
    # use_reentrant=False, where trainable parameters inside the graph are
    # enough to keep the checkpointed segments differentiable.
    return model


def set_lm_trainable(model, flag: bool, lora: bool):
    """Freeze/unfreeze the LM. With LoRA only the adapters are ever trainable,
    so toggling the base weights would either be a no-op or would silently
    un-freeze the whole model."""
    if lora:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(flag)
    else:
        model.requires_grad_(flag)


def flat_grads(params, bucket_bytes: int = 256 << 20):
    """Average .grad across ranks using a few big all-reduces, not one per
    parameter. Groups params into buckets of ~bucket_bytes, copies each
    bucket into one contiguous buffer, reduces it, and copies back."""
    bucket, nbytes = [], 0
    def flush(bs):
        if not bs:
            return
        flat = torch._utils._flatten_dense_tensors([p.grad for p in bs])
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        for p, g in zip(bs, torch._utils._unflatten_dense_tensors(
                flat, [p.grad for p in bs])):
            p.grad.copy_(g)
    for p in params:
        bucket.append(p)
        nbytes += p.grad.numel() * p.grad.element_size()
        if nbytes >= bucket_bytes:
            flush(bucket)
            bucket, nbytes = [], 0
    flush(bucket)


def sample_width(ladder: list[int], step: int, total: int) -> int:
    """Curriculum over code width: start wide, work down the ladder.

    Same motivation as pretrain_superpose.py's slot-count curriculum -- the
    narrow widths are unlearnable from a cold bottleneck, but trivial once
    the wide ones have organised the code space and the nesting means the
    narrow codes are prefixes of the wide ones that already work.
    """
    frac = step / max(total, 1)
    if frac < 0.10:
        avail = ladder[:2]
    elif frac < 0.25:
        avail = ladder[:4]
    elif frac < 0.45:
        avail = ladder[:6]
    else:
        avail = ladder
    return random.choice(avail)


def sample_rho(max_rho: float, noise_prob: float = 0.25) -> float:
    """Load factor for the injected crosstalk: rho = M-1, noise std sqrt(rho).

    Sampled LOG-uniformly in the message count M, not uniformly in rho.
    Uniform-in-rho is uniform in M, which is the wrong measure when the
    frontier is being searched over M = 1..128: it puts half the training
    mass above M=64 and almost none below M=8, so the low-load regime that
    sets the fidelity ceiling goes undertrained while the high-load regime
    that is information-theoretically hopeless soaks up the budget. Log
    spacing gives every octave of M equal weight, which is the scale the
    capacity law (bits/packet ~ P/(2 ln2 (M-1))) actually lives on."""
    if max_rho <= 0 or random.random() >= noise_prob:
        return 0.0
    m = math.exp(random.uniform(0.0, math.log(max_rho + 1.0)))
    return m - 1.0


def save_checkpoint(raw_model, bottleneck, tokenizer, args, path):
    os.makedirs(path, exist_ok=True)
    # Merge adapters so the checkpoint loads as a plain AutoModelForCausalLM
    # everywhere downstream (PackedCodec, LatentCodec, the eval scripts).
    if getattr(args, "lora_r", 0) > 0:
        import copy
        merged = copy.deepcopy(raw_model).merge_and_unload()
        merged.save_pretrained(path)
    else:
        raw_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({"state_dict": bottleneck.state_dict(),
                "config": bottleneck.config(),
                "ladder": args.ladder}, os.path.join(path, "bottleneck.pt"))
    with open(os.path.join(path, "ae_config.json"), "w") as f:
        json.dump({"num_latents": args.num_latents, "bottleneck_dim": None}, f)
    with open(os.path.join(path, "packed_config.json"), "w") as f:
        json.dump({"num_latents": args.num_latents, "ladder": args.ladder,
                   "packet_dim": args.packet_dim, "max_rho": args.max_rho}, f)


@torch.no_grad()
def eval_dev(model, bottleneck, tokenizer, dev_texts, args, prompt_ids,
             li_positions, device, last_layer, embed_layer, n_batches=4):
    """Teacher-forced token accuracy at every ladder width, noiseless
    (BlockPacker operating point). Reported keyed by the message count each
    width corresponds to at the configured packet size."""
    model.eval()
    bottleneck.eval()
    out = {}
    for d in args.ladder:
        accs = []
        for i in range(0, min(len(dev_texts), args.batch_size * n_batches),
                       args.batch_size):
            batch = dev_texts[i:i + args.batch_size]
            if not batch:
                continue
            latents = encode_batch(model, tokenizer, batch, args.num_latents,
                                   args.max_len, device, last_layer)
            code = bottleneck.encode(latents.float(), d)
            recon = bottleneck.decode(code, out_dtype=latents.dtype)
            _, acc = decode_batch_loss(model, tokenizer, batch, recon,
                                       prompt_ids, li_positions, args.max_len,
                                       device, embed_layer)
            accs.append(acc)
        out[d] = sum(accs) / len(accs) if accs else float("nan")
    model.train()
    bottleneck.train()
    return out


_STOP = set("the a an and or but if of to in on at for with by from as is are was "
            "were be been being it its this that these those he she they we you i "
            "his her their our your my not no so than then there here have has had "
            "do does did will would can could should may might must about into over "
            "after before under above s t re ve ll d m".split())


def _content_f1(ref: str, hyp: str) -> float:
    import re
    from collections import Counter
    w = lambda t: [x for x in re.findall(r"[a-z0-9']+", (t or "").lower())
                   if x not in _STOP and len(x) > 2]
    r, h = Counter(w(ref)), Counter(w(hyp))
    if not r or not h:
        return 0.0
    ov = sum((r & h).values())
    if not ov:
        return 0.0
    p, rc = ov / sum(h.values()), ov / sum(r.values())
    return 2 * p * rc / (p + rc)


@torch.no_grad()
def eval_free_running(model, bottleneck, tok, dev_texts, args, device,
                      last_layer, n=6, max_new=None):
    """Greedy free-running reconstruction F1 at each ladder width.

    This is the metric that matters and the one that has to be watched during
    training. Teacher-forced token accuracy is NOT a proxy for it: errors
    compound when the model generates its own prefix, so ~0.47 teacher-forced
    accuracy still yields chance-level free-running output. Judging an earlier
    run by tok_acc alone is precisely what hid a dead channel behind a number
    that looked like it was working.
    """
    # The generation budget MUST track the message length. Scoring a
    # 512-token reference against a 110-token generation reports a low F1 that
    # is purely a truncation artefact, not a channel property.
    if max_new is None:
        max_new = int(args.max_len * 1.2) + 20
    model.eval()
    bottleneck.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    embed = base.get_input_embeddings()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    prompt = "<|im_start|>user\n" + "".join(
        f"<|L{i}|>" for i in range(args.num_latents)
    ) + "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n"
    pid = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)
    pos = [tok.encode(prompt, add_special_tokens=False).index(
        tok.convert_tokens_to_ids(f"<|L{i}|>")) for i in range(args.num_latents)]

    out = {}
    # SHARD across ranks. Evaluating on rank 0 alone (the old behaviour) left
    # the other 7 GPUs blocked at the next step's all-reduce for the whole
    # eval; on FineWeb, where generations rarely hit EOS early and run the full
    # budget, that exceeded the 600s NCCL watchdog and killed the job. Every
    # rank now evaluates its own slice of the SAME dev texts and the per-text
    # scores are gathered, so the collective stays in lockstep and eval is ~8x
    # faster besides. Requires dev_texts loaded on every rank (see main()).
    world = dist.get_world_size() if dist.is_initialized() else 1
    rk = dist.get_rank() if dist.is_initialized() else 0
    texts = dev_texts[:n][rk::world]
    # In TRUE-SUPERPOSITION mode the interesting axis is the LOAD, not the code
    # width: every message already spans the whole packet, so what varies is how
    # many other messages are summed on top of it. Sweep M and inject the
    # matched crosstalk (std sqrt(M-1)) rather than reporting the noiseless
    # M=1 case, which would say nothing about superposition at all.
    if getattr(args, "identity_code", False):
        # Octaves, out to where the capacity law says the channel MUST fail:
        # at P=10240 the AWGN budget is ~477 bits/msg at M=16 but only ~116 at
        # M=64, against ~295 bits of naive unigram entropy per agent message.
        # Sweeping only to M=16 measures the easy side of the cliff and reports
        # a number that says nothing about the scheme.
        # Sweep out past the target M>100. The usable cliff sits near
        # M ~= 1 + P/(2 ln2 bits); at P=196608 (K=48) that is ~150-200 for the
        # ~680-bit fineweb@96 messages, so the sweep must reach 256 to bracket
        # it. Points beyond args.packet_dim (where a full-width code leaves <1
        # slot) are still meaningful here: identity-mode load M is just the
        # crosstalk count, not a code-width division.
        sweep = [("M=%d" % m, args.ladder[0], math.sqrt(max(m - 1, 0)))
                 for m in (1, 8, 32, 64, 128, 200, 256)]
        # PRIOR: decode pure noise, zero signal. Without this row every F1
        # above is unreadable -- on the agent corpus a fixed string scores
        # 0.140 and retrieving the nearest memorised training message scores
        # 0.575, so an "M=32: 0.567" means nothing until you know what zero
        # information already buys. Must be measured per checkpoint, not
        # assumed: it rises as the model memorises the corpus.
        sweep.append(("PRIOR", args.ladder[0], -1.0))
    else:
        sweep = [("d=%d" % d, d, 0.0) for d in args.ladder]
    for label, d, nstd in sweep:
        scores = []
        for t in texts:
            lat = encode_batch(model, tok, [t], args.num_latents, args.max_len,
                               device, last_layer)
            code = bottleneck.encode(lat.float(), d)
            if nstd < 0:                       # PRIOR row: destroy the signal
                code = torch.randn_like(code)
            elif nstd > 0:
                code = code + torch.randn_like(code) * nstd
            z = bottleneck.decode(code, out_dtype=torch.bfloat16)[0]
            emb = embed(pid).clone()
            for i, p in enumerate(pos):
                emb[0, p] = z[i]
            past, cur, gen = None, emb, []
            for _ in range(max_new):
                o = model(inputs_embeds=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nx = o.logits[0, -1].argmax().item()
                if nx == eos:
                    break
                gen.append(nx)
                cur = embed(torch.tensor([[nx]], device=device))
            # Compare against the TRUNCATED source (what the encoder saw),
            # not the full text, or the score charges the channel for tokens
            # it was never given.
            ref = tok.decode(tok.encode(t, add_special_tokens=False)[:args.max_len])
            scores.append(_content_f1(ref, tok.decode(gen, skip_special_tokens=True)))
        out[label] = scores          # LOCAL shard scores; gathered below
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, out)
        out = {label: [s for part in gathered for s in part[label]]
               for label in out}
    out = {label: sum(v) / max(len(v), 1) for label, v in out.items()}
    model.train()
    bottleneck.train()
    return out


def maybe_shorten(text: str, frac: float, lo: int, hi: int) -> str:
    if frac > 0 and random.random() < frac:
        return text[:random.randint(lo, hi)]
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=str,
                    default="data/autoencoder_pretrain_large/final")
    ap.add_argument("--train-data", type=str, default="data/fineweb_ae_large/train.jsonl")
    ap.add_argument("--dev-data", type=str, default="data/fineweb_ae_large/dev.jsonl")
    ap.add_argument("--out-dir", type=str, default="data/packed_matryoshka")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-latents", type=int, default=4)
    ap.add_argument("--ladder", type=int, nargs="+", default=list(DEFAULT_LADDER))
    ap.add_argument("--bottleneck-width", type=int, default=4096,
                    help="Hidden width of the down/up MLPs.")
    ap.add_argument("--identity-code", action="store_true",
                    help="TRUE SUPERPOSITION mode. No dimensional bottleneck at "
                         "all: the code IS the message's full standardised latent "
                         "(d = K*H), so every message occupies the WHOLE packet and "
                         "receivers separate them only by key, through the "
                         "interference. This is the regime disjoint-subspace packing "
                         "cannot reach -- RotorPacker at d=P has capacity P//P=1. "
                         "Crosstalk is then rho = M, i.e. std sqrt(M-1), which is "
                         "what --max-rho must be set from.")
    ap.add_argument("--slot-dim", type=int, default=0,
                    help="Factorised bottleneck: project each of the K slots "
                         "H->slot_dim with shared weights before the joint MLP. "
                         "0 = flat MLP over K*H. Essential at large K (K=16 "
                         "flat is 357M params vs 91M factorised).")
    ap.add_argument("--packet-dim", type=int, default=0,
                    help="P; defaults to num_latents*hidden (the current 40 KB packet).")
    ap.add_argument("--max-rho", type=float, default=1.0,
                    help="Largest FramePacker load factor to train crosstalk for.")
    ap.add_argument("--short-frac", type=float, default=0.5)
    ap.add_argument("--short-min", type=int, default=150)
    ap.add_argument("--short-max", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-5, help="LR for the LM.")
    ap.add_argument("--bottleneck-lr", type=float, default=3e-4,
                    help="LR for the freshly-initialised bottleneck head.")
    ap.add_argument("--mse-warmup-steps", type=int, default=1500,
                    help="Steps of latent-MSE-only training (no decode pass) "
                         "before the LM sees a reconstructed latent.")
    ap.add_argument("--aux-mse-weight", type=float, default=0.0,
                    help="Weight on the latent-reconstruction term. DEFAULT 0: "
                         "at narrow d it actively fights the objective, because "
                         "it demands up(code) reproduce a latent that is provably "
                         "not compressible to d dims, while the decoder only needs "
                         "SOME decodable representation. Non-zero is for the "
                         "wide-d regime or as a short anti-divergence crutch.")
    ap.add_argument("--noise-prob", type=float, default=0.25,
                    help="Fraction of steps carrying FramePacker crosstalk; "
                         "the rotor path has none, so keep this low.")
    ap.add_argument("--lora-r", type=int, default=0,
                    help="LoRA rank on the LM (0 = full fine-tune). See apply_lora.")
    ap.add_argument("--freeze-lm", action="store_true",
                    help="Never unfreeze the LM: train ONLY the bottleneck, "
                         "against a decoder whose exact-latent reading ability "
                         "(tok_acc 0.932) is preserved rather than drifting. "
                         "Also ~3.5x faster -- no 4B-param gradient all-reduce.")
    ap.add_argument("--freeze-lm-steps", type=int, default=300,
                    help="Warm up the random bottleneck alone before letting "
                         "its gradients disturb the pretrained LM.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--reduce-bucket-mb", type=int, default=512,
                    help="Gradient all-reduce bucket size (see flat_grads).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rank, local_rank, world_size, ddp_device = setup_distributed()
    is_main = rank == 0
    device = ddp_device if ddp_device is not None else args.device

    def log(msg):
        if is_main:
            print(msg, flush=True)

    for path in (args.train_data, args.dev_data, args.init_from):
        if not os.path.exists(path):
            print(f"Path {path} not found.", file=sys.stderr)
            sys.exit(1)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed)  # identical bottleneck init on every rank
    train_texts = load_jsonl(args.train_data)[rank::max(world_size, 1)]
    random.shuffle(train_texts)
    # Dev on EVERY rank (identical order) so eval can shard across ranks and
    # stay in NCCL lockstep. load_jsonl is deterministic, so texts[:n][rk::W]
    # partitions the same n texts the same way everywhere.
    dev_texts = load_jsonl(args.dev_data)
    log(f"{len(train_texts)} train texts (rank shard) / {len(dev_texts)} dev")

    tok = AutoTokenizer.from_pretrained(args.init_from)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    raw_model = AutoModelForCausalLM.from_pretrained(
        args.init_from, torch_dtype=torch.bfloat16).to(device)
    # ORDER MATTERS: wrap in LoRA BEFORE enabling gradient checkpointing.
    # peft inspects the base model on wrap and, if checkpointing is already
    # on, installs a make_inputs_require_grad hook -- which turns the
    # embedding output into a leaf requiring grad and makes the decode path's
    # in-place latent write illegal.
    if args.lora_r > 0:
        raw_model = apply_lora(raw_model, args.lora_r)
        if is_main:
            tr = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in raw_model.parameters())
            print(f"LoRA r={args.lora_r}: {tr/1e6:.1f}M trainable / {tot/1e6:.0f}M "
                  f"({100*tr/tot:.2f}%)", flush=True)
    raw_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    raw_model.train()
    H = (raw_model.get_base_model() if args.lora_r > 0 else raw_model).config.hidden_size
    _cfg_model = raw_model.get_base_model() if args.lora_r > 0 else raw_model
    last_layer = _cfg_model.config.num_hidden_layers - 1
    embed_layer = _cfg_model.get_input_embeddings()

    if args.packet_dim <= 0:
        args.packet_dim = args.num_latents * H
    args.ladder = sorted({d for d in args.ladder if d <= args.packet_dim},
                         reverse=True)

    if args.identity_code:
        args.ladder = [args.num_latents * H]
        args.packet_dim = args.num_latents * H
    bottleneck = PackedBottleneck(
        args.num_latents, H, code_dim=max(args.ladder),
        width=args.bottleneck_width, slot_dim=args.slot_dim,
        identity=args.identity_code, dtype=torch.float32).to(device)
    bottleneck.train()

    prompt_ids = decode_prompt_ids(tok, args.num_latents)
    li_positions = latent_token_positions(tok, prompt_ids, args.num_latents)

    # Fit the standardisation buffers from real latents BEFORE any training,
    # identically on every rank (same texts, same order) so the buffers stay
    # in sync without needing to be all-reduced.
    with torch.no_grad():
        # Read the dev file directly rather than reusing `dev_texts` (which
        # is rank-0-only) or `train_texts` (which is rank-SHARDED and
        # independently shuffled) -- either would give each rank different
        # buffers, and these are never all-reduced.
        stat_texts = load_jsonl(args.dev_data)[:64]
        chunks = [encode_batch(raw_model, tok, stat_texts[i:i + 8],
                               args.num_latents, args.max_len, device, last_layer)
                  for i in range(0, len(stat_texts), 8)]
        bottleneck.fit_stats(torch.cat(chunks).float())
    log(f"standardisation fitted: mu|.|={bottleneck.mu.abs().mean():.3f} "
        f"sigma={bottleneck.sigma.mean():.3f}")

    optimizer = torch.optim.AdamW([
        {"params": list(raw_model.parameters()), "lr": args.lr},
        {"params": list(bottleneck.parameters()), "lr": args.bottleneck_lr},
    ])

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    n_params = sum(p.numel() for p in bottleneck.parameters())
    log(f"packet P={args.packet_dim} floats ({args.packet_dim * 4 / 1024:.0f} KB fp32)")
    log(f"ladder (code width -> messages/packet): " + ", ".join(
        f"{d}->{args.packet_dim // d}" for d in args.ladder))
    log(f"bottleneck {n_params/1e6:.1f}M params, width={args.bottleneck_width}")

    n_texts = len(train_texts)
    idx = 0
    acc_stats: dict[int, list] = {}
    mse_stats: list[float] = []
    lm_requires_grad = None  # tracks the requires_grad state we last applied
    total_loss, total_n = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.steps):
        d = sample_width(args.ladder, step, args.steps)
        rho = sample_rho(args.max_rho, args.noise_prob)
        batch = []
        for _ in range(args.batch_size):
            batch.append(maybe_shorten(train_texts[idx % n_texts], args.short_frac,
                                       args.short_min, args.short_max))
            idx += 1
            if idx % n_texts == 0:
                random.shuffle(train_texts)

        # Three phases. `mse_only` trains the bottleneck against a direct
        # latent-reconstruction target with the LM untouched and NO decode
        # pass at all -- it is ~4x faster per step and, more importantly, it
        # means that by the time the LM first sees a reconstructed latent
        # that latent is already close to a real one, so the LM never gets
        # the chance to learn that the latent slots are noise worth ignoring
        # (see PackedBottleneck.recon_loss).
        mse_only = step < args.mse_warmup_steps
        lm_frozen = args.freeze_lm or step < args.mse_warmup_steps + args.freeze_lm_steps
        # Freeze via requires_grad, NOT merely by wrapping the encoder pass in
        # no_grad. Wrapping only the encoder was a real bug that destroyed the
        # decoder: the DECODE pass still populated LM .grad, and because the
        # frozen branch excluded the LM from both the all-reduce and
        # clip_grad_norm_, optimizer.step() then applied those gradients
        # unclipped AND unsynchronised -- so every rank walked a different way
        # under the largest CE of the whole run (the moment the bottleneck is
        # still bad). Measured cost: exact-latent free-running content-F1 fell
        # 0.955 -> 0.020 between step 1000 and 2000, i.e. the LM stopped being
        # able to read latents at all, which flattened every downstream curve.
        if lm_requires_grad != (not lm_frozen):
            lm_requires_grad = not lm_frozen
            set_lm_trainable(raw_model, lm_requires_grad, args.lora_r > 0)
        with torch.no_grad() if lm_frozen else torch.enable_grad():
            latents = encode_batch(raw_model, tok, batch, args.num_latents,
                                   args.max_len, device, last_layer)
        code = bottleneck.encode(latents.float(), d)
        if rho > 0:
            # Codes are unit-RMS, so std=sqrt(rho) IS the frame crosstalk
            # (packing.crosstalk_std); noise only on the live prefix.
            noise = torch.zeros_like(code)
            noise[:, :d] = torch.randn(code.shape[0], d, device=code.device) * (rho ** 0.5)
            code = code + noise

        mse = bottleneck.recon_loss(latents.float(), code)
        if mse_only:
            loss, acc = mse, 0.0
            (loss / args.grad_accum).backward()
        else:
            recon = bottleneck.decode(code, out_dtype=latents.dtype)
            ce, acc = decode_batch_loss(raw_model, tok, batch, recon, prompt_ids,
                                        li_positions, args.max_len, device,
                                        embed_layer)
            loss = ce + args.aux_mse_weight * mse
            (loss / args.grad_accum).backward()
            acc_stats.setdefault(d, []).append(acc)

        total_loss += float(loss)
        total_n += 1
        mse_stats.append(float(mse))

        if (step + 1) % args.grad_accum == 0:
            params = [p for p in bottleneck.parameters() if p.grad is not None]
            if not lm_frozen:
                params += [p for p in raw_model.parameters() if p.grad is not None]
            if world_size > 1:
                # COALESCED all-reduce. One dist.all_reduce per PARAMETER is
                # ~400 separate NCCL calls for a 4B model, and with
                # NCCL_P2P_DISABLE (required on this cluster, see
                # [[superpose-packets]] notes) each call's latency dominates
                # its payload: measured 0.49 -> 3.19 s/step the moment the LM
                # unfroze. Flattening into a few large buffers makes it one
                # transfer per bucket instead, which is bandwidth-bound as it
                # should be.
                flat_grads(params, bucket_bytes=args.reduce_bucket_mb << 20)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            # set_to_none, not zeros: AdamW still applies decoupled weight
            # decay and residual momentum to a param whose .grad is a zero
            # tensor, so zeroed grads are not the same as absent ones.
            optimizer.zero_grad(set_to_none=True)

        if is_main and (step + 1) % args.log_every == 0:
            per_d = "  ".join(
                f"d{k}:{sum(v)/len(v):.3f}" for k, v in sorted(acc_stats.items(), reverse=True))
            phase = ("mse-warmup" if mse_only else "lm-frozen" if lm_frozen else "e2e")
            print(f"step {step+1:5d}/{args.steps} loss={total_loss/max(total_n,1):.4f} "
                  f"mse={sum(mse_stats)/max(len(mse_stats),1):.4f} [{per_d}] ({phase}) "
                  f"({(time.time()-t0)/args.log_every:.2f}s/step)", flush=True)
            acc_stats, mse_stats, total_loss, total_n = {}, [], 0.0, 0
            t0 = time.time()

        if (step + 1) % args.eval_every == 0:
            # eval_dev is teacher-forced, has no collectives, and is cheap:
            # keep it on main. eval_free_running now runs on ALL ranks (it
            # shards the dev texts and all_gather_objects the scores), so it
            # MUST be called outside the is_main guard or the collective hangs.
            # eval_dev (teacher-forced) runs ONLY on rank 0, so it both spikes
            # and imbalances rank-0 memory -- at K=48/P=196608 that pushed rank 0
            # to ~44GB and risked OOM. It is not the metric that matters
            # (free-running is), so skip it unless explicitly requested.
            if is_main and getattr(args, "eval_dev", False):
                res = eval_dev(raw_model, bottleneck, tok, dev_texts, args,
                               prompt_ids, li_positions, device, last_layer,
                               embed_layer)
                print("  [dev tok_acc] " + "  ".join(
                    f"M={args.packet_dim//d}(d={d}): {a:.3f}" for d, a in res.items()),
                    flush=True)
            # n=24 sharded 8 ways = 3 generations/rank. n=6 was far too few to
            # separate F1 0.79 from 0.70; a previous 20-sample estimate here
            # read 0.766 and fell to 0.66 at 36. In-training numbers get quoted.
            fr = eval_free_running(raw_model, bottleneck, tok, dev_texts, args,
                                   device, last_layer, n=24)
            if is_main:
                print("  [FREE-RUNNING F1] " + "  ".join(
                    f"{k}: {a:.3f}" for k, a in fr.items()), flush=True)
            t0 = time.time()

        if is_main and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, f"checkpoint-{step+1}")
            save_checkpoint(raw_model, bottleneck, tok, args, ckpt)
            print(f"  saved {ckpt}", flush=True)
            t0 = time.time()

    if is_main:
        save_checkpoint(raw_model, bottleneck, tok, args,
                        os.path.join(args.out_dir, "final"))
        log(f"Saved to {os.path.join(args.out_dir, 'final')}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
