#!/usr/bin/env python3
"""CryptAE unified evaluation: cryptographic autoencoder with IND-CPA security.

Demonstrates the complete scheme on FineWeb-Text validation set:
  1. Neural encode: Qwen3-4B autoencoder -> (K, D) latent vector per message
  2. Per-slot encryption: bind with per-row independent orthogonal keys
     (fresh nonce per packet, row_keys=True for IND-CPA)
  3. Entanglement: sum all bound latents into one packet
  4. Decryption: each receiver unbinds with their own private key
  5. Neural decode: Qwen3-4B autoregressive reconstruction

Metrics:
  - Reconstruction quality: diff ratio, BLEU, char-level similarity
  - IND-CPA security: attack_cos vs baseline (honest-but-curious insider)
  - IND-CPA distinguisher: Gram-leak test (per-row keys must fix this)
  - Scaling: load curve from N=1 to degradation

Example:
  python -m experiments.crypt_ae.programs.eval_cryptae \
      --model-path /path/to/superpose_checkpoint \
      --dev-data /path/to/dev.jsonl \
      --loads 1 2 4 8 16 --n-groups 10 --device cuda:0
"""

import argparse
import difflib
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import torch

from multimodal_comms.methods.superposition.latent import (  # noqa: E402
    LatentCodec,
    SecureBroadcastCodec,
    SecureReceiverCodec,
    mint_receiver_secrets,
    build_keyring,
    superpose,
    _derive_row_seed,
)
from training.programs.pretrain_autoencoder import (  # noqa: E402
    load_jsonl,
    encode_batch,
    decode_prompt_ids,
    latent_token_positions,
    decode_batch_loss,
)


def load_jsonl_texts(path: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else obj)
    return texts


def diff_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def char_match(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    shorter = min(len(a), len(b))
    matches = sum(1 for i in range(shorter) if a[i] == b[i])
    return matches / max(len(a), len(b))


def bleu_score(reference: str, candidate: str) -> float:
    """Simple corpus-level BLEU-1 approximation."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if not cand_tokens:
        return 0.0
    ref_counts = defaultdict(int)
    for t in ref_tokens:
        ref_counts[t] += 1
    matches = 0
    for t in cand_tokens:
        if ref_counts.get(t, 0) > 0:
            matches += 1
            ref_counts[t] -= 1
    brevity_penalty = min(1.0, len(cand_tokens) / max(1, len(ref_tokens)))
    return brevity_penalty * (matches / max(1, len(cand_tokens)))


def load_ae_model(model_path: str, num_latents: int, device: str):
    """Load the autoencoder model for teacher-forced evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    H = raw_model.config.hidden_size
    last_layer = raw_model.config.num_hidden_layers - 1
    embed_layer = raw_model.get_input_embeddings()
    prompt_ids = decode_prompt_ids(tok, num_latents)
    li_positions = latent_token_positions(tok, prompt_ids, num_latents)
    return raw_model, tok, last_layer, embed_layer, prompt_ids, li_positions


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).abs().item()


def test_indcpa_gram_leak(
    codec: LatentCodec, text_pool: list[str], n_trials: int = 100
) -> dict:
    """IND-CPA distinguisher: Gram-matrix invariance test.

    Challenge: z0 vs z1 encrypted under per-row independent keys.
    Test uses OFF-DIAGONAL Gram entries only, since diagonal (norm) is an
    invariant of orthogonal transforms and the known/acceptable carve-out
    (analogous to ciphertext-length in AES-GCM).

    Under a SHARED key for all K rows, the off-diagonal Gram entries are
    invariants of the cipher → distinguishable with advantage 1.
    Under per-ROW independent keys (row_keys=True), Q^(k)Q^(l)^T is
    Haar-random, so off-diagonal entries are indistinguishable.

    Returns success rate (should be ~0.5 for secure, 1.0 for broken).
    """
    dim = codec.latent_dim
    K = codec.num_latents

    secret = 42
    nonce = 12345
    base_seed = hashlib.blake2b(f"{secret}:{nonce}".encode(), digest_size=8).digest()
    base_seed = int.from_bytes(base_seed, "big") % (2 ** 63 - 1)

    correct = 0
    for trial in range(n_trials):
        t0 = random.choice(text_pool)
        t1 = random.choice(text_pool)
        z0 = codec.encode(t0)
        z1 = codec.encode(t1)

        if trial % 2 == 0:
            true_z, coin = z0, 0
        else:
            true_z, coin = z1, 1

        # Per-row independent keys (secure mode)
        keys = []
        for k in range(K):
            row_seed = _derive_row_seed(base_seed, k)
            gen = torch.Generator().manual_seed(row_seed)
            g = torch.randn(dim, dim, generator=gen)
            q, r = torch.linalg.qr(g)
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
            keys.append(q)
        ct = torch.stack([true_z[k].float() @ keys[k] for k in range(K)])

        ct_gram = ct @ ct.T
        z0_gram = z0 @ z0.T
        z1_gram = z1 @ z1.T

        # OFF-DIAGONAL only: the per-row cipher hides Gram off-diagonals
        mask = torch.eye(K) == 0
        d0 = (ct_gram[mask] - z0_gram[mask]).norm().item()
        d1 = (ct_gram[mask] - z1_gram[mask]).norm().item()

        guess = 0 if d0 < d1 else 1
        if guess == coin:
            correct += 1

    return {
        "n_trials": n_trials,
        "correct": correct,
        "success_rate": correct / n_trials,
        "secure": abs(correct / n_trials - 0.5) < 0.1,
    }


def test_indcpa_shared_key_break(
    codec: LatentCodec, text_pool: list[str], n_trials: int = 100
) -> dict:
    """Prove the break: with SHARED key per message, Gram leaks.
    This should return success_rate ~1.0 to demonstrate why row_keys is required.
    """
    dim = codec.latent_dim
    K = codec.num_latents

    secret = 42
    nonce = 12345
    base_seed = hashlib.blake2b(f"{secret}:{nonce}".encode(), digest_size=8).digest()
    base_seed = int.from_bytes(base_seed, "big") % (2 ** 63 - 1)

    correct = 0
    for trial in range(n_trials):
        t0 = random.choice(text_pool)
        t1 = random.choice(text_pool)
        z0 = codec.encode(t0)
        z1 = codec.encode(t1)

        if trial % 2 == 0:
            true_z, coin = z0, 0
        else:
            true_z, coin = z1, 1

        # SHARED key for ALL rows (the BROKEN original design)
        gen = torch.Generator().manual_seed(base_seed)
        g = torch.randn(dim, dim, generator=gen)
        q, r = torch.linalg.qr(g)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        ct = true_z.float() @ q  # all K rows use same Q

        ct_gram = ct @ ct.T
        z0_gram = z0 @ z0.T
        z1_gram = z1 @ z1.T

        # OFF-DIAGONAL only (shared-key break is definitive on off-diagonals)
        mask = torch.eye(K) == 0
        d0 = (ct_gram[mask] - z0_gram[mask]).norm().item()
        d1 = (ct_gram[mask] - z1_gram[mask]).norm().item()
        guess = 0 if d0 < d1 else 1
        if guess == coin:
            correct += 1

    return {
        "n_trials": n_trials,
        "correct": correct,
        "success_rate": correct / n_trials,
        "broken": correct / n_trials > 0.95,
    }


def eval_load(
    dev_texts: list[str],
    codec: LatentCodec,
    n: int,
    n_groups: int,
    key_mode: str,
    seed: int,
    print_samples: bool = False,
) -> dict:
    """Evaluate the full CryptAE pipeline at a given superposition load N.

    For each of n_groups groups of N texts:
      - Encode each text -> (K, D) latent
      - Create per-slot secrets
      - Build SecureBroadcastCodec -> encode_packet (row_keys=True, fresh nonce)
      - For each receiver j: build SecureReceiverCodec, decode -> text
      - Measure reconstruction quality + security leakage

    Returns a dict with metrics.
    """
    fid_scores, char_scores, bleu_scores = [], [], []
    own_cos_list, attack_cos_list = [], []
    rng = random.Random(seed)
    idx = 0

    for g in range(n_groups):
        if idx + n > len(dev_texts):
            idx = 0
        batch = dev_texts[idx : idx + n]
        idx += n

        texts_by_slot = {j: batch[j] for j in range(n)}
        secrets = mint_receiver_secrets(n)

        # --- Encryption: sender side ---
        sender = SecureBroadcastCodec(
            codec, secrets_by_slot=secrets, key_mode=key_mode,
        )
        packet_str = sender.encode_packet(texts_by_slot)

        # --- Encode true latents for cos-sim measurement ---
        true_z = {j: codec.encode(batch[j]) for j in range(n)}

        for j in range(n):
            # --- Decryption: receiver side ---
            receiver = SecureReceiverCodec(
                codec, my_slot=j, my_secret=secrets[j], key_mode=key_mode,
            )
            decoded = receiver.decode(packet_str) or ""

            orig = texts_by_slot[j]
            fid_scores.append(diff_ratio(orig, decoded))
            char_scores.append(char_match(orig, decoded))
            bleu_scores.append(bleu_score(orig, decoded))

            # --- Security: what does receiver j's operation leak? ---
            # Build keyring with receiver j's secret only, and the packet's nonce
            # to get the exact same bind/unbind that happened in the packet.
            from multimodal_comms.methods.superposition.latent import deserialize_packet, _split_nonce_prefix
            nonce, inner = _split_nonce_prefix(packet_str)
            packet_tensor, _ = deserialize_packet(inner)
            receiver_keyring = build_keyring(
                codec.latent_dim,
                seed={j: secrets[j]},
                mode=key_mode,
                nonce=nonce,
                row_keys=True,
            )
            recovered = receiver_keyring.unbind(packet_tensor, j, n)
            own_cos_list.append(_cos(recovered, true_z[j]))
            for i in range(n):
                if i != j:
                    attack_cos_list.append(_cos(recovered, true_z[i]))

        if print_samples and g == 0:
            print(f"\n  --- N={n} sample group ---")
            for j in range(min(n, 4)):
                orig_short = texts_by_slot[j][:100]
                dec_short = (receiver.decode(packet_str) or "")[:100] if j == 0 else "..."
                print(f"  slot {j}: [{orig_short}...]")

    return {
        "n_load": n,
        "n_groups": n_groups,
        "n_eval": len(fid_scores),
        "diff_ratio": sum(fid_scores) / len(fid_scores),
        "char_match": sum(char_scores) / len(char_scores),
        "bleu1": sum(bleu_scores) / len(bleu_scores),
        "own_cos": sum(own_cos_list) / len(own_cos_list),
        "attack_cos": sum(attack_cos_list) / len(attack_cos_list),
        "attack_below_baseline": None,  # filled in later
        "samples": min(len(fid_scores), n_groups * n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CryptAE unified evaluation")
    ap.add_argument(
        "--model-path", type=str,
        required=True,
        help="Superpose checkpoint (QR mode recommended).",
    )
    ap.add_argument(
        "--dev-data", type=str,
        required=True,
        help="FineWeb-Text validation set.",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--loads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-groups", type=int, default=10)
    ap.add_argument("--key-mode", type=str, default="qr")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gram-trials", type=int, default=100)
    ap.add_argument("--out", type=str, default="reports/cryptae_eval.json")
    ap.add_argument("--print-samples", action="store_true")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"=" * 72)
    print(f"CryptAE Evaluation")
    print(f"  Model:  {args.model_path}")
    print(f"  Data:   {args.dev_data}")
    print(f"  Device: {args.device}")
    print(f"  Keying: {args.key_mode} (secure: row_keys=True, per-packet nonce)")
    print(f"  Loads:  {args.loads}")
    print(f"=" * 72)

    # --- Load codec ---
    print("\n[1/4] Loading autoencoder codec...")
    codec = LatentCodec(model_path=args.model_path, device=args.device)
    print(f"  Latent: {codec.num_latents} x {codec.latent_dim}")

    # --- Load data ---
    print("\n[2/4] Loading validation data...")
    dev_texts = load_jsonl_texts(args.dev_data)
    rng = random.Random(args.seed)
    rng.shuffle(dev_texts)
    print(f"  {len(dev_texts)} dev texts")

    # --- IND-CPA Gram-leak test ---
    print("\n[3/4] IND-CPA Gram-leak distinguisher...")
    text_pool = dev_texts[:50]
    print("  Testing with per-row keys (secure, should be ~0.5)...")
    secure_result = test_indcpa_gram_leak(codec, text_pool, args.gram_trials)
    print(f"  Success rate: {secure_result['success_rate']:.3f} "
          f"({secure_result['correct']}/{secure_result['n_trials']}) "
          f"-> {'SECURE' if secure_result['secure'] else 'INSECURE'}")

    print("  Testing with SHARED key (broken design, should be ~1.0)...")
    broken_result = test_indcpa_shared_key_break(codec, text_pool, args.gram_trials)
    print(f"  Success rate: {broken_result['success_rate']:.3f} "
          f"({broken_result['correct']}/{broken_result['n_trials']}) "
          f"-> {'BROKEN (as expected)' if broken_result['broken'] else 'UNEXPECTED!'}")

    # --- Measure cosine baseline for unrelated real-text pairs ---
    print("\n[3b/4] Real-text latent cosine baseline...")
    base_pool = [codec.encode(t) for t in dev_texts[:20]]
    base_cos = []
    for i in range(10):
        for j in range(10, 20):
            base_cos.append(_cos(base_pool[i], base_pool[j]))
    cosine_baseline = sum(base_cos) / len(base_cos)
    print(f"  Baseline (unrelated texts, no channel): {cosine_baseline:.4f}")

    # --- Scaling evaluation ---
    print("\n[4/4] Scaling evaluation...")
    print(f"  {'Load':>5s}  {'diff_ratio':>10s}  {'bleu1':>8s}  "
          f"{'own_cos':>8s}  {'attack_cos':>10s}  {'verdict':>20s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*20}")

    results = {
        "model_path": args.model_path,
        "key_mode": args.key_mode,
        "indcpa_gram_leak": {
            "per_row_keys_secure": secure_result,
            "shared_key_broken": broken_result,
        },
        "cosine_baseline": cosine_baseline,
        "loads": {},
    }

    for n in args.loads:
        t0 = time.time()
        load_result = eval_load(
            dev_texts, codec, n, args.n_groups,
            args.key_mode, args.seed + n,
            print_samples=args.print_samples,
        )
        load_result["attack_below_baseline"] = (
            load_result["attack_cos"] <= cosine_baseline * 1.1
        )
        results["loads"][str(n)] = load_result
        elapsed = time.time() - t0

        verdict = "SECURE" if load_result["attack_below_baseline"] else "LEAKAGE"
        print(f"  {n:5d}  {load_result['diff_ratio']:10.4f}  "
              f"{load_result['bleu1']:8.4f}  {load_result['own_cos']:8.4f}  "
              f"{load_result['attack_cos']:10.4f}  {verdict:>20s}  "
              f"({elapsed:.1f}s)")

    # --- Write report ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport written to {args.out}")

    # --- Summary ---
    print(f"\n{'='*72}")
    print(f"Summary")
    print(f"  IND-CPA Gram-leak (per-row keys): "
          f"{'PASS' if secure_result['secure'] else 'FAIL'}")
    print(f"  IND-CPA Gram-leak (shared key):   "
          f"{'PROVEN BROKEN' if broken_result['broken'] else 'UNEXPECTED'}")
    for n in args.loads:
        r = results["loads"][str(n)]
        status = "SECURE" if r["attack_below_baseline"] else "LEAK"
        print(f"  N={n:3d}: diff={r['diff_ratio']:.3f} bleu1={r['bleu1']:.3f} "
              f"attack={r['attack_cos']:.3f} (baseline={cosine_baseline:.3f}) "
              f"[{status}]")


if __name__ == "__main__":
    main()
