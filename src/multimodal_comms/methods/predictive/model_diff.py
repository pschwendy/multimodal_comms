"""
Shared codec for the predictive-diff family (PredictiveDiffCompressor,
RateControlledDiffCompressor).

Both sender and receiver hold the same frozen small LM, p_theta (default:
Qwen2.5-0.5B-Instruct - already used as the policy backbone for the GRPO
rewriter compressors elsewhere in this file, but loaded here purely for its
next-token predictions, no training). Given a shared textual context and the
true target message, we do a single teacher-forced forward pass and check,
at every position, whether p_theta's greedy next-token prediction already
matches the true token. Runs of correct guesses cost nothing to transmit;
mismatches become explicit (position, true-token-id) corrections - exactly
the run-length/correction-token bookkeeping of speculative decoding, reused
here as a lossless (or, with omissions, lossy) diff code instead of a
sampling accelerant.

Reconstruction (`replay`) is inherently sequential: at each step the
receiver feeds its own greedy prediction forward UNLESS a correction is
supplied for that position, in which case the given token is spliced in
instead and generation continues from there. This holds regardless of
whether earlier corrections were kept or dropped, which is what lets
RateControlledDiffCompressor omit individual corrections without changing
the wire format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_MODEL_CACHE: dict[str, tuple] = {}


@dataclass
class DiffOp:
    kind: str  # "R" (run of `n` accepted/predicted tokens) or "C" (correction)
    n: int = 0          # run length, only for kind == "R"
    token_id: int = 0   # true token id, only for kind == "C"


def _load(model_path: str, device: str):
    key = f"{model_path}@{device}"
    if key not in _MODEL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[key] = (model, tokenizer)
    return _MODEL_CACHE[key]


class PDiffCodec:
    """Encode/replay against a shared frozen LM. Stateless across calls."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
        max_context_tokens: int = 2048,
    ):
        self.model_path = model_path
        self.device = device
        self.max_context_tokens = max_context_tokens
        self._model = None
        self._tokenizer = None

    def _get(self):
        if self._model is None:
            if self.device is None:
                import torch
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else (
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
            self._model, self._tokenizer = _load(self.model_path, self.device)
        return self._model, self._tokenizer

    def _context_ids(self, context: str) -> list[int]:
        _, tokenizer = self._get()
        if context:
            ids = tokenizer.encode(context, add_special_tokens=False)
            if len(ids) > self.max_context_tokens:
                ids = ids[-self.max_context_tokens:]
        else:
            ids = []
        if not ids:
            # Need at least one token in the sequence to produce a first
            # prediction; eos/bos carries no content, purely a start marker.
            start = tokenizer.bos_token_id
            if start is None:
                start = tokenizer.eos_token_id
            ids = [start]
        return ids

    def encode_diff(self, context: str, target: str) -> list[DiffOp]:
        """Teacher-forced encoding; returns run/correction ops.

        Deliberately mirrors `replay()`'s step-by-step, KV-cached forward
        mechanics exactly (rather than one batched forward over the whole
        sequence) even though both are mathematically supposed to produce
        the same logits. In bf16 they don't always agree in practice: a
        single big forward pass and an incremental one accumulate
        floating-point error differently, and near-tied top-2 logits can
        flip which token argmax picks. That flip silently broke
        losslessness (an audit against a live run found ~20% of messages
        reconstructing with a wrong token) until encode was changed to use
        the identical incremental computation as replay, which makes the
        argmax at every position bit-for-bit reproducible between the two.
        """
        import torch

        model, tokenizer = self._get()
        context_ids = self._context_ids(context)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return []

        ops: list[DiffOp] = []
        run = 0
        with torch.no_grad():
            input_ids = torch.tensor([context_ids], device=self.device)
            out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[0, -1, :]

            for true_id in target_ids:
                pred_id = int(last_logits.argmax().item())
                if pred_id == true_id:
                    run += 1
                else:
                    if run:
                        ops.append(DiffOp(kind="R", n=run))
                        run = 0
                    ops.append(DiffOp(kind="C", token_id=true_id))

                inp = torch.tensor([[true_id]], device=self.device)
                out = model(inp, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                last_logits = out.logits[0, -1, :]

        if run:
            ops.append(DiffOp(kind="R", n=run))
        return ops

    def replay(
        self,
        context: str,
        ops: list[DiffOp],
        keep_mask: Optional[list[bool]] = None,
    ) -> str:
        """Reconstruct text from ops. keep_mask (aligned to the correction
        ops only, in order) selects which corrections are honored; a
        dropped correction lets the model's own greedy guess stand instead,
        so the wire format doesn't change, only fidelity does.
        """
        import torch

        model, tokenizer = self._get()
        context_ids = self._context_ids(context)

        corr_idx = 0
        generated: list[int] = []
        with torch.no_grad():
            input_ids = torch.tensor([context_ids], device=self.device)
            out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[0, -1, :]

            def step(next_id: int):
                nonlocal past_kv, last_logits
                inp = torch.tensor([[next_id]], device=self.device)
                out = model(inp, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                last_logits = out.logits[0, -1, :]

            for op in ops:
                if op.kind == "R":
                    for _ in range(op.n):
                        pred_id = int(last_logits.argmax().item())
                        generated.append(pred_id)
                        step(pred_id)
                else:
                    keep = keep_mask is None or (
                        corr_idx < len(keep_mask) and keep_mask[corr_idx]
                    )
                    corr_idx += 1
                    if keep:
                        generated.append(op.token_id)
                        step(op.token_id)
                    else:
                        pred_id = int(last_logits.argmax().item())
                        generated.append(pred_id)
                        step(pred_id)

        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def num_corrections(ops: list[DiffOp]) -> int:
        return sum(1 for op in ops if op.kind == "C")

    @staticmethod
    def apply_drops(ops: list[DiffOp], keep_mask: list[bool]) -> list[DiffOp]:
        """Fold corrections marked False in keep_mask into the surrounding
        run, so a dropped correction actually shrinks the serialized wire
        format instead of just being ignored at replay time. keep_mask is
        aligned to the correction ops only, in order.
        """
        merged: list[DiffOp] = []
        run = 0
        corr_idx = 0
        for op in ops:
            if op.kind == "R":
                run += op.n
                continue
            keep = corr_idx >= len(keep_mask) or keep_mask[corr_idx]
            corr_idx += 1
            if keep:
                if run:
                    merged.append(DiffOp(kind="R", n=run))
                    run = 0
                merged.append(op)
            else:
                run += 1
        if run:
            merged.append(DiffOp(kind="R", n=run))
        return merged

    @staticmethod
    def serialize(ops: list[DiffOp]) -> str:
        parts = []
        for op in ops:
            parts.append(f"R{op.n}" if op.kind == "R" else f"C{op.token_id}")
        return "|".join(parts)

    @staticmethod
    def deserialize(s: str) -> list[DiffOp]:
        ops = []
        if not s:
            return ops
        for part in s.split("|"):
            if not part:
                continue
            if part[0] == "R":
                ops.append(DiffOp(kind="R", n=int(part[1:])))
            elif part[0] == "C":
                ops.append(DiffOp(kind="C", token_id=int(part[1:])))
        return ops
