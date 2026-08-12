#!/usr/bin/env python3
"""Build the training dataset for the representational-match sentence selector.

Modeled on training/data/build_selector_dataset.py, but the per-sentence label is
NOT a semantic-match heuristic. Instead, for each sentence in each message we
measure how much the frozen proxy model's representation of the message
changes when that one sentence is masked out:

  label = 1 - cosine_similarity( rep(full_message),
                                 rep(message with the sentence removed) )

i.e. the representational distance the sentence is responsible for. Higher =
the sentence mattered more to how the message is represented in-context.

Each message's representation is computed *in context* (its most-recent
preceding messages in the same discussion supplied as `context`), matching how
the deployed compressor sees messages embedded in a running transcript.

The representation is the frozen proxy model's normalized last-token hidden
state at the target layer -- byte-for-byte the same computation as
training/services/repserver.py's `_last_token_hidden` + `F.normalize` (same model, same
layer, same `context\\n\\ntext` input construction). We run that computation
IN-PROCESS here rather than over HTTP: this is a one-shot offline job of ~15k
forward passes, and the shared HTTP rep server proved unreliable at that
volume (an attention-mask cache in the installed transformers leaks GPU memory
across thousands of distinct sequence lengths, and its FastAPI threadpool
occasionally corrupts CUDA state). Loading the identical frozen model here
sidesteps both without touching the shared server (which stays up on its own
GPU for the live compressors). Labels are identical to the server's.

The 16 eval-sweep tasks are strictly excluded.

Output: JSONL of {"sentence", "label", "task", "source"} (same shape as
data/selector_train.jsonl).
"""

import argparse
import json
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

EVAL_REPORT = "reports/sweep_full_identity.json"
FULL_RUN_REPORTS = [
    "reports/hiddenbench_20260712_060139.json",  # Qwen3-4B full run
    "reports/hiddenbench_20260712_062305.json",  # DeepSeek full run
    "reports/results-opus-4.5.json",             # Opus 4.5 reference run
]
OUT = "data/repmatch_train.jsonl"
MODEL_NAME = "Qwen/Qwen3-4B"
LAYER = 14
# Recent-context window. Kept bounded so the appended message text is always
# within the 4096-token limit (long context + text would otherwise truncate the
# trailing message, collapsing full/masked to an identical, spurious ~0 label),
# while still grounding the message in several rounds of preceding discussion.
MAX_CONTEXT_CHARS = 2000


def format_context(messages: list[dict], upto: int) -> str:
    ctx = "\n".join(
        f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages[:upto]
    )
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[-MAX_CONTEXT_CHARS:]
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device}, layer={LAYER}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def rep(text: str, context: str) -> torch.Tensor:
        # Identical to repserver._last_token_hidden + F.normalize.
        full = f"{context}\n\n{text}" if context else text
        enc = tokenizer(full, return_tensors="pt", truncation=True,
                        max_length=4096).to(device)
        out = model(**enc, output_hidden_states=True)
        vec = out.hidden_states[LAYER][0, -1, :].float()
        return F.normalize(vec, dim=0).cpu()

    eval_names = {
        r["task"]["name"] for r in json.load(open(EVAL_REPORT))["results"]
    }

    rows = []
    n_forward = 0
    for path in FULL_RUN_REPORTS:
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            print(f"skip missing {path}", file=sys.stderr)
            continue
        source = path.split("/")[-1]
        for r in d.get("results", []):
            name = r["task"]["name"]
            if name in eval_names:
                continue
            history = r.get("discussion_history", [])
            for i, m in enumerate(history):
                content = m.get("content", "")
                sentences = [s.strip() for s in SENTENCE_SPLIT.split(content) if s.strip()]
                if not sentences:
                    continue
                context = format_context(history, i)
                full_rep = rep(content, context)
                n_forward += 1
                for j, sent in enumerate(sentences):
                    masked = " ".join(s for k, s in enumerate(sentences) if k != j)
                    masked_rep = rep(masked or " ", context)
                    n_forward += 1
                    label = 1.0 - float(torch.dot(full_rep, masked_rep))
                    rows.append({
                        "sentence": sent,
                        "label": label,
                        "task": name,
                        "source": source,
                    })
                if n_forward % 1000 < (len(sentences) + 1):
                    torch.cuda.empty_cache()
                    print(f"  forwards: {n_forward}  rows: {len(rows)}", flush=True)

    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    labels = [r["label"] for r in rows]
    mean = sum(labels) / len(labels) if labels else 0.0
    print(f"sentences: {len(rows)}  mean label(rep-dist): {mean:.4f}  "
          f"min {min(labels):.4f}  max {max(labels):.4f}")
    print(f"unique training tasks: {len({r['task'] for r in rows})}")
    print(f"forwards: {n_forward}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
