#!/usr/bin/env python3
"""Build a BPE grammar codebook from discussion transcripts.

Harvests discussion messages from pre-existing full-benchmark report JSONs,
runs a word-level BPE (iterative replacement of most-frequent bigrams) to
discover recurrent multi-word patterns, and outputs a compact codebook.

Output: data/grammar_codebook.json
  {"<phrase>": "<compact symbol>", ...}

Excludes the 16-task eval-sweep set so training and evaluation tasks are
disjoint (same split as harvest_autoencoder_data.py).

Usage:
  python training/data/build_grammar_codebook.py [-n 256] [-m 5]
"""

import argparse
import json
import re
import sys
from collections import Counter

DEFAULT_FULL_RUNS = [
    "reports/hiddenbench_20260712_060139.json",
    "reports/hiddenbench_20260712_062305.json",
    "reports/results-opus-4.5.json",
]
DEFAULT_EVAL_REPORT = "reports/sweep_full_identity.json"
DEFAULT_OUT = "data/grammar_codebook.json"
MIN_CHARS = 20
MAX_CHARS = 2000
SYMBOLS = 256  # Unicode PUA start will be \ue000 onwards, but for readability
               # we use \u0100 + n (Latin Extended-A, no conflict in agent text)


def harvest_messages(report_paths: list[str], eval_report: str | None = None) -> list[str]:
    """Extract individual message strings from report discussion histories."""
    eval_names: set[str] = set()
    if eval_report is not None:
        try:
            d = json.load(open(eval_report))
            eval_names = {r["task"]["name"] for r in d["results"]}
        except FileNotFoundError:
            pass

    messages: list[str] = []
    seen: set[int] = set()
    for path in report_paths:
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            print(f"skip missing {path}", file=sys.stderr)
            continue
        for r in d.get("results", []):
            name = r["task"]["name"]
            if name in eval_names:
                continue
            for m in r.get("discussion_history", []):
                content = m.get("content", "").strip()
                if MIN_CHARS <= len(content) <= MAX_CHARS:
                    h = hash(content)
                    if h not in seen:
                        seen.add(h)
                        messages.append(content)
    return messages


def tokenize(text: str) -> list[str]:
    """Word-level tokenization preserving punctuation as separate tokens."""
    return re.findall(r"[\w']+|[^\w\s]", text)


def untokenize(tokens: list[str]) -> str:
    """Join tokens back into text, putting spaces between word tokens."""
    parts: list[str] = []
    for tok in tokens:
        if re.match(r"[\w']+", tok):
            if parts and not parts[-1].endswith(" ") and not re.match(r"^[,.;:!?]", tok):
                parts.append(" " + tok)
            else:
                parts.append(tok)
        else:
            parts.append(tok)
    return "".join(parts).strip()


def learn_bpe(
    messages: list[str],
    num_merges: int = 256,
    min_count: int = 3,
    min_consecutive_words: int = 2,
    max_phrase_words: int = 12,
    min_chars: int = 12,
) -> dict[str, str]:
    """Learn a BPE phrase dictionary from the message corpus.

    Iteratively replaces the most frequent adjacent token bigram across all
    messages with a new symbol. The symbol is stored in the codebook only if
    the phrase has enough words and characters to be worth compressing.

    Returns:
        dict mapping phrase -> compact symbol (decoder direction).
    """
    tok_lists = [tokenize(m) for m in messages]

    merge_count = 0
    codebook: dict[str, str] = {}  # phrase -> symbol

    for iteration in range(num_merges * 2):  # allow extra passes for filtering
        bigram_counts: Counter[tuple[str, str]] = Counter()
        for tl in tok_lists:
            for i in range(len(tl) - 1):
                bigram_counts[(tl[i], tl[i + 1])] += 1

        if not bigram_counts:
            break

        (a, b), count = bigram_counts.most_common(1)[0]
        if count < min_count:
            break

        new_token = f"{a} {b}"
        # Replace across all token lists
        for tl in tok_lists:
            i = 0
            while i < len(tl) - 1:
                if tl[i] == a and tl[i + 1] == b:
                    tl[i:i + 2] = [new_token]
                i += 1

        # Only keep as a codebook entry if it's a useful phrase
        word_count = len(new_token.split())
        char_count = len(new_token)
        if word_count >= min_consecutive_words and char_count >= min_chars and word_count <= max_phrase_words:
            symbol = f"§{merge_count}"
            codebook[new_token] = symbol
            merge_count += 1
            if merge_count >= num_merges:
                break

    # Re-sort more frequently to assign symbols to shorter codes for most common phrases
    ranked = sorted(codebook.items(), key=lambda kv: len(kv[0].split()), reverse=True)
    final = {}
    for _, (phrase, sym) in enumerate(ranked):
        final[phrase] = sym
    return final


def tokenize_arbitrary_text(text: str) -> list[str]:
    """Tokenize preserving original text structure for roundtrip fidelity.
    Returns tokens AND their inter-token whitespace/formatting.

    This is a character-level approach: we just scan for occurrences of
    codebook phrases and replace them with the compact symbol. No
    intermediate tokenization needed for compress/decompress.
    """
    return tokenize(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_merges", type=int, default=256,
                    help="number of phrase merges (codebook size)")
    ap.add_argument("-m", "--min-count", type=int, default=3,
                    help="minimum bigram occurrence count")
    ap.add_argument("--min-consecutive-words", type=int, default=2,
                    help="min words a phrase must have to be kept")
    ap.add_argument("--min-chars", type=int, default=12,
                    help="min chars a phrase must have to be kept")
    ap.add_argument("--max-phrase-words", type=int, default=12,
                    help="max words in a kept phrase")
    ap.add_argument("--eval-report", default=DEFAULT_EVAL_REPORT,
                    help="eval-sweep report (tasks to exclude)")
    ap.add_argument("--report", action="append", dest="reports",
                    help="report JSONs to harvest (repeatable)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    reports = args.reports if args.reports else DEFAULT_FULL_RUNS

    print(f"Harvesting messages from {len(reports)} report(s)...")
    messages = harvest_messages(reports, args.eval_report)
    print(f"  -> {len(messages)} unique messages")

    print(f"Learning BPE ({args.num_merges} merges, min_count={args.min_count})...")
    codebook = learn_bpe(
        messages,
        num_merges=args.num_merges,
        min_count=args.min_count,
        min_consecutive_words=args.min_consecutive_words,
        min_chars=args.min_chars,
        max_phrase_words=args.max_phrase_words,
    )
    print(f"  -> {len(codebook)} codebook entries")

    # Compute per-entry stats
    total_saved = 0
    replacements_map: dict[str, int] = Counter()
    for phrase in codebook:
        for msg in messages:
            count_in_msg = msg.count(phrase)
            if count_in_msg:
                replacements_map[phrase] += count_in_msg

    for phrase, count in replacements_map.most_common(20):
        total_saved += count * (len(phrase) - len(codebook[phrase]))
        print(f"  [{count:4d}x] {phrase[:80]} -> {codebook[phrase]} ({len(phrase)} -> {len(codebook[phrase])} chars)")

    total_chars = sum(len(m) for m in messages)
    print(f"\nEstimated savings: {total_saved} chars / {total_chars} total ({100*total_saved/total_chars:.1f}%)")

    # Save
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(codebook)} entries to {args.out}")


if __name__ == "__main__":
    main()
