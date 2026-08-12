#!/usr/bin/env python3
"""Quick, honest regrade pass over existing eval_*.json results.

The original grader used strict normalized exact/substring match against the
ground-truth answer string, which produces false negatives whenever the
model's (correct) conclusion is phrased differently (e.g. "clothing retail"
vs GT "clothes retailer") or wrapped in markdown - and can NEVER pass when
the ground-truth string uses CJK ideographs but the model answers in English
translation (a distinct dataset issue, not a paraphrase issue).

This does not replace a real LLM-judge pass, but a cheap keyword-overlap
heuristic (does the conclusion contain most of the GT's significant tokens,
case/punctuation-insensitive, or vice versa) recovers most of the paraphrase
false negatives cheaply, so the relative comparison across conditions is more
trustworthy. CJK ground-truth items are flagged separately since no cheap grader
can resolve them without translation/judgment.
"""
import glob
import argparse
import json
import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "say", "says", "what",
    "for", "to", "of", "in", "on", "at", "and", "her", "she", "he", "his",
    "i", "am", "have", "has", "that", "this", "it", "be", "as", "with",
}


def norm_tokens(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return {t for t in s.split() if t not in STOPWORDS and len(t) > 1}


def is_cjk(s: str) -> bool:
    import unicodedata

    return any("CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch, "") for ch in s)


def heuristic_correct(answer: str, conclusion: str) -> tuple[bool, str]:
    if is_cjk(answer):
        return (False, "cjk_unresolvable")
    a_tokens = norm_tokens(answer)
    if not a_tokens:
        return (False, "empty_gt")
    c_tokens = norm_tokens(conclusion)
    overlap = a_tokens & c_tokens
    frac = len(overlap) / len(a_tokens)
    # Require most of the GT's content words to appear in the conclusion.
    return (frac >= 0.7, f"overlap={frac:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Independently regrade iAgents reports")
    parser.add_argument("reports", nargs="*", default=["outputs/iagents/reports/eval_*.json"])
    parser.add_argument("--out", help="optional JSON summary destination")
    args = parser.parse_args()
    print(f"{'condition':22s} {'orig_acc':>9s} {'regr_acc':>9s} {'cjk_n':>6s} {'n':>4s}")
    paths = sorted({path for pattern in args.reports for path in glob.glob(pattern)})
    summaries = []
    for path in paths:
        d = json.load(open(path))
        name = d["condition"]
        n = len(d["results"])
        regr_correct = 0
        cjk_n = 0
        for r in d["results"]:
            ok, reason = heuristic_correct(r["answer"], r["conclusion"])
            if reason == "cjk_unresolvable":
                cjk_n += 1
            if ok:
                regr_correct += 1
        orig_acc = d["accuracy"]
        regr_acc = regr_correct / n
        print(f"{name:22s} {orig_acc:>9.3f} {regr_acc:>9.3f} {cjk_n:>6d} {n:>4d}")
        summaries.append({
            "condition": name,
            "original_accuracy": orig_acc,
            "regraded_accuracy": regr_acc,
            "unresolved_ideograph_items": cjk_n,
            "n": n,
        })
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()
