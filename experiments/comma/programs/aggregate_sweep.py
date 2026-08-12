#!/usr/bin/env python3
"""Aggregate COMMA compression-sweep results into a report table.

Usage: aggregate_sweep.py cond=name:save_dir cond=name:save_dir ...
Coverage = fraction of attempted puzzle runs that were ever solved
(the "Puzzle successfully finished" marker), per this project's
Collab-Overcooked precedent. Also reports mean Telehealth score (partial
credit: 0.5 = correct diagnosis, 1.0 = diagnosis+treatment), total prompt /
completion tokens (real DeepSeek usage), transmitted channel chars, and
wall-clock per condition.
"""
import glob
import json
import os
import sys

SUCCESS_MARK = "Puzzle successfully finished"


def load_runs(save_dir):
    runs = []
    for conv in glob.glob(os.path.join(save_dir, "**", "conversation.jsonl"), recursive=True):
        rundir = os.path.dirname(conv)
        puzzle = os.path.basename(os.path.dirname(rundir))
        text = open(conv).read()
        success = SUCCESS_MARK in text
        # telehealth partial score: last ENVIRONMENT score line
        score = None
        for line in text.splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict) and "score" in d:
                try:
                    score = float(d["score"])
                except Exception:
                    pass
        # tokens
        p_tok = c_tok = 0
        rd = os.path.join(rundir, "response_data.jsonl")
        if os.path.exists(rd):
            for line in open(rd):
                try:
                    u = json.loads(line)
                except Exception:
                    continue
                p_tok += u.get("prompt_tokens", 0)
                c_tok += u.get("completion_tokens", 0)
        # channel chars (SOLVER/EXPERT content transmitted)
        chan_chars = 0
        for line in text.splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            frm = str(d.get("from", "")).lower()
            if frm.startswith("solver") or frm.startswith("expert"):
                chan_chars += len(str(d.get("value", "")))
        runs.append({"puzzle": puzzle, "success": success, "score": score,
                     "p_tok": p_tok, "c_tok": c_tok, "chan_chars": chan_chars})
    return runs


def main():
    conds = []
    for arg in sys.argv[1:]:
        # name:save_dir  (strip optional cond= prefix)
        arg = arg[5:] if arg.startswith("cond=") else arg
        name, sd = arg.split(":", 1)
        conds.append((name, sd))

    print(f"{'condition':<20} {'runs':>5} {'cover':>7} {'tele_mean':>9} "
          f"{'prompt_tok':>11} {'compl_tok':>10} {'chan_chars':>11}")
    print("-" * 80)
    base_tok = None
    rows = []
    for name, sd in conds:
        runs = load_runs(sd)
        n = len(runs)
        if n == 0:
            print(f"{name:<20} {0:>5}  (no runs)")
            continue
        cover = sum(r["success"] for r in runs) / n
        teles = [r["score"] for r in runs if r["puzzle"].lower().startswith("tele") and r["score"] is not None]
        tele_mean = sum(teles) / len(teles) if teles else float("nan")
        p = sum(r["p_tok"] for r in runs)
        c = sum(r["c_tok"] for r in runs)
        ch = sum(r["chan_chars"] for r in runs)
        rows.append((name, n, cover, tele_mean, p, c, ch))
        print(f"{name:<20} {n:>5} {cover:>6.1%} {tele_mean:>9.3f} "
              f"{p:>11,} {c:>10,} {ch:>11,}")

    # savings vs identity
    ident = next((r for r in rows if r[0] == "identity"), None)
    if ident:
        print("\nvs identity (prompt_tok / chan_chars):")
        for r in rows:
            if r[4] and ident[4]:
                print(f"  {r[0]:<20} prompt_tok {100*(1-r[4]/ident[4]):+6.1f}%   "
                      f"chan_chars {100*(1-r[6]/ident[6]):+6.1f}%")


if __name__ == "__main__":
    main()
