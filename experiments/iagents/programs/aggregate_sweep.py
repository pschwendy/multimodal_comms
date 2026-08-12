#!/usr/bin/env python3
"""Aggregate per-condition offline-eval reports into a markdown table.

Usage: python experiments/iagents/programs/aggregate_sweep.py reports/eval_*.json --out reports/sweep_needle.md
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", default="reports/sweep_needle.md")
    args = ap.parse_args()

    rows = []
    for path in args.reports:
        d = json.load(open(path))
        rows.append(d)

    # identity is the baseline for char-savings %
    base = next((r for r in rows if r["condition"] == "identity"), None)
    base_tx = base["total_transmitted_chars"] if base else None

    # sort: identity first, then by name
    rows.sort(key=lambda r: (r["condition"] != "identity", r["condition"]))

    lines = []
    lines.append("# Needle_in_the_Persona compression sweep (iAgents / informativeBench)\n")
    lines.append(f"- Dataset: `{rows[0]['dataset']}`  |  tasks/condition: {rows[0]['n']}\n")
    lines.append("| condition | params | acc | correct | errors | tx_chars | char_saving | wall_s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        tx = r["total_transmitted_chars"]
        if base_tx:
            saving = f"{100.0 * (1 - tx / base_tx):.1f}%"
        else:
            saving = "-"
        params = ", ".join(f"{k}={v}" for k, v in (r.get("params") or {}).items()) or "-"
        lines.append(
            f"| {r['condition']} | {params} | {r['accuracy']:.3f} | "
            f"{r['n_correct']}/{r['n']} | {r.get('n_error',0)} | "
            f"{tx} | {saving} | {r['total_wall_s']:.0f} |"
        )
    table = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(table)
    print(table)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
