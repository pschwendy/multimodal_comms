#!/usr/bin/env python3
"""Aggregate compression-sweep reports into a comparison table.

Reads reports/sweep_*.json and prints per-condition means:
accuracy, input/output tokens, wall time, message lengths, channel traffic.
"""

import glob
import json
import os
import sys


def main() -> None:
    reports_dir = sys.argv[1] if len(sys.argv) > 1 else "reports"
    rows = []
    for path in sorted(glob.glob(os.path.join(reports_dir, "sweep_*.json"))):
        d = json.load(open(path))
        name = os.path.basename(path)[len("sweep_"):-len(".json")]
        results = d.get("results", [])
        if not results:
            continue
        n = len(results)

        def mean(fn):
            return sum(fn(r) for r in results) / n

        raw_chars = mean(lambda r: (r["resources"]["channel_stats"] or {}).get("raw_chars", 0))
        raw_msgs = mean(lambda r: (r["resources"]["channel_stats"] or {}).get("raw_messages", 0))
        rows.append({
            "condition": name,
            "status": d["metadata"].get("status"),
            "n": n,
            "pre_acc": mean(lambda r: r["metrics"]["pre_accuracy"]),
            "post_acc": mean(lambda r: r["metrics"]["post_accuracy"]),
            "consensus_rate": sum(
                1 for r in results if r.get("consensus_round") is not None) / n,
            "rounds": mean(lambda r: r["total_rounds"]),
            "in_tok": mean(lambda r: r["resources"]["input_tokens"]),
            "out_tok": mean(lambda r: r["resources"]["output_tokens"]),
            "wall_s": mean(lambda r: r["resources"]["wall_time_seconds"]),
            "msg_len": (raw_chars / raw_msgs) if raw_msgs else 0.0,
            "tx_chars": mean(
                lambda r: (r["resources"]["channel_stats"] or {}).get("transmitted_chars", 0)),
        })

    if not rows:
        print("No sweep reports found.")
        return

    def find(cond: str) -> dict | None:
        return next((r for r in rows if r["condition"] == cond), None)

    def model_prefix(name: str) -> str:
        for p in ("ds4_", "ds_"):
            if name.startswith(p):
                return p
        return ""

    def protocol_baseline_for(name: str) -> dict | None:
        # A compressor's contribution is measured against the identity
        # condition of the SAME protocol (delta alone is a protocol setting,
        # not a compression method).
        prefix = model_prefix(name)
        proto = "delta" if name.removeprefix(prefix).startswith("delta") else "full"
        return find(f"{prefix}{proto}_identity")

    def original_baseline_for(name: str) -> dict | None:
        # Total system effect vs the original (full-history) protocol
        return find(f"{model_prefix(name)}full_identity")

    # Sender-side conditions change the agents' instructions (message style);
    # middleware conditions only change what the channel transmits.
    def is_sender_side(name: str) -> bool:
        return "concise" in name or "schema" in name

    def print_table(title: str, subset: list[dict]) -> None:
        if not subset:
            return
        print(f"\n== {title} ==")
        print("(save_prot: vs same-protocol identity baseline = the compressor's own "
              "contribution; save_orig: vs original full-history protocol)")
        header = (f"{'condition':<22} {'st':<3} {'n':>2} {'pre':>5} {'post':>5} "
                  f"{'cons%':>5} {'rnds':>4} {'in_tok':>9} {'out_tok':>7} "
                  f"{'wall_s':>7} {'msg_len':>7} {'tx_chars':>9} "
                  f"{'save_prot':>9} {'save_orig':>9}")
        print(header)
        print("-" * len(header))
        for r in subset:
            def fmt_save(baseline: dict | None) -> str:
                if baseline and baseline["in_tok"] and baseline is not r:
                    return f"{(1 - r['in_tok'] / baseline['in_tok']) * 100:+.0f}%"
                return "-"
            save_prot = fmt_save(protocol_baseline_for(r["condition"]))
            save_orig = fmt_save(original_baseline_for(r["condition"]))
            status = "ok" if r["status"] == "complete" else "…"
            print(f"{r['condition']:<22} {status:<3} {r['n']:>2} "
                  f"{r['pre_acc']:>5.2f} {r['post_acc']:>5.2f} "
                  f"{r['consensus_rate'] * 100:>4.0f}% {r['rounds']:>4.1f} "
                  f"{r['in_tok']:>9.0f} {r['out_tok']:>7.0f} "
                  f"{r['wall_s']:>7.1f} {r['msg_len']:>7.0f} {r['tx_chars']:>9.0f} "
                  f"{save_prot:>9} {save_orig:>9}")

    print_table(
        "MIDDLEWARE-ONLY (channel controls transmission; sender/receiver untouched)",
        [r for r in rows if not is_sender_side(r["condition"])],
    )
    print_table(
        "SENDER-SIDE REFERENCE (requires changing agent instructions)",
        [r for r in rows if is_sender_side(r["condition"])],
    )


if __name__ == "__main__":
    main()
