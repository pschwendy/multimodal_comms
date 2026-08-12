#!/usr/bin/env python3
"""Join eval_result metrics with channel_stats.json across the compression
sweep. Reports native collaboration and communication metrics together.

Reads logs/collab_sweep/*.done to enumerate completed (run_tag, order) pairs,
pulls eval_result/{run_tag}/{order}/evaluation_result.json (success_rate,
TES/F1, similarity, redundancy, initiate/respond collaboration) and
data/{run_tag}/{order}/channel_stats_*.json (raw/transmitted chars, tokens,
wall clock), and aggregates per (model, condition) and per (model, condition,
complexity level).
"""
import glob
import json
import os
import statistics
import sys

LEVELS = {
    "level_1": ["baked_bell_pepper", "baked_sweet_potato", "boiled_egg", "boiled_mushroom", "boiled_sweet_potato"],
    "level_2": ["baked_potato_slices", "baked_pumpkin_slices", "boiled_corn_slices", "boiled_green_bean_slices", "boiled_potato_slices"],
    "level_3": ["baked_bell_pepper_soup", "baked_carrot_soup", "baked_mushroom_soup", "baked_potato_soup", "baked_pumpkin_soup"],
    "level_4": ["sliced_bell_pepper_and_corn_stew", "sliced_bell_pepper_and_lentil_stew", "sliced_eggplant_and_chickpea_stew", "sliced_pumpkin_and_chickpea_stew", "sliced_zucchini_and_chickpea_stew"],
    "level_5": ["mashed_broccoli_and_bean_patty", "mashed_carrot_and_chickpea_patty", "mashed_cauliflower_and_lentil_patty", "mashed_potato_and_pea_patty", "mashed_sweet_potato_and_bean_patty"],
    "level_6": ["potato_carrot_and_onion_patty", "romaine_lettuce_pea_and_tomato_patty", "sweet_potato_spinach_and_mushroom_patty", "taro_bean_and_bell_pepper_patty", "zucchini_green_pea_and_onion_patty"],
}
ORDER_TO_LEVEL = {o: lvl for lvl, os_ in LEVELS.items() for o in os_}


def load_channel_stats(run_tag: str, order: str) -> dict | None:
    paths = sorted(glob.glob(f"data/{run_tag}/{order}/channel_stats_*.json"))
    if not paths:
        return None
    return json.load(open(paths[-1]))


def load_eval_result(run_tag: str, order: str) -> dict | None:
    # Top-level dict is keyed by order_name (evaluate() writes one entry per
    # order it saw in the log; fix_task mode logs exactly the requested order).
    path = f"eval_result/{run_tag}/{order}/evaluation_result.json"
    if not os.path.exists(path):
        return None
    full = json.load(open(path))
    return full.get(order) or (next(iter(full.values())) if full else None)


def extract_row(run_tag: str, order: str) -> dict | None:
    ch = load_channel_stats(run_tag, order)
    ev = load_eval_result(run_tag, order)
    if ch is None:
        return None
    row = {
        "run_tag": run_tag,
        "order": order,
        "level": ORDER_TO_LEVEL.get(order, "?"),
        "success": ch.get("success"),
        "steps": ch.get("steps"),
        "raw_chars": ch.get("raw_chars", 0),
        "transmitted_chars": ch.get("transmitted_chars", 0),
        "input_tokens": ch.get("input_tokens", 0),
        "output_tokens": ch.get("output_tokens", 0),
        "wall_time_seconds": ch.get("wall_time_seconds", 0),
    }
    if ev:
        tm = ev.get("task_metrics", {})
        row["success_rate"] = tm.get("success_rate")
        row["time_avg"] = tm.get("time_avg")
        avg = ev.get("average", {}).get("similarity_and_redundancy", {})
        for aid in ("agent_0", "agent_1"):
            a = avg.get(aid, {})
            for k in ("similarity", "redundancy", "f1"):
                v = a.get(f"mean_{k}")
                if v is not None:
                    row.setdefault(f"{k}_sum", 0.0)
                    row[f"{k}_sum"] += v
                    row.setdefault(f"{k}_n", 0)
                    row[f"{k}_n"] += 1
        stat = ev.get("statistic", {})
        row["initiate_collaboration"] = stat.get("initiate_collaboration")
        row["respond_collaboration"] = stat.get("respond_collaboration")
    return row


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def main() -> None:
    markers = sorted(glob.glob("logs/collab_sweep/*.done"))
    rows = []
    missing_eval = []
    for m in markers:
        base = os.path.basename(m)[: -len(".done")]
        run_tag, order = base.rsplit("__", 1)
        row = extract_row(run_tag, order)
        if row is None:
            continue
        if "success_rate" not in row:
            missing_eval.append(f"{run_tag}/{order}")
        rows.append(row)

    if not rows:
        print("No completed sweep entries found (logs/collab_sweep/*.done).")
        return

    conditions: dict[str, list[dict]] = {}
    for r in rows:
        conditions.setdefault(r["run_tag"], []).append(r)

    header = (f"{'condition':<38} {'n':>2} {'succ%':>6} {'f1':>6} {'sim':>6} "
              f"{'redun':>6} {'init':>6} {'resp':>6} {'in_tok':>8} {'tx_chars':>9} "
              f"{'wall_s':>7}")
    print(header)
    print("-" * len(header))
    for run_tag, rs in sorted(conditions.items()):
        n = len(rs)
        succ = mean([1.0 if r["success"] else 0.0 for r in rs])
        f1 = mean([r.get("f1_sum", 0) / r["f1_n"] if r.get("f1_n") else None for r in rs])
        sim = mean([r.get("similarity_sum", 0) / r["similarity_n"] if r.get("similarity_n") else None for r in rs])
        redun = mean([r.get("redundancy_sum", 0) / r["redundancy_n"] if r.get("redundancy_n") else None for r in rs])
        init = mean([r.get("initiate_collaboration") for r in rs])
        resp = mean([r.get("respond_collaboration") for r in rs])
        in_tok = mean([r["input_tokens"] for r in rs])
        tx = mean([r["transmitted_chars"] for r in rs])
        wall = mean([r["wall_time_seconds"] for r in rs])

        def fmt(v, pct=False):
            if v is None:
                return "   -  "
            return f"{v * 100:5.1f}%" if pct else f"{v:6.2f}"

        print(f"{run_tag:<38} {n:>2} {fmt(succ, True):>6} {fmt(f1):>6} {fmt(sim):>6} "
              f"{fmt(redun):>6} {fmt(init):>6} {fmt(resp):>6} {in_tok:>8.0f} {tx:>9.0f} "
              f"{wall:>7.1f}")

    if missing_eval:
        print(f"\n[warn] {len(missing_eval)} entries have channel_stats but no eval_result "
              f"(run run_eval_pipeline.sh): {missing_eval[:5]}{'...' if len(missing_eval) > 5 else ''}")

    # per-level breakdown for the frontier conditions
    print("\n== per complexity level (success rate) ==")
    by_tag_level: dict[tuple, list] = {}
    for r in rows:
        by_tag_level.setdefault((r["run_tag"], r["level"]), []).append(r)
    tags = sorted(conditions.keys())
    levels = sorted(LEVELS.keys())
    print(f"{'condition':<38} " + " ".join(f"{lvl:>8}" for lvl in levels))
    for tag in tags:
        vals = []
        for lvl in levels:
            rs = by_tag_level.get((tag, lvl), [])
            s = mean([1.0 if r["success"] else 0.0 for r in rs]) if rs else None
            vals.append(f"{s * 100:7.0f}%" if s is not None else "     -  ")
        print(f"{tag:<38} " + " ".join(vals))


if __name__ == "__main__":
    main()
