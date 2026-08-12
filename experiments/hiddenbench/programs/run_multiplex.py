#!/usr/bin/env python3
"""Scaling sweep for many-to-many superposed-packet communication.

For each T in --num-tasks, runs T HiddenBench tasks in lockstep with one
shared packet per round carrying all T*A agent messages (superposition
load M = T*A), and records reconstruction fidelity, task accuracy, and
bytes on the wire. The AE model is loaded once and shared across sweep
points.

Example:
  python -m experiments.hiddenbench.programs.run_multiplex \
      --config config_deepseek.yaml \
      --model-path data/superpose_pretrain/final \
      --num-tasks 1 2 4 8 --num-rounds 3 --workers 8 \
      --out-dir reports/multiplex
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))

from multimodal_comms.benchmarks.hiddenbench.runtime.config import Config  # noqa: E402
from multimodal_comms.benchmarks.hiddenbench.runtime.multiplex import MultiplexRunner  # noqa: E402
from multimodal_comms.benchmarks.hiddenbench.runtime.superpose import LatentCodec, SuperposedPacketCodec  # noqa: E402


class PassthroughPacketCodec:
    """Identity control: the 'packet' is the plaintext messages themselves.

    Keeps the multiplex protocol byte-for-byte identical (same prompts,
    rounds, slot bookkeeping) while making the channel lossless, so
    lockstep identity baselines are directly comparable to codec runs.
    """

    class _NullCodec:
        num_latents = 0
        latent_dim = 0

    codec = _NullCodec()

    def encode_packet(self, texts_by_slot):
        return json.dumps({str(k): v for k, v in texts_by_slot.items()})

    def decode_slot(self, packet_str, slot):
        return json.loads(packet_str)[str(slot)]
from multimodal_comms.benchmarks.hiddenbench.runtime.task import load_tasks_from_directory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_deepseek.yaml",
                    help="Agent (sender/receiver) provider config. Defaults to "
                         "DeepSeek; config.yaml points at local Qwen3-4B agents, "
                         "which must not be confused with the Qwen-based channel codec.")
    ap.add_argument("--model-path", type=str, default="data/superpose_pretrain/final")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--key-seed", type=int, default=None,
                    help="Default: read from the checkpoint's superpose_config.json.")
    ap.add_argument("--key-mode", type=str, default=None, choices=[None, "qr", "sign"])
    ap.add_argument("--num-tasks", type=int, nargs="+", default=[1, 2, 4],
                    help="Sweep values of T (superposition load = T * num_agents).")
    ap.add_argument("--num-rounds", type=int, default=3)
    ap.add_argument("--num-agents", type=int, default=None,
                    help="Override config.benchmark.num_agents.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Override config.benchmark.max_tokens (reasoning "
                         "providers need headroom beyond the visible reply).")
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--total-tasks", type=int, default=16,
                    help="Fixed task set size evaluated at EVERY load level "
                         "(grouped T at a time), so sweep points are paired.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent LLM calls per round (HTTP providers only).")
    ap.add_argument("--packing", type=str, default="single",
                    choices=["single", "cross"],
                    help="'single': all T*A messages in one packet. 'cross': "
                         "A packets/round, each holding one message per task "
                         "(load T, heterogeneous content).")
    ap.add_argument("--identity", action="store_true",
                    help="Plaintext-channel control: same lockstep protocol, "
                         "lossless packets, no codec model loaded.")
    ap.add_argument("--out-dir", type=str, default="reports/multiplex")
    args = ap.parse_args()

    config = Config.load(Path(args.config))
    if args.num_agents is not None:
        config.benchmark.num_agents = args.num_agents
    if args.max_tokens is not None:
        config.benchmark.max_tokens = args.max_tokens
    A = config.benchmark.num_agents

    # Key parameters must match training; default to the checkpoint's record.
    key_seed, key_mode = args.key_seed, args.key_mode
    sp_cfg_path = os.path.join(args.model_path, "superpose_config.json")
    if os.path.exists(sp_cfg_path):
        sp_cfg = json.load(open(sp_cfg_path))
        key_seed = key_seed if key_seed is not None else sp_cfg.get("key_seed", 1234)
        key_mode = key_mode if key_mode is not None else sp_cfg.get("key_mode", "qr")
        trained_max = sp_cfg.get("max_slots")
    else:
        key_seed = key_seed if key_seed is not None else 1234
        key_mode = key_mode if key_mode is not None else "qr"
        trained_max = None

    max_load = max(args.num_tasks) * A
    if trained_max is not None and max_load > trained_max:
        print(f"WARNING: sweep reaches load {max_load} but checkpoint was "
              f"trained up to max_slots={trained_max}; expect degradation "
              f"beyond that.", file=sys.stderr)

    if args.identity:
        packet_codec = PassthroughPacketCodec()
    else:
        latent_codec = LatentCodec(model_path=args.model_path, device=args.device)
        packet_codec = SuperposedPacketCodec(
            latent_codec, key_seed=key_seed, key_mode=key_mode,
        )
        # Building the keyring loads the model (latent dim must be known);
        # widen its cache so every slot in the sweep stays resident.
        packet_codec.keyring.cache_limit = max_load + 1

    task_pool = load_tasks_from_directory(
        Path(config.benchmark.data_dir), A,
    )
    if config.benchmark.use_custom_tasks:
        tasks_dir = Path(config.benchmark.tasks_dir)
        if tasks_dir.exists():
            existing = {t.id for t in task_pool}
            task_pool.extend(t for t in load_tasks_from_directory(tasks_dir, A)
                             if t.id not in existing)
    if not task_pool:
        print("No tasks found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(task_pool)} tasks; agents per task: {A}")

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []

    pool = task_pool[args.task_offset:args.task_offset + args.total_tasks]
    if len(pool) < args.total_tasks:
        print(f"Only {len(pool)} tasks available; using those.", file=sys.stderr)

    for T in args.num_tasks:
        # Same fixed task set at every load level, packed T at a time --
        # only the packing density (superposition load) varies across sweep
        # points, so accuracies are paired on identical tasks.
        load = T * A
        n_groups = len(pool) // T
        if n_groups == 0:
            print(f"Skipping T={T}: not enough tasks.", file=sys.stderr)
            continue
        print(f"\n=== T={T} per packet (load M={load}), "
              f"{n_groups} groups covering {n_groups * T} tasks ===")
        group_results = []
        for g in range(n_groups):
            tasks = pool[g * T:(g + 1) * T]
            runner = MultiplexRunner(
                config, packet_codec, num_rounds=args.num_rounds,
                workers=args.workers, packing=args.packing,
            )
            result = runner.run(tasks)
            result["sweep"] = {"T": T, "A": A, "load": load, "group": g,
                               "packing": args.packing,
                               "model_path": args.model_path,
                               "key_seed": key_seed, "key_mode": key_mode}
            out_path = os.path.join(
                args.out_dir, f"multiplex_{stamp}_T{T}_g{g}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  group {g + 1}/{n_groups} "
                  f"[{', '.join(t.name[:24] for t in tasks)}]: "
                  f"fid={result['mean_reconstruction_fidelity']:.3f}  "
                  f"pre={result['pre_accuracy']:.2f}  "
                  f"post={result['post_accuracy']:.2f}")
            group_results.append(result)

        # Aggregate over groups: task-weighted accuracies, slot-weighted
        # fidelity, summed bytes.
        n_tasks_run = sum(len(r["tasks"]) for r in group_results)
        all_fid = [f for r in group_results for rec in r["rounds"]
                   for f in rec["slot_fidelity"].values()]
        agg = {
            "T": T, "load": load, "n_groups": n_groups,
            "n_tasks": n_tasks_run,
            "mean_reconstruction_fidelity": sum(all_fid) / len(all_fid),
            "pre_accuracy": sum(t["pre_accuracy"] for r in group_results
                                for t in r["tasks"]) / n_tasks_run,
            "post_accuracy": sum(t["post_accuracy"] for r in group_results
                                 for t in r["tasks"]) / n_tasks_run,
            "bytes": {
                "packet_total": sum(r["bytes"]["packet_total"] for r in group_results),
                "unicast_ae_total": sum(r["bytes"]["unicast_ae_total"] for r in group_results),
            },
            "per_task": [
                {"task": t["task_name"], "group": r["sweep"]["group"],
                 "pre": t["pre_accuracy"], "post": t["post_accuracy"]}
                for r in group_results for t in r["tasks"]
            ],
        }
        agg["bytes"]["broadcast_vs_unicast_ratio"] = (
            agg["bytes"]["packet_total"] / agg["bytes"]["unicast_ae_total"]
            if agg["bytes"]["unicast_ae_total"] else None
        )
        agg_path = os.path.join(args.out_dir, f"multiplex_{stamp}_T{T}_agg.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  == load {load}: fidelity={agg['mean_reconstruction_fidelity']:.3f}  "
              f"pre={agg['pre_accuracy']:.3f}  post={agg['post_accuracy']:.3f}  "
              f"({n_tasks_run} tasks) -> {agg_path}")
        summary_rows.append((load, T, agg))

    if summary_rows:
        md_path = os.path.join(args.out_dir, f"multiplex_{stamp}_summary.md")
        with open(md_path, "w") as f:
            f.write(f"# Superposed-packet scaling sweep ({stamp})\n\n")
            f.write(f"Model: `{args.model_path}`  keys: {key_mode}/{key_seed}  "
                    f"agents/task: {A}  rounds: {args.num_rounds}\n\n")
            f.write("| Load M | Tasks T | Fidelity | Pre acc | Post acc | "
                    "Packet bytes | Unicast-AE bytes | Ratio |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for load, T, res in summary_rows:
                b = res["bytes"]
                ratio = b["broadcast_vs_unicast_ratio"]
                ratio_str = f"{ratio:.3f}" if ratio else "-"
                f.write(f"| {load} | {T} | "
                        f"{res['mean_reconstruction_fidelity']:.3f} | "
                        f"{res['pre_accuracy']:.2f} | {res['post_accuracy']:.2f} | "
                        f"{b['packet_total']} | {b['unicast_ae_total']} | "
                        f"{ratio_str} |\n")
        print(f"\nSummary written to {md_path}")


if __name__ == "__main__":
    main()
