"""Create deterministic, disjoint task directories for selector training and evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source benchmark.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-tasks", type=int, default=16)
    parser.add_argument("--train-tasks", type=int, default=0, help="0 uses all remaining tasks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = json.loads(Path(args.source).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("source benchmark must contain a JSON task list")
    if not 0 < args.eval_tasks < len(tasks):
        raise ValueError("eval-tasks must leave at least one training task")

    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    eval_tasks = tasks[: args.eval_tasks]
    remaining = tasks[args.eval_tasks :]
    train_tasks = remaining[: args.train_tasks] if args.train_tasks else remaining

    root = Path(args.out_dir)
    for name, rows in (("train", train_tasks), ("eval", eval_tasks)):
        destination = root / name / "benchmark.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(train_tasks)} train and {len(eval_tasks)} evaluation tasks to {root}")


if __name__ == "__main__":
    main()
