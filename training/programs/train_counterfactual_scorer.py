#!/usr/bin/env python3
"""Distill counterfactual-importance labels (KL-divergence based, expensive to
compute live) into a lightweight scorer: MiniLM embeddings + Ridge regression
head predicting the continuous importance score. Task-level held-out split
to measure generalization to unseen scenarios (mirrors train_selector.py).
Saves to data/counterfactual_scorer.joblib.
"""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DATA = "data/counterfactual_train.jsonl"
OUT = "data/counterfactual_scorer.joblib"
N_VAL_TASKS = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--validation-tasks", type=int, default=N_VAL_TASKS)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.data)]
    tasks = sorted({r["task"] for r in rows})
    if len(tasks) < 2:
        raise ValueError("counterfactual training requires at least two distinct tasks")
    n_val = max(1, min(args.validation_tasks, len(tasks) // 3))
    random.seed(13)
    val_tasks = set(random.sample(tasks, n_val))

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [r["sentence"] for r in rows]
    embeddings = encoder.encode(sentences, normalize_embeddings=True,
                                 batch_size=256, show_progress_bar=True)
    labels = np.array([r["importance"] for r in rows])
    is_val = np.array([r["task"] in val_tasks for r in rows])

    X_train, y_train = embeddings[~is_val], labels[~is_val]
    X_val, y_val = embeddings[is_val], labels[is_val]

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    val_pred = reg.predict(X_val)

    print(f"train sentences: {len(y_train)}  val sentences: {len(y_val)} "
          f"({len(val_tasks)} held-out tasks)")
    print(f"val R^2: {r2_score(y_val, val_pred):.3f}")
    print(f"label range: min={labels.min():.3f} max={labels.max():.3f} "
          f"mean={labels.mean():.3f} std={labels.std():.3f}")

    reg_full = Ridge(alpha=1.0)
    reg_full.fit(embeddings, labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "regressor": reg_full,
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
