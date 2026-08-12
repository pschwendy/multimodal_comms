#!/usr/bin/env python3
"""Train the representational-match sentence selector (MiniLM + regressor).

This is a regression variant of the sentence selector: the
label is a continuous representational-distance score (see
labels come from `training.data.build_repmatch_dataset`, so logistic regression is replaced with a
regressor. We fit both Ridge and HistGradientBoostingRegressor, report the
task-held-out val R^2 of each, and save whichever is better.

Validation is split by TASK (not sentence), N_VAL_TASKS=8, seed=13, mirroring
train_selector.py and deployment on unseen scenarios.

Saved bundle: {"encoder_name", "regressor"} to data/repmatch_selector.joblib.
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DATA = "data/repmatch_train.jsonl"
OUT = "data/repmatch_selector.joblib"
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
        raise ValueError("representation-match training requires at least two distinct tasks")
    n_val = max(1, min(args.validation_tasks, len(tasks) // 3))
    random.seed(13)
    val_tasks = set(random.sample(tasks, n_val))

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [r["sentence"] for r in rows]
    embeddings = encoder.encode(sentences, normalize_embeddings=True,
                                batch_size=256, show_progress_bar=True)
    labels = np.array([r["label"] for r in rows], dtype=float)
    is_val = np.array([r["task"] in val_tasks for r in rows])

    X_train, y_train = embeddings[~is_val], labels[~is_val]
    X_val, y_val = embeddings[is_val], labels[is_val]

    print(f"train sentences: {len(y_train)}  val sentences: {len(y_val)} "
          f"({n_val} held-out tasks)")
    print(f"label mean/std (train): {y_train.mean():.4f} / {y_train.std():.4f}")

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "hgb": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=None,
            l2_regularization=1.0, random_state=13,
        ),
    }
    results = {}
    for name, reg in candidates.items():
        reg.fit(X_train, y_train)
        pred = reg.predict(X_val)
        r2 = r2_score(y_val, pred)
        results[name] = r2
        print(f"[{name}] val R^2: {r2:.4f}")

    best_name = max(results, key=results.get)
    print(f"best regressor: {best_name} (val R^2 = {results[best_name]:.4f})")
    print("(compare vs Part 4 counterfactual scorer val R^2 = 0.063)")

    # Refit the winning model family on everything for deployment.
    best = candidates[best_name].__class__(**candidates[best_name].get_params())
    best.fit(embeddings, labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "regressor": best,
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
