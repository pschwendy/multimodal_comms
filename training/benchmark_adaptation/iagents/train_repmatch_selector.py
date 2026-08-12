#!/usr/bin/env python3
"""Train the A2 representational-match sentence selector (MiniLM + regressor).

Regression on the
continuous rep-distance label from build_repmatch_dataset.py. Validation split
by TASK (not sentence). Saves {"encoder_name","regressor"} to
data/repmatch_selector.joblib (iAgents-local artifact, per-benchmark).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/repmatch_train.jsonl")
    ap.add_argument("--out", default="data/repmatch_selector.joblib")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    tasks = sorted({r["task"] for r in rows})
    n_val = max(1, min(8, len(tasks) // 3))
    random.seed(13)
    val_tasks = set(random.sample(tasks, n_val))

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [r["sentence"] for r in rows]
    embeddings = encoder.encode(sentences, normalize_embeddings=True,
                                batch_size=256, show_progress_bar=False)
    labels = np.array([r["label"] for r in rows], dtype=float)
    is_val = np.array([r["task"] in val_tasks for r in rows])

    X_train, y_train = embeddings[~is_val], labels[~is_val]
    X_val, y_val = embeddings[is_val], labels[is_val]

    print(f"tasks: {len(tasks)} ({n_val} held out for val)")
    print(f"train sentences: {len(y_train)}  val sentences: {len(y_val)}")
    print(f"label mean/std (train): {y_train.mean():.4f} / {y_train.std():.4f}")

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "hgb": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            random_state=13),
    }
    results = {}
    for name, reg in candidates.items():
        reg.fit(X_train, y_train)
        if len(y_val):
            results[name] = r2_score(y_val, reg.predict(X_val))
        else:
            results[name] = float("nan")
        print(f"[{name}] val R^2: {results[name]:.4f}")

    best_name = max(results, key=lambda k: (results[k] if results[k] == results[k] else -1e9))
    print(f"best regressor: {best_name} (val R^2 = {results[best_name]:.4f})")

    best = candidates[best_name].__class__(**candidates[best_name].get_params())
    best.fit(embeddings, labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "regressor": best,
    }, args.out)

    # Report a threshold suggestion: keep ~ top 60% of sentences by pred.
    preds = best.predict(embeddings)
    print(f"pred label quantiles: p40={np.quantile(preds,0.4):.4f} "
          f"p50={np.quantile(preds,0.5):.4f} p60={np.quantile(preds,0.6):.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
