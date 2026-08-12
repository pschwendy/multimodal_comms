#!/usr/bin/env python3
"""Train the learnable sentence selector (MiniLM embeddings + logistic head).

Validation is split by TASK (not sentence) to measure generalization to
unseen scenarios, mirroring deployment on the 16 eval tasks.
Saves model to data/selector_model.joblib.
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = "data/selector_train.jsonl"
OUT = "data/selector_model.joblib"
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
        raise ValueError("selector training requires at least two distinct tasks")
    n_val = max(1, min(args.validation_tasks, len(tasks) // 3))
    random.seed(13)
    val_tasks = set(random.sample(tasks, n_val))

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [r["sentence"] for r in rows]
    embeddings = encoder.encode(sentences, normalize_embeddings=True,
                                batch_size=256, show_progress_bar=True)
    labels = np.array([r["label"] for r in rows])
    is_val = np.array([r["task"] in val_tasks for r in rows])

    X_train, y_train = embeddings[~is_val], labels[~is_val]
    X_val, y_val = embeddings[is_val], labels[is_val]

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)

    val_scores = clf.predict_proba(X_val)[:, 1]
    print(f"train sentences: {len(y_train)}  val sentences: {len(y_val)} "
          f"({n_val} held-out tasks)")
    print(f"val AUC: {roc_auc_score(y_val, val_scores):.3f}")
    print(f"val acc@0.5: {accuracy_score(y_val, val_scores > 0.5):.3f}")
    print(f"val base rate (keep): {y_val.mean():.3f}")

    # Refit on everything for deployment
    clf_full = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf_full.fit(embeddings, labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "classifier": clf_full,
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
