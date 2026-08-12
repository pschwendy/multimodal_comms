#!/usr/bin/env python3
"""Train the A2 repmatch_selector regressor (MiniLM embeddings + GBR).

Predicts the representational-distance importance label from a raw sentence
embedding. Saves data/repmatch_selector.joblib as {"encoder_name","regressor"}
exactly as RepMatchSelectorCompressor._load expects (it calls .predict and
keeps sentences with score >= tau).

Also prints a suggested tau (a percentile of held-out predictions) so the
sweep keeps a sensible fraction of sentences given COMMA's short turns.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

DATA = "data/repmatch_selector_train.jsonl"
OUT = "data/repmatch_selector.joblib"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.data)]
    print(f"rows: {len(rows)}")
    sources = sorted({r["source"] for r in rows})
    if len(sources) < 2:
        raise ValueError("COMMA selector training requires at least two distinct sources")
    rng = np.random.RandomState(13)
    val_sources = set(rng.choice(sources, max(1, len(sources) // 5), replace=False))

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER, device="cpu")
    sents = [r["sentence"] for r in rows]
    X = enc.encode(sents, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    y = np.array([r["label"] for r in rows])
    is_val = np.array([r["source"] in val_sources for r in rows])

    reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    reg.fit(X[~is_val], y[~is_val])
    if is_val.sum() > 2:
        pred_val = reg.predict(X[is_val])
        print(f"val R^2: {r2_score(y[is_val], pred_val):.3f}  (val n={is_val.sum()})")

    # Refit on everything for deployment
    reg_full = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                         subsample=0.8, random_state=0)
    reg_full.fit(X, y)
    preds = reg_full.predict(X)
    for p in (25, 35, 50, 65):
        print(f"  pred p{p} = {np.percentile(preds, p):.4f}")
    print(f"suggested tau (keeps ~65%): {np.percentile(preds, 35):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"encoder_name": ENCODER, "regressor": reg_full}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
