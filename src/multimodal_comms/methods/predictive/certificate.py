"""
Shared representational-match certificate (Eq. 1 in the proposed-methods note).

A frozen proxy model's first hidden state after reading a message,
h(m | c), is used as a cheap, deterministic acceptance oracle:

    cos( h(m' | c), h(m | c) ) >= 1 - eps

A candidate m' (a pruned/rewritten/reconstructed version of m) is "certified"
when it moves the proxy's representation by less than eps. This has no
trainable components - it's a fixed threshold test against the same
representation server (`training.services.repserver`) used by
repmatch_selector / repmatch_bestofk / saliency, running a frozen Qwen3-4B
that is never itself a sender or receiver.

Fails safe: if the representation server is unreachable, `passes_certificate`
returns False (reject the compression, keep the original) rather than
silently letting an unverifiable candidate through.
"""

from __future__ import annotations

import numpy as np
import requests


def get_reps(texts: list[str], repserver_url: str, timeout: float = 30.0) -> np.ndarray | None:
    """Batched /rep_batch call. Returns an (N, dim) array, or None on failure."""
    if not texts:
        return np.zeros((0, 0))
    try:
        resp = requests.post(
            f"{repserver_url.rstrip('/')}/rep_batch",
            json={"items": [{"text": t} for t in texts]},
            timeout=timeout,
        )
        resp.raise_for_status()
        return np.array(resp.json()["reps"])
    except Exception:
        return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def passes_certificate(
    original_text: str,
    candidate_text: str,
    eps: float,
    repserver_url: str = "http://127.0.0.1:8100",
    timeout: float = 30.0,
) -> bool:
    """True iff cos(h(candidate), h(original)) >= 1 - eps.

    A single batched call scores both texts in one request. Identical text
    trivially passes for any eps >= 0 without a network round trip.
    """
    if candidate_text == original_text:
        return True
    reps = get_reps([original_text, candidate_text], repserver_url, timeout)
    if reps is None or reps.shape[0] < 2:
        return False
    return cosine(reps[0], reps[1]) >= 1.0 - eps


def certificate_scores(
    original_text: str,
    candidate_texts: list[str],
    repserver_url: str = "http://127.0.0.1:8100",
    timeout: float = 30.0,
) -> list[float] | None:
    """Batched cosine similarity of each candidate to the original.

    Returns None (fail-safe) if the server is unreachable. One call scores
    the original plus every candidate together.
    """
    if not candidate_texts:
        return []
    reps = get_reps([original_text] + candidate_texts, repserver_url, timeout)
    if reps is None or reps.shape[0] < 1 + len(candidate_texts):
        return None
    original_rep = reps[0]
    return [cosine(original_rep, reps[i + 1]) for i in range(len(candidate_texts))]
