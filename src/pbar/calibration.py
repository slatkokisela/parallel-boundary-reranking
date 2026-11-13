"""Calibrated rank prior (our extension; not part of the BoundaryMORPH paper).

The paper's prior mu0(pos) = (N-1-pos)/(N-1) implicitly assumes CE scores are on a
rank-percentile scale. For CEs with skewed outputs (e.g. ms-marco-MiniLM on NFCorpus:
median sigmoid score ~0, p99 ~0.03) the prior sits far above every observed score. Each
scored doc then falls out of the top-k, and unscored docs keep inflated posterior means.

The calibrated prior replaces this with g(pos) = E[f_CE(q, d) | MIPS rank of d = pos],
estimated on *other* queries and made monotone non-increasing by isotonic regression (PAV).
It uses CE scores and MIPS ranks only, never relevance labels. To avoid leakage it is fit
on a different split (test runs: fit on dev), or cross-fitted over query folds within a split.
"""
from __future__ import annotations

import zlib

import numpy as np


def pav_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted isotonic regression with a non-increasing constraint (pool adjacent violators)."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    # PAV for non-decreasing on -y
    vals, wts, cnt = [], [], []
    for v, ww in zip(-y, w):
        vals.append(v); wts.append(ww); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(c1 + c2)
    return -np.repeat(vals, cnt)


def fit_rank_curve(score_lists: list[np.ndarray], n: int) -> np.ndarray:
    """Mean CE score at each MIPS rank 0..n-1 over queries (pools shorter than n are skipped
    at missing ranks), then isotonic non-increasing."""
    s = np.zeros(n); c = np.zeros(n)
    for sc in score_lists:
        m = min(n, len(sc))
        s[:m] += sc[:m]; c[:m] += 1
    if (c == 0).any():
        raise ValueError("some ranks have no data")
    return pav_decreasing(s / c, c)


def fold_of(qid: str, n_folds: int, seed: int = 0) -> int:
    return zlib.crc32(f"{seed}:{qid}".encode()) % n_folds


def crossfit_curves(ce_by_qid: dict[str, np.ndarray], n: int, n_folds: int = 2,
                    seed: int = 0) -> dict[str, np.ndarray]:
    """For each qid, the rank curve fit on queries from the *other* folds."""
    folds = {q: fold_of(q, n_folds, seed) for q in ce_by_qid}
    curves = {f: fit_rank_curve([s for q, s in ce_by_qid.items() if folds[q] != f], n)
              for f in range(n_folds)}
    return {q: curves[folds[q]] for q in ce_by_qid}
