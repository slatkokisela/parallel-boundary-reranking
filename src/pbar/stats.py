"""Paired statistics over per-query results."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean. NaNs are dropped."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_test(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired comparison a - b on queries where both values are defined."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    p = float("nan") if np.allclose(d, 0) else float(wilcoxon(d, zero_method="zsplit").pvalue)
    return {"n": int(m.sum()), "delta": mean, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
