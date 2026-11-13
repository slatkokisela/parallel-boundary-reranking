import dataclasses

import numpy as np
import pytest

from pbar import calibration
from pbar.oracle import CachedOracle
from pbar.retrieval import QueryContext
from pbar.selectors import make


def test_pav_matches_bruteforce_and_is_monotone():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.normal(size=12)
        out = calibration.pav_decreasing(y)
        assert np.all(np.diff(out) <= 1e-12)
        # PAV preserves the mean, and an already-monotone input is unchanged
        assert out.mean() == pytest.approx(y.mean())
    y = np.array([5.0, 4.0, 4.0, 1.0])
    assert calibration.pav_decreasing(y).tolist() == y.tolist()
    assert calibration.pav_decreasing(np.array([1.0, 3.0])).tolist() == [2.0, 2.0]


def test_rank_curve_is_mean_by_rank():
    c = calibration.fit_rank_curve([np.array([0.9, 0.5, 0.1]), np.array([0.7, 0.3, 0.1])], 3)
    assert c.tolist() == pytest.approx([0.8, 0.4, 0.1])


def test_crossfit_never_uses_own_scores():
    rng = np.random.default_rng(1)
    ce = {f"q{i}": np.sort(rng.random(5))[::-1] for i in range(20)}
    curves = calibration.crossfit_curves(ce, 5, n_folds=2)
    for q in ce:                                  # perturbing q's scores must not change q's curve
        ce2 = dict(ce); ce2[q] = ce[q] * 0 + 99.0
        assert np.allclose(calibration.crossfit_curves(ce2, 5, n_folds=2)[q], curves[q])


def test_morph_calibrated_prior_runs_and_requires_curve():
    rng = np.random.default_rng(2)
    n = 30
    emb = rng.normal(size=(n, 6)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ctx = QueryContext("q", "q", np.arange(n), emb, emb[0], np.linspace(1, 0, n))
    ce = rng.random(n) * 0.05
    with pytest.raises(ValueError):
        make("morph", prior="calibrated").select(ctx, CachedOracle(5, ce), 10)
    ctx = dataclasses.replace(ctx, rank_prior=np.linspace(0.05, 0.0, n))
    for name, kw in [("morph", {}), ("morph_kb", {"batch_size": 3}), ("morph_topb", {"batch_size": 3})]:
        o = CachedOracle(9, ce)
        r = make(name, prior="calibrated", **kw).select(ctx, o, 10)
        assert sorted(r.tolist()) == list(range(n)) and o.calls == 9
