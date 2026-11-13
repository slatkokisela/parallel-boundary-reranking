"""Tests for the batch BoundaryMORPH variants and the residual GP. Synthetic data only."""
import math

import numpy as np
import pytest

from pbar.gp import ResidualGP, median_kappa, vmf_kernel
from pbar.oracle import CachedOracle
from pbar.retrieval import QueryContext
from pbar.selectors.batch import BATCH_SELECTORS, MorphKrigingBeliever, MorphRandomBoundary, MorphTopB
from pbar.selectors.morph import BoundaryMORPH


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_ctx(n=60, d=8, n_clusters=6, spread=0.15, seed=0):
    """Clustered pool sorted by inner product with a query, plus CE scores correlated with MIPS."""
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(n_clusters, d)))
    emb = _unit(centers[rng.integers(0, n_clusters, n)] + spread * rng.normal(size=(n, d)))
    q = _unit(rng.normal(size=d) + centers[0])
    mips = emb @ q
    order = np.argsort(-mips, kind="stable")
    emb, mips = emb[order], mips[order]
    ce = 1 / (1 + np.exp(-(3 * mips + rng.normal(scale=1.0, size=n))))
    return QueryContext("q", "query", np.arange(n), emb, q, mips), ce


def run(sel, ctx, ce, B, k):
    o = CachedOracle(B, ce)
    r = sel.select(ctx, o, k)
    return r, o


def batches(o):
    """Recover per-round batches from insertion order + round sizes."""
    keys, out, s = list(o.observed), [], 0
    for n in o.round_sizes:
        out.append(keys[s:s + n]); s += n
    return out


# ---- b = 1 equivalence ------------------------------------------------------------------

@pytest.mark.parametrize("cls,kw", [(MorphTopB, {}), (MorphKrigingBeliever, {"fantasy": "mean"})])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [5, 15])
def test_b1_reproduces_sequential(cls, kw, seed, k):
    ctx, ce = make_ctx(seed=seed)
    r_seq, o_seq = run(BoundaryMORPH(batch_size=1), ctx, ce, 20, k)
    r_b, o_b = run(cls(batch_size=1, **kw), ctx, ce, 20, k)
    assert list(o_b.observed) == list(o_seq.observed)       # same picks in the same order
    assert r_b.tolist() == r_seq.tolist()


# ---- budget / round accounting ----------------------------------------------------------

ALL_VARIANTS = [(name, kw) for name in BATCH_SELECTORS for kw in [{}]] + [
    ("morph_kb", {"fantasy": "pessimistic"}), ("morph_kb", {"fantasy": "optimistic"})]


@pytest.mark.parametrize("name,kw", ALL_VARIANTS)
@pytest.mark.parametrize("B,b", [(10, 1), (10, 2), (25, 4), (25, 8), (7, 8)])
def test_exact_budget_and_rounds(name, kw, B, b):
    ctx, ce = make_ctx(seed=3)
    sel = BATCH_SELECTORS[name](batch_size=b, **kw)
    r, o = run(sel, ctx, ce, B, k=10)
    assert o.calls == B
    assert o.rounds == math.ceil(B / b)
    assert o.round_sizes == [b] * (B // b) + ([B % b] if B % b else [])
    assert sorted(r.tolist()) == list(range(ctx.n))


@pytest.mark.parametrize("name", list(BATCH_SELECTORS))
def test_no_doc_scored_twice_and_budget_up_to_pool(name):
    ctx, ce = make_ctx(n=30, seed=4)
    calls = []
    o = CachedOracle(30, ce)
    orig = o._compute
    o._compute = lambda idx: (calls.extend(idx), orig(idx))[1]
    BATCH_SELECTORS[name](batch_size=4).select(ctx, o, 10)
    assert len(calls) == len(set(calls)) == 30            # whole pool, each exactly once


def test_random_boundary_deterministic_and_band():
    ctx, ce = make_ctx(seed=5)
    o1 = run(MorphRandomBoundary(batch_size=4, seed=7), ctx, ce, 12, 10)[1]
    o2 = run(MorphRandomBoundary(batch_size=4, seed=7), ctx, ce, 12, 10)[1]
    o3 = run(MorphRandomBoundary(batch_size=4, seed=8), ctx, ce, 12, 10)[1]
    assert list(o1.observed) == list(o2.observed)
    assert list(o1.observed) != list(o3.observed)
    # The first batch comes from ranks [0, 2k) of the initial posterior mean = MIPS order (query
    # residual is 0), i.e. pool positions < 20.
    assert all(d < 20 for d in batches(o1)[0])


def test_random_boundary_seeded_by_qid_not_call_order():
    import dataclasses
    ctx, ce = make_ctx(seed=5)
    other = dataclasses.replace(ctx, qid=str(ctx.qid) + "_other")
    sel = MorphRandomBoundary(batch_size=4, seed=7)
    a = list(run(sel, ctx, ce, 8, 10)[1].observed)
    run(sel, other, ce, 8, 10)                             # interleave another query
    assert list(run(sel, ctx, ce, 8, 10)[1].observed) == a  # same qid -> same draws
    assert list(run(sel, other, ce, 8, 10)[1].observed) != a


def test_kb_never_overwrites_real_observations():
    ctx, ce = make_ctx(seed=6)
    o = run(MorphKrigingBeliever(batch_size=4, fantasy="optimistic"), ctx, ce, 16, 10)[1]
    for d, v in o.observed.items():
        assert v == ce[d]


# ---- diversity property -----------------------------------------------------------------

def mean_within_batch_similarity(o, K):
    sims = []
    for bt in batches(o):
        if len(bt) > 1:
            sub = K[np.ix_(bt, bt)]
            sims.append(sub[np.triu_indices(len(bt), 1)].mean())
    return float(np.mean(sims))


def test_kb_batches_more_diverse_than_topb():
    kb, tb = [], []
    for seed in range(8):
        ctx, ce = make_ctx(n=80, n_clusters=8, spread=0.08, seed=100 + seed)
        K = vmf_kernel(ctx.doc_emb, median_kappa(ctx.doc_emb))
        kb.append(mean_within_batch_similarity(run(MorphKrigingBeliever(batch_size=4), ctx, ce, 24, 10)[1], K))
        tb.append(mean_within_batch_similarity(run(MorphTopB(batch_size=4), ctx, ce, 24, 10)[1], K))
    assert np.mean(kb) < np.mean(tb), (np.mean(kb), np.mean(tb))


# ---- GP correctness ---------------------------------------------------------------------

def naive_alg1(gp, A, y):
    """Dense reference implementation of paper Alg. 1 (explicit inverse)."""
    A = list(A); y = np.asarray(y, float)
    e = y - gp.mu0[A]
    m = e.mean()
    s = e.std() if len(e) >= 2 else 0.0
    s = 1.0 if s < 1e-8 else s
    z = (e - m) / s
    Kinv = np.linalg.inv(gp.K[np.ix_(A, A)] + gp.noise * np.eye(len(A)))
    Kca = gp.K[: gp.n, A]
    mu_t = Kca @ Kinv @ z
    var_t = np.diag(gp.K)[: gp.n] - np.einsum("ij,jk,ik->i", Kca, Kinv, Kca)
    return gp.mu0[: gp.n] + m + s * mu_t, s ** 2 * np.clip(var_t, 0, None)


@pytest.mark.parametrize("n_obs", [0, 1, 5, 15])
def test_gp_matches_naive_reference(n_obs):
    ctx, ce = make_ctx(seed=9)
    gp = ResidualGP(ctx.doc_emb, ctx.q_emb)
    rng = np.random.default_rng(0)
    obs = rng.choice(ctx.n, n_obs, replace=False).tolist()
    A = [gp.query_index] + obs
    y = [1.0] + ce[obs].tolist()
    mu, var = gp.posterior(A, y)
    mu_r, var_r = naive_alg1(gp, A, y)
    np.testing.assert_allclose(mu, mu_r, atol=1e-6)
    np.testing.assert_allclose(var, var_r, atol=1e-6)


def test_gp_interpolates_observations():
    ctx, ce = make_ctx(seed=10)
    gp = ResidualGP(ctx.doc_emb, ctx.q_emb)
    obs = [0, 7, 19, 33, 50]
    mu, var = gp.posterior([gp.query_index] + obs, [1.0] + ce[obs].tolist())
    np.testing.assert_allclose(mu[obs], ce[obs], atol=2e-2)
    assert np.all(var[obs] < 0.05 * np.median(var))


def test_gp_query_only_posterior_is_prior_mean():
    ctx, _ = make_ctx(seed=11)
    gp = ResidualGP(ctx.doc_emb, ctx.q_emb, mu0_query=1.0)
    mu, var = gp.posterior([gp.query_index], [1.0])
    np.testing.assert_allclose(mu, gp.mu0[: gp.n])          # residual 0 -> prior unchanged
    assert np.all(var <= 1.0 + 1e-12) and np.all(var >= 0)
