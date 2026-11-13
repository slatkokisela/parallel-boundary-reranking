import numpy as np
import pytest

from pbar.oracle import BudgetExceeded, CachedOracle
from pbar.retrieval import QueryContext, build_pools
from pbar.selectors import make


def _ctx(n=10, d=4, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, d)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[0]
    return QueryContext("q", "query", np.arange(n), emb, q, np.linspace(1, 0, n))


def test_oracle_counts_calls_and_rounds():
    o = CachedOracle(3, np.arange(10) / 10)
    assert o.score([1, 2]).tolist() == [0.1, 0.2]
    assert o.score([2]).tolist() == [0.2]        # free re-read, no round
    assert (o.calls, o.rounds) == (2, 1)
    o.score([5])
    assert (o.calls, o.rounds, o.round_sizes) == (3, 2, [2, 1])
    with pytest.raises(BudgetExceeded):
        o.score([7])


def test_oracle_dedups_within_request():
    o = CachedOracle(2, np.arange(5))
    o.score([1, 1, 2])
    assert o.calls == 2


def test_single_stage_reranks_mips_top_b_then_tail():
    ctx = _ctx()
    ce = np.array([0.1, 0.9, 0.5, 0.95, 0, 0, 0, 0, 0, 0])
    o = CachedOracle(3, ce)
    r = make("single_stage").select(ctx, o, k=5)
    assert r.tolist() == [1, 2, 0, 3, 4, 5, 6, 7, 8, 9]   # doc 3 unscored -> stays in tail
    assert o.calls == 3 and o.rounds == 1


def test_single_stage_batch_size_changes_rounds_not_result():
    ctx = _ctx()
    ce = np.random.default_rng(1).random(10)
    o1, o4 = CachedOracle(8, ce), CachedOracle(8, ce)
    r1 = make("single_stage", batch_size=1).select(ctx, o1, 5)
    r4 = make("single_stage").select(ctx, o4, 5)
    assert r1.tolist() == r4.tolist() and o1.rounds == 8 and o4.rounds == 1


def test_full_ce_is_ce_sorted():
    ctx = _ctx()
    ce = np.random.default_rng(2).random(10)
    r = make("full_ce").select(ctx, CachedOracle(10, ce), 3)
    assert r.tolist() == np.argsort(-ce, kind="stable").tolist()


def test_build_pools_order_is_mips_descending():
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(50, 8)); corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    q = corpus[:2] + 0.01
    ctxs = build_pools(["a", "b"], {"a": "x", "b": "y"}, q, corpus, 10)
    for c, qe in zip(ctxs, q):
        full = np.argsort(-(corpus @ qe))[:10]
        assert c.doc_idx.tolist() == full.tolist()
        assert np.all(np.diff(c.mips) <= 0)
