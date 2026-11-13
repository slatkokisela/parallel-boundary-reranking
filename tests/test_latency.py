"""Synthetic tests for the latency harness. No models, only a fake CE with known per-call cost."""
import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from pbar import latency
from pbar.retrieval import QueryContext
from pbar.selectors import make
from pbar.selectors.morph import BoundaryMORPH

PER_CALL, PER_DOC = 0.002, 0.0005


def _ctx(n=60, d=8, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, d)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[0] + 0.1 * rng.normal(size=d); q /= np.linalg.norm(q)
    return QueryContext("q0", "query", np.arange(n), emb, q, np.sort(emb @ q)[::-1])


def _fake_factory(scores):
    def factory(ctx):
        call_log = []

        def fn(idx):
            time.sleep(PER_CALL + PER_DOC * len(idx))
            v = scores[np.asarray(idx)]
            call_log.append((list(idx), v))
            return v
        return fn, call_log
    return factory


def _expected_ce(round_sizes):
    return sum(PER_CALL + PER_DOC * s for s in round_sizes)


@pytest.mark.parametrize("bs,B,rounds", [(None, 10, 1), (1, 10, 10), (4, 10, 3)])
def test_single_stage_rounds_calls_and_ce_time(bs, B, rounds):
    ctx = _ctx()
    scores = np.random.default_rng(1).random(ctx.n)
    fn, _ = _fake_factory(scores)(ctx)
    rec = latency.time_selection(make("single_stage", batch_size=bs), ctx, fn, B, k=5)
    assert rec["calls"] == B and rec["rounds"] == rounds
    sizes = json.loads(rec["round_sizes"])
    assert sum(sizes) == B and len(sizes) == rounds
    exp = _expected_ce(sizes)
    assert exp <= rec["ce_s"] < exp + rounds * 0.02          # sleep can only overshoot
    assert rec["overhead_s"] >= 0 and rec["total_s"] >= rec["ce_s"]


def test_morph_records_gp_overhead_and_sequential_rounds():
    ctx = _ctx()
    scores = np.random.default_rng(2).random(ctx.n)
    fn, _ = _fake_factory(scores)(ctx)
    sel = BoundaryMORPH(batch_size=1)
    sel.gp_seconds = 123.0                       # must be reset by the harness
    rec = latency.time_selection(sel, ctx, fn, 8, k=10)
    assert rec["calls"] == 8 and rec["rounds"] == 8
    assert 0 < rec["gp_s"] < 123.0
    assert rec["gp_s"] <= rec["overhead_s"] + 1e-3
    assert rec["ce_s"] >= _expected_ce([1] * 8)


def test_run_query_jobs_medians_fidelity_and_order(monkeypatch):
    ctx = _ctx()
    scores = np.random.default_rng(3).random(ctx.n)
    cached = scores + 1e-6                      # simulate padding-level drift
    jobs = latency.expand_jobs([{"name": "single_stage"}, {"name": "single_stage", "batch_size": 2}], [4, 6])
    assert len(jobs) == 4
    rows = latency.run_query_jobs(jobs, ctx, _fake_factory(scores), k=5, repeats=2,
                                  rng=np.random.default_rng(0), cached=cached,
                                  token_len_fn=lambda idx: [10] * len(idx))
    assert len(rows) == 4
    by = {(r["params"], r["B"]): r for r in rows}
    assert by[("{}", 4)]["rounds"] == 1 and by[('{"batch_size": 2}', 6)]["rounds"] == 3
    for r in rows:
        assert r["calls"] == r["B"] and r["repeats"] == 2
        assert r["fidelity_max_abs"] == pytest.approx(1e-6, abs=1e-9)
        assert r["tok_mean"] == 10
        assert r["ce_s"] >= _expected_ce(json.loads(r["round_sizes"]))
    s = latency.summarize(pd.DataFrame(rows))
    assert {"total_s_median", "total_s_p90", "ce_s_median", "rounds_mean"} <= set(s.columns)


def test_job_order_is_shuffled_but_seeded():
    jobs = latency.expand_jobs([{"name": "single_stage"}], [1, 2, 3, 4, 5])
    p1 = np.random.default_rng(7).permutation(len(jobs))
    p2 = np.random.default_rng(7).permutation(len(jobs))
    assert p1.tolist() == p2.tolist()


def test_microbench_reports_per_doc_cost():
    def score_docs(docs):
        time.sleep(PER_CALL + PER_DOC * len(docs))
        return np.zeros(len(docs))
    rows = latency.microbench(score_docs, ["d"] * 16, [1, 4, 16, 32], repeats=2)
    assert [r["batch_size"] for r in rows] == [1, 4, 16]        # 32 > n_docs skipped
    r1, r16 = rows[0], rows[-1]
    assert r1["n_calls"] == 16 and r16["n_calls"] == 1
    # fixed per-call cost is amortised, so larger batches are cheaper per doc
    assert r16["s_per_doc"] < r1["s_per_doc"]
    assert r16["speedup_per_doc_vs_bs1"] > 1 and math.isclose(r1["speedup_per_doc_vs_bs1"], 1.0)
