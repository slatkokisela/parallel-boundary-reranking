"""Live latency measurement (M4).

Quality experiments replay cached CE scores. This module runs the *live* CE through a
LiveOracle and measures what each method actually costs on this machine:
  total wall-clock per query = CE time (summed over rounds) + algorithm overhead
One oracle round = one batched CE forward pass (`batch_size = len(idx)`).

Methodology:
  * torch.set_num_threads is fixed from the config and recorded;
  * warm-up queries are run first and discarded;
  * job order (method x B) is shuffled per query and repeat (seeded), so slow drift or
    thermal effects do not favour one method;
  * each (query, job) is repeated `repeats` times and the median is reported;
  * fidelity: live scores are compared with the cached scores (max abs diff). Batch
    padding can cause tiny differences, so these are recorded, not asserted.
A micro-benchmark measures CE seconds per call and per doc against batch size on a fixed doc set.
It answers directly whether batching is faster on this hardware.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import selectors
from .oracle import LiveOracle

log = logging.getLogger(__name__)

ScoreFn = Callable[[list[int]], np.ndarray]


# ---------------------------------------------------------------- core timing (model-free)
def time_selection(selector, ctx, score_fn: ScoreFn, budget: int, k: int) -> dict:
    """Run one selector on one query against a live score function and time it."""
    if hasattr(selector, "gp_seconds"):
        selector.gp_seconds = 0.0
    oracle = LiveOracle(budget, score_fn)
    t0 = time.perf_counter()
    ranking = selector.select(ctx, oracle, k)
    total = time.perf_counter() - t0
    return {
        "total_s": total,
        "ce_s": oracle.ce_seconds,
        "overhead_s": total - oracle.ce_seconds,
        "gp_s": float(getattr(selector, "gp_seconds", float("nan"))),
        "rounds": oracle.rounds,
        "calls": oracle.calls,
        "round_sizes": json.dumps(oracle.round_sizes),
        "_ranking": np.asarray(ranking),
    }


def run_query_jobs(jobs: list[dict], ctx, score_fn_factory, k: int, repeats: int,
                   rng: np.random.Generator, cached: np.ndarray | None = None,
                   token_len_fn: Callable[[list[int]], list[int]] | None = None) -> list[dict]:
    """Time every job on one query.

    jobs: [{"method": name, "params": {...}, "B": int}]
    score_fn_factory(ctx) -> (score_fn, call_log). call_log is a list that score_fn appends
    (idx, values) to. Returns one row per job with medians over repeats.
    """
    samples: dict[int, list[dict]] = {j: [] for j in range(len(jobs))}
    for rep in range(repeats):
        for j in rng.permutation(len(jobs)):
            job = jobs[j]
            sel = selectors.make(job["method"], **job["params"])
            fn, call_log = score_fn_factory(ctx)
            rec = time_selection(sel, ctx, fn, job["B"], k)
            rec.pop("_ranking")
            if cached is not None and call_log:
                rec["fidelity_max_abs"] = max(
                    float(np.max(np.abs(np.asarray(v) - cached[np.asarray(i)]))) for i, v in call_log)
            if token_len_fn is not None and rep == 0 and call_log:
                lens = [token_len_fn(list(i)) for i, _ in call_log]
                rec["tok_max_per_round_mean"] = float(np.mean([max(l) for l in lens]))
                rec["tok_mean"] = float(np.mean([x for l in lens for x in l]))
            samples[j].append(rec)
    rows = []
    for j, recs in samples.items():
        row = {"qid": ctx.qid, "method": jobs[j]["method"],
               "params": json.dumps(jobs[j]["params"], sort_keys=True), "B": jobs[j]["B"], "k": k}
        for key in ("total_s", "ce_s", "overhead_s", "gp_s"):
            row[key] = float(np.median([r[key] for r in recs]))
        row["rounds"], row["calls"], row["round_sizes"] = recs[0]["rounds"], recs[0]["calls"], recs[0]["round_sizes"]
        row["fidelity_max_abs"] = max((r.get("fidelity_max_abs", float("nan")) for r in recs), default=float("nan"))
        for key in ("tok_max_per_round_mean", "tok_mean"):
            row[key] = recs[0].get(key, float("nan"))
        row["repeats"] = len(recs)
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def p90(x):
        return float(np.quantile(x, 0.9))
    g = df.groupby(["method", "params", "B"], dropna=False)
    return g.agg(
        n_q=("qid", "nunique"),
        total_s_median=("total_s", "median"), total_s_p90=("total_s", p90), total_s_mean=("total_s", "mean"),
        ce_s_median=("ce_s", "median"), ce_s_p90=("ce_s", p90),
        overhead_s_median=("overhead_s", "median"), overhead_s_p90=("overhead_s", p90),
        gp_s_median=("gp_s", "median"),
        rounds_mean=("rounds", "mean"), calls_mean=("calls", "mean"),
        fidelity_max_abs=("fidelity_max_abs", "max"),
        tok_max_per_round_mean=("tok_max_per_round_mean", "mean"),
    ).reset_index()


# ---------------------------------------------------------------- live CE helpers
def make_live_factory(scorer, doc_texts: list[str]):
    def factory(ctx):
        call_log: list = []

        def fn(idx: list[int]) -> np.ndarray:
            docs = [doc_texts[ctx.doc_idx[i]] for i in idx]
            vals = scorer.score(ctx.query, docs, batch_size=len(idx))
            call_log.append((list(idx), vals))
            return vals
        return fn, call_log
    return factory


def make_token_len_fn(scorer, ctx, doc_texts, max_length):
    tok = getattr(scorer.model, "tokenizer", None)
    if tok is None:
        return None

    def fn(idx):
        docs = [doc_texts[ctx.doc_idx[i]] for i in idx]
        enc = tok([ctx.query] * len(docs), docs, truncation=True, max_length=max_length)
        return [len(x) for x in enc["input_ids"]]
    return fn


def microbench(score_docs: Callable[[list[str]], np.ndarray], docs: list[str],
               batch_sizes: list[int], repeats: int) -> list[dict]:
    """Score the same fixed docs, split into calls of size bs. Report seconds per call and per doc."""
    rows = []
    for bs in batch_sizes:
        if bs > len(docs):
            continue
        n_calls = len(docs) // bs
        used = docs[: n_calls * bs]
        totals = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for c in range(n_calls):
                score_docs(used[c * bs:(c + 1) * bs])
            totals.append(time.perf_counter() - t0)
        med = float(np.median(totals))
        rows.append({"batch_size": bs, "n_docs": len(used), "n_calls": n_calls,
                     "total_s_median": med, "s_per_call": med / n_calls, "s_per_doc": med / len(used),
                     "total_s_min": float(np.min(totals)), "repeats": repeats})
    base = next((r["s_per_doc"] for r in rows if r["batch_size"] == 1), None)
    for r in rows:
        r["speedup_per_doc_vs_bs1"] = (base / r["s_per_doc"]) if base else float("nan")
    return rows


def expand_jobs(methods: list[dict], budgets: list[int]) -> list[dict]:
    jobs = []
    for m in methods:
        params = {k: v for k, v in m.items() if k != "name"}
        for B in budgets:
            jobs.append({"method": m["name"], "params": params, "B": int(B)})
    return jobs


# ---------------------------------------------------------------- entry point
def run(cfg: dict) -> pd.DataFrame:
    import torch

    from .crossencoder import CEScorer
    from .experiment import config_hash, prepare

    lc = cfg["latency"]
    torch.set_num_threads(int(lc["num_threads"]))
    cfg["_hash"] = config_hash(cfg)
    # calibrated priors are fitted on the full split (cross-fit), then the timed subset is taken
    from .experiment import attach_rank_priors
    full_cfg = dict(cfg, max_queries=None)
    ds, ctxs, ce = prepare(full_cfg)
    ctxs, calib_info = attach_rank_priors(full_cfg, ctxs, ce)
    if cfg.get("max_queries"):
        ctxs = ctxs[: cfg["max_queries"]]
    print(f"latency: timing {len(ctxs)} queries; calibration={calib_info}", flush=True)
    cc = cfg["cross_encoder"]
    scorer = CEScorer(cc["model"], cc.get("max_length", 512), cc.get("batch_size", 32))
    factory = make_live_factory(scorer, ds.doc_texts)
    jobs = expand_jobs(cfg["methods"], lc["budgets"])
    rng = np.random.default_rng(cfg.get("seed", 0))
    out = Path(cfg["results_dir"]) / f"{cfg['name']}_{cfg['_hash']}"
    out.mkdir(parents=True, exist_ok=True)

    # micro-benchmark (fixed random pool docs from the first queries)
    mb = lc.get("microbench")
    mb_rows = []
    if mb:
        mrng = np.random.default_rng(cfg.get("seed", 0))
        for ctx in ctxs[: mb.get("n_queries", 1)]:
            pick = mrng.choice(ctx.n, size=min(mb["n_docs"], ctx.n), replace=False)
            docs = [ds.doc_texts[ctx.doc_idx[i]] for i in pick]
            scorer.score(ctx.query, docs[:8], batch_size=8)   # warm-up
            for r in microbench(lambda d, q=ctx.query: scorer.score(q, d, batch_size=len(d)),
                                docs, mb["batch_sizes"], mb["repeats"]):
                mb_rows.append({"qid": ctx.qid, **r})
        pd.DataFrame(mb_rows).to_csv(out / "microbench.csv", index=False)

    # warm-up (discarded)
    for ctx in ctxs[: lc.get("warmup_queries", 0)]:
        run_query_jobs(jobs, ctx, factory, lc["k"], 1, rng)

    rows = []
    for qi, ctx in enumerate(ctxs):
        rows += run_query_jobs(jobs, ctx, factory, lc["k"], lc["repeats"], rng, cached=ce[ctx.qid],
                               token_len_fn=make_token_len_fn(scorer, ctx, ds.doc_texts,
                                                              cc.get("max_length", 512)))
        log.info("latency: query %d/%d done", qi + 1, len(ctxs))
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_query.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    meta = {"config": cfg, "torch": torch.__version__, "num_threads": torch.get_num_threads(),
            "cpu": platform.processor(), "platform": platform.platform(),
            "python": platform.python_version(), "max_length": cc.get("max_length", 512),
            "cuda": torch.cuda.is_available(), "n_queries": len(ctxs)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s", out)
    return summary
