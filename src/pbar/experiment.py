"""Config-driven experiment runner (quality simulation with cached CE scores)."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import calibration, data, metrics, selectors, stats
from .crossencoder import CEScorer, score_pools
from .oracle import CachedOracle
from .retrieval import DenseRetriever, QueryContext, build_pools

log = logging.getLogger(__name__)


def _slug(s: str) -> str:
    return s.replace("/", "__")


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:10]


def prepare(cfg: dict):
    """Load the dataset, build the candidate pools and the CE score cache for the configured split."""
    ds = data.load(cfg["dataset"]["name"], cfg["dataset"]["data_dir"], cfg["dataset"].get("split_scheme"))
    qids = ds.split_qids(cfg["split"])
    if cfg.get("max_queries"):
        qids = qids[: cfg["max_queries"]]
    cache = Path(cfg["cache_dir"])
    rc, cc = cfg["retriever"], cfg["cross_encoder"]
    _retr = []

    def retriever():  # loaded lazily; with warm caches no model is loaded at all
        if not _retr:
            _retr.append(DenseRetriever(rc["model"], rc.get("query_prefix", ""),
                                        rc.get("doc_prefix", ""), rc.get("batch_size", 64),
                                        fp16=bool(rc.get("fp16", False))))
        return _retr[0]

    tag = f"{_slug(rc['model'])}{'_fp16' if rc.get('fp16') else ''}"
    emb_path = cache / f"emb_{ds.name}_{tag}.npy"
    qemb_path = cache / f"qemb_{ds.name}_{tag}.npz"

    def corpus_emb():
        e = np.load(emb_path) if emb_path.exists() else retriever().encode_corpus(ds.doc_texts, emb_path)
        assert e.shape[0] == len(ds.doc_ids), "stale corpus embedding cache"
        return e

    pc = cfg.get("pool_cache")
    if pc:   # large corpora: ship only pool-level data (see build_pool_cache)
        assert pc["n"] >= cfg["pool_size"], "pool cache smaller than pool_size"
        pool_path = cache / f"pools_{ds.name}_{tag}_N{pc['n']}.npz"
        if not pool_path.exists():
            all_q = sorted({q for sp in ds.qrels for q in ds.split_qids(sp)})
            build_pool_cache(pool_path, all_q, ds, _query_embeddings(retriever, ds, all_q, qemb_path),
                             corpus_emb(), pc["n"])
        ctxs = load_pool_cache(pool_path, qids, ds.queries, cfg["pool_size"])
    else:
        q_emb = _query_embeddings(retriever, ds, qids, qemb_path)
        ctxs = build_pools(qids, ds.queries, q_emb, corpus_emb(), cfg["pool_size"])
    fp16 = bool(cc.get("fp16", False))
    ce_path = cache / (f"ce_{ds.name}_{cfg['split']}_{_slug(cc['model'])}_L{cc.get('max_length', 512)}"
                       f"{'_fp16' if fp16 else ''}.npz")
    ce = score_pools(lambda: CEScorer(cc["model"], cc.get("max_length", 512), cc.get("batch_size", 32),
                                      fp16=fp16),
                     ctxs, ds.doc_texts, ce_path)
    return ds, ctxs, ce


def build_pool_cache(path: Path, qids, ds, q_emb, corpus_emb, n: int) -> None:
    """Top-n pools for all qids, plus embeddings of only the docs that occur in some pool (fp16)."""
    ctxs = build_pools(qids, ds.queries, q_emb, corpus_emb, n)
    uniq = np.unique(np.concatenate([c.doc_idx for c in ctxs]))
    row = {int(d): i for i, d in enumerate(uniq)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, qids=np.array(qids), doc_idx=np.stack([c.doc_idx for c in ctxs]),
             mips=np.stack([c.mips for c in ctxs]), q_emb=np.asarray(q_emb, dtype=np.float32),
             uniq=uniq, uniq_emb=corpus_emb[uniq].astype(np.float16),
             pos=np.array([[row[int(d)] for d in c.doc_idx] for c in ctxs]))


def load_pool_cache(path: Path, qids, queries, n: int):
    with np.load(path, allow_pickle=False) as z:
        index = {q: i for i, q in enumerate(z["qids"].tolist())}
        emb = z["uniq_emb"].astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)    # re-normalise after fp16 storage
        doc_idx, mips, q_emb, pos = z["doc_idx"], z["mips"], z["q_emb"], z["pos"]
    return [QueryContext(q, queries[q], doc_idx[index[q], :n], emb, q_emb[index[q]], mips[index[q], :n],
                         emb_is_corpus=True, emb_idx=pos[index[q], :n]) for q in qids]


def _query_embeddings(retriever, ds, qids, path: Path) -> np.ndarray:
    """Query embeddings cached by qid, so pools are identical across machines."""
    have = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            have = {k: z[k] for k in z.files}
    miss = [q for q in qids if q not in have]
    if miss:
        have.update(zip(miss, retriever().encode_queries([ds.queries[q] for q in miss])))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **have)
    return np.stack([have[q] for q in qids])


def evaluate_one(ctx, ranking, oracle, ce_all, qrel, ds, k, ndcg_k):
    pool_ids = [ds.doc_ids[j] for j in ctx.doc_idx]
    ranked_ids = [pool_ids[i] for i in ranking]
    sel = ranking[:k]
    ce_gain = {i: float(ce_all[i]) for i in range(ctx.n)}
    obs = set(oracle.observed)
    return {
        "ncg_pool": metrics.ncg_at_k(ranked_ids, qrel, k, pool_ids),
        "ncg_corpus": metrics.ncg_at_k(ranked_ids, qrel, k, list(qrel)),
        "ce_ncg": metrics.ncg_at_k(list(sel), ce_gain, k, list(range(ctx.n))),
        f"ndcg@{ndcg_k}": metrics.ndcg_at_k(ranked_ids, qrel, ndcg_k, list(qrel)),
        "calls": oracle.calls,
        "rounds": oracle.rounds,
        # budget efficiency: share of CE calls spent on docs that end up in the final set
        "frac_calls_in_set": (len(obs & set(sel.tolist())) / len(obs)) if obs else float("nan"),
    }


_W: dict = {}   # per-worker state (set by _init_worker)


def _init_worker(ds, ctxs, ce, qrels, ndcg_k):
    _W.update(ds=ds, ctxs=ctxs, ce=ce, qrels=qrels, ndcg_k=ndcg_k)


def _run_job(job) -> list[dict]:
    """One (method, params, B, k) over all queries. Selectors are deterministic per query."""
    name, params, B, k = job
    ds, ctxs, ce, qrels = _W["ds"], _W["ctxs"], _W["ce"], _W["qrels"]
    sel = selectors.make(name, **params)
    rows = []
    for ctx in ctxs:
        budget = ctx.n if name == "full_ce" else (B or 0)
        oracle = CachedOracle(budget, ce[ctx.qid])
        ranking = np.asarray(sel.select(ctx, oracle, k))
        assert sorted(ranking.tolist()) == list(range(ctx.n)), "ranking must permute pool"
        qrel = qrels[ctx.qid]
        rows.append({"method": name, "params": json.dumps(params, sort_keys=True),
                     "B": budget, "k": k, "qid": ctx.qid,
                     "n_rel": sum(g > 0 for g in qrel.values()),
                     **evaluate_one(ctx, ranking, oracle, ce[ctx.qid], qrel, ds, k, _W["ndcg_k"])})
    return rows


def run(cfg: dict, workers: int = 1) -> pd.DataFrame:
    t0 = time.time()
    cfg["_hash"] = config_hash(cfg)
    ds, ctxs, ce = prepare(cfg)
    ctxs, calib_info = attach_rank_priors(cfg, ctxs, ce)
    ev = cfg["eval"]
    qrels = ds.qrels[cfg["split"]]
    jobs = []
    for m in cfg["methods"]:
        params = {k: v for k, v in m.items() if k != "name"}
        uses_budget = selectors.REGISTRY[m["name"]].uses_budget and m["name"] != "full_ce"
        for B in (ev["budgets"] if uses_budget else [None]):
            for k in ev["k"]:
                jobs.append((m["name"], params, B, k))
    import dataclasses
    light = dataclasses.replace(ds, doc_texts=[])           # workers never need document texts
    init_args = (light, ctxs, ce, qrels, ev.get("ndcg_k", 10))
    if workers > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = "1"          # inherited by spawned workers, avoids oversubscription
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=init_args) as ex:
            results = []
            for i, res in enumerate(ex.map(_run_job, jobs)):   # ordered; logs as jobs complete
                results.append(res)
                log.info("job %d/%d done: %s B=%s k=%s (%.0fs elapsed)", i + 1, len(jobs),
                         jobs[i][0], jobs[i][2], jobs[i][3], time.time() - t0)
    else:
        _init_worker(*init_args)
        results = []
        for i, j in enumerate(jobs):
            results.append(_run_job(j))
            log.info("job %d/%d done: %s B=%s k=%s (%.0fs elapsed)", i + 1, len(jobs), j[0], j[2], j[3],
                     time.time() - t0)
    df = pd.DataFrame([r for rows in results for r in rows])
    out = Path(cfg["results_dir"]) / f"{cfg['name']}_{cfg['_hash']}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_query.csv", index=False)
    summary = summarize(df, ev.get("diffuse_min_rel", 20), cfg.get("seed", 0))
    summary.to_csv(out / "summary.csv", index=False)
    pool_stats = pool_report(ds, ctxs, qrels)
    meta = {"config": cfg, "python": platform.python_version(), "platform": platform.platform(),
            "n_queries": len(ctxs), "pool": pool_stats, "calibration": calib_info, "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s", out)
    return summary


def attach_rank_priors(cfg, ctxs, ce):
    """Attach calibrated rank priors if any method uses prior='calibrated'.

    calibration.mode = "file": load per-qid curves from `path` (npz keyed by qid);
    calibration.mode = "crossfit" (default): K query folds within this split, each fitted on
                       the other folds;
                     = "split": fit on all queries of `fit_split` (e.g. dev) for a test run.
    Only CE scores and MIPS ranks are used; no relevance labels."""
    import dataclasses
    if not any(m.get("prior") == "calibrated" for m in cfg["methods"]):
        return ctxs, None
    cal = cfg.get("calibration", {"mode": "crossfit", "folds": 2})
    n = max(c.n for c in ctxs)
    if cal["mode"] == "crossfit":
        assert len(ctxs) >= 2 * cal.get("folds", 2), "too few queries to cross-fit"
        curves = calibration.crossfit_curves({c.qid: ce[c.qid] for c in ctxs}, n,
                                             cal.get("folds", 2), cfg.get("seed", 0))
    elif cal["mode"] == "file":     # precomputed per-qid curves (e.g. shipped to a remote GPU run)
        with np.load(cal["path"]) as z:
            curves = {c.qid: z[c.qid] for c in ctxs}
    elif cal["mode"] == "split":
        assert cal["fit_split"] != cfg["split"], "fit split must differ from eval split"
        fcfg = dict(cfg, split=cal["fit_split"], max_queries=None)
        _, fctxs, fce = prepare(fcfg)
        curve = calibration.fit_rank_curve([fce[c.qid] for c in fctxs], n)
        curves = {c.qid: curve for c in ctxs}
    else:
        raise ValueError(cal["mode"])
    info = {**cal, "curve_head": [round(float(x), 4) for x in next(iter(curves.values()))[:5]]}
    return [dataclasses.replace(c, rank_prior=curves[c.qid]) for c in ctxs], info


def pool_report(ds, ctxs, qrels) -> dict:
    rec = [metrics.recall_of_pool([ds.doc_ids[j] for j in c.doc_idx], qrels[c.qid]) for c in ctxs]
    nrel = [sum(g > 0 for g in qrels[c.qid].values()) for c in ctxs]
    return {"recall@N_mean": float(np.nanmean(rec)), "n_rel_median": float(np.median(nrel)),
            "n_rel_mean": float(np.mean(nrel)), "n_diffuse_ge20": int(sum(n >= 20 for n in nrel))}


METRICS = ["ncg_pool", "ncg_corpus", "ce_ncg", "frac_calls_in_set", "rounds"]


def summarize(df: pd.DataFrame, diffuse_min_rel: int, seed: int) -> pd.DataFrame:
    out = []
    for subset, sub in [("all", df), (f"diffuse>={diffuse_min_rel}", df[df.n_rel >= diffuse_min_rel])]:
        for key, g in sub.groupby(["method", "params", "B", "k"], dropna=False):
            row = dict(zip(["method", "params", "B", "k"], key), subset=subset, n_q=g.qid.nunique())
            for mtr in METRICS + [c for c in g.columns if c.startswith("ndcg@")]:
                if mtr in g:
                    mean, lo, hi = stats.bootstrap_ci(g[mtr].values, seed=seed, n_boot=2000)
                    row[mtr], row[f"{mtr}_lo"], row[f"{mtr}_hi"] = mean, lo, hi
            out.append(row)
    return pd.DataFrame(out)
