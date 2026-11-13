"""Cross-encoder scoring, with an on-disk cache of CE scores.

The cache is keyed by (qid, corpus doc index), not by pool position. Pools built on
another machine, or with a different N, can reuse it. A small float difference in query
embeddings that swaps two near-tied docs cannot misalign scores. Only missing pairs are scored.
File layout (npz): "i:<qid>" -> int64 corpus indices, "s:<qid>" -> float64 scores.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class CEScorer:
    """Wraps a sentence-transformers CrossEncoder. The outputs are sigmoid probabilities in [0,1]."""

    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 32,
                 device: str | None = None, fp16: bool = False):
        import torch
        from sentence_transformers import CrossEncoder
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device,
                                  activation_fn=torch.nn.Sigmoid())
        self.batch_size = batch_size
        self.fp16 = fp16 and device.startswith("cuda")   # autocast; CPU always runs fp32

    def score(self, query: str, docs: list[str], batch_size: int | None = None,
              sort_by_length: bool = True) -> np.ndarray:
        """Score (query, doc) pairs. With sort_by_length, docs are batched in length order to
        minimise padding, and results are returned in input order."""
        if not docs:
            return np.zeros(0)
        order = np.argsort([len(d) for d in docs], kind="stable") if sort_by_length \
            else np.arange(len(docs))
        import contextlib

        import torch
        ctx = torch.autocast("cuda", dtype=torch.float16) if getattr(self, "fp16", False) \
            else contextlib.nullcontext()
        with ctx:
            out = np.asarray(self.model.predict([(query, docs[i]) for i in order],
                                                batch_size=batch_size or self.batch_size,
                                                show_progress_bar=False), dtype=np.float64)
        res = np.empty_like(out)
        res[order] = out
        return res


class CECache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict[int, float]] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as z:
                for key in z.files:
                    if key.startswith("i:"):
                        qid = key[2:]
                        self.data[qid] = dict(zip(z[key].tolist(), z["s:" + qid].tolist()))

    def missing(self, qid: str, doc_idx) -> list[int]:
        have = self.data.get(qid, {})
        return [int(j) for j in doc_idx if int(j) not in have]

    def put(self, qid: str, doc_idx, scores) -> None:
        self.data.setdefault(qid, {}).update(zip(map(int, doc_idx), map(float, scores)))

    def get(self, qid: str, doc_idx) -> np.ndarray:
        d = self.data[qid]
        return np.array([d[int(j)] for j in doc_idx], dtype=np.float64)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for qid, d in self.data.items():
            idx = np.array(sorted(d), dtype=np.int64)
            arrays["i:" + qid] = idx
            arrays["s:" + qid] = np.array([d[i] for i in idx.tolist()], dtype=np.float64)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(self.path)


def score_pools(scorer_factory, contexts, doc_texts: list[str], cache_path: Path) -> dict[str, np.ndarray]:
    """Return CE scores aligned to each context's pool order, scoring only uncached pairs.
    `scorer_factory()` is called lazily, only when something is missing."""
    from tqdm import tqdm
    cache = CECache(cache_path)
    todo = [(c, cache.missing(c.qid, c.doc_idx)) for c in contexts]
    todo = [(c, m) for c, m in todo if m]
    if todo:
        scorer = scorer_factory()
        for i, (ctx, miss) in enumerate(tqdm(todo, desc="CE scoring pools")):
            cache.put(ctx.qid, miss, scorer.score(ctx.query, [doc_texts[j] for j in miss]))
            if (i + 1) % 25 == 0 or i + 1 == len(todo):   # checkpoint
                cache.save()
    return {c.qid: cache.get(c.qid, c.doc_idx) for c in contexts}
