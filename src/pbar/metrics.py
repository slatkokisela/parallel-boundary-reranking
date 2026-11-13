"""Set-selection and ranking metrics.

nCG@k follows BoundaryMORPH Eq. 5:
    nCG@k = sum_{d in S} rel(d) / sum_{d in S*} rel(d)
S is the returned size-k set and S* the ideal size-k set. `ideal_pool` (the paper's
definition, S* ⊆ C) builds S* from the candidate pool, and `ideal_corpus` from all
judged docs (this also charges first-stage recall loss).
CE-nCG@k uses the same formula with CE scores in place of rel, and S† is the CE-optimal set in C.
nDCG@k (ordering quality) uses linear gains and a log2 discount.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def _topk_gain_sum(gains: Sequence[float], k: int) -> float:
    return float(sum(sorted(gains, reverse=True)[:k]))


def ncg_at_k(selected: Sequence, rel: Mapping, k: int, ideal_candidates: Sequence) -> float:
    """Return nCG@k, or NaN when the ideal gain is 0 (undefined; the caller excludes it)."""
    sel = list(selected)[:k]
    if len(set(sel)) != len(sel):
        raise ValueError("selected contains duplicates")
    gain = sum(max(rel.get(d, 0), 0) for d in sel)
    ideal = _topk_gain_sum([max(rel.get(d, 0), 0) for d in ideal_candidates], k)
    return float("nan") if ideal <= 0 else gain / ideal


def ndcg_at_k(ranking: Sequence, rel: Mapping, k: int, ideal_candidates: Sequence) -> float:
    dcg = sum(max(rel.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(list(ranking)[:k]))
    ideal_gains = sorted((max(rel.get(d, 0), 0) for d in ideal_candidates), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return float("nan") if idcg <= 0 else dcg / idcg


def recall_of_pool(pool: Sequence, rel: Mapping) -> float:
    pos = {d for d, g in rel.items() if g > 0}
    return float("nan") if not pos else len(pos & set(pool)) / len(pos)
