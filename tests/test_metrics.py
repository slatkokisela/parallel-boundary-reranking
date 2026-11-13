import math

import numpy as np
import pytest

from pbar import metrics, stats


REL = {"a": 2, "b": 1, "c": 1, "d": 0}
POOL = ["a", "b", "c", "d", "e"]


def test_ncg_perfect_and_partial():
    assert metrics.ncg_at_k(["a", "b"], REL, 2, POOL) == 1.0
    assert metrics.ncg_at_k(["b", "c"], REL, 2, POOL) == pytest.approx(2 / 3)
    assert metrics.ncg_at_k(["d", "e"], REL, 2, POOL) == 0.0


def test_ncg_is_order_invariant_within_k():
    assert metrics.ncg_at_k(["b", "a", "e"], REL, 2, POOL) == metrics.ncg_at_k(["a", "b"], REL, 2, POOL)


def test_ncg_ideal_restricted_to_pool():
    # "a" is missing from the pool, so the pool ideal is lower than the corpus ideal
    pool = ["b", "c", "d"]
    assert metrics.ncg_at_k(["b", "c"], REL, 2, pool) == 1.0
    assert metrics.ncg_at_k(["b", "c"], REL, 2, list(REL)) == pytest.approx(2 / 3)


def test_ncg_undefined_without_relevant():
    assert math.isnan(metrics.ncg_at_k(["d"], REL, 1, ["d", "e"]))


def test_ncg_rejects_duplicates():
    with pytest.raises(ValueError):
        metrics.ncg_at_k(["a", "a"], REL, 2, POOL)


def test_ndcg_known_value():
    # ranking c, a with gains 1, 2 -> dcg = 1 + 2/log2(3); ideal = 2 + 1/log2(3)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(["c", "a"], REL, 2, POOL) == pytest.approx(expected)
    assert metrics.ndcg_at_k(["a", "b"], REL, 2, POOL) == pytest.approx(1.0)


def test_recall_of_pool():
    assert metrics.recall_of_pool(["a", "x"], REL) == pytest.approx(1 / 3)


def test_bootstrap_and_holm():
    m, lo, hi = stats.bootstrap_ci(np.ones(50))
    assert m == lo == hi == 1.0
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
