import numpy as np

from pbar.crossencoder import CECache, score_pools
from pbar.retrieval import QueryContext


class FakeScorer:
    def __init__(self):
        self.calls = 0

    def score(self, query, docs):
        self.calls += len(docs)
        return np.array([len(d) / 100 for d in docs])


def _ctx(qid, idx):
    idx = np.asarray(idx)
    return QueryContext(qid, "q", idx, np.zeros((len(idx), 2)), np.zeros(2), np.zeros(len(idx)))


def test_cache_is_keyed_by_doc_not_position(tmp_path):
    texts = ["a" * i for i in range(1, 11)]
    path = tmp_path / "ce.npz"
    s1 = FakeScorer()
    out1 = score_pools(lambda: s1, [_ctx("q1", [3, 1, 4])], texts, path)
    assert out1["q1"].tolist() == [0.04, 0.02, 0.05] and s1.calls == 3
    # different pool order and a larger pool: only the new doc gets scored
    s2 = FakeScorer()
    out2 = score_pools(lambda: s2, [_ctx("q1", [4, 3, 9, 1])], texts, path)
    assert out2["q1"].tolist() == [0.05, 0.04, 0.10, 0.02] and s2.calls == 1


def test_no_scorer_loaded_when_cache_warm(tmp_path):
    path = tmp_path / "ce.npz"
    c = CECache(path)
    c.put("q", [0, 1], [0.5, 0.6])
    c.save()

    def boom():
        raise AssertionError("scorer should not be constructed")
    assert score_pools(boom, [_ctx("q", [1, 0])], ["x", "y"], path)["q"].tolist() == [0.6, 0.5]


def test_ce_scorer_sort_restores_order():
    from pbar.crossencoder import CEScorer
    sc = CEScorer.__new__(CEScorer)
    sc.batch_size = 2

    class M:
        def predict(self, pairs, **kw):
            return np.array([len(d) for _, d in pairs], dtype=float)
    sc.model = M()
    docs = ["ccc", "a", "bb", "dddd"]
    assert sc.score("q", docs).tolist() == [3, 1, 2, 4]
