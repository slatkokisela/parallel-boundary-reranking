import numpy as np

from pbar.data import Dataset, make_query_split
from pbar.experiment import build_pool_cache, load_pool_cache
from pbar.retrieval import build_pools


def test_query_split_deterministic_disjoint_exact_size():
    src = {f"q{i}": {"d": 1} for i in range(50)}
    t1, e1 = make_query_split(src, 0.4, seed=0)
    t2, e2 = make_query_split(src, 0.4, seed=0)
    assert t1 == t2 and e1 == e2
    assert len(t1) == 20 and len(e1) == 30 and not set(t1) & set(e1)
    assert set(make_query_split(src, 0.4, seed=1)[0]) != set(t1)   # seed changes the partition


def test_pool_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(300, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    qids = ["a", "b", "c"]
    queries = {q: q for q in qids}
    q_emb = corpus[:3] + 0.05
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    ds = Dataset("x", [str(i) for i in range(300)], [""] * 300, queries, {})
    path = tmp_path / "pools.npz"
    build_pool_cache(path, qids, ds, q_emb, corpus, 40)
    ref = build_pools(qids, queries, q_emb, corpus, 30)
    got = load_pool_cache(path, ["c", "a"], queries, 30)          # subset, reordered, smaller N
    for g, r in zip(got, [ref[2], ref[0]]):
        assert g.qid == r.qid and g.doc_idx.tolist() == r.doc_idx.tolist()
        assert np.allclose(g.mips, r.mips)
        assert np.allclose(g.doc_emb, r.doc_emb, atol=2e-3)        # fp16 storage
        assert np.allclose(np.linalg.norm(g.doc_emb, axis=1), 1.0, atol=1e-5)
