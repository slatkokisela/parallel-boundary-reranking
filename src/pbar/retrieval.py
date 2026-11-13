"""Dense first-stage retrieval: corpus embedding cache and MIPS top-N candidate pools."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Everything a selector may see for one query. Pool positions 0..N-1 follow MIPS order."""
    qid: str
    query: str
    doc_idx: np.ndarray     # (N,) corpus indices, MIPS-descending
    emb: np.ndarray         # (N, d) pool embeddings, or the shared corpus matrix if emb_is_corpus
    q_emb: np.ndarray       # (d,) unit-norm
    mips: np.ndarray        # (N,) inner products, descending
    emb_is_corpus: bool = False   # share one corpus matrix across contexts (memory)
    rank_prior: np.ndarray | None = None  # calibrated prior by MIPS rank, fit on other queries

    @property
    def n(self) -> int:
        return len(self.doc_idx)

    @property
    def doc_emb(self) -> np.ndarray:
        """(N, d) unit-norm pool embeddings in pool order."""
        return self.emb[self.doc_idx] if self.emb_is_corpus else self.emb


class DenseRetriever:
    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = "",
                 batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.qp, self.dp, self.bs = query_prefix, doc_prefix, batch_size

    def _enc(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.bs, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 500
                                 ).astype(np.float32)

    def encode_corpus(self, texts: list[str], cache: Path) -> np.ndarray:
        if cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(texts):
                return emb
            log.warning("corpus cache size mismatch, re-encoding")
        emb = self._enc([self.dp + t for t in texts])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._enc([self.qp + t for t in texts])


def build_pools(qids, queries: dict[str, str], q_emb: np.ndarray, corpus_emb: np.ndarray,
                n: int) -> list[QueryContext]:
    n = min(n, corpus_emb.shape[0])
    sims = q_emb @ corpus_emb.T                                  # (Q, D)
    out = []
    for i, qid in enumerate(qids):
        top = np.argpartition(-sims[i], n - 1)[:n]
        # deterministic order: score desc, then corpus index asc
        top = top[np.lexsort((top, -sims[i, top]))]
        out.append(QueryContext(qid, queries[qid], top, corpus_emb, q_emb[i], sims[i, top],
                                emb_is_corpus=True))
    return out
