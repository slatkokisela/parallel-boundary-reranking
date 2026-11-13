"""BEIR NFCorpus download and parsing.

Files are parsed with our own code (jsonl / tsv). The `beir` package is not needed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
SPLITS = ("train", "dev", "test")


@dataclass
class Dataset:
    name: str
    doc_ids: list[str]                  # stable order defines integer doc index
    doc_texts: list[str]                # "title text"
    queries: dict[str, str]             # qid -> text (all splits)
    qrels: dict[str, dict[str, dict[str, int]]]  # split -> qid -> doc_id -> grade

    def split_qids(self, split: str) -> list[str]:
        """Queries of a split that have at least one positive judgment, sorted."""
        q = self.qrels[split]
        return sorted(qid for qid, rels in q.items()
                      if qid in self.queries and any(g > 0 for g in rels.values()))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    out = data_dir / name
    if not zip_path.exists():
        log.info("downloading %s", BEIR_URL.format(name=name))
        try:
            urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
        except OSError as e:  # e.g. TLS chain not trusted by this Python build
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"download failed ({e}). Fetch it manually, e.g.\n  curl -fL -o {zip_path} "
                f"{BEIR_URL.format(name=name)}") from e
    if not (out / "corpus.jsonl").exists():
        with zipfile.ZipFile(zip_path) as zf:
            root = data_dir.resolve()
            for member in zf.namelist():  # refuse path traversal
                if not (data_dir / member).resolve().is_relative_to(root):
                    raise ValueError(f"unsafe path in archive: {member}")
            zf.extractall(data_dir)
    (data_dir / f"{name}.sha256").write_text(_sha256(zip_path) + "\n")
    return out


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def make_query_split(qrels_src: dict, tune_frac: float, seed: int = 0) -> tuple[dict, dict]:
    """Deterministic tune/eval partition of one qrels split: order qids by crc32(seed:qid) and put
    the first round(tune_frac*n) in tune. Used when a dataset ships only a test split."""
    import zlib
    qids = sorted(qrels_src, key=lambda q: (zlib.crc32(f"{seed}:{q}".encode()), q))
    n_tune = round(tune_frac * len(qids))
    tune = {q: qrels_src[q] for q in qids[:n_tune]}
    return tune, {q: qrels_src[q] for q in qids[n_tune:]}


def load(name: str, data_dir: str | Path, split_scheme: dict | None = None) -> Dataset:
    """split_scheme, e.g. {source: test, tune_frac: 0.4, seed: 0}, adds 'tune' and 'eval' splits."""
    root = download(name, Path(data_dir))
    doc_ids, doc_texts = [], []
    for row in _read_jsonl(root / "corpus.jsonl"):
        doc_ids.append(str(row["_id"]))
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        doc_texts.append(f"{title}. {text}" if title else text)
    queries = {str(r["_id"]): r["text"] for r in _read_jsonl(root / "queries.jsonl")}
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for split in SPLITS:
        p = root / "qrels" / f"{split}.tsv"
        if not p.exists():
            continue
        split_q: dict[str, dict[str, int]] = {}
        with open(p, encoding="utf-8") as f:
            next(f)  # header: query-id corpus-id score
            for line in f:
                qid, did, grade = line.rstrip("\n").split("\t")
                split_q.setdefault(qid, {})[did] = int(grade)
        qrels[split] = split_q
    if split_scheme:
        qrels["tune"], qrels["eval"] = make_query_split(
            qrels[split_scheme["source"]], split_scheme["tune_frac"], split_scheme.get("seed", 0))
    return Dataset(name, doc_ids, doc_texts, queries, qrels)
