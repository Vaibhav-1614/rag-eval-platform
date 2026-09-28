"""
BM25 retrieval via rank_bm25; save/load index to models/{ticker}_{chunk_size}_bm25.pkl
"""
from __future__ import annotations

import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

from retrieval import artifacts

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def index_path(ticker: str, chunk_size: int, settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    paths = settings.get("paths") or {}
    base = project_root() / paths.get("bm25_dir", "models")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{ticker.upper()}_{int(chunk_size)}_bm25.pkl"


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self.chunks: list[dict[str, Any]] = chunks or []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if self.chunks:
            self._build()

    def _build(self) -> None:
        self._corpus_tokens = [tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    @classmethod
    def from_chunks(cls, chunks: list[dict[str, Any]]) -> BM25Index:
        return cls(chunks)

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._bm25 or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(text))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out: list[dict[str, Any]] = []
        for idx, score in ranked:
            c = self.chunks[idx]
            out.append(
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "ticker": c["ticker"],
                    "score": float(score),
                    "source": "bm25",
                }
            )
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self.chunks}
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 index: %s", path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        inst = cls(payload.get("chunks", []))
        return inst


_loaded: dict[str, BM25Index] = {}


def load_index(ticker: str, chunk_size: int, settings: dict | None = None) -> BM25Index | None:
    """
    Load a BM25 index once per process (building BM25Okapi re-tokenises the whole
    corpus, so doing it per query dominated sparse latency). Falls back to the
    committed chunk artifacts when the local pickle is absent.
    """
    settings = settings or load_settings()
    p = index_path(ticker, chunk_size, settings)
    key = str(p)
    if key in _loaded:
        return _loaded[key]
    if p.exists():
        idx = BM25Index.load(p)
    else:
        chunks = artifacts.load_chunks(ticker, chunk_size, settings)
        if not chunks:
            return None
        idx = BM25Index.from_chunks(chunks)
    _loaded[key] = idx
    return idx


def build_and_save(chunks: list[dict[str, Any]], settings: dict | None = None) -> Path:
    if not chunks:
        raise ValueError("No chunks")
    ticker = str(chunks[0]["ticker"]).upper()
    cs = int(chunks[0]["chunk_size"])
    idx = BM25Index.from_chunks(chunks)
    p = index_path(ticker, cs, settings)
    idx.save(p)
    _loaded[str(p)] = idx
    artifacts.save_chunks(chunks, settings)
    return p
