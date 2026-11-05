"""
Retrieval behind the API, driven by a frozen RetrievalConfig.

Reuses the repo's index artifacts (data/index/: gzipped chunks + float16 embeddings)
and the BM25 code in retrieval/sparse.py, so the service, the Streamlit dashboard and
the benchmark all score the same chunks. Dense search here is exact cosine over the
prebuilt matrix in numpy (a few thousand vectors per company); the Chroma path in
retrieval/dense.py stays as-is for the benchmark and dashboard.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from rag_service.config import COMPANY_NAMES, ROOT, RetrievalConfig

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval import artifacts, sparse  # noqa: E402

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES = 20  # per retriever before RRF, same as retrieval/hybrid.py

# Names people actually type -> ticker. Ordered longest-first at match time.
_ALIASES: dict[str, str] = {
    "apple": "AAPL",
    "iphone": "AAPL",
    "microsoft": "MSFT",
    "azure": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "youtube": "GOOGL",
    "amazon": "AMZN",
    "aws": "AMZN",
    "meta platforms": "META",
    "meta": "META",
    "facebook": "META",
    "instagram": "META",
    "whatsapp": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "jpmorganchase": "JPM",
    "jpmorgan chase": "JPM",
    "chase": "JPM",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "j&j": "JNJ",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "exxon mobil": "XOM",
}
_ALIAS_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    + "|".join(re.escape(a) for a in sorted(_ALIASES, key=len, reverse=True))
    + r")(?:'s|’s)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(r"\b(" + "|".join(COMPANY_NAMES) + r")\b")  # case-sensitive


def detect_tickers(question: str) -> list[str]:
    """Companies named in the question, in order of first mention."""
    found: list[tuple[int, str]] = []
    for m in _ALIAS_RE.finditer(question):
        found.append((m.start(), _ALIASES[m.group(1).lower()]))
    for m in _TICKER_RE.finditer(question):
        found.append((m.start(), m.group(1)))
    out: list[str] = []
    for _, t in sorted(found):
        if t not in out:
            out.append(t)
    return out


@dataclass
class Passage:
    chunk_id: str
    ticker: str
    text: str
    score: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "ticker": self.ticker,
            "text": self.text,
            "score": round(self.score, 6),
            "rank": self.rank,
        }


class Retriever:
    def __init__(self, cfg: RetrievalConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._bm25: dict[str, sparse.BM25Index] = {}
        self._emb: dict[str, np.ndarray] = {}
        self._chunks: dict[str, list[dict[str, Any]]] = {}
        self._encoder = None

    # ---------- loading ----------
    def warm(self) -> None:
        """Load every index up front so /readyz only passes once queries are fast."""
        for t in self.cfg.tickers:
            self._bm25_for(t)
            if self.cfg.strategy != "sparse":
                self._emb_for(t)
        self._bm25_for("ALL")
        if self.cfg.strategy != "sparse":
            self._get_encoder()

    def _chunks_for(self, ticker: str) -> list[dict[str, Any]]:
        if ticker not in self._chunks:
            ch = artifacts.load_chunks(ticker, self.cfg.chunk_size)
            if not ch:
                raise FileNotFoundError(f"no index artifacts for {ticker} @ {self.cfg.chunk_size}")
            self._chunks[ticker] = ch
        return self._chunks[ticker]

    def _bm25_for(self, ticker: str) -> sparse.BM25Index:
        with self._lock:
            if ticker not in self._bm25:
                if ticker == "ALL":
                    allc = [c for t in self.cfg.tickers for c in self._chunks_for(t)]
                    self._bm25[ticker] = sparse.BM25Index.from_chunks(allc)
                else:
                    self._bm25[ticker] = sparse.BM25Index.from_chunks(self._chunks_for(ticker))
            return self._bm25[ticker]

    def _emb_for(self, ticker: str) -> np.ndarray:
        if ticker not in self._emb:
            e = artifacts.load_embeddings(ticker, self.cfg.chunk_size, str(self.cfg.embedding_model))
            if e is None or len(e) != len(self._chunks_for(ticker)):
                raise FileNotFoundError(f"embeddings missing/misaligned for {ticker}")
            self._emb[ticker] = e
        return self._emb[ticker]

    def _get_encoder(self):
        if self._encoder is None:
            from retrieval.embeddings import _load_st_model, _normalize_model_key

            self._encoder = _load_st_model(_normalize_model_key(str(self.cfg.embedding_model)))
        return self._encoder

    @property
    def ready(self) -> bool:
        need_emb = self.cfg.strategy != "sparse"
        return all(t in self._bm25 for t in self.cfg.tickers) and (
            not need_emb or (self._encoder is not None and len(self._emb) == len(self.cfg.tickers))
        )

    # ---------- search ----------
    def _sparse(self, q: str, ticker: str, k: int) -> list[tuple[str, str, str, float]]:
        hits = self._bm25_for(ticker).query(q, top_k=k)
        return [(h["chunk_id"], h["ticker"], h["text"], h["score"]) for h in hits]

    def _dense(self, q: str, ticker: str, k: int) -> list[tuple[str, str, str, float]]:
        enc = self._get_encoder()
        qv = np.asarray(enc.encode([q], normalize_embeddings=True)[0], dtype=np.float32)
        tickers = self.cfg.tickers if ticker == "ALL" else (ticker,)
        rows: list[tuple[str, str, str, float]] = []
        for t in tickers:
            sims = self._emb_for(t) @ qv
            top = np.argsort(-sims)[:k]
            ch = self._chunks_for(t)
            rows += [(ch[i]["chunk_id"], t, ch[i]["text"], float(sims[i])) for i in top]
        return sorted(rows, key=lambda r: -r[3])[:k]

    def _search_one(self, q: str, ticker: str, k: int) -> list[tuple[str, str, str, float]]:
        if self.cfg.strategy == "sparse":
            return self._sparse(q, ticker, k)
        if self.cfg.strategy == "dense":
            return self._dense(q, ticker, k)
        fused: dict[str, list[Any]] = {}
        for hits in (self._dense(q, ticker, CANDIDATES), self._sparse(q, ticker, CANDIDATES)):
            for r, (cid, t, text, _) in enumerate(hits, start=1):
                row = fused.setdefault(cid, [cid, t, text, 0.0])
                row[3] += 1.0 / (r + RRF_K)
        return [tuple(r) for r in sorted(fused.values(), key=lambda r: -r[3])[:k]]  # type: ignore[misc]

    def search(
        self, question: str, tickers: list[str] | None = None, top_k: int | None = None
    ) -> tuple[list[Passage], list[str]]:
        """
        Returns (passages, scope). With one company -> that company's index; several
        (comparisons) -> top_k from each, interleaved; none -> one index over all ten.
        """
        k = top_k or self.cfg.top_k
        scope = [t for t in (tickers or []) if t in self.cfg.tickers]
        if not scope:
            rows = self._search_one(question, "ALL", k)
        elif len(scope) == 1:
            rows = self._search_one(question, scope[0], k)
        else:
            per = [self._search_one(question, t, k) for t in scope]
            rows = [r for group in zip(*per, strict=False) for r in group][: max(k, 3 * len(scope))]
        passages = [
            Passage(chunk_id=c, ticker=t, text=x, score=s, rank=i)
            for i, (c, t, x, s) in enumerate(rows, start=1)
        ]
        return passages, scope or ["ALL"]
