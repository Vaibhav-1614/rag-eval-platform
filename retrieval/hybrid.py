"""
Hybrid retrieval: dense top-20 + BM25 top-20, merged with RRF (k=60).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings

from retrieval import dense, sparse

RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (rank + RRF_K)


def query(
    text: str,
    ticker: str,
    chunk_size: int,
    embedding_model: str,
    top_k: int = 5,
    settings: dict | None = None,
) -> list[dict[str, Any]]:
    settings = settings or load_settings()
    dense_hits = dense.query(
        text,
        ticker,
        embedding_model,
        chunk_size,
        top_k=20,
        settings=settings,
    )
    bm25_index = sparse.load_index(ticker, chunk_size, settings)
    if bm25_index is None:
        return [
            {
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "ticker": ticker.upper(),
                "score": float(h.get("similarity", 0.0)),
                "source": "dense",
            }
            for h in dense_hits[:top_k]
        ]
    bm25_hits = bm25_index.query(text, top_k=20)

    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    tickers: dict[str, str] = {}

    for r, h in enumerate(dense_hits, start=1):
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(r)
        texts[cid] = h.get("text", "")
        tickers[cid] = str(h.get("metadata", {}).get("ticker", ticker)).upper()

    for r, h in enumerate(bm25_hits, start=1):
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(r)
        texts[cid] = h.get("text", "")
        tickers[cid] = str(h.get("ticker", ticker)).upper()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "chunk_id": cid,
            "text": texts.get(cid, ""),
            "ticker": tickers.get(cid, ticker.upper()),
            "score": float(sc),
            "source": "hybrid",
        }
        for cid, sc in ranked
    ]
