"""
evaluate_config(config) — retrieval-first scoring (RAGAS fields reserved, optional LLM later).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

from evaluation.retrieval_metrics import score_row
from retrieval import dense, hybrid
from retrieval.sparse import load_index

logger = logging.getLogger(__name__)


def _load_testset(path: Path | None = None) -> list[dict[str, Any]]:
    settings = load_settings()
    paths = settings.get("paths") or {}
    p = path or (project_root() / paths.get("testset_path", "data/testset.json"))
    if not p.exists():
        raise FileNotFoundError(f"Missing testset: {p}")
    import json

    return json.loads(p.read_text(encoding="utf-8"))


def retrieve_for_config(
    question: str,
    ticker: str,
    chunk_size: int,
    embedding_model: str,
    retrieval_strategy: str,
    top_k: int = 5,
    settings: dict | None = None,
) -> list[dict[str, Any]]:
    settings = settings or load_settings()
    em = str(embedding_model).strip().lower().replace("_", "-")
    t = ticker.upper()
    cs = int(chunk_size)
    strat = retrieval_strategy.strip().lower()

    if strat == "dense":
        hits = dense.query(question, t, em, cs, top_k=top_k, settings=settings)
        return [
            {
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "ticker": t,
                "score": float(h.get("similarity", 0.0)),
                "source": "dense",
            }
            for h in hits
        ]
    if strat == "sparse":
        idx = load_index(t, cs, settings)
        return idx.query(question, top_k=top_k) if idx else []
    if strat == "hybrid":
        return hybrid.query(question, t, cs, em, top_k=top_k, settings=settings)
    raise ValueError(f"Unknown retrieval_strategy: {retrieval_strategy}")


def evaluate_config(config: dict[str, Any], settings: dict | None = None) -> dict[str, Any]:
    """
    config keys: retrieval_strategy, chunk_size, embedding_model
    """
    settings = settings or load_settings()
    items = _load_testset()
    if not items:
        return {
            "config": {
                "retrieval_strategy": config["retrieval_strategy"],
                "chunk_size": int(config["chunk_size"]),
                "embedding_model": str(config["embedding_model"]),
            },
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
            "avg_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "eval_version": "v2",
        }
    # Warm-up (model load, index hydration) so latency measures steady-state queries
    for t in sorted({row["ticker"] for row in items}):
        retrieve_for_config(
            "warm-up", t, int(config["chunk_size"]), str(config["embedding_model"]),
            str(config["retrieval_strategy"]), top_k=5, settings=settings,
        )
    latencies: list[float] = []
    agg ={"hit_rate": 0.0, "mrr": 0.0, "context_recall": 0.0, "context_precision": 0.0, "ndcg": 0.0}
    n = 0
    for row in items:
        q = row["question"]
        ticker = row["ticker"]
        gold_ctx = row.get("ground_truth_context", "")
        gold_cid = row.get("ground_truth_chunk_id")
        t0 = time.perf_counter()
        retrieved = retrieve_for_config(
            q,
            ticker,
            int(config["chunk_size"]),
            str(config["embedding_model"]),
            str(config["retrieval_strategy"]),
            top_k=5,
            settings=settings,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        scores = score_row(retrieved, gold_ctx, ground_truth_chunk_id=gold_cid)
        for k in agg:
            agg[k] += scores[k]
        n += 1
    if n:
        for k in agg:
            agg[k] /= n
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    out = {
        "config": {
            "retrieval_strategy": config["retrieval_strategy"],
            "chunk_size": int(config["chunk_size"]),
            "embedding_model": str(config["embedding_model"]),
        },
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": agg["context_precision"],
        "context_recall": agg["context_recall"],
        "mrr": agg["mrr"],
        "hit_rate": agg["hit_rate"],
        "ndcg": agg["ndcg"],
        "avg_latency_ms": avg_lat,
        "total_cost_usd": 0.0,
        "eval_version": "v2",
    }
    return out
