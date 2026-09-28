"""
SQLite persistence for benchmark runs at data/results.db
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

TABLE = "benchmark_results"


def _db_path(settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    paths = settings.get("paths") or {}
    p = project_root() / paths.get("results_db", "data/results.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_db(settings: dict | None = None) -> None:
    path = _db_path(settings)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                retrieval_strategy TEXT NOT NULL,
                chunk_size INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                eval_version TEXT NOT NULL DEFAULT 'v1',
                faithfulness REAL,
                answer_relevancy REAL,
                context_precision REAL,
                context_recall REAL,
                mrr REAL,
                hit_rate REAL,
                ndcg REAL,
                avg_latency_ms REAL,
                total_cost_usd REAL,
                created_at TEXT NOT NULL,
                UNIQUE(retrieval_strategy, chunk_size, embedding_model, eval_version)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_result(result: dict[str, Any], settings: dict | None = None) -> None:
    init_db(settings)
    cfg = result.get("config") or {}
    path = _db_path(settings)
    conn = sqlite3.connect(path)
    try:
        row = (
            cfg.get("retrieval_strategy"),
            int(cfg.get("chunk_size", 0)),
            str(cfg.get("embedding_model", "")),
            str(result.get("eval_version", "v1")),
            result.get("faithfulness"),
            result.get("answer_relevancy"),
            float(result.get("context_precision", 0.0)),
            float(result.get("context_recall", 0.0)),
            float(result.get("mrr", 0.0)),
            float(result.get("hit_rate", 0.0)),
            float(result.get("ndcg", 0.0)),
            float(result.get("avg_latency_ms", 0.0)),
            float(result.get("total_cost_usd", 0.0)),
            datetime.now(timezone.utc).isoformat(),
        )
        conn.execute(
            f"""
            INSERT INTO {TABLE} (
                retrieval_strategy, chunk_size, embedding_model, eval_version,
                faithfulness, answer_relevancy, context_precision, context_recall,
                mrr, hit_rate, ndcg, avg_latency_ms, total_cost_usd, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(retrieval_strategy, chunk_size, embedding_model, eval_version) DO UPDATE SET
                faithfulness=excluded.faithfulness,
                answer_relevancy=excluded.answer_relevancy,
                context_precision=excluded.context_precision,
                context_recall=excluded.context_recall,
                mrr=excluded.mrr,
                hit_rate=excluded.hit_rate,
                ndcg=excluded.ndcg,
                avg_latency_ms=excluded.avg_latency_ms,
                total_cost_usd=excluded.total_cost_usd,
                created_at=excluded.created_at
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def load_all_results(settings: dict | None = None) -> pd.DataFrame:
    init_db(settings)
    path = _db_path(settings)
    conn = sqlite3.connect(path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {TABLE} ORDER BY created_at DESC", conn)
    finally:
        conn.close()


def config_key_exists(
    retrieval_strategy: str,
    chunk_size: int,
    embedding_model: str,
    eval_version: str = "v1",
    settings: dict | None = None,
) -> bool:
    init_db(settings)
    path = _db_path(settings)
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            f"SELECT 1 FROM {TABLE} WHERE retrieval_strategy=? AND chunk_size=? AND embedding_model=? AND eval_version=?",
            (retrieval_strategy, chunk_size, embedding_model, eval_version),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()
