import sqlite3

from conftest import ROOT

from rag_service.config import RetrievalConfig


def test_v1_loads(cfg):
    assert cfg.version == "v1"
    assert cfg.strategy == "sparse" and cfg.chunk_size == 256 and cfg.top_k == 5
    assert len(cfg.tickers) == 10
    assert "NVIDIA fiscal 2026" in cfg.corpus_description


def test_v1_is_best_benchmark_config(cfg):
    """The frozen config must be the top-MRR row of the v2 benchmark it was picked from."""
    con = sqlite3.connect(ROOT / "data" / "results.db")
    strategy, size, hit, mrr = con.execute(
        "SELECT retrieval_strategy, chunk_size, hit_rate, mrr FROM benchmark_results "
        "WHERE eval_version='v2' ORDER BY mrr DESC, avg_latency_ms ASC LIMIT 1"
    ).fetchone()
    assert (strategy, size) == (cfg.strategy, cfg.chunk_size)
    assert round(hit, 2) == cfg.benchmark["hit_at_5"]
    assert round(mrr, 3) == cfg.benchmark["mrr"]


def test_hybrid_alternative_loads():
    alt = RetrievalConfig.load(ROOT / "configs" / "retrieval" / "v1-hybrid.yaml")
    assert alt.strategy == "hybrid" and alt.embedding_model == "bge"
