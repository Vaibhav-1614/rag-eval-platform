"""
Seed data/testset.json and data/results.db with SYNTHETIC demo rows (random scores,
eval_version="demo") for UI development only. Real results come from benchmark.run_all.
WARNING: overwrites data/testset.json.
Run from repo root: python scripts/seed_demo_data.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.results_store import save_result
from platform_config import load_settings


def main() -> None:
    settings = load_settings()
    paths = settings.get("paths") or {}
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    testset_path = ROOT / paths.get("testset_path", "data/testset.json")
    tickers = [t.upper() for t in settings.get("companies", [])][:10]
    rows = []
    for ti, ticker in enumerate(tickers):
        for i in range(5):
            ctx = (
                f"Demo excerpt for {ticker} item {i}. The company reported segment "
                f"revenue trends and operational risks in fiscal period {2020 + (ti + i) % 4}."
            )
            rows.append(
                {
                    "question": f"What does the demo excerpt say about {ticker} item {i}?",
                    "ground_truth_answer": "Segment revenue trends and operational risks.",
                    "ground_truth_context": ctx,
                    "ground_truth_chunk_id": f"{ticker}_512_{ti * 5 + i}",
                    "ticker": ticker,
                }
            )
    testset_path.write_text(json.dumps(rows[:50], indent=2), encoding="utf-8")
    print(f"Wrote {testset_path} ({len(rows[:50])} rows)")

    random.seed(42)
    strategies = ["dense", "sparse", "hybrid"]
    sizes = [256, 512, 1024]
    models = ["sentence-transformers", "bge"]
    n = 0
    for s in strategies:
        for z in sizes:
            for m in models:
                base = 0.45 + random.random() * 0.45
                result = {
                    "config": {"retrieval_strategy": s, "chunk_size": z, "embedding_model": m},
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": min(1.0, base + random.random() * 0.1),
                    "context_recall": min(1.0, base - 0.05 + random.random() * 0.15),
                    "mrr": min(1.0, base + random.random() * 0.08),
                    "hit_rate": min(1.0, base - 0.1 + random.random() * 0.2),
                    "ndcg": min(1.0, base + random.random() * 0.05),
                    "avg_latency_ms": 20.0 + random.random() * 180.0 + (50.0 if s == "hybrid" else 0.0),
                    "total_cost_usd": 0.0,
                    "eval_version": "demo",
                }
                save_result(result, settings=settings)
                n += 1
    print(f"Seeded {n} benchmark rows into results DB")


if __name__ == "__main__":
    main()
