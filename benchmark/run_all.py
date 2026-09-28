"""
Run all benchmark configs (18 = 3 strategies x 3 chunk sizes x 2 embedding models).
--dry-run: only 2 configs.
Skips configs already present in SQLite (resumable); --force re-runs them.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from evaluation.ragas_runner import evaluate_config
from evaluation.results_store import config_key_exists, load_all_results, save_result
from platform_config import load_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EVAL_VERSION = "v2"  # v2: stopword-filtered overlap, uniform 0.35 threshold, binary nDCG


def config_label(c: dict) -> str:
    return f"{c['retrieval_strategy']}_{c['chunk_size']}_{c['embedding_model']}"


def all_configs(settings: dict) -> list[dict]:
    strategies = settings.get("retrieval_strategies", ["dense", "sparse", "hybrid"])
    sizes = settings.get("chunk_sizes", [256, 512, 1024])
    emods = (settings.get("embedding_models") or ["sentence-transformers", "bge"])[:2]
    out = []
    for s, z, e in itertools.product(strategies, sizes, emods):
        ek = str(e).strip().lower().replace("_", "-")
        if ek == "e5-small":
            continue
        out.append({"retrieval_strategy": s, "chunk_size": int(z), "embedding_model": ek})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run configs already stored in SQLite")
    args = ap.parse_args()

    settings = load_settings()
    configs = all_configs(settings)
    if args.dry_run:
        configs = configs[:2]
        logger.info("Dry-run: %s configs", len(configs))

    for cfg in tqdm(configs, desc="Benchmark"):
        if not args.force and config_key_exists(
            cfg["retrieval_strategy"],
            cfg["chunk_size"],
            cfg["embedding_model"],
            eval_version=EVAL_VERSION,
            settings=settings,
        ):
            tqdm.write(f"skip {config_label(cfg)}")
            continue
        try:
            result = evaluate_config(cfg, settings=settings)
            result["eval_version"] = EVAL_VERSION
            save_result(result, settings=settings)
        except FileNotFoundError as e:
            tqdm.write(f"FAIL {config_label(cfg)}: {e}\nGenerate testset: python -m evaluation.testset_generator")
            break
        except Exception as e:
            tqdm.write(f"FAIL {config_label(cfg)}: {e}")

    df = load_all_results(settings)
    if df.empty:
        logger.info("No results in DB yet.")
        return
    df = df[df["eval_version"] == EVAL_VERSION].copy()
    df["config_label"] = (
        df["retrieval_strategy"] + "_" + df["chunk_size"].astype(str) + "_" + df["embedding_model"]
    )
    metric_cols = ["context_precision", "context_recall", "mrr", "hit_rate", "ndcg"]
    df["avg_retrieval"] = df[metric_cols].mean(axis=1)
    show = df.sort_values("context_recall", ascending=False)
    sub = show[metric_cols + ["avg_latency_ms", "config_label"]]
    print(tabulate(sub.values.tolist(), headers=list(sub.columns), tablefmt="github"))

    scores = df[metric_cols].fillna(0).mean(axis=1)
    best_row = df.loc[scores.idxmax()]
    print("\nBest config (avg of context_precision, context_recall, mrr, hit_rate, ndcg):")
    print(best_row[["config_label", "avg_latency_ms"] + metric_cols].to_string())


if __name__ == "__main__":
    main()
