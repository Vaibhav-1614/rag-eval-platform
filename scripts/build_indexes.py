"""
Build Chroma collections and BM25 pickles for all tickers, chunk sizes, and embedding models in settings.
Run from repo root: python scripts/build_indexes.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from platform_config import load_settings

from ingestion.chunker import chunk_file_for_ticker
from retrieval import dense, sparse

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()
    tickers = [t.upper() for t in settings.get("companies", [])]
    chunk_sizes = settings.get("chunk_sizes", [256, 512, 1024])
    # Benchmark subset: first two embedding models in yaml
    emb_models = (settings.get("embedding_models") or ["sentence-transformers", "bge"])[:2]

    for t in tickers:
        for cs in chunk_sizes:
            try:
                chunks = chunk_file_for_ticker(t, int(cs), settings)
            except FileNotFoundError as e:
                logger.warning("%s", e)
                continue
            sparse.build_and_save(chunks, settings)
            for em in emb_models:
                em_key = str(em).strip().lower().replace("_", "-")
                if em_key == "e5-small":
                    continue
                logger.info("Chroma ingest %s %s %s (%s chunks)", t, cs, em_key, len(chunks))
                dense.ingest_chunks(chunks, embedding_model=em_key, settings=settings)


if __name__ == "__main__":
    main()
