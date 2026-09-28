"""
Portable index artifacts under data/index/ (committed to git, ~20 MB).

- chunks/{TICKER}_{size}.json.gz      -> list of chunk dicts
- embeddings/{TICKER}_{size}_{model}.npy -> float16 matrix aligned with chunks

scripts/build_indexes.py writes these; a fresh checkout (e.g. Streamlit Cloud)
rehydrates Chroma collections and BM25 indexes from them on first use, so the
306 MB Chroma SQLite file never needs to be versioned.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root


def artifact_dir(settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    paths = settings.get("paths") or {}
    return project_root() / paths.get("index_artifacts", "data/index")


def _chunks_path(ticker: str, chunk_size: int, settings: dict | None = None) -> Path:
    return artifact_dir(settings) / "chunks" / f"{ticker.upper()}_{int(chunk_size)}.json.gz"


def _emb_path(ticker: str, chunk_size: int, embedding_model: str, settings: dict | None = None) -> Path:
    return artifact_dir(settings) / "embeddings" / f"{ticker.upper()}_{int(chunk_size)}_{embedding_model}.npy"


def save_chunks(chunks: list[dict[str, Any]], settings: dict | None = None) -> Path:
    p = _chunks_path(chunks[0]["ticker"], chunks[0]["chunk_size"], settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip bytes deterministic so git only sees real changes
    with open(p, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(json.dumps(chunks, ensure_ascii=False).encode("utf-8"))
    return p


def load_chunks(ticker: str, chunk_size: int, settings: dict | None = None) -> list[dict[str, Any]] | None:
    p = _chunks_path(ticker, chunk_size, settings)
    if not p.exists():
        return None
    with gzip.open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def save_embeddings(
    ticker: str,
    chunk_size: int,
    embedding_model: str,
    embeddings: list[list[float]],
    settings: dict | None = None,
) -> Path:
    p = _emb_path(ticker, chunk_size, embedding_model, settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(embeddings, dtype=np.float16))
    return p


def load_embeddings(
    ticker: str,
    chunk_size: int,
    embedding_model: str,
    settings: dict | None = None,
) -> np.ndarray | None:
    p = _emb_path(ticker, chunk_size, embedding_model, settings)
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)
