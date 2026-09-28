"""
ChromaDB dense retrieval: one collection per (ticker, chunk_size, embedding_model).
Persist under models/chroma/.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

from retrieval import artifacts
from retrieval.embeddings import embed_texts

logger = logging.getLogger(__name__)


def collection_name(ticker: str, chunk_size: int, embedding_model: str) -> str:
    m = embedding_model.strip().lower().replace(" ", "-")
    return f"{ticker.upper()}_{int(chunk_size)}_{m}"


@lru_cache(maxsize=4)
def _client_for_path(path: str) -> chromadb.ClientAPI:
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def _client(settings: dict | None = None) -> chromadb.ClientAPI:
    settings = settings or load_settings()
    paths = settings.get("paths") or {}
    p = project_root() / paths.get("chroma_dir", "models/chroma")
    return _client_for_path(str(p))


def _metadata(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": str(c["ticker"]).upper(),
        "chunk_size": int(c["chunk_size"]),
        "chunk_id": c["chunk_id"],
        "token_count": int(c.get("token_count", 0)),
    }


def _write_collection(
    client: chromadb.ClientAPI,
    name: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> chromadb.Collection:
    """(Re)create a collection from scratch so stale chunk ids never survive a rebuild."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    # Chroma only writes the HNSW graph to disk every `sync_threshold` vectors (default
    # 1000). Smaller collections then live only in the segment cache and fail with
    # "Nothing found on disk" once evicted, so flush every 16 vectors.
    coll = client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine", "sync_threshold": 16, "batch_size": 16}},
    )
    step = max(1, min(client.get_max_batch_size(), 2000))
    for i in range(0, len(chunks), step):
        batch = chunks[i : i + step]
        coll.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_metadata(c) for c in batch],
            embeddings=[list(map(float, e)) for e in embeddings[i : i + step]],
        )
    return coll


def ingest_chunks(
    chunks: list[dict[str, Any]],
    embedding_model: str,
    settings: dict | None = None,
) -> None:
    if not chunks:
        return
    settings = settings or load_settings()
    client = _client(settings)
    first = chunks[0]
    ticker = str(first["ticker"]).upper()
    chunk_size = int(first["chunk_size"])
    name = collection_name(ticker, chunk_size, embedding_model)
    embeddings = embed_texts([c["text"] for c in chunks], embedding_model=embedding_model)
    _write_collection(client, name, chunks, embeddings)
    artifacts.save_embeddings(ticker, chunk_size, embedding_model, embeddings, settings)
    logger.info("Chroma build %s: %s chunks", name, len(chunks))


def _hydrate_from_artifacts(
    client: chromadb.ClientAPI,
    ticker: str,
    chunk_size: int,
    embedding_model: str,
    settings: dict,
) -> chromadb.Collection | None:
    chunks = artifacts.load_chunks(ticker, chunk_size, settings)
    embs = artifacts.load_embeddings(ticker, chunk_size, embedding_model, settings)
    if chunks is None or embs is None or len(chunks) != len(embs):
        return None
    name = collection_name(ticker, chunk_size, embedding_model)
    logger.info("Hydrating Chroma collection %s from artifacts (%s chunks)", name, len(chunks))
    return _write_collection(client, name, chunks, embs.tolist())


def get_collection(
    ticker: str,
    chunk_size: int,
    embedding_model: str,
    settings: dict | None = None,
) -> chromadb.Collection | None:
    settings = settings or load_settings()
    client = _client(settings)
    name = collection_name(ticker, chunk_size, embedding_model)
    try:
        coll = client.get_collection(name=name)
        if coll.count() > 0:
            return coll
    except Exception:
        pass
    coll = _hydrate_from_artifacts(client, ticker, chunk_size, embedding_model, settings)
    if coll is None:
        logger.warning("Missing collection: %s", name)
    return coll


def query(
    text: str,
    ticker: str,
    embedding_model: str,
    chunk_size: int,
    top_k: int = 5,
    settings: dict | None = None,
) -> list[dict[str, Any]]:
    settings = settings or load_settings()
    coll = get_collection(ticker, chunk_size, embedding_model, settings)
    if coll is None:
        return []
    qemb = embed_texts([text], embedding_model=embedding_model, input_type="query")[0]
    res = coll.query(
        query_embeddings=[qemb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") is not None else [None] * len(ids)
    for i, cid in enumerate(ids):
        dist = dists[i] if i < len(dists) else None
        if dist is None:
            sim = 0.0
        else:
            d = float(dist)
            sim = max(0.0, 1.0 - d) if d <= 2.0 else 1.0 / (1.0 + d)
        meta = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else ""
        out.append(
            {
                "chunk_id": cid,
                "text": doc or "",
                "metadata": meta or {},
                "distance": dist,
                "similarity": sim,
                "source": "dense",
            }
        )
    return out
