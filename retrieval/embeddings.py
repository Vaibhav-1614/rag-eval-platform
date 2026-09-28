"""
Free embedding models with joblib.Memory cache.

Models:
- sentence-transformers -> all-MiniLM-L6-v2
- bge -> BAAI/bge-small-en-v1.5
- e5-small -> intfloat/e5-small-v2
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Memory

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

logger = logging.getLogger(__name__)

_MODEL_NAMES: dict[str, str] = {
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
}

_st_model_cache: dict[str, Any] = {}
_memory: Memory | None = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        settings = load_settings()
        paths = settings.get("paths") or {}
        cache_dir = project_root() / paths.get("embedding_cache", "models/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        _memory = Memory(location=str(cache_dir / "joblib"), verbose=0)
    return _memory


def _normalize_model_key(embedding_model: str) -> str:
    k = embedding_model.strip().lower().replace("_", "-")
    if k in ("sentence-transformers", "bge", "e5-small"):
        return k
    raise ValueError(f"Unknown embedding_model: {embedding_model}. Use {list(_MODEL_NAMES)}")


def _model_hf_name(embedding_model_key: str) -> str:
    return _MODEL_NAMES[embedding_model_key]


def _load_st_model(embedding_model_key: str):
    if embedding_model_key in _st_model_cache:
        return _st_model_cache[embedding_model_key]
    from sentence_transformers import SentenceTransformer

    name = _model_hf_name(embedding_model_key)
    logger.info("Loading SentenceTransformer: %s", name)
    m = SentenceTransformer(name)
    _st_model_cache[embedding_model_key] = m
    return m


def _e5_prefix(embedding_model_key: str, input_type: str) -> str | None:
    # E5 is trained with asymmetric "query: " / "passage: " prefixes
    if embedding_model_key == "e5-small":
        return "query: " if input_type == "query" else "passage: "
    return None


def _embed_batch_impl(
    embedding_model_key: str,
    texts: tuple[str, ...],
    batch_size: int,
    input_type: str = "passage",
) -> list[list[float]]:
    model = _load_st_model(embedding_model_key)
    prefix = _e5_prefix(embedding_model_key, input_type)
    to_encode = [f"{prefix}{t}" if prefix else t for t in texts]
    emb = model.encode(
        to_encode,
        batch_size=min(batch_size, len(to_encode)) or 1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    return [row.tolist() for row in arr]


def _make_cached_embed() -> Any:
    return _get_memory().cache(_embed_batch_impl)


_cached_embed_batch = _make_cached_embed()


def embed_texts(
    texts: list[str],
    embedding_model: str = "sentence-transformers",
    *,
    batch_size: int = 32,
    input_type: str = "passage",
) -> list[list[float]]:
    """
    Return list of embedding vectors for each text.
    Caches each batch by (model, content hash). input_type is "passage" or "query".
    """
    if not texts:
        return []
    key = _normalize_model_key(embedding_model)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = tuple(texts[i : i + batch_size])
        vecs = _cached_embed_batch(key, batch, batch_size, input_type)
        out.extend(vecs)
    return out
