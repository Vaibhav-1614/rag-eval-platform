"""
Chunk cleaned text with LangChain RecursiveCharacterTextSplitter.

Overlap = 10% of chunk_size (in character proxy for tokens).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

# Rough chars per token for sizing when exact tokenizer not required
CHARS_PER_TOKEN = 4


def approximate_token_count(text: str) -> int:
    """Simple whitespace token estimate."""
    return len(re.findall(r"\S+", text))


def chunk_text(
    text: str,
    ticker: str,
    chunk_size: int,
    *,
    chunk_overlap_ratio: float = 0.1,
) -> list[dict[str, Any]]:
    chunk_overlap = max(1, int(chunk_size * chunk_overlap_ratio))
    chunk_size_chars = max(100, chunk_size * CHARS_PER_TOKEN)
    overlap_chars = max(10, chunk_overlap * CHARS_PER_TOKEN)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    chunks: list[dict[str, Any]] = []
    for i, piece in enumerate(pieces):
        tid = ticker.upper()
        cid = f"{tid}_{chunk_size}_{i}"
        chunks.append(
            {
                "chunk_id": cid,
                "ticker": tid,
                "text": piece,
                "token_count": approximate_token_count(piece),
                "chunk_size": chunk_size,
            }
        )
    return chunks


def chunk_file_for_ticker(
    ticker: str,
    chunk_size: int,
    settings: dict | None = None,
) -> list[dict[str, Any]]:
    settings = settings or load_settings()
    paths_cfg = settings.get("paths") or {}
    parsed = project_root() / paths_cfg.get("parsed_dir", "data/parsed") / f"{ticker.upper()}.txt"
    if not parsed.exists():
        raise FileNotFoundError(f"Missing parsed file: {parsed}")
    text = parsed.read_text(encoding="utf-8")
    return chunk_text(text, ticker, chunk_size)
