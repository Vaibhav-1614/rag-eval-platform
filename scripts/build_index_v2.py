"""
Build the v2 index: section/table-aware chunks + bge-small embeddings (ONNX, no torch).

    python scripts/build_index_v2.py              # chunks + embeddings
    python scripts/build_index_v2.py --no-embed   # chunks only

Writes data/index/v2/chunks/<TICKER>.json.gz and data/index/v2/embeddings/<TICKER>_bge.npy
(float16, row i = chunk i). Needs data/facts/facts.json (scripts/build_facts.py) for the
company name and fiscal year of each filing.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.structured_chunker import chunk_filing  # noqa: E402

OUT = ROOT / "data" / "index" / "v2"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def index_text(chunk: dict) -> str:
    """What gets indexed (BM25 and embeddings): the context line plus the chunk text."""
    return f"{chunk['context']}\n{chunk['text']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep embeddings that already match the chunk count")
    args = ap.parse_args()
    facts = json.loads((ROOT / "data" / "facts" / "facts.json").read_text())
    (OUT / "chunks").mkdir(parents=True, exist_ok=True)
    (OUT / "embeddings").mkdir(parents=True, exist_ok=True)
    model = None
    if not args.no_embed:
        from fastembed import TextEmbedding

        model = TextEmbedding(EMBED_MODEL)
    total = 0
    for ticker, c in facts.items():
        text = (ROOT / "data" / "parsed" / f"{ticker}.txt").read_text(encoding="utf-8")
        chunks = chunk_filing(text, ticker, c["company"], c["fiscal_year"])
        for ch in chunks:
            ch["source_url"] = c["url"]
        p = OUT / "chunks" / f"{ticker}.json.gz"
        with open(p, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            f.write(json.dumps(chunks, ensure_ascii=False).encode("utf-8"))
        emb_path = OUT / "embeddings" / f"{ticker}_bge.npy"
        if model is not None and not (args.resume and emb_path.exists() and len(np.load(emb_path)) == len(chunks)):
            texts = [index_text(ch) for ch in chunks]
            # Embed in length order so each batch pads to similar lengths (about 2x faster on CPU).
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            vecs = np.zeros((len(texts), 384), dtype=np.float16)
            for i, v in zip(order, model.embed([texts[i] for i in order], batch_size=16), strict=True):
                vecs[i] = v
            np.save(emb_path, vecs)
        total += len(chunks)
        print(f"{ticker}: {len(chunks)} chunks", flush=True)
    print(f"total {total} chunks -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
