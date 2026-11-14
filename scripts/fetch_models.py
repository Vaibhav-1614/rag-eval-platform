"""Download the ONNX models a retrieval config needs (used at Docker build time and in CI).

    python scripts/fetch_models.py configs/retrieval/v2.yaml
"""
from __future__ import annotations

import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
if cfg.get("strategy") != "sparse" and cfg.get("embedding_model"):
    from fastembed import TextEmbedding

    TextEmbedding(cfg["embedding_model"])
    print("embedding model ready:", cfg["embedding_model"])
if (cfg.get("rerank") or {}).get("model"):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    TextCrossEncoder(cfg["rerank"]["model"])
    print("re-ranker ready:", cfg["rerank"]["model"])
