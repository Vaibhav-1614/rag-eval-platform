import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Tests never call real LLMs: strip keys before any settings object is built.
for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.pop(k, None)


@pytest.fixture(scope="session")
def cfg():
    from rag_service.config import RetrievalConfig

    return RetrievalConfig.load(ROOT / "configs" / "retrieval" / "v1.yaml")


@pytest.fixture(scope="session")
def retriever(cfg):
    from rag_service.retriever import Retriever

    r = Retriever(cfg)
    r.warm()
    return r
