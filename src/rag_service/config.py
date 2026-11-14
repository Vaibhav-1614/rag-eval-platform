"""Service settings: frozen retrieval config (YAML) + runtime knobs (env vars)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_CONFIG = ROOT / "configs" / "retrieval" / "v2.yaml"

COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson",
    "XOM": "Exxon Mobil",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class RetrievalConfig:
    version: str
    strategy: str
    chunk_size: int
    embedding_model: str | None
    top_k: int
    corpus_description: str
    tickers: tuple[str, ...]
    benchmark: dict = field(default_factory=dict)
    # --- v2 options (defaults reproduce v1 behaviour) ---
    index: str = "v1"                      # "v1": data/index/ fixed windows; "v2": data/index/v2 structured chunks
    query_expansion: str | None = None     # path to expansion rules (relative to repo root)
    rerank_model: str | None = None        # cross-encoder applied to the fused candidates
    candidates: int = 20                   # per retriever before fusion / rerank
    facts: bool = False                    # answer headline figures from data/facts/facts.json
    min_evidence_score: float | None = None  # refuse below this reranker score (None = off)

    @classmethod
    def load(cls, path: Path | str) -> RetrievalConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        corpus = raw.get("corpus") or {}
        strategy = str(raw["strategy"]).lower()
        if strategy not in {"sparse", "dense", "hybrid"}:
            raise ValueError(f"unknown strategy {strategy!r}")
        if strategy != "sparse" and not raw.get("embedding_model"):
            raise ValueError(f"strategy {strategy!r} needs embedding_model")
        return cls(
            version=str(raw["version"]),
            strategy=strategy,
            chunk_size=int(raw["chunk_size"]),
            embedding_model=raw.get("embedding_model"),
            top_k=int(raw.get("top_k", 5)),
            corpus_description=str(corpus.get("description", "")),
            tickers=tuple(t.upper() for t in corpus.get("tickers", COMPANY_NAMES)),
            benchmark=dict(raw.get("benchmark") or {}),
            index=str(raw.get("index", "v1")),
            query_expansion=raw.get("query_expansion"),
            rerank_model=(raw.get("rerank") or {}).get("model"),
            candidates=int((raw.get("rerank") or {}).get("candidates", raw.get("candidates", 20))),
            facts=bool(raw.get("facts", False)),
            min_evidence_score=(raw.get("abstain") or {}).get("min_evidence_score"),
        )


@dataclass(frozen=True)
class ServiceSettings:
    retrieval_config_path: Path
    prompt_version: str
    groq_model: str
    groq_fallback_model: str
    gemini_model: str
    judge_model: str
    rate_limit: str
    daily_llm_cap: int
    cache_size: int
    cache_ttl_s: int
    feedback_db: Path
    max_question_chars: int
    database_url: str = ""          # Postgres for feedback + usage counter; empty = SQLite file
    api_keys: tuple[str, ...] = ()  # when non-empty, /v1/* requires an X-API-Key header
    trusted_proxy_hops: int = 0     # reverse proxies in front of the app that append X-Forwarded-For

    @classmethod
    def from_env(cls) -> ServiceSettings:
        return cls(
            retrieval_config_path=Path(
                os.environ.get("RETRIEVAL_CONFIG", str(DEFAULT_RETRIEVAL_CONFIG))
            ),
            prompt_version=os.environ.get("PROMPT_VERSION", "v2"),
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_fallback_model=os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile"),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            judge_model=os.environ.get("JUDGE_MODEL", "gemini-2.5-flash"),
            rate_limit=os.environ.get("RATE_LIMIT", "10/minute"),
            # Shared across all callers; protects the free-tier LLM quotas. Above the
            # cap the service still answers, with the extractive (no-LLM) path.
            daily_llm_cap=_env_int("DAILY_LLM_CAP", 500),
            cache_size=_env_int("CACHE_SIZE", 512),
            cache_ttl_s=_env_int("CACHE_TTL_S", 6 * 3600),
            feedback_db=Path(os.environ.get("FEEDBACK_DB", str(ROOT / "data" / "feedback.db"))),
            max_question_chars=_env_int("MAX_QUESTION_CHARS", 500),
            database_url=os.environ.get("DATABASE_URL", "").strip(),
            api_keys=tuple(k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()),
            trusted_proxy_hops=_env_int("TRUSTED_PROXY_HOPS", 0),
        )

    @property
    def groq_key(self) -> str:
        return os.environ.get("GROQ_API_KEY", "").strip()

    @property
    def gemini_key(self) -> str:
        return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


@lru_cache(maxsize=1)
def get_settings() -> ServiceSettings:
    return ServiceSettings.from_env()


@lru_cache(maxsize=4)
def get_retrieval_config(path: str | None = None) -> RetrievalConfig:
    return RetrievalConfig.load(path or get_settings().retrieval_config_path)
