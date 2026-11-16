"""
FastAPI service.

    POST /v1/query         question -> cited answer + passages
    POST /v1/query/stream  same, as server-sent events: passages first, then the answer
    POST /v1/feedback      thumbs up/down on a request_id
    GET  /v1/config        frozen retrieval config, prompt version, which providers are on
    GET  /healthz          process is alive
    GET  /readyz           indexes are loaded and queries will be fast

Run: uvicorn rag_service.api:app --app-dir src --port 8000
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from rag_service import __version__
from rag_service.config import COMPANY_NAMES, get_retrieval_config, get_settings
from rag_service.guardrails import clean_question
from rag_service.pipeline import Pipeline, PipelineResult, Prepared
from rag_service.prompts import load_prompt
from rag_service.store import make_store
from rag_service.tracing import Tracer, configure_logging

logger = logging.getLogger("rag_service.api")


# ---------------- schemas ----------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["How many employees did Tesla have at the end of 2025?"])
    tickers: list[str] | None = Field(
        None, max_length=10, description="Restrict to these companies; omitted = detect from the question."
    )
    top_k: int | None = Field(None, ge=1, le=10)
    include_passages: bool = True

    @field_validator("tickers")
    @classmethod
    def _known(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        up = [t.strip().upper() for t in v]
        bad = [t for t in up if t not in COMPANY_NAMES]
        if bad:
            raise ValueError(f"unknown tickers {bad}; supported: {sorted(COMPANY_NAMES)}")
        return up


class PassageOut(BaseModel):
    rank: int
    chunk_id: str
    ticker: str
    score: float
    text: str
    kind: str = "text"
    section: str | None = None
    source_url: str | None = None
    fiscal_year: int | None = None


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    mode: Literal["llm", "extractive", "refusal"]
    citations: list[int]
    scope: list[str]
    passages: list[PassageOut]
    provider: str | None
    model: str | None
    verification: dict | None = Field(None, description="For LLM answers: sentences checked and dropped by citation verification.")
    retrieval_version: str
    prompt_version: str
    cache_hit: bool
    latency_ms: float


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=64)
    rating: Literal[-1, 1]
    comment: str | None = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    ok: bool
    feedback_id: int


# ---------------- cache ----------------
class TTLCache:
    def __init__(self, maxsize: int, ttl_s: int) -> None:
        self.maxsize, self.ttl = maxsize, ttl_s
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._d.get(key)
            if not item:
                return None
            if time.monotonic() - item[0] > self.ttl:
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return item[1]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.monotonic(), value)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)


# ---------------- identity: client IP and API keys ----------------
settings = get_settings()


def client_ip(request: Request) -> str:
    """
    The caller's address for rate limiting.

    X-Forwarded-For is a list the *client* can pre-fill; only the entries appended by
    our own reverse proxies can be trusted. With TRUSTED_PROXY_HOPS = n, the address is
    the n-th entry from the right (n = 1 behind a single proxy such as Hugging Face
    Spaces or Cloud Run). With 0 the header is ignored and the socket peer is used.
    """
    hops = settings.trusted_proxy_hops
    if hops > 0:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return request.client.host if request.client else "unknown"


def _valid_key(key: str | None) -> bool:
    # Compare bytes: compare_digest raises TypeError on non-ASCII str, which turned a
    # malformed X-API-Key header into a 500 instead of a 401.
    if not key:
        return False
    supplied = key.encode("utf-8", "surrogatepass")
    return any(hmac.compare_digest(supplied, k.encode("utf-8")) for k in settings.api_keys)


def rate_key(request: Request) -> str:
    key = request.headers.get("x-api-key")
    if settings.api_keys and _valid_key(key):
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return "ip:" + client_ip(request)


def require_api_key(request: Request) -> None:
    if settings.api_keys and not _valid_key(request.headers.get("x-api-key")):
        raise HTTPException(401, "missing or invalid X-API-Key header")


limiter = Limiter(key_func=rate_key, default_limits=[])


# ---------------- app ----------------
class State:
    pipeline: Pipeline
    tracer: Tracer
    store: Any
    cache: TTLCache
    started: float


state = State()


def _warm() -> None:
    try:
        state.pipeline.retriever.warm()
        logger.info("indexes_ready", extra={"retrieval_version": state.pipeline.cfg.version})
    except Exception:
        logger.exception("warmup_failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    cfg = get_retrieval_config(str(settings.retrieval_config_path))
    state.pipeline = Pipeline(cfg, settings)
    state.tracer = Tracer()
    state.store = make_store(settings.database_url, settings.feedback_db)
    state.cache = TTLCache(settings.cache_size, settings.cache_ttl_s)
    state.started = time.time()
    # Load indexes off the event loop; /readyz reports when done.
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    title="10-K RAG service",
    version=__version__,
    description="Cited answers over the latest 10-K filings of 10 US companies.",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"rate limit exceeded ({exc.detail}); try again shortly"},
        headers={"Retry-After": "60"},
    )


def _llm_enabled() -> bool:
    return bool(settings.groq_key or settings.gemini_key)


def _llm_gate() -> bool:
    """Asked before every provider call, so the daily cap counts calls, not requests."""
    return state.store.take_llm_call(settings.daily_llm_cap)


def _cache_key(question: str, tickers: list[str] | None, top_k: int | None) -> str:
    cfg = state.pipeline.cfg
    raw = f"{cfg.version}|{settings.prompt_version}|{question.lower()}|{sorted(tickers or [])}|{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _prompt_sha() -> str:
    return load_prompt(f"answer/{settings.prompt_version}").sha


def _trace_payload(rid: str, started_at: str, res: PipelineResult, cache_hit: bool, latency: float) -> dict:
    g = res.generation
    return {
        "request_id": rid,
        "started_at": started_at,
        "question": res.question,
        "scope": res.scope,
        "scope_reason": res.scope_reason,
        "coverage": res.coverage,
        "evidence_score": res.evidence_score,
        "facts_mode": res.facts_mode,
        "chunk_ids": [p.chunk_id for p in res.passages],
        "answer": g.answer,
        "mode": g.mode,
        "provider": g.provider,
        "model": g.model,
        "input_tokens": g.input_tokens,
        "output_tokens": g.output_tokens,
        "llm_ms": g.llm_latency_ms,
        "llm_errors": g.errors,
        "dropped_sentences": len((g.verification or {}).get("dropped", [])),
        "retrieval_ms": res.retrieval_ms,
        "total_ms": res.total_ms,
        "latency_ms": latency,
        "cache_hit": cache_hit,
        "retrieval_version": state.pipeline.cfg.version,
        "prompt_version": settings.prompt_version,
        "prompt_sha": _prompt_sha(),
    }


def _passages_out(res: PipelineResult | Prepared) -> list[dict]:
    return [PassageOut(**p.as_dict()).model_dump() for p in res.passages]


def _response(rid: str, res: PipelineResult, cache_hit: bool, latency: float, include_passages: bool) -> QueryResponse:
    g = res.generation
    return QueryResponse(
        request_id=rid,
        answer=g.answer,
        mode=g.mode,  # type: ignore[arg-type]
        citations=g.citations,
        scope=res.scope,
        passages=_passages_out(res) if include_passages else [],  # type: ignore[arg-type]
        provider=g.provider,
        model=g.model,
        verification=g.verification,
        retrieval_version=state.pipeline.cfg.version,
        prompt_version=settings.prompt_version,
        cache_hit=cache_hit,
        latency_ms=latency,
    )


def _checked_question(body: QueryRequest) -> str:
    if not state.pipeline.retriever.ready:
        raise HTTPException(503, "indexes are still loading; retry in a few seconds")
    try:
        return clean_question(body.question, settings.max_question_chars)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _cacheable(res: PipelineResult) -> bool:
    # Don't pin a degraded answer if an LLM was configured but failed or was capped.
    return not (res.generation.mode == "extractive" and _llm_enabled())


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    q = _checked_question(body)
    rid = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    key = _cache_key(q, body.tickers, body.top_k)
    res: PipelineResult | None = state.cache.get(key)
    cache_hit = res is not None
    if res is None:
        res = state.pipeline.run(q, body.tickers, body.top_k, llm_gate=_llm_gate)
        if _cacheable(res):
            state.cache.put(key, res)
    latency = round((time.perf_counter() - t0) * 1000, 2)
    state.tracer.record(_trace_payload(rid, started_at, res, cache_hit, latency))
    return _response(rid, res, cache_hit, latency, body.include_passages)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/query/stream", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """
    Server-sent events in stages, so a UI can show sources while the answer is written:

        event: passages  -> {request_id, scope, passages}       (after retrieval, ~ms)
        event: answer    -> the same JSON as POST /v1/query      (after generation + verification)
        event: done

    The answer is sent whole, not token by token: it is only released after citation
    verification, which needs the complete text.
    """
    q = _checked_question(body)
    rid = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    key = _cache_key(q, body.tickers, body.top_k)

    def events():
        t0 = time.perf_counter()
        res: PipelineResult | None = state.cache.get(key)
        cache_hit = res is not None
        if res is None:
            prep = state.pipeline.prepare(q, body.tickers, body.top_k)
            yield _sse("passages", {"request_id": rid, "scope": prep.scope, "passages": _passages_out(prep)})
            res = state.pipeline.answer(prep, llm_gate=_llm_gate)
            if _cacheable(res):
                state.cache.put(key, res)
        else:
            yield _sse("passages", {"request_id": rid, "scope": res.scope, "passages": _passages_out(res)})
        latency = round((time.perf_counter() - t0) * 1000, 2)
        state.tracer.record(_trace_payload(rid, started_at, res, cache_hit, latency))
        yield _sse("answer", _response(rid, res, cache_hit, latency, False).model_dump())
        yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def feedback(request: Request, body: FeedbackRequest) -> FeedbackResponse:
    fid = state.store.add_feedback(body.request_id, body.rating, body.comment)
    state.tracer.score(body.request_id, "user_feedback", float(body.rating), body.comment)
    logger.info("feedback", extra={"request_id": body.request_id, "rating": body.rating})
    return FeedbackResponse(ok=True, feedback_id=fid)


@app.get("/v1/config")
def config() -> dict[str, Any]:
    cfg = state.pipeline.cfg
    return {
        "service_version": __version__,
        "retrieval": {
            "version": cfg.version,
            "index": cfg.index,
            "strategy": cfg.strategy,
            "chunk_size": cfg.chunk_size,
            "embedding_model": cfg.embedding_model,
            "rerank_model": cfg.rerank_model,
            "query_expansion": bool(cfg.query_expansion),
            "facts_table": cfg.facts,
            "min_evidence_score": cfg.min_evidence_score,
            "top_k": cfg.top_k,
            "benchmark": cfg.benchmark,
        },
        "corpus": {"description": cfg.corpus_description, "tickers": list(cfg.tickers)},
        "prompt_version": settings.prompt_version,
        "providers": {
            "groq": {"enabled": bool(settings.groq_key), "model": settings.groq_model, "fallback_model": settings.groq_fallback_model},
            "gemini": {"enabled": bool(settings.gemini_key), "model": settings.gemini_model},
            "extractive_fallback": True,
        },
        "limits": {
            "rate_limit": settings.rate_limit,
            "daily_llm_cap": settings.daily_llm_cap,
            "llm_calls_today": state.store.llm_calls_today(),
            "api_key_required": bool(settings.api_keys),
        },
        "storage": state.store.backend,
        "tracing": state.tracer.backend,
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "uptime_s": round(time.time() - state.started, 1)}


@app.get("/readyz")
def readyz() -> JSONResponse:
    ready = state.pipeline.retriever.ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "retrieval_version": state.pipeline.cfg.version, "cache_entries": len(state.cache)},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "10-K RAG", "docs": "/docs", "config": "/v1/config"}
