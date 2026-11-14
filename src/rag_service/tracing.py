"""
Request tracing.

Always: one structured JSON log line per request on stdout (picked up by HF Spaces /
Cloud Run / docker logs).
When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set: the same trace, plus
retrieval and generation observations, is also sent to Langfuse (free Hobby cloud
tier or self-hosted) through its public batch-ingestion HTTP API, from a background
thread so tracing can never slow down or fail a request.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

_log = logging.getLogger("rag_service.trace")


class JsonFormatter(logging.Formatter):
    _SKIP = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update({k: v for k, v in vars(record).items() if k not in self._SKIP})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str | None = None) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LangfuseSink:
    """Minimal async client for POST /api/public/ingestion."""

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        self._auth = (public_key, secret_key)
        self._url = host.rstrip("/") + "/api/public/ingestion"
        self._q: queue.Queue[list[dict]] = queue.Queue(maxsize=1000)
        threading.Thread(target=self._run, daemon=True, name="langfuse-sink").start()

    def submit(self, events: list[dict]) -> None:
        try:
            self._q.put_nowait(events)
        except queue.Full:
            _log.warning("langfuse_queue_full")

    def _run(self) -> None:
        with httpx.Client(timeout=10.0, auth=self._auth) as client:
            while True:
                batch = self._q.get()
                try:
                    r = client.post(self._url, json={"batch": batch})
                    if r.status_code >= 300:
                        _log.warning("langfuse_ingest_failed", extra={"status": r.status_code, "body": r.text[:200]})
                except httpx.HTTPError as e:
                    _log.warning("langfuse_ingest_error", extra={"error": type(e).__name__})


class Tracer:
    def __init__(self) -> None:
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.sink = LangfuseSink(pk, sk, host) if pk and sk else None

    @property
    def backend(self) -> str:
        return "langfuse+json" if self.sink else "json"

    def record(self, trace: dict[str, Any]) -> None:
        """trace: flat dict built by the API layer (see api._trace_payload)."""
        _log.info("query", extra=trace)
        if not self.sink:
            return
        tid = trace["request_id"]
        start = trace.get("started_at") or _now()
        events: list[dict] = [
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": start,
                "body": {
                    "id": tid,
                    "name": "rag-query",
                    "timestamp": start,
                    "input": {"question": trace.get("question"), "tickers": trace.get("scope")},
                    "output": {"answer": trace.get("answer"), "mode": trace.get("mode")},
                    "metadata": {k: trace.get(k) for k in ("retrieval_version", "prompt_version", "prompt_sha", "cache_hit", "scope_reason", "coverage",
                                                                "evidence_score", "facts_mode", "dropped_sentences")},
                    "tags": [trace.get("retrieval_version", ""), trace.get("mode", "")],
                },
            },
            {
                "id": str(uuid.uuid4()),
                "type": "span-create",
                "timestamp": start,
                "body": {
                    "id": f"{tid}-retrieval",
                    "traceId": tid,
                    "name": "retrieval",
                    "startTime": start,
                    "endTime": start,
                    "input": {"question": trace.get("question")},
                    "output": {"chunk_ids": trace.get("chunk_ids")},
                    "metadata": {"latency_ms": trace.get("retrieval_ms")},
                },
            },
        ]
        if trace.get("provider"):
            events.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "generation-create",
                    "timestamp": start,
                    "body": {
                        "id": f"{tid}-generation",
                        "traceId": tid,
                        "name": "answer",
                        "model": trace.get("model"),
                        "startTime": start,
                        "endTime": _now(),
                        "output": trace.get("answer"),
                        "usage": {"input": trace.get("input_tokens"), "output": trace.get("output_tokens"), "unit": "TOKENS"},
                        "metadata": {"provider": trace.get("provider"), "latency_ms": trace.get("llm_ms")},
                    },
                }
            )
        self.sink.submit(events)

    def score(self, request_id: str, name: str, value: float, comment: str | None = None) -> None:
        if not self.sink:
            return
        self.sink.submit(
            [
                {
                    "id": str(uuid.uuid4()),
                    "type": "score-create",
                    "timestamp": _now(),
                    "body": {"id": str(uuid.uuid4()), "traceId": request_id, "name": name, "value": value, "comment": comment},
                }
            ]
        )


def monotonic_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
