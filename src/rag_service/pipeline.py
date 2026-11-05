"""The query pipeline shared by the API and the offline eval: route -> retrieve -> scope check -> generate."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from rag_service.config import RetrievalConfig, ServiceSettings
from rag_service.generation import Generation, generate
from rag_service.guardrails import refusal_message, scope_check
from rag_service.retriever import Passage, Retriever, detect_tickers


@dataclass
class PipelineResult:
    question: str
    scope: list[str]
    passages: list[Passage]
    generation: Generation
    scope_reason: str
    coverage: float
    retrieval_ms: float
    total_ms: float
    timings: dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: RetrievalConfig, settings: ServiceSettings, retriever: Retriever | None = None) -> None:
        self.cfg = cfg
        self.settings = settings
        self.retriever = retriever or Retriever(cfg)
        self.refusal = refusal_message(cfg)

    def run(
        self,
        question: str,
        tickers: list[str] | None = None,
        top_k: int | None = None,
        *,
        llm_gate: Callable[[], bool] | None = None,
        use_llm: bool = True,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        detected = [t.upper() for t in tickers] if tickers else detect_tickers(question)
        passages, scope = self.retriever.search(question, detected, top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        decision = scope_check(question, [p.text for p in passages], detected)
        if not decision.in_scope:
            gen = Generation(answer=self.refusal, mode="refusal")
        else:
            allow = use_llm and (llm_gate() if llm_gate else True)
            gen = generate(question, passages, self.cfg, self.settings, self.refusal, allow_llm=allow)
        return PipelineResult(
            question=question,
            scope=scope,
            passages=passages,
            generation=gen,
            scope_reason=decision.reason,
            coverage=round(decision.coverage, 3),
            retrieval_ms=round(retrieval_ms, 2),
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
