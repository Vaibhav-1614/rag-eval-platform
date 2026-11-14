"""
The query pipeline shared by the API and the offline eval:

    route (companies) -> facts lookup -> retrieve (+ rerank) -> answerability -> generate -> verify
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from rag_service.config import RetrievalConfig, ServiceSettings
from rag_service.facts import fact_answer, fact_passages, find_facts
from rag_service.generation import Generation, generate
from rag_service.guardrails import refusal_message, scope_check
from rag_service.retriever import Passage, detect_tickers, make_retriever


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
    evidence_score: float | None = None  # reranker score of the best text passage
    facts_mode: str = "none"             # "single" | "ranking" | "none"


@dataclass
class Prepared:
    question: str
    scope: list[str]
    passages: list[Passage]
    hits: list
    facts_mode: str
    answerable: bool
    reason: str
    coverage: float
    evidence_score: float | None
    retrieval_ms: float
    t0: float


# A text passage must beat the fact passage by this many re-ranker logits to replace it.
FACT_OVERRIDE_MARGIN = 3.0


class Pipeline:
    def __init__(self, cfg: RetrievalConfig, settings: ServiceSettings, retriever=None) -> None:
        self.cfg = cfg
        self.settings = settings
        self.retriever = retriever or make_retriever(cfg)
        self.refusal = refusal_message(cfg)

    def retrieve(self, question: str, tickers: list[str] | None = None, top_k: int | None = None):
        """Everything before generation. Returns (passages, scope, detected, fact hits, mode, evidence)."""
        detected = [t.upper() for t in tickers] if tickers else detect_tickers(question)
        hits, mode = find_facts(question, detected) if self.cfg.facts else ([], "none")
        text_passages, scope = self.retriever.search(question, detected, top_k)
        # With a reranker the passage score is a calibrated-ish relevance logit.
        evidence = text_passages[0].score if (text_passages and self.cfg.rerank_model) else None
        facts = fact_passages(hits, mode) if hits else []
        passages = facts + text_passages
        for i, p in enumerate(passages, start=1):
            p.rank = i
        return passages, scope, detected, hits, mode, evidence

    def prepare(self, question: str, tickers: list[str] | None = None, top_k: int | None = None) -> Prepared:
        """Route, retrieve and decide answerability. No LLM call happens here."""
        t0 = time.perf_counter()
        passages, scope, detected, hits, mode, evidence = self.retrieve(question, tickers, top_k)
        decision = scope_check(question, [f"{p.context or ''} {p.text}" for p in passages], detected, has_facts=bool(hits))
        reason, answerable = decision.reason, decision.in_scope
        # Evidence-based abstention: nothing retrieved looks like an answer.
        if answerable and not hits and evidence is not None and self.cfg.min_evidence_score is not None:
            if evidence < self.cfg.min_evidence_score:
                answerable, reason = False, "low_evidence"
        return Prepared(question, scope, passages, hits, mode, answerable, reason, round(decision.coverage, 3),
                        None if evidence is None else round(evidence, 3), round((time.perf_counter() - t0) * 1000, 2), t0)

    def answer(self, prep: Prepared, *, llm_gate: Callable[[], bool] | None = None, use_llm: bool = True) -> PipelineResult:
        if not prep.answerable:
            gen = Generation(answer=self.refusal, mode="refusal")
        else:
            scorer = getattr(self.retriever, "rerank_scores", None) if self.cfg.rerank_model else None
            gen = generate(
                prep.question, prep.passages, self.cfg, self.settings, self.refusal,
                allow_llm=use_llm, llm_gate=llm_gate, scorer=scorer,
                direct_answer=self._direct_answer(prep, scorer),
            )
        return PipelineResult(
            question=prep.question, scope=prep.scope, passages=prep.passages, generation=gen,
            scope_reason=prep.reason, coverage=prep.coverage, retrieval_ms=prep.retrieval_ms,
            total_ms=round((time.perf_counter() - prep.t0) * 1000, 2),
            evidence_score=prep.evidence_score, facts_mode=prep.facts_mode,
        )

    def _direct_answer(self, prep: Prepared, scorer) -> str | None:
        """The facts-table sentence for the no-LLM path, unless the re-ranker says a text
        passage answers the question clearly better than the headline figure does
        ("How big is the buyback programme announced in May?" is not last year's buybacks)."""
        if not prep.hits:
            return None
        text = fact_answer(prep.question, prep.hits, prep.facts_mode)
        if scorer and prep.evidence_score is not None and prep.facts_mode == "single":
            fact_score = max(scorer(prep.question, [p.text for p in prep.passages if p.kind == "fact"]))
            if prep.evidence_score > fact_score + FACT_OVERRIDE_MARGIN:
                return None
        return text

    def run(
        self,
        question: str,
        tickers: list[str] | None = None,
        top_k: int | None = None,
        *,
        llm_gate: Callable[[], bool] | None = None,
        use_llm: bool = True,
    ) -> PipelineResult:
        return self.answer(self.prepare(question, tickers, top_k), llm_gate=llm_gate, use_llm=use_llm)
