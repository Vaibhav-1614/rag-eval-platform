"""
Answer generation with a provider chain:

    Groq (primary model) -> Groq (fallback model) -> Gemini -> extractive (no LLM)

Any provider without a key is skipped, so with no keys at all the service still
answers with cited sentences picked from the retrieved passages.
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field

from rag_service.config import ROOT, RetrievalConfig, ServiceSettings
from rag_service.llm import LLMError, LLMResult, gemini_generate, groq_chat
from rag_service.prompts import load_prompt
from rag_service.retriever import Passage

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import token_set  # noqa: E402
from rag_service.guardrails import stems  # noqa: E402

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(“\"])")
_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Generation:
    answer: str
    mode: str  # "llm" | "extractive" | "refusal"
    provider: str | None = None
    model: str | None = None
    citations: list[int] = field(default_factory=list)
    llm_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    errors: list[str] = field(default_factory=list)


def format_context(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.rank}] ({p.ticker}, {p.chunk_id})\n{p.text.strip()}" for p in passages)


def build_prompt(question: str, passages: list[Passage], cfg: RetrievalConfig, refusal: str, version: str) -> tuple[str, str]:
    prompt = load_prompt(f"answer/{version}")
    text = prompt.render(
        question=question,
        context=format_context(passages),
        corpus_description=cfg.corpus_description,
        refusal_message=refusal,
    )
    return text, prompt.sha


def extractive_answer(question: str, passages: list[Passage], max_sentences: int = 3) -> tuple[str, list[int]]:
    """Pick the sentences that best cover the question's content words; cite their passage."""
    q = stems(token_set(question))
    scored: list[tuple[float, int, int, str]] = []
    seen: set[str] = set()
    order = 0
    for p in passages:
        for line in p.text.splitlines():
            for sent in _SENT_SPLIT.split(line):
                sent = sent.strip()
                if not 40 <= len(sent) <= 600 or sent in seen:
                    continue
                seen.add(sent)
                words = stems(token_set(sent))
                hit = len(q & words) / len(q) if q else 0.0
                if hit == 0:
                    continue
                scored.append((hit + 0.05 / p.rank, order, p.rank, sent))
                order += 1
    best = sorted(scored, key=lambda x: -x[0])[:max_sentences]
    best.sort(key=lambda x: x[1])
    cites = sorted({r for _, _, r, _ in best})
    return " ".join(f"{s} [{r}]" for _, _, r, s in best), cites


def _citations(text: str, n_passages: int) -> list[int]:
    return sorted({int(m) for m in _CITE.findall(text) if 1 <= int(m) <= n_passages})


def generate(
    question: str,
    passages: list[Passage],
    cfg: RetrievalConfig,
    settings: ServiceSettings,
    refusal: str,
    *,
    allow_llm: bool = True,
) -> Generation:
    errors: list[str] = []
    if allow_llm:
        prompt, _ = build_prompt(question, passages, cfg, refusal, settings.prompt_version)
        chain = [
            ("groq", settings.groq_model, settings.groq_key),
            ("groq", settings.groq_fallback_model, settings.groq_key),
            ("gemini", settings.gemini_model, settings.gemini_key),
        ]
        for provider, model, key in chain:
            if not key or not model:
                continue
            try:
                fn = groq_chat if provider == "groq" else gemini_generate
                res: LLMResult = fn(prompt, api_key=key, model=model)
            except LLMError as e:
                errors.append(str(e))
                logger.warning("llm_fallback", extra={"provider": provider, "model": model, "error": str(e)})
                continue
            refused = refusal.split(".")[0].lower() in res.text.lower()
            return Generation(
                answer=refusal if refused else res.text,
                mode="refusal" if refused else "llm",
                provider=res.provider,
                model=res.model,
                citations=[] if refused else _citations(res.text, len(passages)),
                llm_latency_ms=res.latency_ms,
                input_tokens=res.input_tokens,
                output_tokens=res.output_tokens,
                errors=errors,
            )
    text, cites = extractive_answer(question, passages)
    if not text:
        return Generation(answer=refusal, mode="refusal", errors=errors)
    return Generation(answer=text, mode="extractive", citations=cites, errors=errors)
