"""
Answer generation with a provider chain:

    Groq (primary model) -> Groq (fallback model) -> Gemini -> extractive (no LLM)

Any provider without a key is skipped, so with no keys at all the service still
answers with cited sentences picked from the retrieved passages.

LLM answers are verified before they are returned (verify_answer): every sentence must
cite a passage, and every number in it must appear in a passage it cites. Sentences that
fail are dropped; if nothing survives, the extractive answer is used instead.
"""
from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
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
_ANSWER_SENT = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z(“\"$])")
_CITE = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?")

Scorer = Callable[[str, list[str]], list[float]]


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
    verification: dict | None = None  # {"sentences": n, "dropped": [..]} for LLM answers


def format_context(passages: list[Passage]) -> str:
    out = []
    for p in passages:
        where = p.context or f"{p.ticker}, {p.chunk_id}"
        if p.context and p.section and p.section not in p.context:
            where = f"{p.context} | {p.section}"
        out.append(f"[{p.rank}] ({where})\n{p.text.strip()}")
    return "\n\n".join(out)


def build_prompt(question: str, passages: list[Passage], cfg: RetrievalConfig, refusal: str, version: str) -> tuple[str, str]:
    prompt = load_prompt(f"answer/{version}")
    text = prompt.render(
        question=question,
        context=format_context(passages),
        corpus_description=cfg.corpus_description,
        refusal_message=refusal,
    )
    return text, prompt.sha


# ---------------- extractive ----------------
def _units(p: Passage) -> list[str]:
    """Answerable units of a passage: sentences for prose; for a table, each row together
    with the table's caption/header lines (a bare row of numbers means nothing)."""
    lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
    if p.kind == "table":
        head = lines[:3] if len(lines) > 4 else lines[:1]
        head_txt = " | ".join(head)
        return [f"{head_txt} | {row}" for row in lines[len(head):]][:40]
    out = []
    for ln in lines:
        out += [s.strip() for s in _SENT_SPLIT.split(ln) if 40 <= len(s.strip()) <= 600]
    return out


def extractive_answer(
    question: str, passages: list[Passage], max_sentences: int = 2, scorer: Scorer | None = None
) -> tuple[str, list[int]]:
    """
    Quote the passage sentences that answer the question, each with its citation.
    With a `scorer` (the cross-encoder) sentences are ranked by relevance and only ones
    close to the best are kept; without one, by overlap with the question's words.
    """
    cands: list[tuple[int, str]] = []
    seen: set[str] = set()
    for p in passages[:4]:
        for u in _units(p):
            if u not in seen:
                seen.add(u)
                cands.append((p.rank, u))
    if not cands:
        return "", []
    cands = cands[:60]
    if scorer:
        scores = scorer(question, [u for _, u in cands])
    else:
        q = stems(token_set(question))
        scores = []
        for rank, u in cands:
            words = stems(token_set(u))
            scores.append((len(q & words) / len(q) if q else 0.0) + 0.05 / rank)
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    best = scores[order[0]]
    if not scorer and best <= 0.06:
        return "", []
    margin = 2.5 if scorer else 0.15
    keep = [i for i in order[:max_sentences] if scores[i] >= best - margin]
    keep.sort()
    cites = sorted({cands[i][0] for i in keep})
    return " ".join(f"{cands[i][1]} [{cands[i][0]}]" for i in keep), cites


# ---------------- verification ----------------
def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _number_supported(x: float, source: set[float]) -> bool:
    if x in source:
        return True
    # Unit restatements: "$416.2 billion" for 416,161 (millions), "58 thousand" for 58,000,
    # and rounding to the digits the answer shows.
    for p in source:
        for scale in (1e3, 1e6, 1e9, 1e-3):
            v = p / scale
            for nd in (0, 1, 2, 3):
                if round(v, nd) == x:
                    return True
        for nd in (0, 1, 2):
            if round(p, nd) == x:
                return True
    return False


def verify_answer(answer: str, passages: list[Passage]) -> tuple[str, dict]:
    """
    Keep only sentences that (a) cite at least one existing passage and (b) contain no
    number that is missing from the passages they cite. Returns (verified answer, report).
    Small integers (< 10) and citation markers are not treated as figures.
    """
    by_rank = {p.rank: p for p in passages}
    kept: list[str] = []
    dropped: list[dict] = []
    sentences = [s.strip() for s in _ANSWER_SENT.split(answer.strip()) if s.strip()]
    for sent in sentences:
        cites = [int(c) for c in _CITE.findall(sent)]
        valid = [c for c in cites if c in by_rank]
        body = _CITE.sub("", sent)
        if not valid:
            dropped.append({"sentence": sent, "reason": "no valid citation"})
            continue
        source_text = " ".join(f"{by_rank[c].context or ''} {by_rank[c].text}" for c in valid)
        source = set(_numbers(source_text))
        missing = [n for n in _numbers(body) if n >= 10 and not _number_supported(n, source)]
        if missing:
            dropped.append({"sentence": sent, "reason": f"numbers not in cited passages: {missing[:4]}"})
            continue
        kept.append(sent)
    return " ".join(kept), {"sentences": len(sentences), "dropped": dropped}


def _citations(text: str, n_passages: int) -> list[int]:
    return sorted({int(m) for m in _CITE.findall(text) if 1 <= int(m) <= n_passages})


# ---------------- chain ----------------
def generate(
    question: str,
    passages: list[Passage],
    cfg: RetrievalConfig,
    settings: ServiceSettings,
    refusal: str,
    *,
    allow_llm: bool = True,
    llm_gate: Callable[[], bool] | None = None,
    scorer: Scorer | None = None,
    direct_answer: str | None = None,
) -> Generation:
    """
    `llm_gate` is asked before every provider call (the shared daily cap counts calls,
    not requests). `direct_answer` is a ready, cited sentence from the facts table, used
    by the no-LLM path instead of sentence extraction.
    """
    errors: list[str] = []

    def fallback(verification: dict | None = None) -> Generation:
        if direct_answer:
            return Generation(direct_answer, "extractive", citations=_citations(direct_answer, len(passages)),
                              errors=errors, verification=verification)
        text, cites = extractive_answer(question, passages, scorer=scorer)
        if not text:
            return Generation(answer=refusal, mode="refusal", errors=errors, verification=verification)
        return Generation(answer=text, mode="extractive", citations=cites, errors=errors, verification=verification)

    if not allow_llm:
        return fallback()
    prompt, _ = build_prompt(question, passages, cfg, refusal, settings.prompt_version)
    chain = [
        ("groq", settings.groq_model, settings.groq_key),
        ("groq", settings.groq_fallback_model, settings.groq_key),
        ("gemini", settings.gemini_model, settings.gemini_key),
    ]
    for provider, model, key in chain:
        if not key or not model:
            continue
        if llm_gate and not llm_gate():
            errors.append("daily LLM cap reached")
            break
        try:
            fn = groq_chat if provider == "groq" else gemini_generate
            res: LLMResult = fn(prompt, api_key=key, model=model)
        except LLMError as e:
            errors.append(str(e))
            logger.warning("llm_fallback", extra={"provider": provider, "model": model, "error": str(e)})
            continue
        meta = {"provider": res.provider, "model": res.model, "llm_latency_ms": res.latency_ms,
                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "errors": errors}
        if refusal.split(".")[0].lower() in res.text.lower():
            return Generation(answer=refusal, mode="refusal", **meta)
        verified, report = verify_answer(res.text, passages)
        if not verified:
            logger.warning("answer_failed_verification", extra={"model": res.model, "dropped": report["dropped"][:3]})
            g = fallback(report)
            g.provider, g.model, g.llm_latency_ms = res.provider, res.model, res.latency_ms
            g.input_tokens, g.output_tokens = res.input_tokens, res.output_tokens
            return g
        return Generation(answer=verified, mode="llm", citations=_citations(verified, len(passages)),
                          verification=report, **meta)
    return fallback()
