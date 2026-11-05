"""
Input validation and a pre-generation scope check.

The scope check is a cheap, deterministic filter for questions the corpus cannot
answer (other companies, market data, periods after the filings). It runs before any
LLM call so refusals cost nothing; the answer prompt also tells the LLM to refuse,
so the LLM is the second line of defence when keys are configured.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from rag_service.config import ROOT, RetrievalConfig

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import token_set  # noqa: E402

LATEST_FILING_YEAR = 2026  # NVIDIA fiscal 2026 is the newest period in the corpus
MIN_COVERAGE = 0.3  # share of question terms (stemmed) found in the retrieved passages

_YEAR = re.compile(r"\b(19[5-9]\d|20\d\d)\b")
_MARKET = re.compile(
    r"\b(stock|share) price\b|\btrading at\b|\bmarket cap(italization)? (today|now)\b"
    r"|\b(today|right now|currently trading|this week)\b",
    re.I,
)
_FORECAST = re.compile(r"\bwill\b.*\b(be|pay|deliver|earn|report|reach)\b|\bforecast\b|\bpredict", re.I)
_QUARTER = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[- ]quarter\b|\bQ[1-4]\b", re.I)
_FINANCIAL = re.compile(r"\b(revenue|revenues|sales|income|earnings|profit|margin|eps|cash flow|dividend)s?\b", re.I)
_CALL = re.compile(r"\bearnings call\b|\bpress release\b|\binterview\b|\btweet", re.I)
# Words in a question that carry no topical signal for coverage.
_QWORDS = frozenset(
    "what how much many which who whom where when why does did do was were is are the its "
    "their latest report reported recent describe explain tell".split()
)
_CAP_WORD = re.compile(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,}(?:[A-Z][a-z]+)?)\b")
_ALIAS_WORDS = frozenset(
    "apple microsoft alphabet google amazon meta nvidia tesla jpmorgan chase johnson exxon "
    "mobil exxonmobil facebook instagram whatsapp youtube azure aws platforms".split()
)


def stems(words: set[str]) -> set[str]:
    # 5-char prefixes: cheap stemming so "headquartered" matches "headquarters".
    return {w[:5] for w in words}


@dataclass
class ScopeDecision:
    in_scope: bool
    reason: str
    coverage: float


def refusal_message(cfg: RetrievalConfig) -> str:
    return (
        "I can't answer that from the indexed filings. This service only covers the "
        f"{cfg.corpus_description} for: {', '.join(cfg.tickers)}."
    )


def clean_question(q: str, max_chars: int) -> str:
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        raise ValueError("question is empty")
    if len(q) > max_chars:
        raise ValueError(f"question longer than {max_chars} characters")
    return q


def scope_check(question: str, passages_text: list[str], detected: list[str]) -> ScopeDecision:
    ctx = " ".join(passages_text)
    ctx_stems = stems(token_set(ctx))
    q_terms = stems({t for t in token_set(question) if t not in _QWORDS and t not in _ALIAS_WORDS})
    coverage = len(q_terms & ctx_stems) / len(q_terms) if q_terms else 0.0

    if _MARKET.search(question):
        return ScopeDecision(False, "market_data", coverage)
    if _CALL.search(question):
        return ScopeDecision(False, "not_in_filing", coverage)
    years = [int(y) for y in _YEAR.findall(question)]
    if any(y > LATEST_FILING_YEAR for y in years):
        return ScopeDecision(False, "future_period", coverage)
    # 10-Ks are annual; quarterly figures for periods after the filings can't be there.
    if _QUARTER.search(question) and any(y >= LATEST_FILING_YEAR for y in years):
        return ScopeDecision(False, "future_period", coverage)
    if years and not all(str(y) in ctx for y in years):
        return ScopeDecision(False, "period_not_in_context", coverage)
    # Financial statements cover the last 3 fiscal years (5 at most in some tables).
    if _FINANCIAL.search(question) and any(y < LATEST_FILING_YEAR - 5 for y in years):
        return ScopeDecision(False, "period_not_in_corpus", coverage)
    if _FORECAST.search(question) and not detected:
        return ScopeDecision(False, "forecast", coverage)
    if not detected:
        # A capitalised name that never appears in what we retrieved is most likely a
        # company or product outside the corpus ("Netflix", "Starbucks").
        # Requiring two mentions filters names that only appear in passing (a competitor list).
        names = {m.group(1).lower() for m in _CAP_WORD.finditer(question)} - _ALIAS_WORDS
        if any(ctx.lower().count(n) < 2 for n in names):
            return ScopeDecision(False, "unknown_entity", coverage)
    if coverage < MIN_COVERAGE:
        return ScopeDecision(False, "low_coverage", coverage)
    return ScopeDecision(True, "ok", coverage)
