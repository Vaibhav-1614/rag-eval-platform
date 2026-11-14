"""
Structured path for headline figures.

Questions like "What was Apple's revenue?" or "Which company has the most employees?"
are answered from data/facts/facts.json (exact XBRL values of the indexed 10-K filings,
built by scripts/build_facts.py) instead of hoping a table fragment gets retrieved.
A matching fact is added to the passages as a `fact` passage, so the LLM (or the
extractive path) cites it like any other source.

The match is deliberately narrow: a company-wide metric, with no segment or product
qualifier. "iPhone revenue" or "AWS operating income" is left to text retrieval.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from rag_service.config import ROOT
from rag_service.retriever import Passage

FACTS_PATH = ROOT / "data" / "facts" / "facts.json"

# Ordered: the first pattern that matches wins, so specific metrics come before general ones.
METRIC_PATTERNS: list[tuple[str, str]] = [
    ("eps_diluted", r"\b(diluted (eps|earnings per share)|eps|earnings? per (diluted )?share|earn(ed|s)? per share)\b"),
    ("eps_basic", r"\bbasic (eps|earnings per share)\b"),
    ("dividends_per_share", r"\bdividends? (declared |paid )?per share\b"),
    ("gross_margin", r"\bgross (profit )?margin( percentage)?\b"),
    ("operating_margin", r"\boperating (profit )?margin\b"),
    ("net_margin", r"\b(net|profit) margin\b"),
    ("gross_profit", r"\bgross profit\b"),
    ("operating_cash_flow", r"\b(operating cash flow|cash (flow )?(from|provided by|generated (by|from)) operat\w+)\b"),
    ("operating_income", r"\boperating (income|profit|earnings)\b"),
    ("capex", r"\b(capex|capital (expenditures?|spending))\b"),
    ("rd_expense", r"\b(r\s?&\s?d|research (and|&) development)\b"),
    ("share_repurchases", r"\b(share|stock) (repurchases?|buy ?backs?)\b|\bbuy ?backs?\b"),
    ("income_tax", r"\b(income tax(es)? (expense|provision)|provision for income taxes|tax expense)\b"),
    ("cost_of_revenue", r"\bcost of (revenue|sales|goods sold)\b"),
    ("long_term_debt", r"\blong[- ]term debt\b"),
    ("total_assets", r"\btotal assets\b"),
    ("total_liabilities", r"\btotal liabilities\b"),
    ("equity", r"\b(stockholders'?|shareholders'?) equity\b"),
    ("employees", r"\b(employees?|headcount|workforce|staff|workers|people (work|does|do|did|are employed)|employs?)\b"),
    ("cash", r"\bcash( and cash equivalents| on hand| position)?\b"),
    ("net_income", r"\b(net (income|profit|earnings)|profits?|bottom line|earn(ed|s)?)\b"),
    ("revenue", r"\b(revenues?|net sales|sales|turnover|top line)\b"),
]
_METRICS = [(name, re.compile(pat, re.I)) for name, pat in METRIC_PATTERNS]

# Lower-case words that turn a company total into a part of it.
_SEGMENT_WORDS = frozenset(
    "advertising ad ads cloud services service subscription subscriptions product products hardware software "
    "automotive energy gaming segment segments international domestic foreign online retail licensing interest "
    "noninterest trading banking upstream downstream chemical chemicals pharmaceutical medtech device devices "
    "search app apps store stores regional china europe americas asia us u.s japan india quarterly deferred "
    "unearned restricted segment's".split()
)
_PRODUCT_ALIASES = frozenset({"iphone", "azure", "aws", "youtube", "instagram", "whatsapp"})
_AFTER_PREP = re.compile(r"^\s*(?:\w+\s+){0,3}?(from|by|per|in|for|of|among|across|within)\s+((?:[\w&.'’-]+\s*){1,3})", re.I)
_NOT_NARROWING = re.compile(
    r"^(fiscal|fy|(19|20)\d\d|the (last|latest|most|past|prior|previous|year|fiscal|company)|last|its|their|total|all|each|"
    r"during|over|a year|that year|this year|millions?|billions?|dollars|usd|research|operations|operating)\b", re.I)


def _narrowed_after(rest: str) -> bool:
    m = _AFTER_PREP.match(rest)
    if not m:
        return False
    phrase = m.group(2).strip()
    if m.group(1).lower() in {"among", "across", "within"}:
        return True
    first = re.sub(r"['’]s$", "", phrase.split()[0].lower()) if phrase else ""
    if _NOT_NARROWING.match(phrase) or _is_company_word(first) or first in {"the", "a", "an"} and _is_company_word(
        re.sub(r"['’]s$", "", (phrase.split() + [""])[1].lower())
    ):
        return False
    return True


_NOT_ANNUAL_ACTUAL = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[- ]quarter\b|\bq[1-4]\b|\bquarterly\b|\bmonth(ly)?\b"
    r"|\bwill\b|\bnext (year|quarter)\b|\bforecast|\bguidance\b|\bexpect(s|ed)? to\b|\bproject(ed|ion)\b|\btarget\b",
    re.I,
)
_SUPERLATIVE = re.compile(
    r"\b(highest|most|largest|biggest|greatest|top|leads?|lowest|least|smallest|fewest|rank(ed|ing|s)?|bigger|larger|"
    r"more|higher|lower|less|fewer)\b",
    re.I,
)
_LOWEST = re.compile(r"\b(lowest|least|smallest|fewest|lower|less|fewer)\b", re.I)
_YEAR = re.compile(r"\b(?:fy\s?|fiscal (?:year )?)?((?:19|20)\d\d)\b", re.I)
_GROWTH = re.compile(r"\b(grow(th|n|ing)?|grew|increase[ds]?|decrease[ds]?|declin\w+|change[ds]?|rise|rose|fell|drop\w*|"
                     r"compar\w+ (to|with) (last|prior|previous)|year[- ]over[- ]year|yoy|trend)\b", re.I)


@dataclass
class FactHit:
    ticker: str
    company: str
    metric: str
    label: str
    unit: str
    tag: str
    values: dict[int, float]  # fiscal year -> value
    year: int                 # the year the question is about
    filed: str
    url: str
    evidence: str | None = None


@lru_cache(maxsize=1)
def load_facts() -> dict:
    return json.loads(FACTS_PATH.read_text(encoding="utf-8")) if FACTS_PATH.exists() else {}


def fmt(value: float, unit: str) -> str:
    if unit == "USD":
        sign = "-" if value < 0 else ""
        m = abs(value) / 1e6
        text = f"{sign}${m:,.0f} million"
        if m >= 1000:
            text += f" ({sign}${m / 1000:,.2f} billion)"
        return text
    if unit == "USD/shares":
        return f"${value:,.2f}"
    if unit == "percent":
        return f"{value:.1f}%"
    if unit == "employees":
        return f"{value:,.0f}"
    return f"{value:,}"


def detect_metric(question: str, alias_spans: list[tuple[int, int]] | None = None) -> str | None:
    """The company-wide metric a question asks about, or None (no metric, or it is
    narrowed to a segment/product/region and must come from the text)."""
    for name, pat in _METRICS:
        m = pat.search(question)
        if not m:
            continue
        # Qualifier right before the metric: a proper noun ("iPhone revenue", "Data Center
        # revenue") or a common segment word ("advertising revenue", "cloud revenue").
        before = list(re.finditer(r"[A-Za-z&']+", question[: m.start()]))[-2:]
        for w in before[-1:]:
            word = re.sub(r"['’]s$", "", w.group(0))
            if word != w.group(0):
                continue  # possessive: "Apple's revenue", "the company's revenue"
            proper = word[:1].isupper() and w.start() > 0 and not _is_company_word(word.lower())
            if proper or word.lower() in _SEGMENT_WORDS or word.lower() in _PRODUCT_ALIASES:
                return None
        # Qualifier after it ("revenue from advertising", "sales in China", "revenue among customers").
        if _narrowed_after(question[m.end():]):
            return None
        return name
    return None


def _is_company_word(w: str) -> bool:
    from rag_service.retriever import _ALIASES

    if w in _PRODUCT_ALIASES:
        return False  # "AWS operating income" is a segment figure, not Amazon's total
    return any(w in alias.split() for alias in _ALIASES) or w.upper() in load_facts()


def find_facts(question: str, tickers: list[str]) -> tuple[list[FactHit], str]:
    """Returns (hits, mode): mode is "single" (named companies), "ranking" (no company
    named + a superlative: all companies, sorted) or "none"."""
    facts = load_facts()
    metric = detect_metric(question) if facts else None
    # The table holds full-year actuals only: no quarters, no forecasts.
    if not metric or _NOT_ANNUAL_ACTUAL.search(question):
        return [], "none"
    years = [int(y) for y in _YEAR.findall(question)]
    ranking = not tickers and bool(_SUPERLATIVE.search(question))
    if not tickers and not ranking:
        return [], "none"
    hits: list[FactHit] = []
    for t in tickers or list(facts):
        c = facts.get(t)
        m = (c or {}).get("metrics", {}).get(metric)
        if not m:
            continue
        values = {int(y): v for y, v in m["values"].items()}
        year = years[0] if years else c["fiscal_year"]
        if year not in values:
            if ranking:
                continue
            return [], "none"  # asked for a year this filing does not report
        hits.append(FactHit(t, c["company"], metric, m["label"], m["unit"], m["tag"], values, year,
                            c["filed"], c["url"], m.get("evidence")))
    if ranking:
        hits.sort(key=lambda h: h.values[h.year], reverse=not _LOWEST.search(question))
    return hits, ("ranking" if ranking else "single") if hits else "none"


def fact_text(h: FactHit) -> str:
    series = "; ".join(f"fiscal {y}: {fmt(v, h.unit)}" for y, v in sorted(h.values.items(), reverse=True))
    src = "stated in the filing text" if h.tag == "filing text" else f"XBRL tag {h.tag}"
    text = f"{h.company} — {h.label} (company-wide): {series}. Source: Form 10-K filed {h.filed}, {src}."
    if h.evidence:
        text += f' Filing sentence: "{h.evidence}"'
    return text


def fact_passages(hits: list[FactHit], mode: str) -> list[Passage]:
    if mode == "ranking":
        lines = [f"{i}. {h.company}: {fmt(h.values[h.year], h.unit)} (fiscal {h.year})" for i, h in enumerate(hits, 1)]
        text = (f"{hits[0].label}, company-wide, latest fiscal year in each indexed 10-K, ranked:\n" + "\n".join(lines)
                + "\nSource: XBRL data / filing text of each company's Form 10-K. Fiscal years end on different dates.")
        return [Passage("FACTS_ranking_" + hits[0].metric, "ALL", text, 1.0, 1, kind="fact",
                        section="Structured financial data (XBRL)", context="Cross-company ranking")]
    return [
        Passage(f"{h.ticker}_fact_{h.metric}", h.ticker, fact_text(h), 1.0, i, kind="fact",
                section="Structured financial data (XBRL)", context=f"{h.company} 10-K, fiscal {max(h.values)}",
                source_url=h.url, fiscal_year=max(h.values))
        for i, h in enumerate(hits, 1)
    ]


def fact_answer(question: str, hits: list[FactHit], mode: str) -> str:
    """A direct, cited sentence for the no-LLM path."""
    if mode == "ranking":
        low = bool(_LOWEST.search(question))
        top = hits[0]
        rest = ", ".join(f"{h.company} ({fmt(h.values[h.year], h.unit)})" for h in hits[1:4])
        word = "lowest" if low else "highest"
        return (f"{top.company} has the {word} {top.label.lower()} of the ten indexed companies: "
                f"{fmt(top.values[top.year], top.unit)} in fiscal {top.year} [1]. Next: {rest} [1].")
    parts = []
    for i, h in enumerate(hits, 1):
        s = f"{h.company}'s {h.label.lower()} was {fmt(h.values[h.year], h.unit)} in fiscal {h.year}"
        prev = h.values.get(h.year - 1)
        if prev is not None and (_GROWTH.search(question) or len(hits) == 1):
            if h.unit == "percent":
                s += f", compared with {fmt(prev, h.unit)} in fiscal {h.year - 1}"
            elif prev:
                s += f", {'up' if h.values[h.year] >= prev else 'down'} {abs(h.values[h.year] / prev - 1) * 100:.1f}% from {fmt(prev, h.unit)} in fiscal {h.year - 1}"
        parts.append(f"{s} [{i}].")
    if len(hits) == 2 and hits[0].unit == hits[1].unit:
        a, b = sorted(hits, key=lambda h: -h.values[h.year])
        parts.append(f"{a.company}'s was higher.")
    return " ".join(parts)
