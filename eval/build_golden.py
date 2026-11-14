"""
Build eval/golden.jsonl — the regression set the CI gate runs on.

Parts (see eval/golden_items.py for the v2 additions: numeric, comparison,
cross-company, lookup, thematic and more unanswerable questions, split dev/test):
  * benchmark (50): the natural-language questions from data/testset.json. These
    picked the frozen config, so they guard against regressions but are optimistic.
  * heldout (12): new questions written from the filings after the config was frozen.
  * unanswerable (12): out-of-corpus companies, periods after the filings, market data,
    and forecasts. The correct behaviour is the refusal message.

For every answerable item `evidence` is a verbatim span of data/parsed/<TICKER>.txt
(checked here; build fails otherwise) and `gold_chunk_ids` lists the 256-token chunks
that contain at least EVIDENCE_WINDOW characters of it, i.e. chunk-size independent.

    python eval/build_golden.py           # rewrite eval/golden.jsonl
    python eval/build_golden.py --check   # verify the committed file is up to date
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import artifacts  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.jsonl"
EVIDENCE_WINDOW = 60

HELDOUT = [
    ("MSFT", "How many full-time employees did Microsoft have at the end of its fiscal year?",
     "As of June 30, 2025, we employed approximately 228,000 people on a full-time basis"),
    ("AMZN", "How big is Amazon's workforce?",
     "As of December 31, 2025, we employed approximately 1,576,000 full-time and part-time employees."),
    ("JNJ", "Roughly how many people work at Johnson & Johnson?",
     "Our approximately 138,200 employees are critical drivers of the Company’s success."),
    ("GOOGL", "What was Alphabet's headcount at the end of 2025?",
     "As of December 31, 2025, Alphabet had 190,820 employees."),
    ("TSLA", "How many employees did Tesla have worldwide at the end of 2025?",
     "As of December 31, 2025, our employee headcount worldwide was 134,785."),
    ("JPM", "How many people does JPMorgan Chase employ globally?",
     "As of December 31, 2025, JPMorganChase had 318,512 employees globally."),
    ("AAPL", "In which city is Apple headquartered?",
     "The Company’s headquarters is located in Cupertino, California."),
    ("META", "When and where was Meta incorporated?",
     "We were incorporated in Delaware in July 2004."),
    ("NVDA", "Where was NVIDIA originally incorporated, and when did it move to Delaware?",
     "NVIDIA was incorporated in California in April 1993 and reincorporated in Delaware in April 1998."),
    ("TSLA", "Which city is Tesla's headquarters in?",
     "We are headquartered in Austin, Texas."),
    ("JPM", "Where is JPMorgan Chase's headquarters building?",
     "JPMorganChase’s headquarters is located in New York City at 270 Park Avenue, a 60-story office building owned by the Firm."),
    ("XOM", "In what state and year was Exxon Mobil incorporated?",
     "Exxon Mobil Corporation was incorporated in the State of New Jersey in 1882."),
]

UNANSWERABLE = [
    (None, "What was Netflix's paid subscriber count at the end of 2025?", "other_company"),
    (None, "How much revenue did Walmart report for its latest fiscal year?", "other_company"),
    (None, "How many Starbucks stores operate in China?", "other_company"),
    ("AAPL", "What is Apple's stock price today?", "market_data"),
    ("NVDA", "What will NVIDIA's data center revenue be in fiscal 2029?", "forecast"),
    ("TSLA", "How many vehicles will Tesla deliver in 2030?", "forecast"),
    ("JNJ", "What dividend per share will Johnson & Johnson pay in 2029?", "forecast"),
    ("MSFT", "What was Microsoft's revenue in fiscal 2015?", "other_period"),
    ("JPM", "What was JPMorgan Chase's net income in the third quarter of 2026?", "other_period"),
    ("AMZN", "What was AWS operating income in the first quarter of 2027?", "other_period"),
    ("META", "What did Meta's CEO say on the second-quarter 2026 earnings call?", "not_in_filing"),
    ("XOM", "What is Exxon Mobil's market share in Brazilian EV charging?", "not_in_filing"),
]


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def chunk_has_evidence(chunk_text: str, evidence: str, window: int = EVIDENCE_WINDOW) -> bool:
    """True if the chunk contains the whole evidence or any `window`-char slice of it."""
    c, e = _norm(chunk_text), _norm(evidence)
    if e in c:
        return True
    if len(e) <= window:
        return False
    return any(e[i : i + window] in c for i in range(0, len(e) - window + 1, 10))


def _gold_ids(ticker: str, evidence: str) -> list[str]:
    chunks = artifacts.load_chunks(ticker, 256) or []
    return [c["chunk_id"] for c in chunks if chunk_has_evidence(c["text"], evidence)]


def number_forms(value: float, unit: str) -> tuple[str, list[str]]:
    """(the form printed in the filing's tables, forms an answer may use)."""
    if unit == "USD":
        m = value / 1e6
        filing = f"{m:,.0f}"
        forms = [filing, f"{m / 1e3:,.2f} billion", f"{m / 1e3:,.1f} billion"]
        if m >= 1e6:
            forms += [f"{m / 1e6:.2f} trillion", f"{m / 1e6:.1f} trillion"]
        return filing, forms
    if unit == "USD/shares":
        return f"{value:.2f}", [f"{value:.2f}"]
    if unit == "employees":
        return f"{value:,.0f}", [f"{value:,.0f}", f"{value / 1000:,.0f} thousand", f"{value / 1e6:.2f} million"]
    raise ValueError(unit)


SHORT_NAME = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta",
              "NVDA": "NVIDIA", "TSLA": "Tesla", "JPM": "JPMorgan", "JNJ": "Johnson & Johnson", "XOM": "Exxon"}


def build() -> list[dict]:
    from eval.golden_items import COMPARISON, CROSS_COMPANY, LOOKUP, NUMERIC, THEMATIC, UNANSWERABLE_V2

    rows: list[dict] = []
    filings = {p.stem: _norm(p.read_text(encoding="utf-8")) for p in (ROOT / "data" / "parsed").glob("*.txt")}
    facts = json.loads((ROOT / "data" / "facts" / "facts.json").read_text(encoding="utf-8"))

    def base(qid: str, split: str, kind: str, source: str, tickers: list[str], q: str, **kw) -> dict:
        row = {"id": qid, "split": split, "type": kind, "source": source, "tickers": tickers, "question": q,
               "answerable": True, "reference_answer": None, "evidence_any": None, "section_rule": None,
               "answer_contains": None, "gold_chunk_ids": []}
        row.update(kw)
        return row

    def lookup(qid: str, split: str, source: str, ticker: str, q: str, evidence: str, contains: str | None = None) -> dict:
        if _norm(evidence) not in filings[ticker]:
            raise SystemExit(f"{qid}: evidence not verbatim in data/parsed/{ticker}.txt")
        ids = _gold_ids(ticker, evidence)
        if not ids:
            raise SystemExit(f"{qid}: no 256-token chunk contains the evidence")
        if contains and _norm(contains) not in _norm(evidence):
            raise SystemExit(f"{qid}: answer string {contains!r} is not in the evidence")
        return base(qid, split, "lookup", source, [ticker], q, reference_answer=evidence, evidence_any=[evidence],
                    answer_contains=[[contains]] if contains else None, gold_chunk_ids=ids)

    def fact(qid: str, ticker: str, metric: str, year: int | None) -> tuple[str, list[str], str]:
        c = facts[ticker]
        m = c["metrics"][metric]
        y = str(year or c["fiscal_year"])
        if y not in m["values"]:
            raise SystemExit(f"{qid}: {ticker} {metric} has no value for {y}")
        filing_form, forms = number_forms(m["values"][y], m["unit"])
        # The number as printed in the filing; for headcount stated in words ("58 thousand")
        # fall back to the sentence it comes from.
        evidence = filing_form if _norm(filing_form) in filings[ticker] else m.get("evidence", filing_form)
        if _norm(evidence) not in filings[ticker]:
            raise SystemExit(f"{qid}: {evidence!r} not found in data/parsed/{ticker}.txt")
        ref = f"{c['company']} {m['label'].lower()}, fiscal {y}: {forms[0]}" + (" (USD millions)" if m["unit"] == "USD" else "")
        return evidence, forms, ref

    def split_of(i: int) -> str:
        return "dev" if i % 2 == 0 else "test"

    # --- v1 golden set: all in dev (it picked the v1 config and tuned the v1 rules) ---
    bench = json.loads((ROOT / "data" / "testset.json").read_text(encoding="utf-8"))
    for i, item in enumerate(bench, start=1):
        rows.append(lookup(f"bench-{i:03d}", "dev", "benchmark", item["ticker"], item["question"], item["ground_truth_answer"]))
    for i, (t, q, ev) in enumerate(HELDOUT, start=1):
        rows.append(lookup(f"held-{i:03d}", "dev", "heldout", t, q, ev))
    for i, (t, q, cat) in enumerate(UNANSWERABLE, start=1):
        rows.append(base(f"unans-{i:03d}", "dev", "unanswerable", "unanswerable", [t] if t else [], q,
                         answerable=False, reference_answer="UNANSWERABLE", category=cat))

    # --- v2 additions: alternate dev / test ---
    for i, (t, metric, year, q) in enumerate(NUMERIC):
        qid = f"num-{i + 1:03d}"
        ev, forms, ref = fact(qid, t, metric, year)
        rows.append(base(qid, split_of(i), "numeric", "v2", [t], q, reference_answer=ref, evidence_any=[ev],
                         answer_contains=[forms], metric=metric))
    for i, ((a, b), metric, q) in enumerate(COMPARISON):
        qid = f"cmp-{i + 1:03d}"
        (ev_a, forms_a, ref_a), (ev_b, forms_b, ref_b) = fact(qid, a, metric, None), fact(qid, b, metric, None)
        rows.append(base(qid, split_of(i), "comparison", "v2", [a, b], q, reference_answer=f"{ref_a}; {ref_b}",
                         evidence_any=[ev_a, ev_b], answer_contains=[forms_a, forms_b], metric=metric))
    for i, (metric, direction, q) in enumerate(CROSS_COMPANY):
        qid = f"xco-{i + 1:03d}"
        have = {t: c["metrics"][metric]["values"][str(c["fiscal_year"])] for t, c in facts.items() if metric in c["metrics"]}
        winner = (max if direction == "max" else min)(have, key=have.get)
        ev, forms, ref = fact(qid, winner, metric, None)
        rows.append(base(qid, split_of(i), "cross_company", "v2", [], q, reference_answer=f"{SHORT_NAME[winner]}: {ref}",
                         evidence_any=[ev], answer_contains=[[SHORT_NAME[winner]], forms], metric=metric))
    for i, (t, q, ev, contains) in enumerate(LOOKUP):
        rows.append(lookup(f"look-{i + 1:03d}", split_of(i), "v2", t, q, ev, contains))
    for i, (t, q, sections, pattern) in enumerate(THEMATIC):
        rows.append(base(f"theme-{i + 1:03d}", split_of(i), "thematic", "v2", [t], q,
                         reference_answer=f"Discussed in Item {'/'.join(sections)} of the 10-K.",
                         section_rule={"sections": sections, "pattern": pattern}))
    for i, (t, q, cat) in enumerate(UNANSWERABLE_V2):
        rows.append(base(f"unans2-{i + 1:03d}", split_of(i), "unanswerable", "v2", [t] if t else [], q,
                         answerable=False, reference_answer="UNANSWERABLE", category=cat))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build())
    if args.check:
        if not GOLDEN.exists() or GOLDEN.read_text(encoding="utf-8") != text:
            raise SystemExit("eval/golden.jsonl is stale: run python eval/build_golden.py")
        print("golden.jsonl up to date")
        return
    GOLDEN.write_text(text, encoding="utf-8")
    print(f"wrote {GOLDEN.relative_to(ROOT)}: {text.count(chr(10))} items")


if __name__ == "__main__":
    main()
