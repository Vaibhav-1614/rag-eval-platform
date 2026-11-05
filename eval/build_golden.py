"""
Build eval/golden.jsonl — the regression set the CI gate runs on.

Three parts:
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


def build() -> list[dict]:
    rows: list[dict] = []
    filings = {p.stem: _norm(p.read_text(encoding="utf-8")) for p in (ROOT / "data" / "parsed").glob("*.txt")}

    def answerable(qid: str, ticker: str, q: str, evidence: str, source: str) -> dict:
        if _norm(evidence) not in filings[ticker]:
            raise SystemExit(f"{qid}: evidence not verbatim in data/parsed/{ticker}.txt")
        ids = _gold_ids(ticker, evidence)
        if not ids:
            raise SystemExit(f"{qid}: no 256-token chunk contains the evidence")
        return {"id": qid, "ticker": ticker, "question": q, "answerable": True,
                "reference_answer": evidence, "evidence": evidence,
                "gold_chunk_ids": ids, "source": source, "category": "fact"}

    bench = json.loads((ROOT / "data" / "testset.json").read_text(encoding="utf-8"))
    for i, item in enumerate(bench, start=1):
        rows.append(answerable(f"bench-{i:03d}", item["ticker"], item["question"],
                               item["ground_truth_answer"], "benchmark"))
    for i, (t, q, ev) in enumerate(HELDOUT, start=1):
        rows.append(answerable(f"held-{i:03d}", t, q, ev, "heldout"))
    for i, (t, q, cat) in enumerate(UNANSWERABLE, start=1):
        rows.append({"id": f"unans-{i:03d}", "ticker": t, "question": q, "answerable": False,
                     "reference_answer": "UNANSWERABLE", "evidence": None, "gold_chunk_ids": [],
                     "source": "unanswerable", "category": cat})
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
