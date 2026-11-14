"""
Build data/facts/facts.json: exact headline figures for the indexed 10-K filings.

Source: SEC "company facts" XBRL API (free, no key):
    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Only values tagged in the *same filing that is indexed* (accession number from
data/raw/<TICKER>/filing_meta.json) are kept, so the facts table and the text index
always describe the same document. Each kept fact carries its XBRL tag, period and
accession number, which is what the service cites.

    python scripts/build_facts.py                 # fetch from SEC and rebuild
    python scripts/build_facts.py --from-dir DIR  # rebuild from saved <TICKER>.json files

SEC asks automated clients to identify themselves; the User-Agent comes from
SEC_USER_AGENT or config/settings.yml (sec.user_agent).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from platform_config import load_settings  # noqa: E402

OUT = ROOT / "data" / "facts" / "facts.json"
API = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# metric -> (label, kind, unit, candidate us-gaap tags in priority order)
# kind: "flow" = over the fiscal year, "instant" = at fiscal year end
METRICS: dict[str, tuple[str, str, str, list[str]]] = {
    "revenue": ("Total revenue", "flow", "USD", [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "RevenuesNetOfInterestExpense", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax"]),
    "cost_of_revenue": ("Cost of revenue", "flow", "USD", [
        "CostOfRevenue", "CostOfGoodsAndServicesSold"]),
    "gross_profit": ("Gross profit", "flow", "USD", ["GrossProfit"]),
    "operating_income": ("Operating income", "flow", "USD", ["OperatingIncomeLoss"]),
    "net_income": ("Net income", "flow", "USD", ["NetIncomeLoss", "ProfitLoss"]),
    "rd_expense": ("Research and development expense", "flow", "USD", [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"]),
    "income_tax": ("Provision for income taxes", "flow", "USD", ["IncomeTaxExpenseBenefit"]),
    "operating_cash_flow": ("Net cash provided by operating activities", "flow", "USD", [
        "NetCashProvidedByUsedInOperatingActivities"]),
    "capex": ("Capital expenditures (purchases of property and equipment)", "flow", "USD", [
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]),
    "share_repurchases": ("Share repurchases", "flow", "USD", [
        "PaymentsForRepurchaseOfCommonStock"]),
    "eps_diluted": ("Diluted earnings per share", "flow", "USD/shares", ["EarningsPerShareDiluted"]),
    "eps_basic": ("Basic earnings per share", "flow", "USD/shares", ["EarningsPerShareBasic"]),
    "dividends_per_share": ("Dividends declared per share", "flow", "USD/shares", [
        "CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"]),
    "cash": ("Cash and cash equivalents", "instant", "USD", [
        "CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks"]),
    "total_assets": ("Total assets", "instant", "USD", ["Assets"]),
    "total_liabilities": ("Total liabilities", "instant", "USD", ["Liabilities"]),
    "equity": ("Total stockholders' equity", "instant", "USD", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "long_term_debt": ("Long-term debt", "instant", "USD", [
        "LongTermDebtNoncurrent", "LongTermDebt"]),
}
# name -> (label, numerator, denominator); computed only when both inputs exist
RATIOS = {
    "gross_margin": ("Gross margin", "gross_profit", "revenue"),
    "operating_margin": ("Operating margin", "operating_income", "revenue"),
    "net_margin": ("Net profit margin", "net_income", "revenue"),
}


# Headcount is not an XBRL tag; it comes from the filing text. Each sentence below is
# checked verbatim against data/parsed/<TICKER>.txt at build time (the build fails if a
# filing no longer contains it), and the number is the one stated in that sentence.
EMPLOYEES: dict[str, tuple[int, str]] = {
    "AAPL": (166000, "As of September 27, 2025, the Company had approximately 166,000 full-time equivalent employees."),
    "MSFT": (228000, "As of June 30, 2025, we employed approximately 228,000 people on a full-time basis"),
    "GOOGL": (190820, "As of December 31, 2025, Alphabet had 190,820 employees."),
    "AMZN": (1576000, "As of December 31, 2025, we employed approximately 1,576,000 full-time and part-time employees."),
    "META": (78865, "Headcount was 78,865 as of December 31, 2025"),
    "NVDA": (42000, "As of the end of fiscal year 2026, we had approximately 42,000 employees in 38 countries"),
    "TSLA": (134785, "As of December 31, 2025, our employee headcount worldwide was 134,785."),
    "JPM": (318512, "As of December 31, 2025, JPMorganChase had 318,512 employees globally."),
    "JNJ": (138200, "Our approximately 138,200 employees are critical drivers of the Company’s success."),
    "XOM": (58000, "The number of regular employees was 58 thousand, 61 thousand, and 62 thousand at years ended 2025, 2024, and 2023, respectively."),
}


# SEC's entityName is upper-case for some filers ("MICROSOFT CORP"); use readable names.
COMPANY = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.", "META": "Meta Platforms, Inc.", "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.", "JPM": "JPMorgan Chase & Co.", "JNJ": "Johnson & Johnson",
    "XOM": "Exxon Mobil Corporation",
}


def _norm(s: str) -> str:
    return " ".join(s.replace("’", "'").split()).lower()


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _fetch(cik: str, user_agent: str) -> dict:
    import httpx

    r = httpx.get(API.format(cik=cik), headers={"User-Agent": user_agent}, timeout=120)
    r.raise_for_status()
    return r.json()


def _rows(gaap: dict, tag: str, unit: str, accn: str) -> list[dict]:
    units = (gaap.get(tag) or {}).get("units") or {}
    return [r for r in units.get(unit, []) if r.get("accn") == accn and r.get("form") == "10-K"]


def build_company(ticker: str, meta: dict, raw: dict) -> dict:
    accn = meta["accessionNumber"]
    gaap = raw["facts"].get("us-gaap", {})

    # Fiscal year end = latest end date of any full-year fact tagged in this filing.
    year_ends: set[str] = set()
    for tag in ("NetIncomeLoss", "ProfitLoss", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"):
        for r in _rows(gaap, tag, "USD", accn):
            if "start" in r and 340 <= _days(r["start"], r["end"]) <= 380:
                year_ends.add(r["end"])
    if not year_ends:
        raise SystemExit(f"{ticker}: no annual facts found for accession {accn}")
    ends = sorted(year_ends, reverse=True)[:3]  # current year + comparatives in the filing
    fy_of = {e: int(e[:4]) for e in ends}  # all ten companies name the fiscal year by its end year

    metrics: dict[str, dict] = {}
    for name, (label, kind, unit, tags) in METRICS.items():
        for tag in tags:
            values: dict[int, float] = {}
            for r in _rows(gaap, tag, unit, accn):
                if r["end"] not in fy_of:
                    continue
                if kind == "flow" and not ("start" in r and 340 <= _days(r["start"], r["end"]) <= 380):
                    continue
                if kind == "instant" and "start" in r:
                    continue
                values[fy_of[r["end"]]] = r["val"]
            if fy_of[ends[0]] in values:
                metrics[name] = {"label": label, "kind": kind, "unit": unit, "tag": f"us-gaap:{tag}",
                                 "values": {str(y): values[y] for y in sorted(values, reverse=True)}}
                break
    if "gross_profit" not in metrics and {"revenue", "cost_of_revenue"} <= metrics.keys():
        rev, cost = metrics["revenue"]["values"], metrics["cost_of_revenue"]["values"]
        vals = {y: rev[y] - cost[y] for y in rev if y in cost}
        if vals:
            metrics["gross_profit"] = {"label": "Gross profit", "kind": "flow", "unit": "USD",
                                       "tag": "derived: revenue - cost of revenue", "values": vals}
    for name, (label, num, den) in RATIOS.items():
        if num in metrics and den in metrics:
            n, d = metrics[num]["values"], metrics[den]["values"]
            vals = {y: round(100 * n[y] / d[y], 2) for y in n if y in d and d[y]}
            if vals:
                metrics[name] = {"label": label, "kind": "flow", "unit": "percent",
                                 "tag": f"derived: {num} / {den}", "values": vals}
    count, sentence = EMPLOYEES[ticker]
    parsed = (ROOT / "data" / "parsed" / f"{ticker}.txt").read_text(encoding="utf-8")
    if _norm(sentence) not in _norm(parsed):
        raise SystemExit(f"{ticker}: headcount sentence not found verbatim in the parsed filing")
    metrics["employees"] = {"label": "Number of employees", "kind": "instant", "unit": "employees",
                            "tag": "filing text", "evidence": sentence,
                            "values": {str(fy_of[ends[0]]): count}}
    return {
        "ticker": ticker,
        "company": COMPANY.get(ticker) or raw.get("entityName", ticker),
        "cik": meta["cik"],
        "accession": accn,
        "filed": meta["filingDate"],
        "url": meta["url"],
        "fiscal_year": fy_of[ends[0]],
        "period_end": ends[0],
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", help="directory with saved <TICKER>.json company-facts files")
    args = ap.parse_args()
    settings = load_settings()
    ua = os.environ.get("SEC_USER_AGENT") or settings["sec"]["user_agent"]

    out: dict[str, dict] = {}
    for ticker in settings["companies"]:
        meta = json.loads((ROOT / "data" / "raw" / ticker / "filing_meta.json").read_text())
        if args.from_dir:
            raw = json.loads((Path(args.from_dir) / f"{ticker}.json").read_text())
        else:
            raw = _fetch(meta["cik"], ua)
            time.sleep(0.3)  # SEC fair-access: stay far below 10 requests/s
        out[ticker] = build_company(ticker, meta, raw)
        c = out[ticker]
        print(f"{ticker}: FY{c['fiscal_year']} (ends {c['period_end']}), {len(c['metrics'])} metrics")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
