"""
Pull latest 10-K per company from SEC EDGAR.

- Tickers from config/settings.yml
- Ticker -> CIK via https://www.sec.gov/files/company_tickers.json
- Latest 10-K via https://data.sec.gov/submissions/CIK{cik_padded}.json
- Download primary document to data/raw/{ticker}/
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Repo root on path for platform_config
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

load_dotenv(project_root() / ".env")

logger = logging.getLogger(__name__)

SEC_DATA_BASE = "https://data.sec.gov"
SEC_WWW_BASE = "https://www.sec.gov"
TICKER_MAP_URL = f"{SEC_WWW_BASE}/files/company_tickers.json"
SUBMISSIONS_URL = f"{SEC_DATA_BASE}/submissions/CIK{{cik}}.json"
ARCHIVES_BASE = f"{SEC_WWW_BASE}/Archives/edgar/data"


class RateLimiter:
    """Simple token bucket: max `rate` requests per second."""

    def __init__(self, rate: float = 10.0) -> None:
        self.rate = rate
        self.min_interval = 1.0 / rate if rate > 0 else 0
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def _user_agent(settings: dict[str, Any]) -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if ua:
        return ua
    sec = settings.get("sec") or {}
    return (sec.get("user_agent") or "RAG-Eval-Platform contact@example.com").strip()


def _session(settings: dict[str, Any]) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": _user_agent(settings),
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return s


def _get_with_retry(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    max_retries: int,
    limiter: RateLimiter,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        limiter.wait()
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                wait = min(2**attempt, 60)
                logger.warning("HTTP %s for %s, retry in %ss", r.status_code, url, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except (requests.RequestException, OSError) as e:
            last_exc = e
            wait = min(2**attempt, 60)
            logger.warning("Request failed %s: %s, retry in %ss", url, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed after retries: {url}") from last_exc


def load_ticker_to_cik(
    settings: dict[str, Any],
    limiter: RateLimiter,
    timeout: float,
    max_retries: int,
) -> dict[str, str]:
    """Map upper-case ticker -> zero-padded 10-digit CIK string."""
    sess = _session(settings)
    r = _get_with_retry(
        sess,
        TICKER_MAP_URL,
        timeout=timeout,
        max_retries=max_retries,
        limiter=limiter,
    )
    data = r.json()
    rows: list[dict[str, Any]]
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and "data" in data and "fields" in data:
        fields = data["fields"]
        rows = [dict(zip(fields, row)) for row in data["data"]]
    elif isinstance(data, dict):
        rows = [v for v in data.values() if isinstance(v, dict)]
    else:
        rows = []
    out: dict[str, str] = {}
    for row in rows:
        t = str(row.get("ticker", "")).upper()
        if not t:
            continue
        cik_int = int(row["cik_str"])
        out[t] = f"{cik_int:010d}"
    return out


def find_latest_10k(
    settings: dict[str, Any],
    cik_padded: str,
    limiter: RateLimiter,
    timeout: float,
    max_retries: int,
) -> dict[str, str]:
    """Return accessionNumber, primaryDocument, filingDate for latest 10-K."""
    sess = _session(settings)
    url = SUBMISSIONS_URL.format(cik=cik_padded)
    r = _get_with_retry(
        sess,
        url,
        timeout=timeout,
        max_retries=max_retries,
        limiter=limiter,
    )
    j = r.json()
    recent = j.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    primaries = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    for i, form in enumerate(forms):
        if form == "10-K" and i < len(accs) and i < len(primaries):
            return {
                "accessionNumber": accs[i],
                "primaryDocument": primaries[i],
                "filingDate": dates[i] if i < len(dates) else "",
            }
    raise LookupError(f"No 10-K found in recent filings for CIK {cik_padded}")


def build_archive_url(cik_padded: str, accession: str, primary_doc: str) -> str:
    cik_num = str(int(cik_padded))  # strip leading zeros for path
    acc_nodash = accession.replace("-", "")
    return f"{ARCHIVES_BASE}/{cik_num}/{acc_nodash}/{primary_doc}"


def download_filing(
    settings: dict[str, Any],
    url: str,
    dest_dir: Path,
    limiter: RateLimiter,
    timeout: float,
    max_retries: int,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("/")[-1] or "filing.html"
    name = re.sub(r"[^\w.\-]", "_", name)
    dest = dest_dir / name
    sess = _session(settings)
    r = _get_with_retry(
        sess,
        url,
        timeout=timeout,
        max_retries=max_retries,
        limiter=limiter,
    )
    dest.write_bytes(r.content)
    logger.info("Saved %s (%s bytes)", dest, len(r.content))
    return dest


def fetch_all(settings: dict[str, Any] | None = None) -> list[Path]:
    settings = settings or load_settings()
    sec = settings.get("sec") or {}
    rate = float(sec.get("rate_limit_per_sec", 10))
    timeout = float(sec.get("request_timeout_sec", 60))
    max_retries = int(sec.get("max_retries", 5))
    paths_cfg = settings.get("paths") or {}
    raw_rel = paths_cfg.get("raw_dir", "data/raw")
    raw_base = project_root() / raw_rel

    limiter = RateLimiter(rate)
    tickers = [t.upper() for t in settings.get("companies", [])]
    logger.info("Loading ticker map from SEC...")
    t2c = load_ticker_to_cik(settings, limiter, timeout, max_retries)

    saved: list[Path] = []
    for ticker in tickers:
        if ticker not in t2c:
            logger.error("Unknown ticker: %s", ticker)
            continue
        cik = t2c[ticker]
        try:
            info = find_latest_10k(settings, cik, limiter, timeout, max_retries)
        except Exception as e:
            logger.exception("Failed to find 10-K for %s: %s", ticker, e)
            continue
        url = build_archive_url(cik, info["accessionNumber"], info["primaryDocument"])
        logger.info(
            "Downloading 10-K for %s dated %s -> %s",
            ticker,
            info.get("filingDate", ""),
            url,
        )
        meta = {
            "ticker": ticker,
            "cik": cik,
            "filingDate": info.get("filingDate", ""),
            "accessionNumber": info["accessionNumber"],
            "primaryDocument": info["primaryDocument"],
            "url": url,
        }
        out_dir = raw_base / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "filing_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            p = download_filing(settings, url, out_dir, limiter, timeout, max_retries)
            saved.append(p)
        except Exception as e:
            logger.exception("Download failed for %s: %s", ticker, e)
    return saved


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    fetch_all()


if __name__ == "__main__":
    main()
