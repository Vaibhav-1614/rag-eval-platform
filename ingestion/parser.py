"""
Parse SEC filing from data/raw/{ticker}/ (HTML, TXT, or PDF) -> clean text.

Saves to data/parsed/{ticker}.txt and returns cleaned string.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

load_dotenv(project_root() / ".env")

logger = logging.getLogger(__name__)

# iXBRL filings declare an XML prolog; lxml's HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Common SEC page furniture (headers, TOC links, page numbers)
_BOILERPLATE_PATTERNS = [
    re.compile(r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION", re.IGNORECASE),
    re.compile(r"Table\s+of\s+Contents", re.IGNORECASE),
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # isolated page numbers
]


def _strip_boilerplate(text: str) -> str:
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BLOCK_TAGS = [
    "p", "div", "tr", "li", "table", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol",
]


def _is_hidden(style: str | None) -> bool:
    return bool(style) and "display:none" in style.replace(" ", "").lower()


def parse_html_bytes(data: bytes) -> str:
    soup = BeautifulSoup(data, "lxml")
    # Inline XBRL filings carry a hidden <ix:header> block of taxonomy URIs,
    # contexts and dates; it is machine metadata, not filing text.
    for tag in soup(["script", "style", "noscript", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=_is_hidden):
        tag.decompose()
    # Break lines only at block boundaries so inline <span>s (which iXBRL uses
    # heavily) stay in the same sentence; table cells are joined with spaces.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append("\n")
    for tag in soup.find_all(["td", "th"]):
        tag.append(" ")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return soup.get_text().replace("\xa0", " ")


def parse_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing")
    doc = fitz.open(path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts)


def clean_text(raw: str) -> str:
    raw = _strip_boilerplate(raw)
    # Remove repeated SEC header lines
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if len(s) < 2:
            continue
        if re.fullmatch(r"\d+", s):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = _normalize_whitespace(text)
    return text


def parse_file(path: Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = parse_pdf(path)
    else:
        raw = parse_html_bytes(path.read_bytes())
    return clean_text(raw)


def parse_ticker(ticker: str, settings: dict | None = None) -> str:
    settings = settings or load_settings()
    paths_cfg = settings.get("paths") or {}
    raw_dir = project_root() / paths_cfg.get("raw_dir", "data/raw") / ticker.upper()
    parsed_dir = project_root() / paths_cfg.get("parsed_dir", "data/parsed")
    parsed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

    # Prefer primary document from meta, else largest file
    meta_path = raw_dir / "filing_meta.json"
    candidates: list[Path] = []
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prim = meta.get("primaryDocument")
        if prim:
            p = raw_dir / prim
            if p.exists():
                candidates.append(p)
    for p in sorted(raw_dir.iterdir(), key=lambda x: x.stat().st_size, reverse=True):
        if p.is_file() and p.suffix.lower() in {".htm", ".html", ".txt", ".pdf"}:
            if p not in candidates:
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No htm/html/txt/pdf in {raw_dir}")

    combined: list[str] = []
    for c in candidates[:1]:  # primary filing only
        logger.info("Parsing %s", c)
        combined.append(parse_file(c))

    cleaned = clean_text("\n\n".join(combined))
    out = parsed_dir / f"{ticker.upper()}.txt"
    out.write_text(cleaned, encoding="utf-8")
    logger.info("Wrote %s (%s chars)", out, len(cleaned))
    return cleaned


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", help="Single ticker")
    ap.add_argument("--all", action="store_true", help="All companies from settings")
    args = ap.parse_args()
    settings = load_settings()
    tickers = [t.upper() for t in settings.get("companies", [])]
    if args.all:
        for t in tickers:
            try:
                parse_ticker(t, settings)
            except Exception as e:
                logger.exception("%s: %s", t, e)
    elif args.ticker:
        parse_ticker(args.ticker.upper(), settings)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
