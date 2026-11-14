"""
Section- and table-aware chunking of a parsed 10-K (index "v2").

Compared with ingestion/chunker.py (fixed-size character windows) this:
  * tracks the 10-K item each line belongs to (Item 1A Risk Factors, Item 7 MD&A, ...)
    and the nearest sub-heading;
  * keeps tables together, with their caption and header rows; a table too big for one
    chunk is split on row boundaries and the caption + header are repeated in each piece;
  * packs prose by paragraph/sentence instead of cutting mid-sentence;
  * drops page footers ("Apple Inc. | 2025 Form 10-K | 22");
  * gives every chunk a context line ("Apple Inc. 10-K, fiscal 2025 | Item 7 ... | Segment
    Operating Performance") that is indexed together with the text.

Input is data/parsed/<TICKER>.txt, where every line is one block (paragraph, heading
or table row) as written by ingestion/parser.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TARGET_CHARS = 1000   # ~256 tokens of prose
MAX_TABLE_CHARS = 1800
OVERLAP_CHARS = 150
HEADING_ONLY_CHARS = 90

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity and Issuer Purchases", "6": "Reserved",
    "7": "Management's Discussion and Analysis", "7A": "Market Risk Disclosures",
    "8": "Financial Statements and Notes", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Related Transactions and Director Independence",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
_ITEM = re.compile(r"^\s*item\s+(\d{1,2}[abc]?)\s*[\.:\-–—]?\s*(\S.*)?$", re.I)
_FOOTER = re.compile(r"^.{0,80}\|\s*\d{1,3}\s*$|^\s*\d{1,3}\s*$|^.{0,60}form 10-k\s*\|?\s*\d{0,3}\s*$", re.I)
_NUM = re.compile(r"\(?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?|—|–")
_WORD = re.compile(r"[A-Za-z]{2,}")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(“\"])")
# Some filers (JPMorgan, Exxon, NVIDIA) put MD&A and the statements after Item 15/16
# and leave Items 7/8 as page references. These headings re-anchor the section there.
_REANCHOR = [
    (re.compile(r"^management.s discussion and analysis", re.I), "7"),
    (re.compile(r"^(notes to (the )?consolidated financial statements|consolidated (statements?|balance sheets?)\b"
                r"|report of independent registered public accounting firm)", re.I), "8"),
]


@dataclass
class _Line:
    text: str
    section: str
    heading: str
    kind: str  # "prose" | "row" | "label"


@dataclass
class _Block:
    kind: str  # "text" | "table"
    section: str
    heading: str
    lines: list[str] = field(default_factory=list)
    caption: str = ""


def _is_row(line: str) -> bool:
    """A table row: several numeric cells and few words."""
    if len(line) > 220:
        return False
    nums = _NUM.findall(line)
    words = _WORD.findall(line)
    if len(nums) >= 2 and len(words) <= 2 * len(nums) + 6:
        return not (line.rstrip().endswith(".") and len(words) > 12)
    return False


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not 3 <= len(s) <= 90 or s[-1] in ".,;:" or _NUM.search(s) and len(_NUM.findall(s)) > 1:
        return False
    words = s.split()
    if len(words) > 12:
        return False
    caps = sum(1 for w in words if w[:1].isupper() or not w[:1].isalpha())
    return caps >= max(1, int(0.6 * len(words)))


_CELL = re.compile(r"^\(?\$?\s?\d[\d,]*(?:\.\d+)?\s?\)?\s?%?$|^[—–-]$|^\$$|^%$|^\)$")


def _join_cells(lines: list[str]) -> list[str]:
    """Some filings come out of the parser with one table cell per line. Put each row
    back on one line: a short label followed by its numeric cells, or a run of 2+ bare
    numeric cells (a header row of years)."""
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        j = i + 1
        while j < n and _CELL.match(lines[j].strip()):
            j += 1
        cells = [lines[k].strip() for k in range(i + 1, j)]
        is_label = bool(s) and len(s) < 100 and not s.endswith(".") and not _CELL.match(s)
        if is_label and cells and (len(cells) >= 2 or not _FOOTER.match(cells[0])):
            out.append(s + " " + " ".join(cells))
            i = j
        elif _CELL.match(s) and len(cells) >= 1:
            out.append(s + " " + " ".join(cells))
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


def _toc_end(lines: list[str]) -> int:
    """Index of the last line of a table of contents that lists items with titles, else -1.

    A TOC shows up as 8+ item headings with at most 3 lines between neighbours."""
    cand = [i for i, raw in enumerate(lines) if (m := _ITEM.match(raw.strip())) and m.group(2) and len(raw) < 160]
    run_start = 0
    for k in range(1, len(cand) + 1):
        if k == len(cand) or cand[k] - cand[k - 1] > 3:
            if k - run_start >= 8:
                return cand[k - 1]
            if k < len(cand) and cand[k] > 400:
                break
            run_start = k
    return -1


def _classify(lines: list[str]) -> list[_Line]:
    out: list[_Line] = []
    section, heading = "cover", ""
    seen_items: list[str] = []
    toc_end = _toc_end(lines)
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not s or _FOOTER.match(s):
            continue
        m = _ITEM.match(s) if idx > toc_end else None
        if m and m.group(2) and len(s) < 160:
            code = m.group(1).upper()
            # First titled occurrence in reading order; ignores table-of-contents repeats
            # and cross-references back to earlier items.
            if code in ITEM_TITLES and code not in seen_items:
                seen_items.append(code)
                section, heading = code, ""
                continue
        for pat, code in _REANCHOR:
            if pat.match(s) and len(s) < 120 and section in {"15", "16", "8", "7"}:
                section = code
        if _is_row(s):
            kind = "row"
        elif len(s) < 100 and not s.endswith("."):
            kind = "label"
            if _is_heading(s):
                heading = s
        else:
            kind = "prose"
        out.append(_Line(s, section, heading, kind))
    return out


def _blocks(lines: list[_Line]) -> list[_Block]:
    """Group classified lines into text blocks and table blocks."""
    blocks: list[_Block] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        # A table = run of row/label lines containing at least 3 rows.
        j, rows = i, 0
        while j < n and lines[j].kind in ("row", "label") and lines[j].section == ln.section:
            rows += lines[j].kind == "row"
            j += 1
        if rows >= 3:
            # Trailing labels belong to what follows (usually the next heading).
            while j > i and lines[j - 1].kind == "label":
                j -= 1
            caption = ""
            if blocks and blocks[-1].kind == "text" and blocks[-1].lines:
                last = blocks[-1].lines[-1]
                if len(last) < 400 and (last.endswith(":") or re.search(r"\b(table|following|summar|as follows)\b", last, re.I)):
                    caption = last
            blocks.append(_Block("table", ln.section, ln.heading, [x.text for x in lines[i:j]], caption))
            i = j
            continue
        if blocks and blocks[-1].kind == "text" and blocks[-1].section == ln.section and blocks[-1].heading == ln.heading:
            blocks[-1].lines.append(ln.text)
        else:
            blocks.append(_Block("text", ln.section, ln.heading, [ln.text]))
        i += 1
    return blocks


def _split_text(lines: list[str]) -> list[str]:
    """Pack paragraphs (splitting long ones on sentences) into ~TARGET_CHARS pieces with overlap."""
    units: list[str] = []
    for ln in lines:
        if len(ln) <= TARGET_CHARS:
            units.append(ln)
        else:
            units.extend(s for s in _SENT.split(ln) if s.strip())
    pieces: list[str] = []
    cur: list[str] = []
    size = 0
    for u in units:
        if cur and size + len(u) > TARGET_CHARS:
            pieces.append("\n".join(cur))
            tail = cur[-1]
            cur = [tail] if len(tail) <= OVERLAP_CHARS else []
            size = sum(len(x) for x in cur)
        while len(u) > 2 * TARGET_CHARS:  # pathological: no sentence breaks
            pieces.append(u[:TARGET_CHARS])
            u = u[TARGET_CHARS:]
        cur.append(u)
        size += len(u)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _split_table(b: _Block) -> list[str]:
    # Header = leading lines up to (not including) the first row that starts with a word label.
    head: list[str] = []
    for ln in b.lines[:4]:
        first = ln.split()[0] if ln.split() else ""
        is_data = _is_row(ln) and first[:1].isalpha() and not re.fullmatch(r"(19|20)\d\d", first)
        if is_data:
            break
        head.append(ln)
    body = b.lines[len(head):]
    prefix = ([b.caption] if b.caption else []) + head
    pre_len = sum(len(x) + 1 for x in prefix)
    pieces: list[str] = []
    cur: list[str] = []
    size = pre_len
    for ln in body:
        if cur and size + len(ln) > MAX_TABLE_CHARS:
            pieces.append("\n".join(prefix + cur))
            cur, size = [], pre_len
        cur.append(ln)
        size += len(ln) + 1
    if cur or not pieces:
        pieces.append("\n".join(prefix + cur))
    return pieces


def section_label(code: str) -> str:
    if code == "cover":
        return "Cover page and table of contents"
    return f"Item {code}. {ITEM_TITLES.get(code, '')}".strip()


def chunk_filing(text: str, ticker: str, company: str, fiscal_year: int) -> list[dict[str, Any]]:
    blocks = _blocks(_classify(_join_cells(text.splitlines())))
    chunks: list[dict[str, Any]] = []

    def emit(piece: str, b: _Block) -> None:
        ctx = f"{company} 10-K, fiscal {fiscal_year} | {section_label(b.section)}"
        if b.heading and b.heading.lower() not in ctx.lower():
            ctx += f" | {b.heading}"
        chunks.append(
            {
                "chunk_id": f"{ticker}_v2_{len(chunks)}",
                "ticker": ticker,
                "text": piece,
                "kind": b.kind,
                "section": b.section,
                "section_label": section_label(b.section),
                "heading": b.heading,
                "context": ctx,
                "fiscal_year": fiscal_year,
            }
        )

    # A lone heading line ("Liquidity and Capital Resources") retrieves badly on its own:
    # it is glued to the front of the next piece in the same section. Anything longer is
    # a real paragraph and stays its own chunk under its own heading.
    carry: tuple[str, _Block] | None = None
    for b in blocks:
        pieces = _split_table(b) if b.kind == "table" else _split_text(b.lines)
        if carry and carry[1].section != b.section:
            if len(carry[0]) >= 40:
                emit(*carry)
            carry = None
        for piece in pieces:
            if carry:
                piece, carry = f"{carry[0]}\n{piece}", None
            if b.kind == "text" and len(piece) < HEADING_ONLY_CHARS:
                carry = (piece, b)
                continue
            emit(piece, b)
    if carry and len(carry[0]) >= 40:
        emit(*carry)
    return chunks
