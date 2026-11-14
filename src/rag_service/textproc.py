"""Text processing for the v2 index: stemming tokenizer, query expansion, BM25."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import snowballstemmer
import yaml

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:&[A-Za-z]+)?")
_POSSESSIVE = re.compile(r"['’]s\b")
_STEMMER = snowballstemmer.stemmer("english")
# Function words only. Domain words ("company", "fiscal") stay: BM25's IDF already
# discounts them and they matter inside phrases.
STOPWORDS = frozenset(
    "a an and are as at be been by did do does for from had has have how in is it its of on or "
    "that the their this to was were what when where which who whom why will with would".split()
)


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    out = []
    text = _POSSESSIVE.sub("", text)
    for m in _TOKEN.finditer(text):
        w = m.group(0).lower()
        if drop_stopwords and w in STOPWORDS:
            continue
        out.append(_stem(w) if stem and not w.isdigit() else w)
    return out


class QueryExpander:
    """Adds filing vocabulary for the words people actually type ("R&D" -> "research and
    development"). Rules live in a YAML file: each has `match` (regex on the lower-cased
    question) and `add` (terms appended to the lexical query)."""

    def __init__(self, path: Path | None) -> None:
        self.rules: list[tuple[re.Pattern[str], str]] = []
        if path:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            for rule in raw.get("rules", []):
                self.rules.append((re.compile(rule["match"], re.I), str(rule["add"])))

    def expand(self, question: str) -> str:
        extra = [add for pat, add in self.rules if pat.search(question)]
        return question if not extra else f"{question} {' '.join(extra)}"


class BM25:
    """Okapi BM25 over an inverted index (numpy postings), so a query only touches the
    documents that contain its terms."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.n = len(docs_tokens)
        self.k1, self.b = k1, b
        lengths = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(lengths.mean()) if self.n else 0.0
        norm = k1 * (1 - b + b * lengths / (self.avgdl or 1.0))
        postings: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(docs_tokens):
            for t in toks:
                p = postings.setdefault(t, {})
                p[i] = p.get(i, 0) + 1
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for t, p in postings.items():
            idx = np.fromiter(p.keys(), dtype=np.int32, count=len(p))
            tf = np.fromiter(p.values(), dtype=np.float32, count=len(p))
            idf = math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            self.postings[t] = (idx, idf * tf * (k1 + 1) / (tf + norm[idx]))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for t in set(query_tokens):
            hit = self.postings.get(t)
            if hit is not None:
                s[hit[0]] += hit[1]
        return s

    def top(self, query_tokens: list[str], k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(query_tokens)
        if mask is not None:
            s = np.where(mask, s, 0.0)
        k = min(k, self.n)
        if k <= 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]
