"""
Generate data/testset.json: 5 Q&A per ticker (50 total).

- Default: deterministic keyword questions from random prose 512-token chunks.
- --use-llm: Gemini (GOOGLE_API_KEY) for JSON Q&A, with retries.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

from collections import Counter

import httpx
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from platform_config import load_settings, project_root

from evaluation.retrieval_metrics import _STOPWORDS
from ingestion.chunker import chunk_file_for_ticker

load_dotenv(project_root() / ".env")

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ALPHA = re.compile(r"[A-Za-z]")
# Verbs/qualifiers that make poor question topics
_GENERIC = frozenset(
    """
    additionally however still face faces considered described based bases reported presented
    related certain various significant significantly such include expect expects believe
    believes could would might result results affect affected adversely negatively require
    requires required make makes made recognize recognized subject using used within
    """.split()
)


def _prose_ratio(text: str) -> float:
    """Share of characters on lines that read like sentences rather than table rows."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    prose = sum(len(ln) for ln in lines if len(ln) > 120 and ln.rstrip().endswith((".", ":", ";")))
    return prose / max(1, sum(len(ln) for ln in lines))


def _candidate_sentences(text: str) -> list[str]:
    out = []
    for sent in _SENT_SPLIT.split(text.replace("\n", " ")):
        sent = sent.strip()
        if not 90 <= len(sent) <= 320:
            continue
        if len(_ALPHA.findall(sent)) / len(sent) < 0.75:
            continue
        out.append(sent)
    return out


def _keywords(sentence: str, df: Counter, n_docs: int, k: int = 4) -> list[str]:
    """Distinctive (mid-IDF) content words, kept in sentence order."""
    seen: list[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", sentence):
        lw = w.lower()
        if lw in _STOPWORDS or lw in _GENERIC or lw.endswith("ly") or lw in {x.lower() for x in seen}:
            continue
        # skip words in >15% of chunks (generic) and hapaxes (trivial exact-match keys)
        if not 2 <= df.get(lw, 0) <= max(2, int(0.15 * n_docs)):
            continue
        seen.append(w)
    if len(seen) <= k:
        return seen
    ranked = sorted(seen, key=lambda w: df.get(w.lower(), 0))[:k]
    return [w for w in seen if w in ranked]


def _join_topics(words: list[str]) -> str:
    words = [w if w.isupper() else w.lower() for w in words]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def deterministic_qa(
    chunk: dict[str, Any],
    company: str,
    df: Counter,
    n_docs: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """
    Keyword-style question about one sentence of the chunk. The question names the
    sentence's distinctive terms rather than quoting it, so lexical (BM25) retrieval
    gets no verbatim-copy advantage over dense retrieval.
    """
    sents = _candidate_sentences(chunk["text"])
    rng.shuffle(sents)
    for sent in sents:
        kws = _keywords(sent, df, n_docs)
        if len(kws) < 3:
            continue
        return {
            "question": f"What does {company}'s 10-K say about {_join_topics(kws)}?",
            "ground_truth_answer": sent,
            "ground_truth_context": chunk["text"][:8000],
            "ground_truth_chunk_id": chunk["chunk_id"],
            "ticker": chunk["ticker"],
            "question_source": "template",
        }
    return None


async def _gemini_one(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    chunk_text: str,
) -> dict[str, Any] | None:
    url = GEMINI_URL.format(model=model)
    prompt = (
        "Based on this excerpt from a SEC 10-K filing, generate one specific factual "
        "question and its exact answer. Return JSON only with keys: "
        'question, answer, context. Use "context" equal to a short verbatim quote from the excerpt.\n\n'
        f"EXCERPT:\n{chunk_text[:6000]}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    r = await client.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=120.0)
    r.raise_for_status()
    data = r.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    raw = parts[0].get("text", "") if parts else ""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


async def generate_with_gemini(
    chunks: list[dict[str, Any]],
    model: str,
    api_key: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(4)

    async def one(client: httpx.AsyncClient, c: dict[str, Any]) -> dict[str, Any] | None:
        async with sem:
            for attempt in range(3):
                try:
                    obj = await _gemini_one(client, api_key, model, c["text"])
                    if obj and "question" in obj and "answer" in obj:
                        return {
                            "question": str(obj["question"]),
                            "ground_truth_answer": str(obj.get("answer", "")),
                            "ground_truth_context": str(obj.get("context", c["text"][:2000])),
                            "ground_truth_chunk_id": c["chunk_id"],
                            "ticker": c["ticker"],
                            "question_source": model,
                        }
                except Exception as e:
                    logger.warning("Gemini attempt %s failed: %s", attempt + 1, e)
                await asyncio.sleep(1.5 * (attempt + 1))
        return None

    async with httpx.AsyncClient() as client:
        tasks = [one(client, c) for c in chunks]
        for coro in asyncio.as_completed(tasks):
            row = await coro
            if row:
                out.append(row)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-llm", action="store_true", help="Use Gemini for Q&A JSON")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    settings = load_settings()
    paths = settings.get("paths") or {}
    out_path = project_root() / paths.get("testset_path", "data/testset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    tickers = [t.upper() for t in settings.get("companies", [])]
    names = settings.get("company_names") or {}
    per_ticker = 5
    by_t: dict[str, list[dict[str, Any]]] = {t: [] for t in tickers}
    sample_chunks: list[dict[str, Any]] = []

    for t in tickers:
        try:
            chunks_512 = chunk_file_for_ticker(t, 512, settings)
        except FileNotFoundError as e:
            logger.warning("%s", e)
            continue
        df: Counter = Counter()
        for c in chunks_512:
            df.update({w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", c["text"])})
        # Skip the cover page / TOC region and table-heavy chunks
        start = max(1, len(chunks_512) // 25)
        prose = [c for c in chunks_512[start:] if _prose_ratio(c["text"]) >= 0.5]
        rng.shuffle(prose)
        company = names.get(t, t)
        for c in prose:
            if len(by_t[t]) >= per_ticker:
                break
            row = deterministic_qa(c, company, df, len(chunks_512), rng)
            if row:
                by_t[t].append(row)
                sample_chunks.append(c)

    use_llm = args.use_llm
    if use_llm:
        import os

        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            logger.error("GOOGLE_API_KEY missing; using deterministic testset")
            use_llm = False

    if use_llm:
        import os

        free = settings.get("free_llm") or {}
        model = str(free.get("model", "gemini-2.5-flash"))
        rows = asyncio.run(generate_with_gemini(sample_chunks, model, os.environ["GOOGLE_API_KEY"]))
        # Prefer LLM rows; keep deterministic rows only where Gemini failed
        llm_by_cid = {r["ground_truth_chunk_id"]: r for r in rows}
        for t in tickers:
            by_t[t] = [llm_by_cid.get(r["ground_truth_chunk_id"], r) for r in by_t[t]]

    final = [r for t in tickers for r in by_t.get(t, [])[:per_ticker]]
    final = final[:50]
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%s items)", out_path, len(final))
    for i, row in enumerate(final[:3], 1):
        logger.info("Preview %s: Q=%s…", i, row["question"][:100])


if __name__ == "__main__":
    main()
