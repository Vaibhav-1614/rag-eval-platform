"""Optional Gemini answers (free tier), with a key-free extractive fallback."""
from __future__ import annotations

import os
import re

import httpx

DEFAULT_MODEL = "gemini-2.5-flash"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WORD = re.compile(r"[A-Za-z0-9]+")


def gemini_answer(prompt: str, model: str | None = None) -> tuple[str, float]:
    """Return (text, estimated_cost_usd). No key or API error -> ("", 0.0)."""
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        return "", 0.0
    model = model or DEFAULT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, json=body, headers={"x-goog-api-key": key})
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        return "", 0.0
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = parts[0].get("text", "") if parts else ""
    return str(text).strip(), 0.0


def extractive_answer(question: str, passages: list[str], max_sentences: int = 3) -> str:
    """
    Free fallback: return the retrieved sentences that best cover the question's
    content words (with a small bonus for higher-ranked passages).
    """
    from evaluation.retrieval_metrics import token_set

    q = token_set(question)
    if not q:
        return ""
    scored: list[tuple[float, int, str]] = []
    order = 0
    seen: set[str] = set()
    for rank, passage in enumerate(passages):
        # Lines are block boundaries (headings, table rows) from the parser
        sents = [x for line in passage.splitlines() for x in _SENT_SPLIT.split(line)]
        for sent in sents:
            sent = sent.strip()
            # overlapping chunks repeat sentences
            if not 40 <= len(sent) <= 600 or sent in seen:
                continue
            seen.add(sent)
            words = token_set(sent)
            if not words:
                continue
            hit = len(q & words) / len(q)
            if hit == 0:
                continue
            scored.append((hit + 0.05 / (rank + 1), order, sent))
            order += 1
    best = sorted(scored, key=lambda x: -x[0])[:max_sentences]
    # Present in reading order so the answer flows naturally
    return " ".join(s for _, _, s in sorted(best, key=lambda x: x[1]))
