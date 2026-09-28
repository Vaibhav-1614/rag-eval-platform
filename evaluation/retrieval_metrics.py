"""
Retrieval-first metrics @k (default k=5).

Uses content-word overlap (F1) between retrieved chunks and ground-truth context.
A chunk counts as relevant when overlap >= RELEVANCE_THRESHOLD; random chunk pairs
from the same filing clear 0.35 only ~1-3% of the time, true matches ~0.7.
"""
from __future__ import annotations

import math
import re
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9]+")

RELEVANCE_THRESHOLD = 0.35

# Function words plus words that appear on nearly every 10-K page. Without this,
# two unrelated chunks from the same filing overlap enough to count as "relevant".
_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into
    is it its itself just may me might more most must my no nor not of off on once only
    or other our ours out over own same shall she should so some such than that the their
    theirs them then there these they this those through to too under until up upon us
    very was we were what when where which while who whom why will with within would you
    your yours s t including include includes company companys inc form 10 k fiscal year
    years ended million millions billion billions percent total see item part note notes
    """.split()
)


def token_set(text: str) -> set[str]:
    return {t for t in (w.lower() for w in _WORD.findall(text)) if t not in _STOPWORDS}


def overlap_f1(pred: str, gold: str) -> float:
    p, g = token_set(pred), token_set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def hit_rate(
    retrieved_texts: list[str],
    ground_truth_context: str,
    *,
    ground_truth_chunk_id: str | None = None,
    retrieved_ids: list[str] | None = None,
    overlap_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    if ground_truth_chunk_id and retrieved_ids:
        if ground_truth_chunk_id in retrieved_ids[:5]:
            return 1.0
    for t in retrieved_texts[:5]:
        if overlap_f1(t, ground_truth_context) >= overlap_threshold:
            return 1.0
    return 0.0


def mrr(
    retrieved_texts: list[str],
    ground_truth_context: str,
    *,
    overlap_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    for i, t in enumerate(retrieved_texts[:5], start=1):
        if overlap_f1(t, ground_truth_context) >= overlap_threshold:
            return 1.0 / i
    return 0.0


def context_recall(
    retrieved_texts: list[str],
    ground_truth_context: str,
    *,
    overlap_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Best overlap of any single retrieved chunk with gold (proxy for recall)."""
    if not ground_truth_context.strip():
        return 0.0
    best = 0.0
    for t in retrieved_texts[:5]:
        best = max(best, overlap_f1(t, ground_truth_context))
    return 1.0 if best >= overlap_threshold else best


def context_precision(
    retrieved_texts: list[str],
    ground_truth_context: str,
    *,
    overlap_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    if not retrieved_texts:
        return 0.0
    hits = sum(1 for t in retrieved_texts[:5] if overlap_f1(t, ground_truth_context) >= overlap_threshold)
    return hits / min(5, len(retrieved_texts))


def dcg(rels: list[float]) -> float:
    s = 0.0
    for i, rel in enumerate(rels, start=1):
        s += (2**rel - 1) / math.log2(i + 1)
    return s


def ndcg(
    retrieved_texts: list[str],
    ground_truth_context: str,
    *,
    overlap_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Binary-relevance nDCG@5; ideal ranking puts every relevant chunk first."""
    rels = [
        1.0 if overlap_f1(t, ground_truth_context) >= overlap_threshold else 0.0
        for t in retrieved_texts[:5]
    ]
    n_rel = int(sum(rels))
    if n_rel == 0:
        return 0.0
    return dcg(rels) / dcg([1.0] * n_rel)


def score_row(
    retrieved: list[dict[str, Any]],
    ground_truth_context: str,
    *,
    ground_truth_chunk_id: str | None = None,
) -> dict[str, float]:
    texts = [r.get("text", "") for r in retrieved]
    ids = [r.get("chunk_id", "") for r in retrieved]
    return {
        "hit_rate": hit_rate(
            texts,
            ground_truth_context,
            ground_truth_chunk_id=ground_truth_chunk_id,
            retrieved_ids=ids,
        ),
        "mrr": mrr(texts, ground_truth_context),
        "context_recall": context_recall(texts, ground_truth_context),
        "context_precision": context_precision(texts, ground_truth_context),
        "ndcg": ndcg(texts, ground_truth_context),
    }
