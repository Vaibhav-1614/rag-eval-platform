"""
Offline eval of the production pipeline on eval/golden.jsonl, gated against eval/baseline.json.

Deterministic metrics need no keys and run on every PR:
  * retrieval: Hit@5 / MRR / nDCG@5, overall, per question type and per split
  * refusal: does the service answer what it should and refuse what it cannot know
  * answer checks: for questions with a known figure or key phrase, is it in the answer
--generation adds LLM answers and LLM-judge scores (needs GROQ_API_KEY and/or GEMINI_API_KEY).

    python eval/run_eval.py                         # production config, compare to baseline
    python eval/run_eval.py --generation            # + LLM answers and judges
    python eval/run_eval.py --update-baseline       # accept current numbers as the new baseline
    python eval/run_eval.py --config configs/retrieval/v1.yaml --no-gate
    python eval/run_eval.py --split dev             # only the tuning split

The `test` split is never used to tune rules or thresholds; report it, don't fit it.
Exit code 1 when a gated metric drops more than its tolerance below baseline.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.build_golden import _norm, chunk_has_evidence  # noqa: E402
from rag_service.config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig, get_settings  # noqa: E402
from rag_service.generation import format_context  # noqa: E402
from rag_service.judges import judge  # noqa: E402
from rag_service.pipeline import Pipeline  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.jsonl"
BASELINE = ROOT / "eval" / "baseline.json"
REPORT_DIR = ROOT / "eval" / "reports"
K = 5

# metric -> max allowed drop vs baseline (absolute). Latency is reported, not gated
# (CI runners are noisy).
TOLERANCE = {
    "retrieval.hit_at_5": 0.02,
    "retrieval.mrr": 0.02,
    "retrieval.ndcg_at_5": 0.02,
    "retrieval.test_hit_at_5": 0.03,
    "answers.accuracy": 0.03,
    "refusal.accuracy": 0.02,
    "refusal.unanswerable_recall": 0.04,
    "refusal.false_refusal_rate": -0.02,  # negative: lower is better, may rise by at most 0.02
    "refusal_llm.accuracy": 0.05,
    "generation.faithfulness_mean": 0.15,
    "generation.correctness_mean": 0.15,
}


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SectionIndex:
    """Which 10-K item a passage comes from, for any index version: a slice from the
    middle of the passage is looked up in the per-item text of the v2 chunks."""

    def __init__(self) -> None:
        self._text: dict[str, dict[str, str]] = {}

    def _load(self, ticker: str) -> dict[str, str]:
        if ticker not in self._text:
            with gzip.open(ROOT / "data" / "index" / "v2" / "chunks" / f"{ticker}.json.gz", "rb") as f:
                chunks = json.loads(f.read().decode("utf-8"))
            by: dict[str, list[str]] = {}
            for c in chunks:
                by.setdefault(c["section"], []).append(_norm(c["text"]))
            self._text[ticker] = {s: " \n ".join(parts) for s, parts in by.items()}
        return self._text[ticker]

    def in_sections(self, ticker: str, passage_text: str, sections: list[str]) -> bool:
        t = _norm(passage_text)
        if len(t) < 40:
            return False
        mid = len(t) // 2
        probes = [t[max(0, mid - 40): mid + 40], t[:80], t[-80:]]
        secs = self._load(ticker)
        return any(pr in secs.get(s, "") for s in sections for pr in probes)


def is_relevant(p, item: dict, sections: SectionIndex) -> bool:
    if item.get("evidence_any"):
        if any(chunk_has_evidence(p.text, ev) for ev in item["evidence_any"]):
            return True
    rule = item.get("section_rule")
    if rule and p.kind != "fact" and p.ticker in item["tickers"]:
        return bool(re.search(rule["pattern"], p.text, re.I)) and sections.in_sections(p.ticker, p.text, rule["sections"])
    return False


def answer_ok(answer: str, groups: list[list[str]]) -> bool:
    a = _norm(answer).replace("$", "")
    return all(any(_norm(alt) in a for alt in group) for group in groups)


def _ndcg(rels: list[int]) -> float:
    n = sum(rels)
    if not n:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / math.log2(i + 2) for i in range(n))
    return dcg / ideal


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else 0.0


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(statistics.mean(xs), 4) if xs else None


def evaluate(cfg_path: Path, generation: bool, limit: int | None = None, split: str | None = None,
             pipeline: Pipeline | None = None) -> tuple[dict, list[dict]]:
    settings = get_settings()
    cfg = RetrievalConfig.load(cfg_path)
    pipe = pipeline or Pipeline(cfg, settings)
    pipe.retriever.warm()
    sections = SectionIndex()
    golden = [g for g in load_golden() if not split or g["split"] == split]
    golden = golden[:limit] if limit else golden

    rows: list[dict] = []
    for g in golden:
        res = pipe.run(g["question"], use_llm=generation)
        row = {
            "id": g["id"], "split": g["split"], "type": g["type"], "source": g["source"],
            "answerable": g["answerable"], "question": g["question"], "scope": res.scope,
            "chunk_ids": [p.chunk_id for p in res.passages], "mode": res.generation.mode,
            "answer": res.generation.answer, "provider": res.generation.provider,
            "scope_reason": res.scope_reason, "evidence_score": res.evidence_score,
            "facts_mode": res.facts_mode, "retrieval_ms": res.retrieval_ms, "total_ms": res.total_ms,
        }
        if g["answerable"]:
            rels = [int(is_relevant(p, g, sections)) for p in res.passages[:K]]
            first = next((i for i, r in enumerate(rels, start=1) if r), None)
            row.update(hit=int(first is not None), rr=1 / first if first else 0.0, ndcg=_ndcg(rels))
            if g.get("answer_contains"):
                row["answer_ok"] = int(res.generation.mode != "refusal" and answer_ok(res.generation.answer, g["answer_contains"]))
        if generation:
            ctx = format_context(res.passages)
            f = judge("faithfulness", settings, question=g["question"], context=ctx, answer=res.generation.answer)
            c = judge("correctness", settings, question=g["question"], reference=g["reference_answer"], answer=res.generation.answer)
            row.update(faithfulness=f.score, correctness=c.score, judge_model=f.judge_model or c.judge_model,
                       verification=res.generation.verification)
            time.sleep(float(os.environ.get("EVAL_SLEEP_S", "1.0")))  # stay under free-tier RPM
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    refused = [r for r in rows if r["mode"] == "refusal"]
    correct_refusals = [r for r in refused if not r["answerable"]]
    checked = [r for r in ans if "answer_ok" in r]
    test = [r for r in ans if r["split"] == "test"]
    metrics: dict = {
        "retrieval": {
            "n": len(ans),
            "hit_at_5": _mean(r["hit"] for r in ans),
            "mrr": _mean(r["rr"] for r in ans),
            "ndcg_at_5": _mean(r["ndcg"] for r in ans),
            "dev_hit_at_5": _mean(r["hit"] for r in ans if r["split"] == "dev"),
            "test_hit_at_5": _mean(r["hit"] for r in test),
            "test_mrr": _mean(r["rr"] for r in test),
            "p50_ms": _pct([r["retrieval_ms"] for r in rows], 0.5),
            "p95_ms": _pct([r["retrieval_ms"] for r in rows], 0.95),
        },
        "by_type": {
            t: {"n": sum(r["type"] == t for r in ans),
                "hit_at_5": _mean(r["hit"] for r in ans if r["type"] == t),
                "mrr": _mean(r["rr"] for r in ans if r["type"] == t),
                "answer_accuracy": _mean(r["answer_ok"] for r in checked if r["type"] == t)}
            for t in sorted({r["type"] for r in ans})
        },
        "answers": {
            "n_checked": len(checked),
            "accuracy": _mean(r["answer_ok"] for r in checked),
            "test_accuracy": _mean(r["answer_ok"] for r in checked if r["split"] == "test"),
            "wrong": [r["id"] for r in checked if not r["answer_ok"]],
        },
        "refusal": {
            "accuracy": round(sum((r["mode"] == "refusal") != r["answerable"] for r in rows) / len(rows), 4),
            "unanswerable_recall": round(len(correct_refusals) / len(unans), 4) if unans else None,
            "test_unanswerable_recall": _mean(int(r["mode"] == "refusal") for r in unans if r["split"] == "test"),
            "false_refusal_rate": round(sum(r["mode"] == "refusal" for r in ans) / len(ans), 4) if ans else None,
            "false_refusals": [r["id"] for r in refused if r["answerable"]],
            "missed_refusals": [r["id"] for r in unans if r["mode"] != "refusal"],
        },
    }
    if generation:
        # With an LLM in the loop refusals also come from the model; keep them apart from
        # the deterministic numbers so baselines don't mix.
        metrics["refusal_llm"] = metrics.pop("refusal")
        metrics["answers_llm"] = metrics.pop("answers")
        f = [r["faithfulness"] for r in rows if r.get("faithfulness")]
        c = [r["correctness"] for r in rows if r.get("correctness")]
        llm_rows = [r for r in rows if r.get("verification")]
        metrics["generation"] = {
            "n_judged": len(f),
            "faithfulness_mean": round(statistics.mean(f), 3) if f else None,
            "correctness_mean": round(statistics.mean(c), 3) if c else None,
            "test_correctness_mean": _mean(r["correctness"] for r in rows if r.get("correctness") and r["split"] == "test"),
            "modes": {m: sum(r["mode"] == m for r in rows) for m in ("llm", "extractive", "refusal")},
            "answers_with_dropped_sentences": sum(bool(r["verification"]["dropped"]) for r in llm_rows),
            "judge_model": next((r.get("judge_model") for r in rows if r.get("judge_model")), None),
        }
    metrics["meta"] = {"retrieval_version": cfg.version, "golden_items": len(rows), "split": split or "all",
                       "prompt_version": settings.prompt_version}
    return metrics, rows


def compare(metrics: dict, baseline: dict) -> list[str]:
    failures = []
    for key, tol in TOLERANCE.items():
        sect, name = key.split(".")
        cur = (metrics.get(sect) or {}).get(name)
        base = (baseline.get(sect) or {}).get(name)
        if cur is None or base is None:
            continue
        if tol < 0:  # lower is better
            if cur > base - tol + 1e-9:
                failures.append(f"{key}: {cur} > baseline {base} + tol {-tol}")
        elif cur < base - tol - 1e-9:
            failures.append(f"{key}: {cur} < baseline {base} - tol {tol}")
    return failures


def to_markdown(metrics: dict, baseline: dict | None, failures: list[str]) -> str:
    m = metrics["meta"]
    lines = [f"### RAG eval — retrieval `{m['retrieval_version']}`, prompt `{m['prompt_version']}`, "
             f"{m['golden_items']} questions ({m['split']})", ""]
    lines += ["| metric | current | baseline |", "|---|---|---|"]
    for sect in ("retrieval", "answers", "refusal", "refusal_llm", "answers_llm", "generation"):
        for k, v in (metrics.get(sect) or {}).items():
            if isinstance(v, (list, dict)):
                continue
            b = ((baseline or {}).get(sect) or {}).get(k, "")
            lines.append(f"| {sect}.{k} | {v} | {b} |")
    lines += ["", "| question type | n | Hit@5 | MRR | answer check |", "|---|---|---|---|---|"]
    for t, v in metrics["by_type"].items():
        lines.append(f"| {t} | {v['n']} | {v['hit_at_5']} | {v['mrr']} | {v['answer_accuracy'] if v['answer_accuracy'] is not None else '–'} |")
    lines += ["", "**Result:** " + ("FAIL\n- " + "\n- ".join(failures) if failures else "pass")]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="write the full report JSON here instead of eval/reports/latest.json")
    args = ap.parse_args()

    metrics, rows = evaluate(Path(args.config), args.generation, args.limit, args.split)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else REPORT_DIR / "latest.json"
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=1), encoding="utf-8")

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    partial = bool(args.split or args.limit)
    if args.update_baseline:
        if partial:
            sys.exit("refusing to update the baseline from a partial run (--split/--limit)")
        merged = dict(baseline or {})
        merged.update({k: v for k, v in metrics.items() if k != "meta"})
        merged["meta"] = metrics["meta"]
        BASELINE.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"baseline updated: {BASELINE.relative_to(ROOT)}")
    gate = not (args.no_gate or baseline is None or args.update_baseline or partial)
    failures = compare(metrics, baseline) if gate else []
    md = to_markdown(metrics, baseline if gate else None, failures)
    (REPORT_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
