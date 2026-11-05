"""
Offline eval of the production pipeline on eval/golden.jsonl, gated against eval/baseline.json.

Retrieval + refusal metrics are deterministic and need no keys (run on every PR).
--generation adds LLM answers and LLM-judge scores (needs GROQ_API_KEY and/or GEMINI_API_KEY).

    python eval/run_eval.py                         # retrieval + refusal, compare to baseline
    python eval/run_eval.py --generation            # + LLM answers and judges
    python eval/run_eval.py --update-baseline       # accept current numbers as the new baseline
    python eval/run_eval.py --config configs/retrieval/v1-hybrid.yaml --no-gate

Exit code 1 when a gated metric drops more than its tolerance below baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.build_golden import chunk_has_evidence  # noqa: E402
from rag_service.config import RetrievalConfig, get_settings  # noqa: E402
from rag_service.generation import format_context  # noqa: E402
from rag_service.judges import judge  # noqa: E402
from rag_service.pipeline import Pipeline  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.jsonl"
BASELINE = ROOT / "eval" / "baseline.json"
REPORT_DIR = ROOT / "eval" / "reports"

# metric -> max allowed drop vs baseline (absolute). Latency is reported, not gated
# (CI runners are noisy).
TOLERANCE = {
    "retrieval.hit_at_5": 0.02,
    "retrieval.mrr": 0.02,
    "retrieval.ndcg_at_5": 0.02,
    "retrieval.heldout_hit_at_5": 0.0,
    "refusal.accuracy": 0.03,
    "refusal.unanswerable_recall": 0.0,
    "refusal_llm.accuracy": 0.05,
    "generation.faithfulness_mean": 0.15,
    "generation.correctness_mean": 0.15,
}


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank_of_first_relevant(passages, evidence: str) -> int | None:
    for p in passages:
        if chunk_has_evidence(p.text, evidence):
            return p.rank
    return None


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


def evaluate(cfg_path: Path, generation: bool, limit: int | None = None) -> tuple[dict, list[dict]]:
    settings = get_settings()
    cfg = RetrievalConfig.load(cfg_path)
    pipe = Pipeline(cfg, settings)
    pipe.retriever.warm()
    golden = load_golden()[:limit] if limit else load_golden()

    rows: list[dict] = []
    for g in golden:
        res = pipe.run(g["question"], use_llm=generation)
        row = {
            "id": g["id"],
            "source": g["source"],
            "answerable": g["answerable"],
            "question": g["question"],
            "scope": res.scope,
            "chunk_ids": [p.chunk_id for p in res.passages],
            "mode": res.generation.mode,
            "answer": res.generation.answer,
            "provider": res.generation.provider,
            "scope_reason": res.scope_reason,
            "retrieval_ms": res.retrieval_ms,
            "total_ms": res.total_ms,
        }
        if g["answerable"]:
            rels = [int(chunk_has_evidence(p.text, g["evidence"])) for p in res.passages[:5]]
            first = _rank_of_first_relevant(res.passages[:5], g["evidence"])
            row.update(hit=int(first is not None), rr=1 / first if first else 0.0, ndcg=_ndcg(rels))
        if generation:
            ctx = format_context(res.passages)
            f = judge("faithfulness", settings, question=g["question"], context=ctx, answer=res.generation.answer)
            c = judge("correctness", settings, question=g["question"], reference=g["reference_answer"], answer=res.generation.answer)
            row.update(faithfulness=f.score, correctness=c.score, judge_model=f.judge_model or c.judge_model)
            time.sleep(float(os.environ.get("EVAL_SLEEP_S", "1.0")))  # stay under free-tier RPM
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    held = [r for r in ans if r["source"] == "heldout"]
    unans = [r for r in rows if not r["answerable"]]
    refused = [r for r in rows if r["mode"] == "refusal"]
    correct_refusals = [r for r in refused if not r["answerable"]]
    metrics: dict = {
        "retrieval": {
            "n": len(ans),
            "hit_at_5": round(statistics.mean(r["hit"] for r in ans), 4) if ans else None,
            "mrr": round(statistics.mean(r["rr"] for r in ans), 4) if ans else None,
            "ndcg_at_5": round(statistics.mean(r["ndcg"] for r in ans), 4) if ans else None,
            "heldout_hit_at_5": round(statistics.mean(r["hit"] for r in held), 4) if held else None,
            "p50_ms": _pct([r["retrieval_ms"] for r in rows], 0.5),
            "p95_ms": _pct([r["retrieval_ms"] for r in rows], 0.95),
        },
        "refusal": {
            "accuracy": round(sum((r["mode"] == "refusal") != r["answerable"] for r in rows) / len(rows), 4),
            "unanswerable_recall": round(len(correct_refusals) / len(unans), 4) if unans else None,
            "precision": round(len(correct_refusals) / len(refused), 4) if refused else None,
            "false_refusals": [r["id"] for r in refused if r["answerable"]],
            "missed_refusals": [r["id"] for r in unans if r["mode"] != "refusal"],
        },
    }
    if generation:
        # With an LLM in the loop refusals also come from the model; keep them apart from
        # the deterministic (scope-check + extractive) numbers so baselines don't mix.
        metrics["refusal_llm"] = metrics.pop("refusal")
        f = [r["faithfulness"] for r in rows if r.get("faithfulness")]
        c = [r["correctness"] for r in rows if r.get("correctness")]
        metrics["generation"] = {
            "n_judged": len(f),
            "faithfulness_mean": round(statistics.mean(f), 3) if f else None,
            "correctness_mean": round(statistics.mean(c), 3) if c else None,
            "modes": {m: sum(r["mode"] == m for r in rows) for m in ("llm", "extractive", "refusal")},
            "judge_model": next((r.get("judge_model") for r in rows if r.get("judge_model")), None),
        }
    metrics["meta"] = {"retrieval_version": cfg.version, "golden_items": len(rows), "prompt_version": settings.prompt_version}
    return metrics, rows


def compare(metrics: dict, baseline: dict) -> list[str]:
    failures = []
    for key, tol in TOLERANCE.items():
        sect, name = key.split(".")
        cur = (metrics.get(sect) or {}).get(name)
        base = (baseline.get(sect) or {}).get(name)
        if cur is None or base is None:
            continue
        if cur < base - tol - 1e-9:
            failures.append(f"{key}: {cur} < baseline {base} - tol {tol}")
    return failures


def to_markdown(metrics: dict, baseline: dict | None, failures: list[str]) -> str:
    lines = [f"### RAG eval — retrieval `{metrics['meta']['retrieval_version']}`, prompt `{metrics['meta']['prompt_version']}`", ""]
    lines += ["| metric | current | baseline |", "|---|---|---|"]
    for sect in ("retrieval", "refusal", "refusal_llm", "generation"):
        for k, v in (metrics.get(sect) or {}).items():
            if isinstance(v, (list, dict)):
                continue
            b = ((baseline or {}).get(sect) or {}).get(k, "")
            lines.append(f"| {sect}.{k} | {v} | {b} |")
    lines += ["", "**Result:** " + ("FAIL\n- " + "\n- ".join(failures) if failures else "pass")]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "retrieval" / "v1.yaml"))
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    metrics, rows = evaluate(Path(args.config), args.generation, args.limit)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "latest.json").write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=1), encoding="utf-8")

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    if args.update_baseline:
        merged = dict(baseline or {})
        merged.update({k: v for k, v in metrics.items() if k != "meta"})
        merged["meta"] = metrics["meta"]
        BASELINE.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"baseline updated: {BASELINE.relative_to(ROOT)}")
    failures = [] if (args.no_gate or baseline is None or args.update_baseline) else compare(metrics, baseline)
    md = to_markdown(metrics, baseline, failures)
    (REPORT_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
