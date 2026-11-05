"""
Agreement between the LLM faithfulness judge and human labels (Cohen's kappa).

    python eval/calibrate_judge.py            # needs GEMINI_API_KEY or GROQ_API_KEY

Reads eval/calibration/labeling_sheet.csv (human_faithfulness filled in by a person),
asks the judge for the same rows, and writes eval/calibration/judge_agreement.json.
Rule of thumb used here: quadratic-weighted kappa >= 0.6 before trusting judge scores
as a CI gate; below that, keep them report-only and fix the rubric.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from rag_service.config import get_settings  # noqa: E402
from rag_service.judges import judge  # noqa: E402

SHEET = ROOT / "eval" / "calibration" / "labeling_sheet.csv"
OUT = ROOT / "eval" / "calibration" / "judge_agreement.json"
LABELS = (1, 2, 3)


def cohen_kappa(a: list[int], b: list[int], weights: str | None = None) -> float:
    n = len(a)
    if n == 0:
        raise ValueError("no pairs")
    k = len(LABELS)
    idx = {v: i for i, v in enumerate(LABELS)}

    def w(i: int, j: int) -> float:
        if weights == "quadratic":
            return ((i - j) ** 2) / ((k - 1) ** 2)
        if weights == "linear":
            return abs(i - j) / (k - 1)
        return 0.0 if i == j else 1.0

    obs = Counter((idx[x], idx[y]) for x, y in zip(a, b, strict=False))
    pa, pb = Counter(idx[x] for x in a), Counter(idx[y] for y in b)
    observed = sum(w(i, j) * c for (i, j), c in obs.items()) / n
    expected = sum(w(i, j) * pa[i] * pb[j] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 if expected == 0 else 1 - observed / expected


def main() -> None:
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8")))
    labeled = [r for r in rows if r["human_faithfulness"].strip()]
    if len(labeled) < len(rows):
        sys.exit(f"{len(rows) - len(labeled)} rows still need a human_faithfulness label (1/2/3) in {SHEET}")
    settings = get_settings()
    human, model, detail = [], [], []
    for r in labeled:
        v = judge("faithfulness", settings, question=r["question"], context=r["context"], answer=r["answer"])
        if v.score is None:
            sys.exit(f"judge failed on {r['id']}: {v.reason}")
        human.append(int(r["human_faithfulness"]))
        model.append(v.score)
        detail.append({"id": r["id"], "human": human[-1], "judge": v.score, "reason": v.reason})
    result = {
        "n": len(human),
        "judge_model": v.judge_model,
        "raw_agreement": round(sum(h == m for h, m in zip(human, model, strict=False)) / len(human), 3),
        "kappa": round(cohen_kappa(human, model), 3),
        "kappa_quadratic": round(cohen_kappa(human, model, "quadratic"), 3),
        "rows": detail,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
