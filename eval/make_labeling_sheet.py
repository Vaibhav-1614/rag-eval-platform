"""
Make the 30-item human labeling sheet used to calibrate the LLM judge.

Picks 30 golden items (stratified: 18 benchmark, 6 held-out, 6 unanswerable; fixed
seed), runs the pipeline (LLM if keys are set, else extractive) and writes
eval/calibration/labeling_sheet.csv with an empty `human_faithfulness` column.

Fill that column by hand with 1/2/3 using the rubric in prompts/judge/faithfulness_v1.md,
then run eval/calibrate_judge.py. Never fill it with model output: the point is to
measure how far the judge is from a human.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from eval.run_eval import load_golden  # noqa: E402
from rag_service.config import get_retrieval_config, get_settings  # noqa: E402
from rag_service.generation import format_context  # noqa: E402
from rag_service.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "eval" / "calibration" / "labeling_sheet.csv"
QUOTA = {"benchmark": 18, "heldout": 6, "unanswerable": 6}


def main() -> None:
    rng = random.Random(30)
    golden = load_golden()
    picked = []
    for src, n in QUOTA.items():
        pool = [g for g in golden if g["source"] == src]
        picked += rng.sample(pool, n)
    pipe = Pipeline(get_retrieval_config(), get_settings())
    pipe.retriever.warm()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "answer_mode", "reference_answer", "context", "human_faithfulness", "notes"])
        for g in picked:
            r = pipe.run(g["question"])
            w.writerow([g["id"], g["question"], r.generation.answer, r.generation.mode,
                        g["reference_answer"], format_context(r.passages), "", ""])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(picked)} rows)")


if __name__ == "__main__":
    main()
