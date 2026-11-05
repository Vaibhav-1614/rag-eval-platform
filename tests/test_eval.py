import json

import pytest
from conftest import ROOT

from eval.build_golden import build, chunk_has_evidence
from eval.calibrate_judge import cohen_kappa
from eval.run_eval import compare
from rag_service.judges import parse_verdict


def test_golden_file_matches_builder():
    expected = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build())
    assert (ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8") == expected


def test_golden_shape():
    rows = build()
    assert len(rows) >= 60
    unans = [r for r in rows if not r["answerable"]]
    assert len(unans) == 12
    assert all(r["gold_chunk_ids"] for r in rows if r["answerable"])
    assert len({r["id"] for r in rows}) == len(rows)


def test_chunk_has_evidence_partial_overlap():
    ev = "The quick brown fox jumps over the lazy dog and keeps running far away into the forest."
    assert chunk_has_evidence("prefix " + ev + " suffix", ev)
    assert chunk_has_evidence("... jumps over the lazy dog and keeps running far away into the forest. Next", ev)
    assert not chunk_has_evidence("nothing related", ev)


def test_compare_flags_regressions():
    base = {"retrieval": {"hit_at_5": 0.9, "mrr": 0.8}, "refusal": {"accuracy": 0.95}}
    assert compare({"retrieval": {"hit_at_5": 0.89, "mrr": 0.8}, "refusal": {"accuracy": 0.95}}, base) == []
    fails = compare({"retrieval": {"hit_at_5": 0.85, "mrr": 0.8}, "refusal": {"accuracy": 0.95}}, base)
    assert len(fails) == 1 and fails[0].startswith("retrieval.hit_at_5")


def test_parse_verdict():
    assert parse_verdict('```json\n{"score": 3, "reason": "ok"}\n```') == (3, "ok")
    assert parse_verdict('{"score": 7}')[0] is None
    assert parse_verdict("no json")[0] is None


def test_kappa():
    assert cohen_kappa([1, 2, 3, 3], [1, 2, 3, 3]) == pytest.approx(1.0)
    k = cohen_kappa([1, 1, 2, 3, 3, 2], [1, 2, 2, 3, 2, 2])
    assert 0 < k < 1
    assert cohen_kappa([1, 2, 3, 3], [1, 2, 3, 2], "quadratic") > cohen_kappa([1, 2, 3, 3], [1, 2, 3, 2])
