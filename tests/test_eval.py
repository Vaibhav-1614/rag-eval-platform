import json

import pytest
from conftest import ROOT

from eval.build_golden import build, chunk_has_evidence
from eval.calibrate_judge import cohen_kappa
from eval.run_eval import answer_ok, compare
from rag_service.judges import parse_verdict


def test_golden_file_matches_builder():
    expected = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build())
    assert (ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8") == expected


def test_golden_shape():
    rows = build()
    assert len(rows) >= 170
    assert len({r["id"] for r in rows}) == len(rows)
    by_type = {t: [r for r in rows if r["type"] == t] for t in {r["type"] for r in rows}}
    assert set(by_type) == {"lookup", "numeric", "comparison", "cross_company", "thematic", "unanswerable"}
    assert len(by_type["unanswerable"]) == 28
    # every answerable item can be scored for retrieval
    assert all(r["evidence_any"] or r["section_rule"] for r in rows if r["answerable"])
    # the v1 questions stay in dev; every v2 type has a held-out half
    assert all(r["split"] == "dev" for r in rows if r["source"] != "v2")
    for t in ("numeric", "comparison", "cross_company", "thematic", "unanswerable"):
        assert any(r["split"] == "test" for r in by_type[t]), t


def test_answer_ok():
    groups = [["416,161", "416.16 billion"], ["Apple"]]
    assert answer_ok("Apple Inc.'s revenue was $416,161 million [1].", groups)
    assert answer_ok("apple made $416.16 billion", groups)
    assert not answer_ok("Apple made a lot of money", groups)


def test_chunk_has_evidence_partial_overlap():
    ev = "The quick brown fox jumps over the lazy dog and keeps running far away into the forest."
    assert chunk_has_evidence("prefix " + ev + " suffix", ev)
    assert chunk_has_evidence("... jumps over the lazy dog and keeps running far away into the forest. Next", ev)
    assert not chunk_has_evidence("nothing related", ev)


def test_compare_flags_regressions():
    base = {"retrieval": {"hit_at_5": 0.9, "mrr": 0.8}, "refusal": {"accuracy": 0.95, "false_refusal_rate": 0.01}}
    ok = {"retrieval": {"hit_at_5": 0.89, "mrr": 0.8}, "refusal": {"accuracy": 0.95, "false_refusal_rate": 0.02}}
    assert compare(ok, base) == []
    fails = compare({"retrieval": {"hit_at_5": 0.85, "mrr": 0.8}, "refusal": {"accuracy": 0.95, "false_refusal_rate": 0.01}}, base)
    assert len(fails) == 1 and fails[0].startswith("retrieval.hit_at_5")
    # "lower is better" metrics fail when they rise past the tolerance
    fails = compare({"retrieval": {"hit_at_5": 0.9, "mrr": 0.8}, "refusal": {"accuracy": 0.95, "false_refusal_rate": 0.08}}, base)
    assert len(fails) == 1 and fails[0].startswith("refusal.false_refusal_rate")


def test_parse_verdict():
    assert parse_verdict('```json\n{"score": 3, "reason": "ok"}\n```') == (3, "ok")
    assert parse_verdict('{"score": 7}')[0] is None
    assert parse_verdict("no json")[0] is None


def test_kappa():
    assert cohen_kappa([1, 2, 3, 3], [1, 2, 3, 3]) == pytest.approx(1.0)
    k = cohen_kappa([1, 1, 2, 3, 3, 2], [1, 2, 2, 3, 2, 2])
    assert 0 < k < 1
    assert cohen_kappa([1, 2, 3, 3], [1, 2, 3, 2], "quadratic") > cohen_kappa([1, 2, 3, 3], [1, 2, 3, 2])
