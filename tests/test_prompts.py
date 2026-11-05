import pytest

from rag_service.prompts import load_prompt


def test_answer_prompt_renders():
    p = load_prompt("answer/v1")
    out = p.render(question="Q?", context="[1] ctx", corpus_description="corpus", refusal_message="NOPE")
    assert "Question: Q?" in out and "[1] ctx" in out and "NOPE" in out
    assert "{" not in out  # every placeholder filled
    assert len(p.sha) == 12


def test_judge_prompt_keeps_json_braces():
    out = load_prompt("judge/faithfulness_v1").render(question="q", context="c", answer="a")
    assert '{"score": <1|2|3>' in out


def test_missing_input_raises():
    with pytest.raises(KeyError):
        load_prompt("answer/v1").render(question="q")
