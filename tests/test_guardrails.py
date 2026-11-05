import pytest

from rag_service.guardrails import clean_question, refusal_message, scope_check

CTX = ["As of December 31, 2025, our employee headcount worldwide was 134,785. We are headquartered in Austin, Texas."]


def test_refusal_message_states_corpus(cfg):
    msg = refusal_message(cfg)
    assert msg.startswith("I can't answer that from the indexed filings")
    assert "fiscal 2025" in msg and "NVIDIA fiscal 2026" in msg and "XOM" in msg


@pytest.mark.parametrize(
    "q,detected,reason",
    [
        ("What is Tesla's stock price today?", ["TSLA"], "market_data"),
        ("How many cars will Tesla deliver in 2030?", ["TSLA"], "future_period"),
        ("What was Tesla's revenue in the third quarter of 2026?", ["TSLA"], "future_period"),
        ("What was Tesla's revenue in 2012?", ["TSLA"], "period_not_in_context"),
        ("What did Tesla's CEO say on the earnings call?", ["TSLA"], "not_in_filing"),
        ("How many Starbucks stores are in Texas?", [], "unknown_entity"),
        ("What is Tesla's policy on pet insurance reimbursements?", ["TSLA"], "low_coverage"),
    ],
)
def test_out_of_scope(q, detected, reason):
    d = scope_check(q, CTX, detected)
    assert not d.in_scope
    assert d.reason == reason


def test_in_scope_with_stemming():
    d = scope_check("Where is Tesla headquartered?", CTX, ["TSLA"])
    assert d.in_scope, d
    d = scope_check("How many employees did Tesla have at the end of 2025?", CTX, ["TSLA"])
    assert d.in_scope and d.coverage >= 0.5


def test_clean_question():
    assert clean_question("  What\x00 is\n\nthis? ", 100) == "What is this?"
    with pytest.raises(ValueError):
        clean_question("   ", 100)
    with pytest.raises(ValueError):
        clean_question("x" * 101, 100)
