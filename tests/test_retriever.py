import pytest

from rag_service.retriever import detect_tickers


@pytest.mark.parametrize(
    "q,expected",
    [
        ("How many employees did Tesla have?", ["TSLA"]),
        ("Compare Apple's and Microsoft's R&D spending", ["AAPL", "MSFT"]),
        ("What does AWS contribute to Amazon's income?", ["AMZN"]),
        ("What risks does J&J face from talc litigation?", ["JNJ"]),
        ("How does JPMorgan Chase manage liquidity?", ["JPM"]),
        ("What is NVDA's data center revenue?", ["NVDA"]),
        ("How does the company handle metadata and purchase obligations?", []),
        ("What was Netflix's subscriber count?", []),
    ],
)
def test_detect_tickers(q, expected):
    assert detect_tickers(q) == expected


def test_single_company_search(retriever):
    passages, scope = retriever.search("employee headcount worldwide", ["TSLA"])
    assert scope == ["TSLA"]
    assert len(passages) == 5
    assert all(p.ticker == "TSLA" for p in passages)
    assert [p.rank for p in passages] == [1, 2, 3, 4, 5]
    assert any("134,785" in p.text for p in passages)


def test_multi_company_interleaves(retriever):
    passages, scope = retriever.search("research and development expense", ["AAPL", "MSFT"])
    assert scope == ["AAPL", "MSFT"]
    assert {p.ticker for p in passages[:2]} == {"AAPL", "MSFT"}


def test_global_search_when_no_company(retriever):
    passages, scope = retriever.search("Ethicon Physiomesh hernia mesh", [])
    assert scope == ["ALL"]
    assert passages[0].ticker == "JNJ"


def test_unknown_tickers_fall_back_to_global(retriever):
    _, scope = retriever.search("revenue", ["ZZZZ"])
    assert scope == ["ALL"]
