"""v2 index: chunker, text processing, facts table, structured retriever, pipeline."""
import json

import pytest
from conftest import ROOT

from ingestion.structured_chunker import chunk_filing
from rag_service.config import RetrievalConfig, get_settings
from rag_service.facts import detect_metric, fact_answer, find_facts, load_facts
from rag_service.pipeline import Pipeline
from rag_service.retriever import detect_tickers, is_cross_company
from rag_service.textproc import BM25, QueryExpander, tokenize

FILING = """Cover text for the annual report of Example Corp for the fiscal year.
Item 1. Business
Example Corp designs and sells widgets worldwide. The company was founded in 1999 and has grown through acquisitions in many markets over time.
Example Corp | 2025 Form 10-K | 3
Item 1A. Risk Factors
Competition
The widget market is highly competitive and competitors may have greater resources than we do, which could adversely affect our results.
Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
The following table shows net sales by category (dollars in millions):
2025 2024 2023
Widgets $ 1,200 $ 1,100 $ 1,000
Gadgets 300 250 200
Total net sales $ 1,500 $ 1,350 $ 1,200
Net sales grew because of higher widget demand in all regions during the year compared with the prior year.
"""


@pytest.fixture(scope="module")
def pipe():
    p = Pipeline(RetrievalConfig.load(ROOT / "configs" / "retrieval" / "v2-bm25.yaml"), get_settings())
    p.retriever.warm()
    return p


# ---------- chunker ----------
def test_chunker_sections_tables_and_footers():
    chunks = chunk_filing(FILING, "EXM", "Example Corp", 2025)
    by_section = {c["section"] for c in chunks}
    assert {"1", "1A", "7"} <= by_section
    table = next(c for c in chunks if c["kind"] == "table")
    # caption + header row + every row stay together
    assert table["text"].startswith("The following table shows net sales by category")
    assert "2025 2024 2023" in table["text"] and "Total net sales $ 1,500" in table["text"]
    assert table["context"] == "Example Corp 10-K, fiscal 2025 | Item 7. Management's Discussion and Analysis"
    assert not any("Form 10-K | 3" in c["text"] for c in chunks)  # page footer dropped
    risk = next(c for c in chunks if c["section"] == "1A")
    assert risk["context"].endswith("| Competition")


def test_chunker_rejoins_one_cell_per_line_tables():
    text = "Item 8. Financial Statements\nINCOME STATEMENTS\nRevenue\n281,724\n245,122\nCost of revenue\n87,831\n74,114\nNet income\n101,832\n88,136\n"
    chunks = chunk_filing(text, "EXM", "Example Corp", 2025)
    assert any("Revenue 281,724 245,122" in c["text"] and c["kind"] == "table" for c in chunks)


def test_committed_v2_chunks_match_the_chunker():
    import gzip

    facts = load_facts()
    with gzip.open(ROOT / "data" / "index" / "v2" / "chunks" / "AAPL.json.gz", "rb") as f:
        stored = json.loads(f.read().decode())
    fresh = chunk_filing((ROOT / "data" / "parsed" / "AAPL.txt").read_text(encoding="utf-8"), "AAPL",
                         facts["AAPL"]["company"], facts["AAPL"]["fiscal_year"])
    assert [c["text"] for c in stored] == [c["text"] for c in fresh]
    assert {c["section"] for c in stored} >= {"1", "1A", "7", "8"}


# ---------- text processing ----------
def test_tokenize_stems_and_keeps_compounds():
    assert tokenize("Where is Tesla's R&D headquartered?") == ["tesla", "r&d", "headquart"]
    assert tokenize("headquarters") == tokenize("headquartered")
    assert tokenize("2025 employees") == ["2025", "employe"]


def test_bm25_ranks_and_masks():
    import numpy as np

    b = BM25([tokenize("the cat sat"), tokenize("dogs chase cats"), tokenize("stock markets fell")])
    assert [i for i, _ in b.top(tokenize("cats"), 3)] == [0, 1]
    assert [i for i, _ in b.top(tokenize("cats"), 3, mask=np.array([False, True, True]))] == [1]
    assert b.top(tokenize("zebra"), 3) == []


def test_query_expansion():
    ex = QueryExpander(ROOT / "configs" / "query_expansion.yaml")
    assert "research and development" in ex.expand("Compare R&D spending")
    assert "employees" in ex.expand("How big is the workforce?")
    assert ex.expand("What is the fiscal year end?") == "What is the fiscal year end?"


# ---------- facts ----------
@pytest.mark.parametrize(
    "q,metric",
    [
        ("What was Apple's total net sales in fiscal 2025?", "revenue"),
        ("How much does J&J invest in research and development?", "rd_expense"),
        ("what did apple earn per share", "eps_diluted"),
        ("Compare the total assets of JPMorgan and Exxon Mobil.", "total_assets"),
        ("What is NVIDIA's gross margin?", "gross_margin"),
        # narrowed to a product, segment or region -> must come from the text, not the total
        ("What was Apple's iPhone revenue?", None),
        ("What was AWS operating income?", None),
        ("What was NVIDIA's Data Center revenue?", None),
        ("How much revenue does Meta get from advertising?", None),
        ("What was Meta's advertising revenue?", None),
        ("What were JPMorgan's sales in China?", None),
        ("How concentrated is NVIDIA's revenue among its biggest customers?", None),
        ("What are Tesla's main risks?", None),
    ],
)
def test_detect_metric(q, metric):
    assert detect_metric(q) == metric


def test_facts_match_the_filing_and_answer_directly():
    q = "What was Apple's total net sales in fiscal 2025?"
    hits, mode = find_facts(q, detect_tickers(q))
    assert mode == "single" and hits[0].values[2025] == 416161000000
    assert "$416,161 million ($416.16 billion) in fiscal 2025" in fact_answer(q, hits, mode)
    assert "416,161" in (ROOT / "data" / "parsed" / "AAPL.txt").read_text(encoding="utf-8")


def test_facts_ranking_and_guards():
    q = "Which company has the most employees?"
    hits, mode = find_facts(q, [])
    assert mode == "ranking" and hits[0].ticker == "AMZN" and len(hits) == 10
    assert fact_answer(q, hits, mode).startswith("Amazon.com, Inc. has the highest number of employees")
    assert find_facts("Which company had the lowest net income?", [])[0][0].ticker == "TSLA"
    # no figures for years the filing does not report, quarters, or forecasts
    assert find_facts("What was Tesla's revenue in 2019?", ["TSLA"]) == ([], "none")
    assert find_facts("What was Apple's revenue in the second quarter of 2025?", ["AAPL"]) == ([], "none")
    assert find_facts("What will Amazon's revenue be next year?", ["AMZN"]) == ([], "none")
    # no company named and no superlative: nothing to look up
    assert find_facts("What is revenue?", []) == ([], "none")


def test_every_fact_is_tagged_to_the_indexed_filing():
    for t, c in load_facts().items():
        meta = json.loads((ROOT / "data" / "raw" / t / "filing_meta.json").read_text())
        assert c["accession"] == meta["accessionNumber"]
        assert {"revenue", "net_income", "total_assets", "employees"} <= c["metrics"].keys(), t


# ---------- retrieval + pipeline ----------
def test_cross_company_detection():
    assert is_cross_company("Which company spends the most on lobbying?")
    assert not is_cross_company("What does Apple spend on lobbying?")


def test_structured_retrieval_finds_the_right_section(pipe):
    ps, scope = pipe.retriever.search("What are the main risks Tesla faces from competition?", ["TSLA"])
    assert scope == ["TSLA"] and ps[0].section == "Item 1A. Risk Factors"
    ps, _ = pipe.retriever.search("Compare R&D spending", ["AAPL"])  # needs the R&D expansion
    assert "Research and Development" in ps[0].context
    assert all(p.source_url and p.fiscal_year == 2025 for p in ps)


def test_pipeline_fact_question(pipe):
    r = pipe.run("How much cash does Amazon have?", use_llm=False)
    assert r.facts_mode == "single" and r.passages[0].kind == "fact" and r.passages[0].rank == 1
    assert r.generation.mode == "extractive" and "$86,810 million" in r.generation.answer
    assert [p.rank for p in r.passages] == list(range(1, len(r.passages) + 1))


def test_pipeline_comparison_and_ranking(pipe):
    r = pipe.run("Who spends more on R&D, Apple or Microsoft?", use_llm=False)
    assert "34,550" in r.generation.answer and "32,488" in r.generation.answer
    r = pipe.run("Which company earned the most net income?", use_llm=False)
    assert r.facts_mode == "ranking" and r.generation.answer.startswith("Alphabet Inc.")


def test_pipeline_refuses_out_of_scope(pipe):
    for q in ("What is Tesla's current market capitalization?", "What was Apple's net income in 2010?",
              "What was Netflix's subscriber count in 2025?"):
        assert pipe.run(q, use_llm=False).generation.mode == "refusal", q
