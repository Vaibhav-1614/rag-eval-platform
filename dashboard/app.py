"""
Streamlit dashboard: Benchmark results, Live query, Dataset explorer.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.ragas_runner import retrieve_for_config
from evaluation.results_store import load_all_results
from free_llm import extractive_answer, gemini_answer
from platform_config import load_settings
from retrieval import artifacts

REPO_URL = "https://github.com/Vaibhav-1614/rag-eval-platform"
METRICS = ["hit_rate", "mrr", "ndcg", "context_precision", "context_recall"]
METRIC_LABELS = {
    "hit_rate": "Hit Rate@5",
    "mrr": "MRR",
    "ndcg": "nDCG@5",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}
STRATEGY_COLORS = {"dense": "#4C9AFF", "sparse": "#F5A623", "hybrid": "#4CAF50"}
EMB_LABELS = {"sentence-transformers": "MiniLM", "bge": "BGE-small"}


@st.cache_data
def cached_settings() -> dict:
    return load_settings()


@st.cache_data
def cached_results() -> pd.DataFrame:
    return load_all_results()


@st.cache_data
def cached_testset() -> list[dict]:
    settings = cached_settings()
    p = ROOT / (settings.get("paths") or {}).get("testset_path", "data/testset.json")
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def corpus_stats() -> pd.DataFrame:
    settings = cached_settings()
    names = settings.get("company_names") or {}
    rows = []
    for t in settings.get("companies", []):
        meta_p = ROOT / "data" / "raw" / t / "filing_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        parsed = ROOT / "data" / "parsed" / f"{t}.txt"
        row = {
            "Ticker": t,
            "Company": names.get(t, t),
            "Filed": meta.get("filingDate", ""),
            "Parsed chars": len(parsed.read_text(encoding="utf-8")) if parsed.exists() else 0,
        }
        for cs in settings.get("chunk_sizes", [256, 512, 1024]):
            chunks = artifacts.load_chunks(t, cs, settings) or []
            row[f"{cs}-tok chunks"] = len(chunks)
        rows.append(row)
    return pd.DataFrame(rows)


def label(strategy: str, chunk_size: int, emb: str) -> str:
    if strategy == "sparse":
        return f"sparse · {chunk_size} · BM25"
    return f"{strategy} · {chunk_size} · {EMB_LABELS.get(emb, emb)}"


def results_frame() -> pd.DataFrame:
    df = cached_results()
    if df.empty:
        return df
    df = df.copy()
    df["config"] = [
        label(s, c, e) for s, c, e in zip(df["retrieval_strategy"], df["chunk_size"], df["embedding_model"])
    ]
    df["avg_score"] = df[METRICS].mean(axis=1)
    return df


# --------------------------------------------------------------------------- benchmark


def page_benchmark() -> None:
    df_all = results_frame()
    st.title("Benchmark Results")
    if df_all.empty:
        st.warning("No benchmark results yet. Run `python -m benchmark.run_all`.")
        return
    versions = sorted(df_all["eval_version"].dropna().unique(), reverse=True)
    n_q = len(cached_testset())
    n_t = len(cached_settings().get("companies", []))

    top = st.columns([3, 1])
    ev = top[1].selectbox("Eval version", versions, index=0)
    df = df_all[df_all["eval_version"] == ev].copy()
    top[0].caption(
        f"{len(df)} retrieval configs × {n_q} questions over {n_t} SEC 10-K filings · "
        "3 strategies × 3 chunk sizes × 2 embedding models · metrics @k=5 · "
        "BM25 runs are embedding-independent, so they are shown once per chunk size"
    )
    df = df.drop_duplicates(subset="config")

    best = df.loc[df["avg_score"].idxmax()]
    k = st.columns(5)
    short = best["config"].split(" · ")
    k[0].metric(
        "Best config",
        f"{short[2] if best['retrieval_strategy'] == 'sparse' else short[0]} {short[1]}",
        help=f"{best['config']}: highest mean of all 5 metrics",
    )
    k[1].metric("Hit Rate@5", f"{df['hit_rate'].max():.0%}", help=df.loc[df["hit_rate"].idxmax(), "config"])
    k[2].metric("MRR", f"{df['mrr'].max():.3f}", help=df.loc[df["mrr"].idxmax(), "config"])
    k[3].metric("nDCG@5", f"{df['ndcg'].max():.3f}", help=df.loc[df["ndcg"].idxmax(), "config"])
    k[4].metric(
        "Latency (best config)",
        f"{best['avg_latency_ms']:.0f} ms",
        help="Mean per-query retrieval latency, local CPU",
    )

    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.subheader("Strategy × chunk size")
        metric_h = st.selectbox(
            "Metric", METRICS, index=0, format_func=METRIC_LABELS.get, key="heat_metric"
        )
        heat = df.groupby(["chunk_size", "retrieval_strategy"], as_index=False)[metric_h].mean()
        pivot = heat.pivot(index="chunk_size", columns="retrieval_strategy", values=metric_h)
        pivot = pivot.reindex(columns=[c for c in ["dense", "sparse", "hybrid"] if c in pivot.columns])
        pivot.index = [f"{i} tokens" for i in pivot.index]
        fig_h = px.imshow(
            pivot,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(x="Retrieval strategy", y="Chunk size", color=METRIC_LABELS[metric_h]),
        )
        fig_h.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        fig_h.update_traces(textfont_size=16)
        st.plotly_chart(fig_h, width="stretch")
        st.caption("Mean over embedding models (BM25 is embedding-independent).")
    with c2:
        st.subheader("Config leaderboard")
        board = df.sort_values("avg_score", ascending=True)
        fig_l = px.bar(
            board,
            x="avg_score",
            y="config",
            color="retrieval_strategy",
            orientation="h",
            color_discrete_map=STRATEGY_COLORS,
            labels={"avg_score": "Mean of 5 retrieval metrics", "config": "", "retrieval_strategy": "Strategy"},
            hover_data={m: ":.3f" for m in METRICS},
        )
        fig_l.update_layout(
            height=470,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text=""),
        )
        st.plotly_chart(fig_l, width="stretch")

    c3, c4 = st.columns([1.25, 1])
    with c3:
        st.subheader("Metric breakdown by strategy")
        by_s = df.groupby("retrieval_strategy", as_index=False)[METRICS].mean()
        melted = by_s.melt(id_vars="retrieval_strategy", var_name="metric", value_name="score")
        melted["metric"] = melted["metric"].map(METRIC_LABELS)
        fig_b = px.bar(
            melted,
            x="metric",
            y="score",
            color="retrieval_strategy",
            barmode="group",
            color_discrete_map=STRATEGY_COLORS,
            text_auto=".2f",
            labels={"metric": "", "score": "Score", "retrieval_strategy": "Strategy"},
        )
        fig_b.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 1])
        st.plotly_chart(fig_b, width="stretch")
    with c4:
        st.subheader("Latency vs quality")
        fig_s = px.scatter(
            df,
            x="avg_latency_ms",
            y="avg_score",
            color="retrieval_strategy",
            symbol="chunk_size",
            color_discrete_map=STRATEGY_COLORS,
            hover_data=["config"],
            labels={
                "avg_latency_ms": "Mean latency per query (ms)",
                "avg_score": "Mean retrieval score",
                "retrieval_strategy": "Strategy",
                "chunk_size": "Chunk",
            },
        )
        fig_s.update_traces(marker=dict(size=13, line=dict(width=1, color="#0e1117")))
        fig_s.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_s, width="stretch")

    st.success(
        f"**Recommended config: {best['config']}** — highest mean retrieval score "
        f"({best['avg_score']:.3f}; Hit@5 {best['hit_rate']:.0%}, MRR {best['mrr']:.3f}) "
        f"at {best['avg_latency_ms']:.0f} ms/query."
    )

    st.subheader("All configurations")
    table = df.sort_values("avg_score", ascending=False)[
        ["config", "avg_score", *METRICS, "avg_latency_ms"]
    ]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "config": st.column_config.TextColumn("Config (strategy · chunk · embedding)"),
            "avg_score": st.column_config.ProgressColumn("Mean", min_value=0, max_value=1, format="%.3f"),
            **{
                m: st.column_config.NumberColumn(METRIC_LABELS[m], format="%.3f")
                for m in METRICS
            },
            "avg_latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
        },
    )


# --------------------------------------------------------------------------- live query


def _answer(question: str, hits: list[dict]) -> tuple[str, str]:
    ctx = "\n\n".join(h["text"][:1500] for h in hits[:5])
    prompt = (
        "Answer the question using only the SEC 10-K excerpts below. Be concise; "
        f"say so if the excerpts do not contain the answer.\n\nExcerpts:\n{ctx}\n\nQuestion: {question}"
    )
    model = (cached_settings().get("free_llm") or {}).get("model")
    ans, _ = gemini_answer(prompt, model=model)
    if ans:
        return ans, f"Generated by {model} from the top-5 chunks"
    return (
        extractive_answer(question, [h["text"] for h in hits]),
        "Extractive answer: best-matching sentences from the retrieved chunks (no LLM key configured)",
    )


def _render_hits(hits: list[dict], compact: bool = False) -> None:
    for i, h in enumerate(hits, 1):
        score = h.get("score", h.get("similarity"))
        title = f"#{i} · {h.get('chunk_id', '')} · score {score:.3f}" if score is not None else h.get("chunk_id", "")
        with st.expander(title, expanded=(i == 1 and not compact)):
            st.write(h.get("text", "")[:1800])


def page_live() -> None:
    st.title("Live Query")
    st.caption("Ask a question against one company's 10-K with any retrieval configuration.")
    settings = cached_settings()
    names = settings.get("company_names") or {}
    tickers = [t.upper() for t in settings.get("companies", [])]
    testset = cached_testset()

    c = st.columns(4)
    ticker = c[0].selectbox("Company", tickers, format_func=lambda t: f"{t} · {names.get(t, t)}")
    strat = c[1].selectbox("Retrieval strategy", ["hybrid", "dense", "sparse"])
    cs = c[2].selectbox("Chunk size (tokens)", [512, 256, 1024])
    em = c[3].selectbox(
        "Embedding model",
        ["sentence-transformers", "bge"],
        format_func=lambda e: EMB_LABELS.get(e, e),
        disabled=(strat == "sparse"),
    )

    samples = [r["question"] for r in testset if r["ticker"] == ticker]
    sample = st.selectbox("Sample questions from the test set", ["(write your own)"] + samples, index=1 if samples else 0)
    default_q = "" if sample == "(write your own)" else sample
    q = st.text_input("Question", value=default_q, placeholder="e.g. What are the main risks to the supply chain?")
    compare = st.toggle("Also compare best vs worst benchmark config")

    if not (st.button("Run query", type="primary") and q.strip()):
        return

    with st.spinner("Retrieving…"):
        t0 = time.perf_counter()
        hits = retrieve_for_config(q, ticker, int(cs), em, strat, top_k=5, settings=settings)
        latency = (time.perf_counter() - t0) * 1000
    if not hits:
        st.error("No index found for this configuration.")
        return
    with st.spinner("Answering…"):
        ans, how = _answer(q, hits)

    st.subheader("Answer")
    st.info(ans or "No matching sentence found in the retrieved chunks.")
    st.caption(f"{how} · retrieval {latency:.0f} ms · cost $0.00 (local embeddings)")

    if compare:
        df = results_frame()
        if df.empty:
            st.info("No benchmark results for comparison.")
        else:
            df = df[df["eval_version"] == df["eval_version"].max()].drop_duplicates(subset="config")
            best = df.loc[df["avg_score"].idxmax()]
            worst = df.loc[df["avg_score"].idxmin()]
            cols = st.columns(2)
            for col, row, tag in [(cols[0], best, "Best"), (cols[1], worst, "Worst")]:
                with col:
                    st.markdown(f"**{tag} benchmark config:** `{row['config']}` (mean {row['avg_score']:.3f})")
                    hh = retrieve_for_config(
                        q, ticker, int(row["chunk_size"]), str(row["embedding_model"]),
                        str(row["retrieval_strategy"]), top_k=5, settings=settings,
                    )
                    st.info(_answer(q, hh)[0] or "—")
                    _render_hits(hh, compact=True)
        return

    st.subheader("Retrieved chunks")
    _render_hits(hits)


# --------------------------------------------------------------------------- dataset


def page_dataset() -> None:
    st.title("Dataset Explorer")
    stats = corpus_stats()
    rows = cached_testset()

    k = st.columns(4)
    chunk_cols = [c for c in stats.columns if c.endswith("chunks")]
    k[0].metric("Filings", len(stats))
    k[1].metric("Parsed text", f"{stats['Parsed chars'].sum() / 1e6:.1f}M chars")
    k[2].metric("Indexed chunks", f"{int(stats[chunk_cols].sum().sum()):,}", help="Across 256/512/1024-token sizes")
    k[3].metric("Test questions", len(rows))

    st.subheader("Corpus — latest 10-K per company (SEC EDGAR)")
    st.dataframe(
        stats,
        hide_index=True,
        width="stretch",
        column_config={"Parsed chars": st.column_config.NumberColumn(format="%d")},
    )

    st.subheader("Evaluation test set")
    if not rows:
        st.warning("No testset.json yet. Run `python -m evaluation.testset_generator`.")
        return
    df = pd.DataFrame(rows)
    c = st.columns([1, 3])
    t = c[0].selectbox("Filter ticker", ["ALL"] + sorted(df["ticker"].unique().tolist()))
    qf = c[1].text_input("Search questions")
    if t != "ALL":
        df = df[df["ticker"] == t]
    if qf:
        df = df[df["question"].str.contains(qf, case=False, na=False)]
    st.dataframe(
        df[["ticker", "question", "ground_truth_answer", "ground_truth_chunk_id"]],
        hide_index=True,
        width="stretch",
        height=460,
        column_config={
            "ticker": "Ticker",
            "question": st.column_config.TextColumn("Question", width="large"),
            "ground_truth_answer": st.column_config.TextColumn("Ground-truth answer (verbatim)", width="large"),
            "ground_truth_chunk_id": "Gold chunk",
        },
    )


# --------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="RAG Eval Platform", page_icon="📊", layout="wide")
    with st.sidebar:
        st.title("RAG Eval Platform")
        st.markdown(
            "Benchmarks **dense** (ChromaDB), **sparse** (BM25) and **hybrid** (RRF) retrieval "
            "on SEC 10-K filings with free local embeddings (MiniLM, BGE)."
        )
        page = st.radio("Page", ["Benchmark Results", "Live Query", "Dataset Explorer"])
        st.divider()
        st.markdown(f"[Source on GitHub]({REPO_URL})")

    if page == "Benchmark Results":
        page_benchmark()
    elif page == "Live Query":
        page_live()
    else:
        page_dataset()


if __name__ == "__main__":
    main()
