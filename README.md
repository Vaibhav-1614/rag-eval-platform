# RAG Evaluation Platform — SEC 10-K

**Live demo:** https://vaibhav-1614-rag-eval-platform-dashboardapp-k5zaiu.streamlit.app

Benchmarks **dense** (ChromaDB), **sparse** (BM25) and **hybrid** (Reciprocal Rank Fusion) retrieval
over the latest 10-K filings of 10 large-cap companies, across 3 chunk sizes and 2 free local
embedding models: **18 configurations**, scored with retrieval-first metrics stored in SQLite and
explored in a 3-page Streamlit dashboard.

## Results

50 questions × 10 filings, top-5 retrieval, local CPU (18 configs; BM25 rows are identical across
embedding models, so they are listed once per chunk size).

| Strategy | Chunk | Embedding | Hit@5 | MRR | nDCG@5 | Precision | Recall | Latency |
|---|---|---|---|---|---|---|---|---|
| sparse | 256 | — (BM25) | 0.98 | 0.846 | 0.870 | 0.308 | 0.986 | 6 ms |
| sparse | 512 | — (BM25) | 0.98 | 0.785 | 0.833 | 0.300 | 0.984 | 5 ms |
| sparse | 1024 | — (BM25) | 0.96 | 0.762 | 0.801 | 0.252 | 0.972 | 3 ms |
| hybrid | 256 | BGE-small | 0.90 | 0.751 | 0.781 | 0.268 | 0.919 | 30 ms |
| hybrid | 512 | BGE-small | 0.88 | 0.768 | 0.779 | 0.264 | 0.907 | 25 ms |
| hybrid | 256 | MiniLM | 0.88 | 0.711 | 0.745 | 0.268 | 0.908 | 30 ms |
| hybrid | 1024 | BGE-small | 0.82 | 0.650 | 0.690 | 0.204 | 0.866 | 26 ms |
| hybrid | 512 | MiniLM | 0.76 | 0.633 | 0.652 | 0.232 | 0.818 | 25 ms |
| hybrid | 1024 | MiniLM | 0.74 | 0.543 | 0.586 | 0.192 | 0.805 | 20 ms |
| dense | 256 | BGE-small | 0.66 | 0.548 | 0.572 | 0.196 | 0.732 | 14 ms |
| dense | 512 | BGE-small | 0.68 | 0.512 | 0.549 | 0.184 | 0.753 | 14 ms |
| dense | 256 | MiniLM | 0.66 | 0.524 | 0.550 | 0.188 | 0.736 | 13 ms |
| dense | 1024 | BGE-small | 0.64 | 0.456 | 0.496 | 0.156 | 0.711 | 13 ms |
| dense | 512 | MiniLM | 0.62 | 0.387 | 0.444 | 0.176 | 0.706 | 13 ms |
| dense | 1024 | MiniLM | 0.46 | 0.352 | 0.380 | 0.120 | 0.580 | 14 ms |

**Takeaways**

- **BM25 wins on 10-K text** (Hit@5 0.97 vs 0.83 hybrid and 0.62 dense, averaged
  over configs). Filings are dense with exact entities — product names, rule numbers, subsidiaries —
  that lexical matching nails and small general-purpose embedding models blur.
- **Hybrid RRF recovers most of the gap** over dense-only (+21 pts Hit@5) but trails pure BM25 because
  fusing in the weaker dense ranking pushes some correct BM25 hits down.
- **Smaller chunks rank better:** 256/512-token chunks lead in every strategy, and 1024-token
  chunks are the weakest in all of them because they dilute the relevant passage.
- **BGE-small beats MiniLM** in all 6 dense/hybrid pairings at the same chunk size.
- All of it runs for **$0**: local embeddings, no LLM calls needed for evaluation.


## How it works

```mermaid
flowchart LR
    A[SEC EDGAR API] -->|latest 10-K| B[Parser<br/>BeautifulSoup / PyMuPDF]
    B -->|clean text| C[Chunker<br/>256 / 512 / 1024 tokens]
    C --> D[(ChromaDB<br/>MiniLM, BGE)]
    C --> E[(BM25 index)]
    D --> F{Retriever<br/>dense / sparse / hybrid RRF}
    E --> F
    G[Test set<br/>50 Q&A] --> H[Evaluator<br/>Hit@5, MRR, nDCG, precision, recall]
    F --> H --> I[(SQLite results)] --> J[Streamlit dashboard]
```

| Stage | Details |
|---|---|
| **Ingestion** | Ticker → CIK via `company_tickers.json`, latest 10-K from the EDGAR submissions API, rate-limited to 10 req/s with exponential backoff. |
| **Parsing** | Strips the hidden inline-XBRL header (6–19% of each filing is taxonomy metadata), breaks lines only at block elements so sentences and table rows stay intact. |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter`, 256/512/1024-token targets with 10% overlap. |
| **Dense** | `all-MiniLM-L6-v2` and `bge-small-en-v1.5` embeddings (cached with joblib) in cosine ChromaDB collections — one per ticker × chunk size × model (60 total). |
| **Sparse** | `rank_bm25` Okapi BM25, one index per ticker × chunk size, loaded once per process. |
| **Hybrid** | Dense top-20 + BM25 top-20 fused with RRF (k = 60). |
| **Answering** | Optional Gemini (`GOOGLE_API_KEY`); otherwise an extractive answer from the best-matching retrieved sentences. |

### Evaluation methodology

- **Test set:** 50 questions (5 per company) over randomly sampled *prose* chunks (cover pages and
  number tables are skipped). Answers are verbatim sentences from the filing; the gold context is
  the source 512-token chunk. Committed questions were written by an LLM as natural paraphrases —
  never copying the source sentence, so BM25 gets no verbatim-match advantage.
  `python -m evaluation.testset_generator` regenerates the sample (free keyword-template
  questions, or `--use-llm` for Gemini).
- **Relevance:** a retrieved chunk is relevant when its content-word overlap F1 with the gold
  context is ≥ 0.35 (stopwords and 10-K boilerplate removed). This is chunk-size agnostic;
  random chunk pairs from the same filing clear it only ~1–3% of the time, true matches score ~0.7.
- **Metrics @5:** Hit Rate, MRR, binary nDCG, context precision (relevant share of top-5),
  context recall (best single-chunk overlap, capped at 1 above threshold), mean query latency.
- **Resumable:** each config is upserted into SQLite keyed by
  (strategy, chunk size, embedding model, eval version); re-runs skip finished configs (`--force` to redo).
  `docs/schema_postgres.sql` has the equivalent PostgreSQL DDL.

## Dashboard

- **Benchmark Results** — KPI tiles, strategy × chunk-size heatmap, config leaderboard, per-metric
  breakdown, latency-vs-quality scatter and the full results table.
- **Live Query** — ask any question against a company's 10-K with any config; shows the answer
  (Gemini if `GOOGLE_API_KEY` is set, extractive otherwise), latency and the top-5 chunks, and can
  compare the best and worst benchmark configs side by side.
- **Dataset Explorer** — corpus stats per filing and the searchable 50-question test set.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The repo ships the parsed filings, test set, results DB and compact index artifacts
(`data/index/`: gzipped chunks + float16 embeddings, ~20 MB), so the dashboard works immediately:
ChromaDB collections and BM25 indexes are hydrated from the artifacts on first use.

### Full pipeline

```bash
copy .env.example .env                   # set SEC_USER_AGENT="Your Name you@example.com"
python -m ingestion.edgar_fetcher        # download latest 10-K per ticker
python -m ingestion.parser --all         # HTML/PDF -> data/parsed/{ticker}.txt
python scripts/build_indexes.py          # chunk, embed, build Chroma + BM25 + artifacts
python -m evaluation.testset_generator   # optional: new test set (--use-llm for Gemini)
python -m benchmark.run_all              # 18 configs -> data/results.db (--dry-run, --force)
streamlit run dashboard/app.py
```

Tickers, chunk sizes, strategies, models and paths are configured in `config/settings.yml`.

## Project structure

```
ingestion/     edgar_fetcher.py, parser.py, chunker.py
retrieval/     embeddings.py, dense.py (Chroma), sparse.py (BM25), hybrid.py (RRF), artifacts.py
evaluation/    testset_generator.py, retrieval_metrics.py, ragas_runner.py, results_store.py (SQLite)
benchmark/     run_all.py — resumable 18-config sweep
dashboard/     app.py — Streamlit (Benchmark Results, Live Query, Dataset Explorer)
scripts/       build_indexes.py, seed_demo_data.py (synthetic rows for UI dev only)
```

## License

MIT
