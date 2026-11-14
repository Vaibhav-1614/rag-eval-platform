# Decision: production retrieval config `v2`

**Status:** accepted · **Config:** [`configs/retrieval/v2.yaml`](../../configs/retrieval/v2.yaml) · **Supersedes:** [v1](retrieval-v1-decision.md)

## Why v1 was not enough

v1 (BM25 over fixed 256-token windows) scored well on single-fact lookups but failed on ordinary questions:

| Failure | Example | Cause |
|---|---|---|
| Vocabulary mismatch | "R&D spending" never matched "research and development" | No stemming, no synonyms |
| Thematic questions | "main risks from competition" returned boilerplate | Chunks carried no section information |
| Headline numbers | "How much cash does Amazon have?" returned acquisition cash | Keyword match with no notion of the balance sheet |
| Table fragments | A row of numbers without its header or units | Fixed windows cut tables apart |
| Cross-company questions | "Which company has the most employees?" returned junk | One index searched once |

## What v2 changes

1. **Section- and table-aware chunks.** Every chunk knows its 10-K item and nearest heading, and that context line is indexed with the text. Tables stay whole with their caption and header rows.
2. **Facts table.** Company-wide figures come from the SEC XBRL data of the same filing (`data/facts/facts.json`): 14–22 metrics per company, including headcount taken from a verified sentence in the filing. Comparisons and cross-company rankings are answered from it.
3. **Lexical search that understands wording.** Stemmed BM25 plus query-expansion rules (`configs/query_expansion.yaml`).
4. **Cross-encoder re-ranker** over the BM25 top 20 (`Xenova/ms-marco-MiniLM-L-6-v2`, ONNX, CPU).
5. **Evidence-based refusal.** The re-ranker's score for the best passage is thresholded (`min_evidence_score: 0.75`, chosen on the dev split).

## Candidates compared

All four ran on the **dev split** (125 questions). Latency is per query on a 4-core CPU.

| Config | Hit@5 | MRR | Refusal accuracy | p50 latency |
|---|---|---|---|---|
| `v2-bm25` (no re-ranker) | 0.971 | 0.889 | 0.992 | 1 ms |
| **`v2` (BM25 + re-ranker)** | **0.981** | **0.939** | **1.000** | **470 ms** |
| hybrid (BM25 + bge-small, RRF), no re-ranker | 0.981 | 0.920 | 0.976 | 90 ms |
| `v2-hybrid-rerank` | 0.990 | 0.938 | 1.000 | 800 ms |

**Chosen: BM25 + re-ranker.**

- The re-ranker is what lifts ranking quality (thematic MRR 0.73 → 1.0) and what makes evidence-based refusal possible.
- Adding dense embeddings on top found one more dev question (1 of 105) and cost about 330 ms more per query. That is not worth it on this corpus, so hybrid stays as a switchable alternative (`v2-hybrid-rerank.yaml`), and the embeddings are committed.
- `v2-bm25.yaml` is the fast option (about 1 ms) when latency matters more than ranking quality. It has no evidence-based refusal.

## Result on the full golden set

176 questions; v1 was run on the same set for comparison.

| Metric | v1 | v2 |
|---|---|---|
| Hit@5 | 0.730 | 0.986 |
| MRR | 0.580 | 0.945 |
| Hit@5, held-out test split | 0.628 | 1.000 |
| Answer check (known figure or phrase present), no LLM | 0.206 | 0.971 |
| Unanswerable questions refused | 0.786 | 1.000 |
| Answerable questions wrongly refused | 0.014 | 0.000 |

## Caveats

- **The test split was not used for tuning, with one exception.** The first test run had one false refusal (a dividend-policy question tripped the old word-overlap rule although the re-ranker scored the passage 5.0). I then let the evidence score replace that rule whenever a re-ranker is present. The numbers above include that fix; before it, test answer accuracy was 0.971 and one of 148 answerable questions was wrongly refused.
- **I wrote the new questions**, knowing how the facts lookup detects metrics. The test split measures generalisation across phrasings I chose, not across real users. Real traffic and the `/v1/feedback` data should feed new questions.
- **The answer check is run without an LLM.** LLM answer quality (faithfulness, correctness) is still unmeasured: it needs API keys.
- **The evidence threshold was set on 20 dev unanswerable questions.** Expect it to need adjusting.
- **Latency.** p50 is about 0.5 s and p95 about 1.2 s per query on 4 CPU cores, nearly all of it the re-ranker. Two-company comparisons are the slow case.
- **Parser gaps remain.** Microsoft's tables lost their year header row and closing brackets on negatives in `data/parsed/`. The facts table covers its headline figures, but segment tables are harder to read. Fixing this means re-parsing the raw filing.
- **The corpus is ageing.** Apple filed a newer 10-K in November 2026, after its filing here was indexed.
