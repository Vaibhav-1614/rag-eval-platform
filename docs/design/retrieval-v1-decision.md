# Decision: frozen retrieval config `v1` = BM25 @ 256 tokens

**Status:** accepted · **Config:** [`configs/retrieval/v1.yaml`](../../configs/retrieval/v1.yaml) · **Alternative kept:** [`v1-hybrid.yaml`](../../configs/retrieval/v1-hybrid.yaml)

## Context

The design started out preferring hybrid retrieval (dense + BM25 merged with RRF), because hybrid usually holds up better on paraphrased questions. Before freezing a config we ran the week-1 benchmark: 18 configs (3 strategies × 3 chunk sizes × 2 embedding models) on 50 natural-language questions. Results are in `data/results.db`, `eval_version = 'v2'`.

| config | Hit@5 | MRR | nDCG@5 | avg latency |
|---|---|---|---|---|
| **sparse 256** | **0.98** | **0.846** | **0.870** | **6.4 ms** |
| sparse 512 | 0.98 | 0.785 | 0.833 | 4.7 ms |
| hybrid 512 bge | 0.88 | 0.768 | 0.779 | 24.5 ms |
| hybrid 256 bge | 0.90 | 0.751 | 0.781 | 29.9 ms |
| dense 256 bge | 0.66 | 0.548 | 0.572 | 14.3 ms |

(Sparse rows are identical for both embedding models, because BM25 doesn't use embeddings.)

## Decision

We freeze **sparse BM25, 256-token chunks, top-5** as `v1`, which goes against the design's original preference for hybrid:

1. **Quality.** BM25 beats the best hybrid config by +8 pp Hit@5 and +0.095 MRR. 10-K questions are full of exact terms (product names, "Physiomesh", "ASU 2023-09", figures). BM25 matches those directly, while small general-purpose embedders blur them.
2. **Cost and operations.** BM25 needs no model in memory and no torch. The image is about 100 MB compressed instead of about 1 GB, it starts cold in under 2 s, and it fits any free host.
3. **Latency.** It runs about 5× faster than hybrid.

`tests/test_config.py::test_v1_is_best_benchmark_config` checks that `v1` is still the top-MRR row of the benchmark DB.

## Known weakness, and what would flip the decision

The 12 **held-out** golden questions were written after the freeze. They deliberately paraphrase ("workforce", "people work at", "headcount") instead of using the filing's words. On them BM25 gets **Hit@5 = 0.67 (8/12)**, against 0.96 on the benchmark questions: the classic vocabulary-mismatch failure that dense retrieval exists to fix.

The next config candidate is therefore `v1-hybrid` (or BM25 with query expansion). Promote it only if `python eval/run_eval.py --config configs/retrieval/v1-hybrid.yaml` (needs `requirements-dense.txt`) meets both conditions:

- raises held-out Hit@5;
- costs at most 0.02 overall Hit@5/MRR on the golden set.

Ship it as `v2.yaml` with an updated baseline, and never edit `v1.yaml` in place.

## Not decided here

- **Re-ranking.** A cross-encoder over BM25 top-20 is the cheapest likely win. It is CPU-only, but at about 80 MB it would bring torch back.
- **Chunk size per question type.** Numeric facts tend to like small chunks and explanations bigger ones. It is untested.
