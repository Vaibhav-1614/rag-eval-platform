-- PostgreSQL equivalent of the SQLite schema in evaluation/results_store.py,
-- for running the benchmark store on a shared production database.

CREATE TABLE IF NOT EXISTS benchmark_results (
    id                 BIGSERIAL PRIMARY KEY,
    retrieval_strategy TEXT        NOT NULL CHECK (retrieval_strategy IN ('dense', 'sparse', 'hybrid')),
    chunk_size         INTEGER     NOT NULL CHECK (chunk_size > 0),
    embedding_model    TEXT        NOT NULL,
    eval_version       TEXT        NOT NULL DEFAULT 'v1',
    faithfulness       DOUBLE PRECISION,
    answer_relevancy   DOUBLE PRECISION,
    context_precision  DOUBLE PRECISION,
    context_recall     DOUBLE PRECISION,
    mrr                DOUBLE PRECISION,
    hit_rate           DOUBLE PRECISION,
    ndcg               DOUBLE PRECISION,
    avg_latency_ms     DOUBLE PRECISION,
    total_cost_usd     NUMERIC(12, 6) NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (retrieval_strategy, chunk_size, embedding_model, eval_version)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_version ON benchmark_results (eval_version, created_at DESC);

-- Resumable upsert used by the benchmark runner
-- INSERT INTO benchmark_results (retrieval_strategy, chunk_size, embedding_model, eval_version,
--     context_precision, context_recall, mrr, hit_rate, ndcg, avg_latency_ms, total_cost_usd)
-- VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
-- ON CONFLICT (retrieval_strategy, chunk_size, embedding_model, eval_version) DO UPDATE SET
--     context_precision = EXCLUDED.context_precision,
--     context_recall    = EXCLUDED.context_recall,
--     mrr               = EXCLUDED.mrr,
--     hit_rate          = EXCLUDED.hit_rate,
--     ndcg              = EXCLUDED.ndcg,
--     avg_latency_ms    = EXCLUDED.avg_latency_ms,
--     total_cost_usd    = EXCLUDED.total_cost_usd,
--     created_at        = now();

-- Per-strategy comparison with a rolling average across chunk sizes and the delta to the
-- next-larger chunk size (window functions)
-- SELECT retrieval_strategy, embedding_model, chunk_size, hit_rate,
--        AVG(hit_rate) OVER (PARTITION BY retrieval_strategy, embedding_model
--                            ORDER BY chunk_size ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS rolling_hit_rate,
--        LEAD(hit_rate) OVER (PARTITION BY retrieval_strategy, embedding_model ORDER BY chunk_size) - hit_rate
--            AS delta_to_next_chunk_size
-- FROM benchmark_results
-- WHERE eval_version = 'v2'
-- ORDER BY retrieval_strategy, embedding_model, chunk_size;
