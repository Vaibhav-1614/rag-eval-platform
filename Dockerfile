# 10-K RAG service. Default build = frozen v1 (BM25), no torch: ~250 MB image, <2 s cold start.
#   docker build -t rag-service .
#   docker build --build-arg WITH_DENSE=1 -t rag-service:hybrid .   # for v1-hybrid.yaml
FROM python:3.11-slim AS base

ARG WITH_DENSE=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    HF_HOME=/app/.hf-cache \
    FEEDBACK_DB=/tmp/rag/feedback.db \
    PORT=7860

WORKDIR /app

COPY requirements-service.txt requirements-dense.txt ./
RUN pip install -r requirements-service.txt \
 && if [ "$WITH_DENSE" = "1" ]; then pip install -r requirements-dense.txt; fi

# Embedding weights are baked at build time so the container never downloads at runtime.
RUN if [ "$WITH_DENSE" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"; \
    fi

# Only what the service imports + the prebuilt index (chunks; embeddings only for dense).
COPY platform_config.py ./
COPY config/ config/
COPY configs/ configs/
COPY prompts/ prompts/
COPY retrieval/ retrieval/
COPY evaluation/__init__.py evaluation/retrieval_metrics.py evaluation/
COPY src/ src/
COPY data/index/chunks/ data/index/chunks/
COPY data/index/embeddings/ /tmp/embeddings/
RUN if [ "$WITH_DENSE" = "1" ]; then mv /tmp/embeddings data/index/embeddings; else rm -rf /tmp/embeddings; fi

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 app && mkdir -p /tmp/rag models && chown -R app /tmp/rag /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/readyz', timeout=4)"

CMD ["sh", "-c", "exec uvicorn rag_service.api:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
