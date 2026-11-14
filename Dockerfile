# 10-K RAG service (production config configs/retrieval/v2.yaml). CPU only, no torch:
# the embedding model and the re-ranker are ONNX models run by fastembed.
#   docker build -t rag-service .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    FASTEMBED_CACHE_PATH=/app/.model-cache \
    HF_HUB_DISABLE_XET=1 \
    FEEDBACK_DB=/tmp/rag/feedback.db \
    PORT=7860

WORKDIR /app

COPY requirements-service.txt ./
RUN pip install -r requirements-service.txt

# Model weights are baked at build time, so the container never downloads at runtime
# (cold starts stay fast and the service works without outbound access to Hugging Face).
COPY configs/ configs/
COPY scripts/fetch_models.py scripts/fetch_models.py
RUN python scripts/fetch_models.py configs/retrieval/v2.yaml

# Only what the service imports + the prebuilt indexes and facts table.
COPY platform_config.py ./
COPY config/ config/
COPY prompts/ prompts/
COPY retrieval/ retrieval/
COPY evaluation/__init__.py evaluation/retrieval_metrics.py evaluation/
COPY src/ src/
COPY data/index/chunks/ data/index/chunks/
COPY data/index/v2/ data/index/v2/
COPY data/facts/ data/facts/

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 app && mkdir -p /tmp/rag models && chown -R app /tmp/rag /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/readyz', timeout=4)"

# No --proxy-headers: the app reads X-Forwarded-For itself and only trusts the last
# TRUSTED_PROXY_HOPS entries (see rag_service.api.client_ip).
CMD ["sh", "-c", "exec uvicorn rag_service.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
