---
title: 10-K RAG API
emoji: 📑
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers over the latest 10-K filings of 10 US companies
---

FastAPI service from https://github.com/Vaibhav-1614/rag-eval-platform (deployed by GitHub Actions).

- `POST /v1/query` — `{"question": "How many employees did Tesla have at the end of 2025?"}`
- `POST /v1/feedback`, `GET /v1/config`, `GET /healthz`, `GET /readyz`
- Interactive docs: `/docs`
