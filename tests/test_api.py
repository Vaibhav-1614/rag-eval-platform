import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    import time

    os.environ["FEEDBACK_DB"] = str(tmp_path_factory.mktemp("db") / "fb.db")
    os.environ["RATE_LIMIT"] = "20/minute"
    from rag_service.config import get_settings

    get_settings.cache_clear()
    import importlib

    import rag_service.api as api

    importlib.reload(api)
    with TestClient(api.app) as c:
        for _ in range(100):
            if c.get("/readyz").status_code == 200:
                break
            time.sleep(0.1)
        yield c


def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "ok"
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["retrieval_version"] == "v1"


def test_config(client):
    c = client.get("/v1/config").json()
    assert c["retrieval"]["strategy"] == "sparse"
    assert c["providers"]["groq"]["enabled"] is False
    assert c["tracing"] == "json"


def test_query_extractive_without_keys(client):
    r = client.post("/v1/query", json={"question": "How many employees did Tesla have worldwide at the end of 2025?"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["mode"] == "extractive" and "134,785" in b["answer"]
    assert b["scope"] == ["TSLA"] and len(b["passages"]) == 5
    assert b["citations"] and b["cache_hit"] is False
    again = client.post("/v1/query", json={"question": "How many employees did Tesla have worldwide at the end of 2025?"})
    assert again.json()["cache_hit"] is True


def test_query_refusal(client):
    b = client.post("/v1/query", json={"question": "What is Apple's stock price today?"}).json()
    assert b["mode"] == "refusal" and "can't answer" in b["answer"]


def test_query_validation(client):
    assert client.post("/v1/query", json={"question": "hi"}).status_code == 422
    assert client.post("/v1/query", json={"question": "Revenue?", "tickers": ["ZZZZ"]}).status_code == 422
    assert client.post("/v1/query", json={"question": "Revenue?", "top_k": 50}).status_code == 422


def test_feedback(client):
    rid = client.post("/v1/query", json={"question": "Where is Tesla headquartered?"}).json()["request_id"]
    r = client.post("/v1/feedback", json={"request_id": rid, "rating": 1, "comment": "good"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.post("/v1/feedback", json={"request_id": rid, "rating": 5}).status_code == 422


def test_rate_limit(client):
    codes = [client.post("/v1/query", json={"question": f"Tesla revenue question {i}?"}).status_code for i in range(25)]
    assert 429 in codes
