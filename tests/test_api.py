import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    import time

    os.environ["FEEDBACK_DB"] = str(tmp_path_factory.mktemp("db") / "fb.db")
    os.environ["RATE_LIMIT"] = "20/minute"
    # BM25-only v2 config: same pipeline as production without downloading models.
    os.environ["RETRIEVAL_CONFIG"] = "configs/retrieval/v2-bm25.yaml"
    os.environ.pop("API_KEYS", None)
    os.environ.pop("DATABASE_URL", None)
    from rag_service.config import get_settings

    get_settings.cache_clear()
    from rag_service.config import get_retrieval_config

    get_retrieval_config.cache_clear()
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
    assert r.status_code == 200 and r.json()["retrieval_version"] == "v2-bm25"


def test_config(client):
    c = client.get("/v1/config").json()
    assert c["retrieval"]["strategy"] == "sparse" and c["retrieval"]["index"] == "v2"
    assert c["retrieval"]["facts_table"] is True and c["storage"] == "sqlite"
    assert c["limits"]["api_key_required"] is False
    assert c["providers"]["groq"]["enabled"] is False
    assert c["tracing"] == "json"


def test_query_extractive_without_keys(client):
    r = client.post("/v1/query", json={"question": "How many employees did Tesla have worldwide at the end of 2025?"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["mode"] == "extractive" and "134,785" in b["answer"]
    assert b["scope"] == ["TSLA"] and len(b["passages"]) == 6  # 1 fact + 5 text
    first = b["passages"][0]
    assert first["kind"] == "fact" and first["source_url"].startswith("https://www.sec.gov/")
    assert all(p["section"] for p in b["passages"])
    assert b["citations"] == [1] and b["cache_hit"] is False
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



def test_stream_sends_passages_then_answer(client):
    with client.stream("POST", "/v1/query/stream", json={"question": "What business segments does Tesla report?"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event: ")]
    assert events == ["passages", "answer", "done"]
    import json

    payloads = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert payloads[0]["passages"] and payloads[1]["request_id"] == payloads[0]["request_id"]
    assert payloads[1]["mode"] in ("extractive", "refusal")


class _Req:
    def __init__(self, xff=None, peer="10.0.0.9", key=None):
        self.headers = {k: v for k, v in (("x-forwarded-for", xff), ("x-api-key", key)) if v}
        self.client = type("C", (), {"host": peer})()


def test_client_ip_ignores_spoofed_forwarded_for(client, monkeypatch):
    import dataclasses

    import rag_service.api as api

    # No trusted proxy: the header is attacker-controlled and must be ignored.
    assert api.client_ip(_Req(xff="1.2.3.4")) == "10.0.0.9"
    # One trusted proxy: it appends the real peer last; anything before it is spoofable.
    monkeypatch.setattr(api, "settings", dataclasses.replace(api.settings, trusted_proxy_hops=1))
    assert api.client_ip(_Req(xff="6.6.6.6, 203.0.113.7")) == "203.0.113.7"
    assert api.client_ip(_Req(xff=None)) == "10.0.0.9"


def test_api_key_required_when_configured(client, monkeypatch):
    import dataclasses

    import rag_service.api as api

    monkeypatch.setattr(api, "settings", dataclasses.replace(api.settings, api_keys=("secret-1",)))
    q = {"question": "Where is Tesla headquartered?"}
    assert client.post("/v1/query", json=q).status_code == 401
    assert client.post("/v1/query", json=q, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/v1/query", json=q, headers={"X-API-Key": "secret-1"}).status_code == 200
    assert client.get("/healthz").status_code == 200  # health stays open
    assert api.rate_key(_Req(key="secret-1")).startswith("key:")


def test_rate_limit(client):
    codes = [client.post("/v1/query", json={"question": f"Tesla revenue question {i}?"}).status_code for i in range(25)]
    assert 429 in codes
