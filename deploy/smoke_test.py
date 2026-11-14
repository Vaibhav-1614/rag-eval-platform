"""Post-deploy smoke test: wait for /readyz, then check one answer and one refusal."""
from __future__ import annotations

import sys
import time

import httpx

base = sys.argv[1].rstrip("/")
deadline = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 900)
while True:
    try:
        r = httpx.get(f"{base}/readyz", timeout=10)
        if r.status_code == 200:
            break
    except httpx.HTTPError:
        pass
    if time.time() > deadline:
        sys.exit(f"{base} not ready in time")
    time.sleep(15)

ok = httpx.post(f"{base}/v1/query", json={"question": "How many employees did Tesla have worldwide at the end of 2025?"}, timeout=60).json()
assert ok["mode"] in ("llm", "extractive"), ok
assert "134,785" in ok["answer"], ok["answer"]
no = httpx.post(f"{base}/v1/query", json={"question": "What is Apple's stock price today?"}, timeout=60).json()
assert no["mode"] == "refusal", no
cfg = httpx.get(f"{base}/v1/config", timeout=10).json()
theme = httpx.post(f"{base}/v1/query", json={"question": "What are the main risks Tesla faces from competition?"}, timeout=90).json()
assert theme["mode"] in ("llm", "extractive"), theme
assert any("Risk Factors" in (p.get("section") or "") for p in theme["passages"]), [p.get("section") for p in theme["passages"]]
print(f"smoke ok: mode={ok['mode']} provider={ok['provider']} retrieval={cfg['retrieval']['version']} "
      f"storage={cfg['storage']} tracing={cfg['tracing']}")
