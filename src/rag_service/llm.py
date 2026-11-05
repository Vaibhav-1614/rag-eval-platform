"""
Free-tier LLM clients (plain httpx, no SDKs): Groq (OpenAI-compatible) and Gemini.

Each call returns an LLMResult or raises LLMError; the caller decides the fallback chain.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


def groq_chat(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 600,
    timeout: float = 30.0,
    json_mode: bool = False,
) -> LLMResult:
    if not api_key:
        raise LLMError("groq: no key")
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            GROQ_URL, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )
    except httpx.HTTPError as e:
        raise LLMError(f"groq transport: {type(e).__name__}") from e
    if r.status_code != 200:
        raise LLMError(f"groq http {r.status_code}: {r.text[:200]}")
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = data.get("usage") or {}
    if not text.strip():
        raise LLMError("groq: empty completion")
    return LLMResult(
        text=text.strip(),
        provider="groq",
        model=model,
        latency_ms=(time.perf_counter() - t0) * 1000,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


def gemini_generate(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 600,
    timeout: float = 45.0,
    json_mode: bool = False,
) -> LLMResult:
    if not api_key:
        raise LLMError("gemini: no key")
    gen_cfg: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    if model.startswith("gemini-2.5"):
        # Thinking tokens count against maxOutputTokens; grounded QA doesn't need them.
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            GEMINI_URL.format(model=model),
            json=body,
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise LLMError(f"gemini transport: {type(e).__name__}") from e
    if r.status_code != 200:
        raise LLMError(f"gemini http {r.status_code}: {r.text[:200]}")
    data = r.json()
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        raise LLMError("gemini: empty completion")
    usage = data.get("usageMetadata") or {}
    return LLMResult(
        text=text.strip(),
        provider="gemini",
        model=model,
        latency_ms=(time.perf_counter() - t0) * 1000,
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
    )
