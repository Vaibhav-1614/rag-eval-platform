"""LLM-as-judge (rubric prompts in prompts/judge/). Gemini first, Groq as fallback."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag_service.config import ServiceSettings
from rag_service.llm import LLMError, gemini_generate, groq_chat
from rag_service.prompts import load_prompt

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class Verdict:
    score: int | None  # 1..3, None if no judge available / unparsable
    reason: str
    judge_model: str | None


def _call(prompt: str, settings: ServiceSettings) -> tuple[str, str]:
    errors = []
    if settings.gemini_key:
        try:
            r = gemini_generate(prompt, api_key=settings.gemini_key, model=settings.judge_model, json_mode=True, max_tokens=300)
            return r.text, r.model
        except LLMError as e:
            errors.append(str(e))
    if settings.groq_key:
        try:
            r = groq_chat(prompt, api_key=settings.groq_key, model=settings.groq_model, json_mode=True, max_tokens=300)
            return r.text, r.model
        except LLMError as e:
            errors.append(str(e))
    raise LLMError("; ".join(errors) or "no judge key configured")


def parse_verdict(text: str) -> tuple[int | None, str]:
    m = _JSON.search(text)
    if not m:
        return None, "unparsable"
    try:
        obj = json.loads(m.group(0))
        score = int(obj.get("score"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None, "unparsable"
    if score not in (1, 2, 3):
        return None, f"out of range: {score}"
    return score, str(obj.get("reason", ""))[:300]


def judge(kind: str, settings: ServiceSettings, **inputs: str) -> Verdict:
    """kind: 'faithfulness' (question, context, answer) or 'correctness' (question, reference, answer)."""
    prompt = load_prompt(f"judge/{kind}_v1").render(**inputs)
    try:
        text, model = _call(prompt, settings)
    except LLMError as e:
        return Verdict(None, str(e), None)
    score, reason = parse_verdict(text)
    return Verdict(score, reason, model)
