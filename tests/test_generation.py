from rag_service import generation
from rag_service.config import ServiceSettings
from rag_service.llm import LLMError, LLMResult
from rag_service.retriever import Passage

PASSAGES = [
    Passage("TSLA_256_66", "TSLA", "As of December 31, 2025, our employee headcount worldwide was 134,785. Other text here.", 9.0, 1),
    Passage("TSLA_256_208", "TSLA", "We are headquartered in Austin, Texas, where our main campus is located.", 5.0, 2),
]
REFUSAL = "I can't answer that from the indexed filings. Corpus: X."


def _settings(monkeypatch, groq="", gemini=""):
    monkeypatch.setenv("GROQ_API_KEY", groq)
    monkeypatch.setenv("GEMINI_API_KEY", gemini)
    return ServiceSettings.from_env()


def test_extractive_cites_passages():
    text, cites = generation.extractive_answer("How many employees did Tesla have worldwide?", PASSAGES)
    assert "134,785" in text and "[1]" in text
    assert 1 in cites


def test_no_keys_uses_extractive(cfg, monkeypatch):
    g = generation.generate("Tesla employee headcount?", PASSAGES, cfg, _settings(monkeypatch), REFUSAL)
    assert g.mode == "extractive" and g.provider is None


def test_chain_falls_back_groq_to_gemini(cfg, monkeypatch):
    calls = []

    def bad_groq(prompt, **kw):
        calls.append(("groq", kw["model"]))
        raise LLMError("groq http 429")

    def ok_gemini(prompt, **kw):
        calls.append(("gemini", kw["model"]))
        return LLMResult("Tesla had 134,785 employees [1].", "gemini", kw["model"], 12.0, 100, 10)

    monkeypatch.setattr(generation, "groq_chat", bad_groq)
    monkeypatch.setattr(generation, "gemini_generate", ok_gemini)
    g = generation.generate("Tesla headcount?", PASSAGES, cfg, _settings(monkeypatch, "g", "m"), REFUSAL)
    assert [c[0] for c in calls] == ["groq", "groq", "gemini"]
    assert g.mode == "llm" and g.provider == "gemini" and g.citations == [1]
    assert len(g.errors) == 2


def test_llm_refusal_is_normalised(cfg, monkeypatch):
    monkeypatch.setattr(
        generation, "groq_chat", lambda p, **kw: LLMResult("I can't answer that from the indexed filings.", "groq", kw["model"], 1.0)
    )
    g = generation.generate("q?", PASSAGES, cfg, _settings(monkeypatch, "g"), REFUSAL)
    assert g.mode == "refusal" and g.answer == REFUSAL and g.citations == []


def test_all_providers_fail_falls_back_to_extractive(cfg, monkeypatch):
    def boom(prompt, **kw):
        raise LLMError("down")

    monkeypatch.setattr(generation, "groq_chat", boom)
    monkeypatch.setattr(generation, "gemini_generate", boom)
    g = generation.generate("Tesla employee headcount?", PASSAGES, cfg, _settings(monkeypatch, "g", "m"), REFUSAL)
    assert g.mode == "extractive" and len(g.errors) == 3
