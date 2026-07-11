import config
import rag_pipeline


def test_call_llm_missing_anthropic_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)

    result = rag_pipeline._call_llm("does this exist?")

    assert result.startswith("ERROR: ANTHROPIC_API_KEY is not set")


def test_call_llm_missing_gemini_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)

    result = rag_pipeline._call_llm("does this exist?")

    assert result.startswith("ERROR: GEMINI_API_KEY is not set")


def test_call_llm_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "bogus-provider")

    result = rag_pipeline._call_llm("does this exist?")

    assert "ERROR: Unknown LLM_PROVIDER 'bogus-provider'" in result
