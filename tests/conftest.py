"""Suite-wide isolation: the tests are OFFLINE and deterministic.

A developer (or CI runner) may have ANTHROPIC_API_KEY / OPENAI_API_KEY exported. Without this fixture the agents
would pick the real provider up through core.harness.llm.get_llm() and the suite would spend money, need network
and stop being reproducible. Tests that exercise the real-provider path opt back in explicitly with monkeypatch.
"""
import pytest


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
