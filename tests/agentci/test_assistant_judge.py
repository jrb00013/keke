"""Agentic CI: exercise Assistant against Ollama and judge answer quality."""

from __future__ import annotations

import os

import pytest

from ollama_client import judge_json, judge_model, openai_base_url

pytestmark = pytest.mark.agentci

openai = pytest.importorskip("openai", reason="openai package not installed")


@pytest.fixture
def ollama_env(monkeypatch):
    """Point the Assistant OpenAI client at local Ollama."""
    monkeypatch.setenv("OPENAI_BASE_URL", openai_base_url())
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("OPENAI_MODEL", judge_model())
    monkeypatch.setenv("OLLAMA_JUDGE_MODEL", judge_model())
    # Ensure a stale real key does not override the local default path.
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    yield


def test_assistant_summary_quality(sample_sheet_df, ollama_env):
    from assistant import Assistant

    assistant = Assistant()
    assert assistant.openai_available is True
    assert assistant.openai_client is not None

    result = assistant.process_natural_language_query(
        "Summarize this dataset for me",
        sample_sheet_df,
    )
    assert "error" not in result or not result.get("error")
    response = result.get("response") or ""
    assert response.strip(), f"empty assistant response: {result!r}"

    rubric = (
        "The subject is an assistant answer about a 3-row employee sheet with "
        "columns Name, Age, Department, Salary. "
        "PASS if the answer is broadly correct about shape/columns/types and "
        "does not invent unrelated columns (e.g. Latitude, SKU, ticker symbols). "
        "Minor wording differences are fine. "
        "score >= 0.6 should accompany pass=true."
    )
    verdict = judge_json(rubric, response)
    assert verdict["pass"] is True, {
        "verdict": verdict,
        "assistant_response": response,
        "insights": result.get("data_insights"),
    }
    assert float(verdict["score"]) >= 0.6, verdict
