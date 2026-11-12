"""Session fixtures for agentic CI tests against local Ollama."""

from __future__ import annotations

import os
import sys

import pytest

_AGENTCI_DIR = os.path.dirname(__file__)
_API_DIR = os.path.join(_AGENTCI_DIR, "..", "..", "api")
sys.path.insert(0, os.path.abspath(_API_DIR))
sys.path.insert(0, _AGENTCI_DIR)

from ollama_client import judge_model, ollama_host, is_reachable  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def require_ollama():
    """Skip the entire agentci session if Ollama is unreachable."""
    host = ollama_host()
    if not is_reachable():
        pytest.skip(
            f"Ollama unreachable at {host} "
            f"(set OLLAMA_HOST / start ollama to run agentci)"
        )
    return {"host": host, "model": judge_model()}


@pytest.fixture
def sample_sheet_df():
    """Deterministic mini sheet used by assistant / judge tests."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35],
            "Department": ["IT", "HR", "IT"],
            "Salary": [50000, 60000, 70000],
        }
    )
