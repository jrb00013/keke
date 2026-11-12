"""Agentic CI: use Ollama to judge a fixed excel-analysis narrative."""

from __future__ import annotations

import pytest

from ollama_client import judge_json

pytestmark = pytest.mark.agentci

# Deterministic "API-ish" fact about known columns — the narrative must stay
# consistent with this ground truth for the judge to pass.
KNOWN_COLUMNS = ["Name", "Age", "Department", "Salary"]
FACTUAL_SUMMARY = (
    "The uploaded sheet has 4 columns: Name, Age, Department, and Salary. "
    "There are 3 employee rows. Age and Salary are numeric; Name and "
    "Department are categorical. The IT department appears twice."
)
INCONSISTENT_SUMMARY = (
    "The uploaded sheet has 12 columns including Revenue, SKU, and Latitude. "
    "There are 5000 rows of stock tickers with no employee fields."
)

RUBRIC = (
    f"The subject is a summary of a spreadsheet whose ONLY columns are "
    f"{KNOWN_COLUMNS} and which has exactly 3 rows of employee data. "
    f"PASS only if the summary is factually consistent with that sheet "
    f"(mentions those columns / small row count / employee-like data). "
    f"FAIL if it invents unrelated columns, huge row counts, or wrong domain. "
    f"score >= 0.7 should accompany pass=true."
)


def test_judge_factual_summary_passes():
    result = judge_json(RUBRIC, FACTUAL_SUMMARY)
    assert result["pass"] is True, result
    assert float(result["score"]) >= 0.7, result
    assert isinstance(result["reasons"], list)


def test_judge_inconsistent_summary_fails():
    result = judge_json(RUBRIC, INCONSISTENT_SUMMARY)
    assert result["pass"] is False, result
    assert float(result["score"]) < 0.7, result
