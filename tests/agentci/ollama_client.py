"""Thin Ollama / OpenAI-compatible chat helpers for agentic CI tests."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_JUDGE_MODEL = "qwen2.5:3b"


def ollama_host() -> str:
    """Return OLLAMA_HOST with trailing slash stripped."""
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def judge_model() -> str:
    return os.getenv("OLLAMA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def openai_base_url() -> str:
    """OpenAI-compatible base URL (…/v1) derived from OPENAI_BASE_URL or Ollama."""
    explicit = os.getenv("OPENAI_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    return f"{ollama_host()}/v1"


def is_reachable(timeout: float = 2.0) -> bool:
    """True if Ollama answers GET /api/tags."""
    url = f"{ollama_host()}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    use_openai_compat: bool = True,
) -> str:
    """
    Send a chat completion and return the assistant message text.

    Prefer Ollama's OpenAI-compatible `/v1/chat/completions`; fall back to
    native `/api/chat` if the compat path fails.
    """
    model = model or judge_model()

    if use_openai_compat:
        try:
            result = _post_json(
                f"{openai_base_url()}/chat/completions",
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=timeout,
            )
            return result["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, TypeError):
            pass

    result = _post_json(
        f"{ollama_host()}/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    return result["message"]["content"]


def judge_json(
    rubric: str,
    subject: str,
    *,
    model: Optional[str] = None,
    timeout: float = 120.0,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Ask the judge model to score `subject` against `rubric`.

    Expects a JSON object with keys: pass (bool), score (0-1 float), reasons (list).
    Retries once or twice if the model returns unparseable JSON.
    """
    system = (
        "You are a strict factual judge for CI. "
        "Reply with ONLY a single JSON object, no markdown fences, no trailing text. "
        'Exact shape: {"pass": true, "score": 0.9, "reasons": ["short reason"]}. '
        "Use a boolean for pass, a number for score, and a JSON array of strings "
        "for reasons. Do not nest arrays."
    )
    user = f"Rubric:\n{rubric}\n\nSubject to judge:\n{subject}"
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. "
                        'Reply again with ONLY: {"pass": bool, "score": 0.0, '
                        '"reasons": ["one string"]}.'
                    ),
                }
            )
        raw = chat(messages, model=model, temperature=0.0, timeout=timeout)
        try:
            return _parse_judge_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def _repair_jsonish(text: str) -> str:
    """Best-effort cleanup of common small-model JSON mistakes."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]

    # Trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Extra closing brackets: {"reasons": ["a"]]} -> {"reasons": ["a"]}
    text = re.sub(r"\]\s*\](\s*})", r"]\1", text)
    text = re.sub(r"\}\s*\}(\s*)$", r"}\1", text)
    # Single-quoted keys/strings → double quotes (naive but helps tiny models)
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    return text


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    text = _repair_jsonish(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Second pass after another repair round on the original slice
        repaired = _repair_jsonish(raw.replace("]]", "]").replace("}}", "}"))
        parsed = json.loads(repaired)

    if not isinstance(parsed, dict):
        raise ValueError(f"judge response is not an object: {parsed!r}")
    if "pass" not in parsed or "score" not in parsed:
        raise ValueError(f"judge response missing pass/score: {parsed!r}")
    if "reasons" not in parsed:
        parsed["reasons"] = []
    # Coerce pass from stringy bools some models emit
    if isinstance(parsed["pass"], str):
        parsed["pass"] = parsed["pass"].strip().lower() in {"true", "1", "yes", "pass"}
    parsed["score"] = float(parsed["score"])
    if not isinstance(parsed["reasons"], list):
        parsed["reasons"] = [str(parsed["reasons"])]
    return parsed
