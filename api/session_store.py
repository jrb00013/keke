"""Persistent on-disk session storage for uploaded spreadsheets."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

SESSION_ROOT = Path(
    os.getenv(
        "KEKE_SESSION_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "sessions"),
    )
)

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{8,128}$")


def _validate_session_id(session_id: str) -> str:
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError("Invalid session ID")
    return session_id


def _safe_sheet_filename(sheet_name: str) -> str:
    safe = re.sub(r"[^\w\-.]", "_", sheet_name).strip("._")
    return safe or "sheet"


def get_session_dir(session_id: str) -> Path:
    return SESSION_ROOT / _validate_session_id(session_id)


def session_exists(session_id: str) -> bool:
    try:
        return (get_session_dir(session_id) / "metadata.json").is_file()
    except ValueError:
        return False


def save_session(
    session_id: str,
    dataframes: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
) -> None:
    session_dir = get_session_dir(session_id)
    sheets_dir = session_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    sheet_files = {}
    for name, df in dataframes.items():
        fname = f"{_safe_sheet_filename(name)}.parquet"
        df.to_parquet(sheets_dir / fname, index=False)
        sheet_files[name] = fname

    metadata = dict(metadata)
    metadata["session_id"] = session_id
    metadata["sheet_files"] = sheet_files

    with open(session_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, default=str)


def load_session(session_id: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    if not session_exists(session_id):
        raise FileNotFoundError(f"Session not found: {session_id}")

    session_dir = get_session_dir(session_id)
    with open(session_dir / "metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)

    dataframes: Dict[str, pd.DataFrame] = {}
    sheets_dir = session_dir / "sheets"
    sheet_files = metadata.get("sheet_files", {})

    for name, fname in sheet_files.items():
        path = sheets_dir / fname
        if path.is_file():
            dataframes[name] = pd.read_parquet(path)

    if not dataframes and sheets_dir.is_dir():
        for path in sorted(sheets_dir.glob("*.parquet")):
            name = path.stem.replace("_", " ")
            dataframes[name] = pd.read_parquet(path)

    return dataframes, metadata


def delete_session(session_id: str) -> None:
    session_dir = get_session_dir(session_id)
    if session_dir.is_dir():
        shutil.rmtree(session_dir, ignore_errors=True)
