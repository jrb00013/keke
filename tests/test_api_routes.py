"""Session-backed Excel processing integration tests (Python layer)."""

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from excel_processor import ExcelProcessor  # noqa: E402
from session_store import load_session, session_exists  # noqa: E402


@pytest.fixture
def session_root(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv('KEKE_SESSION_DIR', tmp)
        import session_store as store

        monkeypatch.setattr(store, 'SESSION_ROOT', __import__('pathlib').Path(tmp))
        yield tmp


@pytest.fixture
def sample_xlsx(session_root):
    data = pd.DataFrame({
        'Name': ['Alice', 'Bob'],
        'Score': [10, 20],
    })
    path = os.path.join(session_root, 'sample.xlsx')
    data.to_excel(path, index=False)
    return path


def test_upload_analyze_flow(sample_xlsx, session_root):
    session_id = 'flow-session-001'
    processor = ExcelProcessor()
    result = processor.load_file_into_session(sample_xlsx, session_id, 'sample.xlsx')

    assert result['session_id'] == session_id
    assert session_exists(session_id)

    reloaded = ExcelProcessor.from_session(session_id)
    analysis = reloaded.analyze_data('Sheet1')
    assert analysis['basic_info']['rows'] == 2
    assert 'data_quality' in analysis

    frames, meta = load_session(session_id)
    assert meta['original_name'] == 'sample.xlsx'
    assert 'Sheet1' in frames
