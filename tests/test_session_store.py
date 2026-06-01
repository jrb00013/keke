import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from session_store import (  # noqa: E402
    load_session,
    save_session,
    session_exists,
)


@pytest.fixture
def session_root(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv('KEKE_SESSION_DIR', tmp)
        import session_store as store

        monkeypatch.setattr(store, 'SESSION_ROOT', __import__('pathlib').Path(tmp))
        yield tmp


def test_save_and_load_session(session_root):
    frames = {
        'Sheet1': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
    }
    metadata = {'original_name': 'test.xlsx', 'total_sheets': 1}

    save_session('test-session-123', frames, metadata)
    assert session_exists('test-session-123')

    loaded_frames, loaded_meta = load_session('test-session-123')
    assert list(loaded_frames['Sheet1']['A']) == [1, 2]
    assert loaded_meta['original_name'] == 'test.xlsx'

    meta_path = os.path.join(session_root, 'test-session-123', 'metadata.json')
    with open(meta_path, encoding='utf-8') as handle:
        on_disk = json.load(handle)
    assert 'sheet_files' in on_disk


def test_invalid_session_id_rejected(session_root):
    with pytest.raises(ValueError):
        save_session('../bad', {}, {})
