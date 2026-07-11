import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# server.py mounts StaticFiles/Jinja2Templates using paths relative to the
# process CWD, not __file__. Anchor CWD to the repo root so the app can be
# constructed regardless of where pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import config
import knowledge_base
import rag_pipeline
import server

FIXTURE_CAMPUS_DATA = Path(__file__).resolve().parent / "fixtures" / "campus_data.json"


@pytest.fixture
def mock_call_llm(monkeypatch):
    """Stubs the only function in the app that makes real network calls to an LLM."""
    mock = MagicMock(return_value="MOCKED_ANSWER")
    monkeypatch.setattr(rag_pipeline, "_call_llm", mock)
    return mock


@pytest.fixture
def client(tmp_path, monkeypatch, mock_call_llm):
    """A TestClient wired to an isolated, temporary SQLite DB and Chroma collection."""
    monkeypatch.setattr(config, "SQLITE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(config, "CAMPUS_DATA_PATH", str(FIXTURE_CAMPUS_DATA))

    # knowledge_base.load_campus_data's `path` default is bound to
    # config.CAMPUS_DATA_PATH at import time, not at call time, so the
    # monkeypatch above alone does NOT redirect get_or_create_collection()
    # (which calls load_campus_data() with no args) to the fixture file.
    # Patch the function itself so ChromaDB seeding is isolated too.
    def _load_fixture_campus_data(path=None):
        with open(FIXTURE_CAMPUS_DATA) as f:
            return json.load(f)

    monkeypatch.setattr(knowledge_base, "load_campus_data", _load_fixture_campus_data)

    with TestClient(server.app) as c:
        yield c
