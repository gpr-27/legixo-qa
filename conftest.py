"""Shared pytest fixtures. Living at the repo root so `import app` resolves."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent


class FakeGroq:
    """Minimal stand-in for the Groq client that returns canned message content."""

    def __init__(self, content: str):
        self._content = content

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def corpus_dir() -> Path:
    return ROOT / "corpus"


@pytest.fixture
def fake_groq():
    return FakeGroq


@pytest.fixture
def settings_stub():
    return SimpleNamespace(answer_model="test-model")


@pytest.fixture
def make_client():
    """Return a factory that builds a TestClient whose graph returns a fixed result.

    Used without lifespan, so no real clients/keys are needed.
    """
    from app.main import app, get_graph

    class FakeGraph:
        def __init__(self, result):
            self.result = result

        def invoke(self, state, config=None):
            return self.result

    def _make(result):
        app.dependency_overrides[get_graph] = lambda: FakeGraph(result)
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()
