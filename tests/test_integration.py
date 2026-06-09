"""End-to-end test against the real services. Skipped unless API keys are present.

Requires that `python -m app.ingest` has already populated the index.
"""

import os

import pytest

_HAVE_KEYS = bool(
    os.getenv("PINECONE_API_KEY")
    and os.getenv("GROQ_API_KEY")
    and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
)

pytestmark = pytest.mark.skipif(not _HAVE_KEYS, reason="live API keys not set")


def test_grounded_answer_end_to_end():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:  # lifespan builds the real clients + graph
        r = client.post("/ask", json={
            "question": "What is the notice period in the Bluecrest employment agreement?"
        })
    assert r.status_code == 200
    body = r.json()
    assert "60" in body["answer"]
    assert any(c["source"] == "02_employment_agreement_excerpt.md"
               for c in body["citations"])


def test_out_of_corpus_is_refused():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "What is the population of Riverside city?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == []
    assert "cannot find" in body["answer"].lower()
