from app.llm import REFUSAL, enforce_grounding, grade_relevance

CHUNKS = [
    {"chunk_id": "a#1", "source": "a.md", "text": "alpha", "score": 0.9},
    {"chunk_id": "b#1", "source": "b.md", "text": "beta", "score": 0.8},
]


def test_enforce_grounding_keeps_valid_citation():
    answer, cited = enforce_grounding("an answer", [{"chunk_id": "a#1"}], CHUNKS)
    assert answer == "an answer"
    assert [c["chunk_id"] for c in cited] == ["a#1"]


def test_enforce_grounding_drops_fabricated_citation():
    answer, cited = enforce_grounding("made up", [{"chunk_id": "ghost#9"}], CHUNKS)
    assert answer == REFUSAL and cited == []


def test_enforce_grounding_deduplicates():
    _, cited = enforce_grounding("x", [{"chunk_id": "a#1"}, {"chunk_id": "a#1"}], CHUNKS)
    assert len(cited) == 1


def test_enforce_grounding_passes_refusal_through():
    assert enforce_grounding(REFUSAL, [], CHUNKS) == (REFUSAL, [])


def test_grade_relevance_keeps_only_chosen_chunks(fake_groq, settings_stub):
    kept = grade_relevance(fake_groq('{"relevant": [2]}'), settings_stub, "q?", CHUNKS)
    assert [c["chunk_id"] for c in kept] == ["b#1"]


def test_grade_relevance_handles_no_chunks(fake_groq, settings_stub):
    assert grade_relevance(fake_groq('{"relevant": []}'), settings_stub, "q?", []) == []
