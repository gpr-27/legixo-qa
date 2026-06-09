def test_ask_returns_answer_and_citations(make_client):
    result = {
        "answer": "Either party may give 60 days written notice.",
        "citations": [{
            "source": "02_employment_agreement_excerpt.md",
            "chunk_id": "02_employment_agreement_excerpt#notice-period",
            "score": 0.71, "snippet": "Either party may end this agreement...",
        }],
        "steps": ["retrieve", "grade", "generate"],
    }
    client = make_client(result)
    r = client.post("/ask", json={"question": "What is the notice period?"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Either party")
    assert body["citations"][0]["source"] == "02_employment_agreement_excerpt.md"
    assert body["trace"] is None          # omitted unless ?trace=true


def test_ask_includes_trace_when_requested(make_client):
    result = {"answer": "x", "citations": [], "steps": ["retrieve", "grade", "give_up"]}
    client = make_client(result)
    r = client.post("/ask?trace=true", json={"question": "anything"})
    assert r.json()["trace"] == ["retrieve", "grade", "give_up"]


def test_blank_question_is_rejected(make_client):
    client = make_client({"answer": "", "citations": [], "steps": []})
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_overlong_question_is_rejected(make_client):
    client = make_client({"answer": "", "citations": [], "steps": []})
    assert client.post("/ask", json={"question": "x" * 5000}).status_code == 422


def test_health(make_client):
    client = make_client({"answer": "", "citations": [], "steps": []})
    assert client.get("/health").json() == {"status": "ok"}


def test_home_serves_ui(make_client):
    client = make_client({"answer": "", "citations": [], "steps": []})
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Legixo" in r.text
