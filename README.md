# Legixo Q&A API

A small HTTP API that answers questions **only** from a set of legal-style documents.
Every answer names the exact chunk it came from, and if the documents don't contain the
answer it says so instead of guessing.

Built with **Python · LangGraph · Pinecone · FastAPI**, on a fully free stack
(**Groq** for the answer LLM, **Gemini** for embeddings, **Pinecone** Starter for the
vector index — none require a credit card).

> The corpus in `corpus/` is fiction (made-up parties and courts) from the take-home brief.

**Demo video (5–10 min):** _add link here_ — install → ingest → start the API → call
`/ask` (curl + Postman + web UI) → a few good answers with citations → one question the
documents can't answer → a walk through the LangGraph layout.

---

## How it works

```
Ingest:  corpus/*.md -> chunk by section -> Gemini embeddings -> Pinecone (with metadata)

Ask:     POST /ask -> LangGraph:
           expand   -> turn the question into several diverse search queries
           retrieve -> embed each, search Pinecone, pool + dedupe above a score floor
           grade    -> LLM keeps only the chunks that actually answer the question
           decide   -+- good      -> generate answer with verified citations
                     +- bad       -> refine (broaden the queries) and retry, bounded
                     +- exhausted -> "I cannot find this in the documents."
```

The graph, its branch, and the two loop guards are documented in
[`docs/langgraph.md`](docs/langgraph.md). The diagram:

```
                  +-----------------------------------------------------+
                  |                                                     |
                  v                                                     |
START -> expand -> retrieve -> grade -> decide --+-- good ------> generate --> END
                                                 +-- bad --------> refine ----+ (loop)
                                                 +-- exhausted --> give_up --> END
```

---

## Quickstart (a new person, from scratch)

### 1. Get the free API keys

| Service | Where | Env var |
|---------|-------|---------|
| Groq (answer LLM) | <https://console.groq.com> | `GROQ_API_KEY` |
| Google Gemini (embeddings) | <https://aistudio.google.com/apikey> — the free **AI Studio** key | `GEMINI_API_KEY` |
| Pinecone (vector DB) | <https://app.pinecone.io> | `PINECONE_API_KEY` |

### 2. Install (Python 3.10+; developed on 3.12/3.13)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and paste your three keys
```

### 4. Ingest the corpus into Pinecone

```bash
python -m app.ingest
```

This creates the Pinecone index on first run (serverless, AWS `us-east-1`, dimension 768,
cosine), chunks every `*.md` in `corpus/`, embeds them, and upserts them with metadata.
See the **Pinecone checklist** below.

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

- **Web UI** — <http://localhost:8000/> (ask in the browser; shows the live graph flow)
- **Interactive API docs** — <http://localhost:8000/docs>
- **Health check** — <http://localhost:8000/health>

---

## Call the API

### With curl

A grounded answer with a citation:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What is the notice period at Bluecrest?"}'
```
```json
{
  "answer": "60 days written notice.",
  "citations": [
    { "source": "02_employment_agreement_excerpt.md",
      "chunk_id": "02_employment_agreement_excerpt#notice-period",
      "score": 0.7673,
      "snippet": "Employment agreement excerpt — Bluecrest Analytics ... ## Notice period Either party may end this agreement by giving 60 days written notice..." }
  ],
  "trace": null
}
```

Another one:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Can the tenant sublet the Harbor View unit?"}'
```
```json
{
  "answer": "No, the tenant cannot sublet the Harbor View unit without written consent of the lessor.",
  "citations": [
    { "source": "06_property_lease_clause.md",
      "chunk_id": "06_property_lease_clause#subletting",
      "score": 0.7342,
      "snippet": "Lease excerpt — Unit 4B, Harbor View Tower ... ## Subletting Subletting is not allowed without written consent of the lessor." }
  ],
  "trace": null
}
```

A question the documents **can't** answer (note the empty `citations` — no guessing):

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What is the population of Riverside city?"}'
```
```json
{ "answer": "I cannot find this in the documents.", "citations": [], "trace": null }
```

See the graph's node-by-node steps by adding `?trace=true`:

```bash
curl -s 'localhost:8000/ask?trace=true' -H 'content-type: application/json' \
  -d '{"question": "When is the next hearing in Arvind Mehta v. Northfield?"}'
```
```jsonc
// trace shows the path taken, e.g.:
// "expand -> 4 queries [...]", "retrieve(4 queries) -> 4 hits", "grade -> kept 1/4", "generate"
```

### With Postman

1. **Import** → drag in [`docs/legixo-qa.postman_collection.json`](docs/legixo-qa.postman_collection.json).
2. The collection has a `baseUrl` variable (default `http://localhost:8000`) and ready-made requests:
   *Health*, *answerable* (notice period, sublet), *unanswerable* (abstains), and *with graph trace*.
3. With the server running, hit **Send** on any request.

Or build it by hand: `POST {{baseUrl}}/ask`, header `Content-Type: application/json`, raw JSON body
`{ "question": "What is the notice period at Bluecrest?" }`.

### With the web UI

Open <http://localhost:8000/>, type a question (or click a sample chip), and press Enter.
Each answer shows its **citations** and a **"Behind the scenes"** panel that visualizes the
exact LangGraph path the request took — expand → retrieve → grade → branch → generate/abstain,
with the generated search queries and per-step results. Tick **show raw steps** for the raw trace.

---

## Pinecone checklist

- **Create the index:** `python -m app.ingest` creates it automatically if missing
  (`dimension=768`, `metric=cosine`, `ServerlessSpec(cloud="aws", region="us-east-1")`).
  You can also create it manually in the Pinecone console with those settings.
- **Region:** the free Starter plan only allows **AWS `us-east-1`** — it is the default.
- **Env vars:** `PINECONE_API_KEY` (required); optional `PINECONE_INDEX_NAME`,
  `PINECONE_NAMESPACE` — see `.env.example`.
- **Running ingest twice:** vector ids are the deterministic chunk ids
  (`<file>#<section>`), so a second run **overwrites the same ids — no duplicates**.
  Unchanged chunks are skipped (a `content_hash` is stored in metadata). Use
  `python -m app.ingest --reset` to wipe the namespace and rebuild from scratch.
- **Note:** Starter indexes pause after ~3 weeks idle; re-run ingest to wake one.

---

## Self-test (eval)

`eval/sample_test_cases.json` holds the gold-set questions (in-corpus + out-of-corpus).
With the server running:

```bash
python eval/run_eval.py
```

It posts each question and reports **citation correctness**, **fact recall**, and
**abstain correctness**, writing per-question pass/fail to `eval/results.md`.

## Tests

```bash
pytest
```

Unit tests (chunking, grading, citation guard, routing) and an API test run fully offline
with mocked providers. `tests/test_integration.py` runs end-to-end and is skipped unless
API keys are set.

---

## How this maps to the scoring

| What's scored | Where it lives |
|---------------|----------------|
| **Does the LangGraph make sense — clear steps, branch, limit?** | `app/graph.py` + [`docs/langgraph.md`](docs/langgraph.md): five named nodes, a single `decide` branch backed by the pure `route()` function, and **two** loop guards (the `retries` counter and `recursion_limit`). |
| **Are answers tied to real chunks? Any fake citations?** | Grounding is enforced **in code** (`enforce_grounding` in `app/llm.py`), not just by the prompt: any citation whose chunk id wasn't actually retrieved is dropped, and if none remain the answer becomes the refusal. A fabricated citation cannot reach the response. Verified by `tests/test_llm.py`. |
| **Does Pinecone ingest and search work?** | `python -m app.ingest` (chunk → embed → upsert, idempotent) and `app/retrieval.py` (embed query, search, pool above the score floor). The Pinecone checklist above covers index settings. |
| **Can a new person run it from the README?** | The Quickstart above (keys → install → configure → ingest → start) plus curl / Postman / web-UI examples. |

---

## Project layout

```
app/
  main.py        FastAPI app + /ask + /health + web UI at /
  static/        single-page chat UI with the live graph-flow visualization (no build step)
  graph.py       LangGraph: expand -> retrieve -> grade -> branch -> generate / refine / give_up
  retrieval.py   embed each query + Pinecone search + pool above the score floor
  llm.py         query expansion, relevance grader, grounded answer, citation guard
  chunking.py    header-aware markdown splitter
  ingest.py      python -m app.ingest
  clients.py     Pinecone / Gemini / Groq constructors
  schemas.py     request/response models
  settings.py    config from .env
corpus/          the documents
docs/            langgraph.md, the Postman collection, screenshots
eval/            gold-set questions + run_eval.py
```

## Configuration

All optional, via `.env` (defaults shown):

| Var | Default | Meaning |
|-----|---------|---------|
| `ANSWER_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `EMBED_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `TOP_K` | `4` | chunks retrieved per query |
| `SCORE_THRESHOLD` | `0.55` | cosine floor for a "good" chunk |
| `QUERY_FANOUT` | `3` | reworded query variants per search (the original is always included too) |
| `MAX_LOOPS` | `2` | broadened retries before giving up |

---

## Design notes

- **Grounding is enforced in code, not just by the prompt.** After the LLM answers, any
  citation whose chunk id was not actually retrieved is dropped; if nothing valid remains,
  the answer becomes the refusal. Fabricated citations cannot reach the response.
- **Multi-query expansion.** Each question is fanned out into several diverse search
  queries (the original plus reworded variants) and searched together, then the hits are
  pooled and deduped — better recall in one retrieval step. The `refine` node only kicks in
  as a fallback to broaden the search if that wide net still comes up empty.
- **Chunking** is per `##` section, with the document title and front-matter prepended to
  each chunk — so a question about the rent *and* the parties can be answered (and cited)
  from a single lease chunk.
- **Embeddings** use Gemini at 768 dimensions, L2-normalised (the model doesn't normalise
  truncated outputs), matching a cosine Pinecone index.

## What I'd add with more time

- **Hybrid search** (dense + sparse) for exact-term legal lookups.
- **A reranker** (e.g. Pinecone `bge-reranker-v2-m3`) after retrieval to sharpen the good/bad gate.
- **LangSmith** tracing for richer observability of graph runs.
