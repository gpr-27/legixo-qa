# Demo video script (5–8 min)

A shot-by-shot script for the walkthrough. Read the **say** lines aloud and run the
**do** commands. Keep your API keys off-screen (don't open `.env` on camera).

Before you hit record: have two terminals ready in the repo root, the Pinecone
console open in a browser tab, and Postman (or `/docs`) ready.

---

## 1 · Intro (~30s)
- **say:** "This is a grounded Q&A API over a set of fictional legal documents,
  built with Python, LangGraph, and Pinecone. It answers only from the documents
  and cites the chunk each answer came from — and says 'I cannot find this' when
  the documents don't cover it."
- **do:** show the repo tree and scroll the top of `README.md`.

## 2 · Setup (~1m)
- **say:** "Dependencies are pinned in requirements.txt, and the three API keys —
  Groq, Gemini, Pinecone — live in a local .env that's gitignored."
- **do:** show `requirements.txt`; mention `.env` exists (do **not** open it).

## 3 · Ingest + Pinecone (~1.5m)
- **do:** `python -m app.ingest`
- **say:** "Ingest chunks the six documents by section, embeds them with Gemini,
  and upserts them into Pinecone with citation metadata. It reports 12 chunks."
- **do:** switch to **app.pinecone.io** → index `legixo-qa` → namespace `legal-docs`.
- **say:** "Here are the 12 vectors in Pinecone. Each one stores its chunk_id, the
  source file, the section header, and the text — that's what citations point back to."
- **say (idempotency):** "Re-running ingest overwrites the same ids, so the count
  stays 12 — no duplicates."

## 4 · Start the API (~30s)
- **do:** `uvicorn app.main:app --reload`
- **do:** open `http://localhost:8000/` (the web UI) — also mention `/docs` and `/health`.
- **say:** "The backend is up, with a small chat UI, a POST /ask endpoint, and a health check."

## 5 · Ask questions (~2m) — use the web UI at / (or Postman / /docs)
Good answers (show the **citations** each time):
- **Q:** `What is the notice period in the Bluecrest employment agreement?`
  → 60 days, cites `02_employment_agreement_excerpt.md`.
- **Q:** `If a contract fixes no interest rate, what rate applies to delayed payments?`
  → 9% per year, cites `04_statute_style_excerpt_fictional.md`.

Show the graph trace:
- **do:** call `http://localhost:8000/ask?trace=true` with
  `Can the tenant sublet the Harbor View unit?`
- **say:** "With trace on, you can see the graph's steps: retrieve, grade, generate."

Out-of-corpus (the grounding demo):
- **Q:** `What is the population of Riverside city?`
- **say:** "Nothing in the documents covers this, so it refuses with empty
  citations instead of guessing — no fabricated answer."

## 6 · Point at the LangGraph layout (~1.5m)
- **do:** open `docs/langgraph.md`, then `app/graph.py`.
- **say:** "The flow is a LangGraph StateGraph: retrieve finds chunks, grade checks
  if they're relevant, then the branch decides — good chunks go to generate with
  citations; if nothing's relevant it rewrites the query and retries; after two
  tries it gives up honestly. The loop is bounded by a retry counter and a
  recursion limit, so it can't spin forever."

## 7 · (optional) Eval + wrap (~30s)
- **do:** `python eval/run_eval.py`
- **say:** "This runs the 19 gold-set questions and reports citation correctness,
  fact recall, and abstain correctness." Then wrap up.

---

After recording: upload (Loom / YouTube unlisted / Drive) and paste the link into
the **Demo video** section of `README.md`.
