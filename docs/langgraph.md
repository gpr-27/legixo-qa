# LangGraph layout

The Q&A flow is a `StateGraph` defined in [`app/graph.py`](../app/graph.py). It
expands the question into several search queries, retrieves and pools chunks, grades
them, and branches: answer from good chunks, or broaden the search and try again, or
give up honestly.

```
                  +-----------------------------------------------------+
                  |                                                     |
                  v                                                     |
START -> expand -> retrieve -> grade -> decide --+-- good ------> generate --> END
                                                 |
                                                 +-- bad --------> refine ----+ (loop)
                                                 |
                                                 +-- exhausted --> give_up --> END
```

## State

A `TypedDict` carried through every node:

| field | purpose |
|-------|---------|
| `question` | the original user question (never rewritten) |
| `queries` | the current search queries — the fan-out from `expand`, or the broadened set from `refine` |
| `documents` | chunks kept after grading: `{chunk_id, source, text, score}` |
| `answer` | the final answer string |
| `citations` | `{source, chunk_id, score, snippet}` for the answer |
| `retries` | how many times the search has been broadened |
| `steps` | trace of nodes visited (returned as `trace` when `?trace=true`) |

## Nodes

| node | what it does |
|------|--------------|
| **expand** | Asks Groq to turn the question into several diverse search queries (synonyms, formal/statutory phrasing, key parties). The original question is always included. One LLM call casts a wide net up front. |
| **retrieve** | Embeds every query with Gemini, searches Pinecone (`top_k` each), and **pools** the results — deduped by chunk id, keeping each chunk's best score, dropping anything below the cosine `score_threshold`. |
| **grade** | Asks Groq to mark which pooled chunks are actually relevant; keeps those. This is the "good vs bad chunks" gate — a score floor *and* an LLM judgement, which catches plausible-but-absent questions. |
| **generate** | Writes a grounded answer from the kept chunks and cites them. Citations are validated in code, so a fabricated one can't slip through. |
| **refine** | The fallback: if the wide search found nothing, asks Groq for broader/synonym-rich queries, increments `retries`, and loops back to retrieve. |
| **give_up** | Emits the fixed refusal (`"I cannot find this in the documents."`) with no citations. |

## The branch

`decide(state)` (backed by the pure, unit-tested `route()` function) returns:

- `generate` — at least one chunk survived grading (the good path);
- `refine` — nothing survived but `retries < MAX_LOOPS` (the bad path: broaden and retry);
- `give_up` — nothing survived and the retry budget is spent (abstain).

## The limit (so it cannot spin forever)

Two independent guards:

1. **`retries` counter** — the intended stop. After `MAX_LOOPS` (default 2) broadened
   retries with no good chunks, `decide` routes to `give_up`.
2. **`recursion_limit`** — a safety net passed at invoke time
   (`max_loops * 3 + 7`). If it is ever exceeded, LangGraph raises
   `GraphRecursionError`, which the API catches and turns into the same clean refusal.

## Why expand up front?

An earlier version refined **one** query at a time, sequentially, only after a miss.
Generating several diverse queries at once gives better recall in a single retrieval
step and lower latency on hard questions, and it removes a fragile detail (the old
rewrite ran at `temperature=0` and only varied by an attempt counter). `refine` is now
just the safety net for the rare case where even the wide search comes up empty.
