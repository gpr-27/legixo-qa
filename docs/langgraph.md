# LangGraph layout

The Q&A flow is a `StateGraph` defined in [`app/graph.py`](../app/graph.py). It
retrieves chunks, grades them, and branches: answer from good chunks, or rewrite
the query and try again, or give up honestly.

```
            +-------------------------------------------------+
            |                                                 |
            v                                                 |
START --> retrieve --> grade --> decide --+-- good ------> generate --> END
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
| `query` | current retrieval query (the rewritten one after a refine) |
| `documents` | chunks kept after grading: `{chunk_id, source, text, score}` |
| `answer` | the final answer string |
| `citations` | `{source, chunk_id, score, snippet}` for the answer |
| `retries` | how many times the query has been rewritten |
| `steps` | trace of nodes visited (returned as `trace` when `?trace=true`) |

## Nodes

| node | what it does |
|------|--------------|
| **retrieve** | Embeds `query` with Gemini, searches Pinecone (`top_k`), keeps matches at or above the cosine `score_threshold`. |
| **grade** | Asks Groq to mark which retrieved chunks are actually relevant; keeps those. This is the "good vs bad chunks" gate — a score floor *and* an LLM judgement, which catches plausible-but-absent questions. |
| **generate** | Writes a grounded answer from the kept chunks and cites them. Citations are validated in code, so a fabricated one can't slip through. |
| **refine** | Rewrites the question into a fresh query and increments `retries`, then loops back to retrieve. |
| **give_up** | Emits the fixed refusal (`"I cannot find this in the documents."`) with no citations. |

## The branch

`decide(state)` (backed by the pure, unit-tested `route()` function) returns:

- `generate` — at least one chunk survived grading (the good path);
- `refine` — nothing survived but `retries < MAX_LOOPS` (the bad path: rewrite and retry);
- `give_up` — nothing survived and the retry budget is spent (abstain).

## The limit (so it cannot spin forever)

Two independent guards:

1. **`retries` counter** — the intended stop. After `MAX_LOOPS` (default 2) rewrites
   with no good chunks, `decide` routes to `give_up`.
2. **`recursion_limit`** — a safety net passed at invoke time
   (`max_loops * 3 + 5`). If it is ever exceeded, LangGraph raises
   `GraphRecursionError`, which the API catches and turns into the same clean refusal.
