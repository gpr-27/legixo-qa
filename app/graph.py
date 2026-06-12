"""The LangGraph flow: expand -> retrieve -> grade -> branch (generate / broaden-loop / give up).

    START -> expand -> retrieve -> grade -> decide -+-- good ------> generate -> END
                          ^                          +-- bad -------> refine (loops to retrieve)
                          |                          +-- exhausted -> give_up -> END
                          +--------------------------- refine -------------------+

`expand` turns the question into several diverse search queries up front, so retrieval
casts a wide net in one shot. `refine` is now only a fallback: if that wide search still
finds nothing, it generates broader queries and retries, bounded by the `retries` counter
(the intended stop) and `recursion_limit` at invoke time (a safety net).
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from . import llm, retrieval
from .llm import REFUSAL


class GraphState(TypedDict):
    question: str                            # original question, never rewritten
    queries: list[str]                       # current search queries (fanned out / broadened)
    documents: list[dict]                    # chunks kept after grading
    answer: str
    citations: list[dict]
    retries: int
    steps: Annotated[list[str], operator.add]  # trace; appended to by every node


def route(documents: list, retries: int, max_loops: int) -> str:
    """Pure branch decision, factored out so it can be unit-tested directly."""
    if documents:
        return "generate"
    if retries < max_loops:
        return "refine"
    return "give_up"


def _dedupe(queries: list[str]) -> list[str]:
    """Drop blank/duplicate queries (case-insensitive), preserving order."""
    seen, out = set(), []
    for q in queries:
        q = q.strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def _fmt(queries: list[str]) -> str:
    """Render the query list for the trace (and the UI flow)."""
    return "[" + " | ".join(queries) + "]"


def _snippet(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_graph(index, embedder, groq, settings):
    """Compile the graph, closing over the service clients it needs."""

    def expand_node(state: GraphState):
        variants = llm.generate_search_queries(groq, settings, state["question"])
        queries = _dedupe([state["question"], *variants])
        return {"queries": queries,
                "steps": [f"expand -> {len(queries)} queries {_fmt(queries)}"]}

    def retrieve_node(state: GraphState):
        docs = retrieval.retrieve(index, embedder, settings, state["queries"])
        return {"documents": docs,
                "steps": [f"retrieve({len(state['queries'])} queries) -> {len(docs)} hits"]}

    def grade_node(state: GraphState):
        kept = llm.grade_relevance(groq, settings, state["question"], state["documents"])
        return {"documents": kept,
                "steps": [f"grade -> kept {len(kept)}/{len(state['documents'])}"]}

    def generate_node(state: GraphState):
        answer, cited = llm.generate_answer(
            groq, settings, state["question"], state["documents"]
        )
        citations = [
            {"source": c["source"], "chunk_id": c["chunk_id"],
             "score": round(c["score"], 4), "snippet": _snippet(c["text"])}
            for c in cited
        ]
        return {"answer": answer, "citations": citations, "steps": ["generate"]}

    def refine_node(state: GraphState):
        attempt = state["retries"] + 1
        queries = _dedupe(llm.generate_search_queries(groq, settings, state["question"], broaden=True))
        return {"queries": queries, "retries": attempt,
                "steps": [f"refine(attempt={attempt}) -> {len(queries)} broadened queries {_fmt(queries)}"]}

    def give_up_node(state: GraphState):
        return {"answer": REFUSAL, "citations": [], "steps": ["give_up"]}

    def decide(state: GraphState) -> str:
        return route(state["documents"], state["retries"], settings.max_loops)

    builder = StateGraph(GraphState)
    builder.add_node("expand", expand_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)
    builder.add_node("refine", refine_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "expand")
    builder.add_edge("expand", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade", decide,
        {"generate": "generate", "refine": "refine", "give_up": "give_up"},
    )
    builder.add_edge("refine", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("give_up", END)
    return builder.compile()
