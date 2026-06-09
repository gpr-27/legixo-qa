"""The LangGraph flow: retrieve -> grade -> branch (generate / refine-loop / give up).

    START -> retrieve -> grade -> decide -+-- good ------> generate -> END
                ^                          +-- bad -------> refine (loops to retrieve)
                |                          +-- exhausted -> give_up -> END
                +--------------------------- refine -------------------+

The branch lives in `decide`; the loop is bounded twice — by the `retries` counter
(the intended stop) and by `recursion_limit` at invoke time (a safety net).
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from . import llm, retrieval
from .llm import REFUSAL


class GraphState(TypedDict):
    question: str                            # original question, never rewritten
    query: str                               # current retrieval query (may be rewritten)
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


def _snippet(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_graph(index, embedder, groq, settings):
    """Compile the graph, closing over the service clients it needs."""

    def retrieve_node(state: GraphState):
        docs = retrieval.retrieve(index, embedder, settings, state["query"])
        return {"documents": docs,
                "steps": [f"retrieve(query={state['query']!r}) -> {len(docs)} hits"]}

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
        new_query = llm.rewrite_query(groq, settings, state["question"], attempt)
        return {"query": new_query, "retries": attempt,
                "steps": [f"refine(attempt={attempt}) -> {new_query!r}"]}

    def give_up_node(state: GraphState):
        return {"answer": REFUSAL, "citations": [], "steps": ["give_up"]}

    def decide(state: GraphState) -> str:
        return route(state["documents"], state["retries"], settings.max_loops)

    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)
    builder.add_node("refine", refine_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade", decide,
        {"generate": "generate", "refine": "refine", "give_up": "give_up"},
    )
    builder.add_edge("refine", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("give_up", END)
    return builder.compile()
