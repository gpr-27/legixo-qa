"""Groq-backed steps: relevance grading, query rewriting, and grounded answering.

The model is only ever asked to work from the chunks we pass it. Grounding is then
enforced in code (`enforce_grounding`) so a fabricated citation can never reach the
response, regardless of what the model returns.
"""

import json
import logging

_log = logging.getLogger(__name__)

REFUSAL = "I cannot find this in the documents."

_GRADER_SYSTEM = (
    "You are a strict relevance grader for a retrieval system. For each numbered "
    "context chunk, decide whether it could help answer the question. "
    'Reply as JSON: {"relevant": [<numbers of the relevant chunks>]}. '
    "Return an empty list if none are relevant."
)

_EXPAND_SYSTEM = (
    "You turn a user's question into several short search queries for a semantic "
    "search over legal-style documents. Make them diverse: use synonyms, the formal "
    "or statutory phrasing, and the key parties/terms likely written in the documents. "
    "Each query is a short phrase, not a sentence; no quotes or surrounding punctuation. "
    'Reply as JSON: {"queries": [<query strings>]}.'
)

_ANSWER_SYSTEM = (
    "You answer questions using ONLY the numbered context chunks provided.\n"
    "- Use only facts found in the chunks; never use outside knowledge and never guess.\n"
    "- Answer fully: include the relevant conditions, amounts, dates, and obligations "
    "from the chunks, not just a one-word reply.\n"
    "- Cite the chunk_id of every chunk you relied on.\n"
    "- If the chunks cover the topic but not the specific detail asked, state what the "
    "documents do say and that the detail is not given — do not invent it.\n"
    f'- If the chunks do not address the question at all, set "answer" to exactly '
    f'"{REFUSAL}" and cite nothing.\n'
    'Reply as JSON: {"answer": <string>, '
    '"citations": [{"chunk_id": <string>, "source": <string>}]}.'
)


def _numbered_context(chunks: list[dict]) -> str:
    blocks = [
        f"### Chunk {i}\n[chunk_id: {c['chunk_id']} | source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n".join(blocks)


def _chat_json(groq, model: str, system: str, user: str) -> dict:
    response = groq.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(response.choices[0].message.content)


def grade_relevance(groq, settings, question: str, chunks: list[dict]) -> list[dict]:
    """Keep only the chunks the model judges relevant to the question."""
    if not chunks:
        return []
    user = f"Question: {question}\n\n{_numbered_context(chunks)}"
    data = _chat_json(groq, settings.answer_model, _GRADER_SYSTEM, user)
    try:
        keep = {int(n) for n in data.get("relevant", [])}
    except (TypeError, ValueError):
        keep = set()
    return [c for i, c in enumerate(chunks, 1) if i in keep]


def generate_search_queries(groq, settings, question: str, broaden: bool = False) -> list[str]:
    """Turn the question into several diverse search queries (one LLM call).

    With ``broaden=True`` it is asked for wider, more general phrasings — used by the
    fallback after the first wide search found nothing.
    """
    n = settings.query_fanout
    hint = (
        " The first search found nothing, so go broader: more general terms, "
        "alternative wordings, and related concepts."
        if broaden else ""
    )
    user = f"Generate {n} search queries for this question.{hint}\n\nQuestion: {question}"
    data = _chat_json(groq, settings.answer_model, _EXPAND_SYSTEM, user)
    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
    return queries[:n]


def generate_answer(groq, settings, question: str, chunks: list[dict]):
    """Generate a grounded answer and return (answer, cited_chunks)."""
    user = f"Question: {question}\n\n{_numbered_context(chunks)}"
    data = _chat_json(groq, settings.answer_model, _ANSWER_SYSTEM, user)
    answer = (data.get("answer") or "").strip()
    return enforce_grounding(answer, data.get("citations") or [], chunks)


def enforce_grounding(answer: str, citations: list[dict], chunks: list[dict]):
    """Drop citations not backed by a retrieved chunk; abstain if none remain valid."""
    by_id = {c["chunk_id"]: c for c in chunks}
    cited, seen = [], set()
    for citation in citations:
        cid = citation.get("chunk_id")
        if cid in by_id and cid not in seen:
            seen.add(cid)
            cited.append(by_id[cid])

    fabricated = sum(1 for c in citations if c.get("chunk_id") not in by_id)
    if fabricated:
        _log.warning("dropped %d ungrounded citation(s)", fabricated)

    if answer == REFUSAL or not cited:
        return REFUSAL, []
    return answer, cited
