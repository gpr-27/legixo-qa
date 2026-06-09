"""Dense retrieval: embed the query, search Pinecone, keep matches above the floor."""

from .clients import Embedder
from .settings import Settings


def retrieve(index, embedder: Embedder, settings: Settings, query: str) -> list[dict]:
    """Return chunks scoring at or above the cosine threshold, best first."""
    vector = embedder.embed_query(query)
    response = index.query(
        vector=vector,
        top_k=settings.top_k,
        include_metadata=True,
        namespace=settings.pinecone_namespace,
    )

    documents = []
    for match in response.matches:
        if match.score < settings.score_threshold:
            continue
        meta = match.metadata or {}
        documents.append(
            {
                "chunk_id": meta.get("chunk_id", match.id),
                "source": meta.get("source", ""),
                "text": meta.get("text", ""),
                "score": float(match.score),
            }
        )
    return documents
