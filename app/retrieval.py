"""Dense retrieval: embed each query, search Pinecone, pool the matches above the floor."""

from .clients import Embedder
from .settings import Settings


def retrieve(index, embedder: Embedder, settings: Settings, queries: list[str]) -> list[dict]:
    """Search the index with every query and return the pooled chunks above the floor.

    Each query is embedded and searched independently; the results are merged, deduped
    by chunk id (keeping the best score a chunk earned across queries), and sorted best
    first. Searching several phrasings at once widens recall in a single retrieval step.
    """
    pooled: dict[str, dict] = {}
    for query in queries:
        vector = embedder.embed_query(query)
        response = index.query(
            vector=vector,
            top_k=settings.top_k,
            include_metadata=True,
            namespace=settings.pinecone_namespace,
        )
        for match in response.matches:
            if match.score < settings.score_threshold:
                continue
            meta = match.metadata or {}
            chunk_id = meta.get("chunk_id", match.id)
            existing = pooled.get(chunk_id)
            if existing is None or match.score > existing["score"]:
                pooled[chunk_id] = {
                    "chunk_id": chunk_id,
                    "source": meta.get("source", ""),
                    "text": meta.get("text", ""),
                    "score": float(match.score),
                }
    return sorted(pooled.values(), key=lambda d: d["score"], reverse=True)
