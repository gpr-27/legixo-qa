"""Load the markdown corpus into Pinecone.

    python -m app.ingest            # idempotent: re-running overwrites, no duplicates
    python -m app.ingest --reset    # wipe the namespace first (clean rebuild)

Vector ids are the deterministic chunk ids, so a second run upserts the same ids in
place. Unchanged chunks are skipped via a stored content hash to save embedding quota.
"""

import argparse
from pathlib import Path

from . import clients
from .chunking import chunk_corpus
from .settings import get_settings

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into Pinecone.")
    parser.add_argument("--reset", action="store_true",
                        help="delete all vectors in the namespace before ingesting")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR,
                        help="path to the markdown corpus directory")
    args = parser.parse_args()

    settings = get_settings()
    pc = clients.get_pinecone(settings)
    index = clients.ensure_index(pc, settings)
    embedder = clients.Embedder(settings)

    chunks = chunk_corpus(args.corpus)
    if not chunks:
        raise SystemExit(f"No .md files found in {args.corpus}")

    if args.reset:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
        print(f"Cleared namespace '{settings.pinecone_namespace}'.")

    stored_hashes = _stored_hashes(index, settings, [c.chunk_id for c in chunks])
    todo = [c for c in chunks if stored_hashes.get(c.chunk_id) != c.content_hash]
    print(f"{len(chunks)} chunks total; {len(todo)} new or changed.")

    if todo:
        embeddings = embedder.embed_documents([c.text for c in todo])
        vectors = [
            {
                "id": c.chunk_id,
                "values": vector,
                "metadata": {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "doc_title": c.doc_title,
                    "section_header": c.section_header,
                    "text": c.text,
                    "content_hash": c.content_hash,
                },
            }
            for c, vector in zip(todo, embeddings)
        ]
        for start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[start:start + 100],
                         namespace=settings.pinecone_namespace)

    print(
        f"Done: {len(todo)} vectors upserted into "
        f"'{settings.pinecone_index_name}' / namespace '{settings.pinecone_namespace}'."
    )


def _stored_hashes(index, settings, ids: list[str]) -> dict[str, str]:
    """Return {chunk_id: content_hash} already in the index (empty on a fresh index)."""
    if not ids:
        return {}
    try:
        fetched = index.fetch(ids=ids, namespace=settings.pinecone_namespace)
    except Exception:
        return {}  # best-effort optimisation: on any failure we just re-embed everything
    vectors = getattr(fetched, "vectors", {}) or {}
    return {
        vid: (v.metadata or {}).get("content_hash")
        for vid, v in vectors.items()
    }


if __name__ == "__main__":
    main()
