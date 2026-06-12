"""Thin constructors for the external services: Pinecone, Gemini, Groq.

Kept separate from the graph and the API so both the server (app.main) and the
ingest command can build the same clients without duplicating setup.
"""

import time

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

from .settings import Settings

_RETRIES = 5


def get_pinecone(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(pc: Pinecone, settings: Settings):
    """Create the serverless index if it is missing, then wait until it is queryable."""
    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
        time.sleep(1)
    return pc.Index(settings.pinecone_index_name)


def get_index(pc: Pinecone, settings: Settings):
    """Return a handle to an existing index. The API assumes ingest already ran."""
    if not pc.has_index(settings.pinecone_index_name):
        raise RuntimeError(
            f"Pinecone index '{settings.pinecone_index_name}' not found — "
            "run `python -m app.ingest` first."
        )
    return pc.Index(settings.pinecone_index_name)


def get_groq(settings: Settings) -> Groq:
    # max_retries lets the SDK back off and retry on free-tier 429s automatically.
    return Groq(api_key=settings.groq_api_key, max_retries=_RETRIES)


class Embedder:
    """Gemini text embeddings, L2-normalised to unit length so cosine search behaves.

    gemini-embedding-001 does not normalise when the output is truncated below its
    native dimension, so we do it ourselves.
    """

    def __init__(self, settings: Settings):
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.embed_model
        self._dim = settings.embed_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self._call_with_backoff(
            lambda: self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._dim
                ),
            )
        )
        if len(response.embeddings) != len(texts):
            raise RuntimeError(
                f"expected {len(texts)} embeddings, got {len(response.embeddings)}"
            )
        return [_normalise(e.values) for e in response.embeddings]

    @staticmethod
    def _call_with_backoff(call):
        """Retry the Gemini call on rate-limit / server errors with exponential backoff."""
        for attempt in range(_RETRIES):
            try:
                return call()
            except genai_errors.APIError as exc:
                transient = isinstance(exc, genai_errors.ServerError) or "RESOURCE_EXHAUSTED" in str(exc)
                if not transient or attempt == _RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # 1, 2, 4, 8s


def _normalise(values) -> list[float]:
    vec = np.asarray(values, dtype="float32")
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm else vec.tolist()
