"""Request/response models for the Q&A API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    source: str        # source file name
    chunk_id: str
    score: float       # cosine similarity of the retrieved chunk
    snippet: str       # short excerpt of the chunk text


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: list[str] | None = None   # node-by-node steps; included when ?trace=true
