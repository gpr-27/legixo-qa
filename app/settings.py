"""Application configuration, loaded from the environment / a local .env file."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets (no defaults — the app refuses to start without them).
    groq_api_key: str
    gemini_api_key: str = Field(
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    pinecone_api_key: str

    # Pinecone index. Region/cloud are fixed to the only values the free tier allows.
    pinecone_index_name: str = "legixo-qa"
    pinecone_namespace: str = "legal-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Models.
    answer_model: str = "llama-3.3-70b-versatile"   # Groq
    embed_model: str = "gemini-embedding-001"        # Gemini
    embed_dim: int = 768

    # Retrieval / graph behaviour.
    top_k: int = 4
    score_threshold: float = 0.55   # cosine floor for a "good" match; tune on the eval set
    max_loops: int = 2              # query rewrites before giving up


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance; raises clearly if a required key is missing."""
    return Settings()
