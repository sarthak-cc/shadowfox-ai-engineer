"""Typed application settings (pydantic-settings).

Every value can be overridden with an environment variable of the same name
(case-insensitive) or a `.env` file. Invalid values fail fast at startup.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore")

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_fallback_models: list[str] = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
    embedding_provider: Literal["gemini", "local"] = Field(
        "gemini", description="gemini = API embeddings (free tier: 100/min, 1,000/day); "
                              "local = open-source model on this machine (no limits, no network)")
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = Field(768, ge=128, le=3072)
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_dim: int = 384
    request_timeout_s: float = Field(25, gt=0)
    hedge_after_s: float = Field(6, gt=0, description="Start a backup model if no response by then")
    embed_per_minute: int = Field(90, ge=1, description="Free tier allows 100 embedded texts/minute")

    # --- Ingestion ---
    chunk_size: int = Field(1200, ge=200, le=4000)
    chunk_overlap: int = Field(200, ge=0, le=1000)
    max_file_mb: float = Field(20, gt=0)
    max_documents: int = Field(50, ge=1)

    # --- Retrieval ---
    candidates_per_query: int = Field(20, ge=5, le=100, description="Dense + BM25 hits per search query")
    rerank_pool: int = Field(20, ge=5, le=60, description="Fused candidates sent to the reranker")
    top_k: int = Field(5, ge=1, le=12, description="Passages given to the generator")
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_min_score: float = Field(0.02, ge=0, le=1, description="Absolute relevance floor (0-1)")
    rerank_relative_floor: float = Field(0.05, ge=0, le=1, description="Keep passages >= best * this")
    min_dense_score: float = Field(0.5, ge=-1, le=1, description="Used only if the reranker is unavailable")

    # --- Generation / checks ---
    groundedness_threshold: float = Field(0.75, ge=0, le=1)
    max_revisions: int = Field(1, ge=0, le=2)
    history_turns: int = Field(4, ge=0, le=20)

    # --- Paths ---
    data_dir: Path = BACKEND_ROOT / "data"
    samples_dir: Path = BACKEND_ROOT.parent / "samples"
    model_cache_dir: Path = BACKEND_ROOT / ".model_cache"

    cors_origins: list[str] = ["*"]

    @property
    def vector_dim(self) -> int:
        return self.local_embedding_dim if self.embedding_provider == "local" else self.embedding_dim

    @property
    def embedding_id(self) -> str:
        """Identifies the vector space; an index built with a different one can't be reused."""
        if self.embedding_provider == "local":
            return f"local:{self.local_embedding_model}"
        return f"gemini:{self.embedding_model}:{self.embedding_dim}"

    @field_validator("gemini_fallback_models", "cors_origins", mode="before")
    @classmethod
    def split_csv(cls, v):
        if isinstance(v, str) and not v.strip().startswith("["):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, v, info):
        size = info.data.get("chunk_size", 1200)
        if v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
