"""Settings loaded from environment variables / .env."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


class Settings:
    def __init__(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        self.fallback_models = _list("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
        self.request_timeout_s = float(os.getenv("REQUEST_TIMEOUT_S", "25"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "1"))
        # Gemini free tier: 100 embedded texts per minute. Keep a margin for question embeddings.
        self.embed_per_minute = int(os.getenv("EMBED_PER_MINUTE", "90"))

        # Ingestion
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_file_mb = float(os.getenv("MAX_FILE_MB", "15"))
        self.max_documents = int(os.getenv("MAX_DOCUMENTS", "20"))

        # Retrieval
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.min_score = float(os.getenv("MIN_SCORE", "0.55"))

        self.data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "data")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
