"""Application settings loaded from environment variables / .env file."""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        self.request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "60"))
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "1"))
        fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash")
        self.fallback_models: list[str] = [m.strip() for m in fallbacks.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
