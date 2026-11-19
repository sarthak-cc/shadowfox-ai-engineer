"""Shared Gemini client and error mapping."""
import logging

from google import genai
from google.genai import errors, types

from ..core.config import get_settings

log = logging.getLogger("rag.gemini")
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """An error that is safe to show to the user."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    s = get_settings()
    if not s.gemini_api_key:
        raise LLMError("The server is missing GEMINI_API_KEY. Add it to .env and restart.", 500)
    if _client is None:
        # SDK retries are disabled: retries, fallbacks and hedging are handled by this app,
        # so a stalled call fails after one timeout instead of silently retrying for minutes.
        _client = genai.Client(api_key=s.gemini_api_key, http_options=types.HttpOptions(
            timeout=int(s.request_timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)))
    return _client


def friendly(err: errors.APIError) -> LLMError:
    code = err.code or 0
    if code in (401, 403):
        return LLMError("The AI service rejected the API key. Check GEMINI_API_KEY.", 500)
    if code == 404:
        return LLMError("A configured AI model is not available. Check the model names in .env.", 500)
    if code == 429:
        return LLMError("The AI service is busy (rate limit reached). Please wait a moment and try again.", 429)
    if code == 400:
        return LLMError("The AI service could not process this request.", 400)
    return LLMError("The AI service is temporarily unavailable. Please try again.", 503)


def retry_delay(err: errors.APIError) -> float | None:
    """Seconds the API asks us to wait (RetryInfo.retryDelay), if present."""
    try:
        for d in (err.details or {}).get("error", {}).get("details", []):
            if d.get("@type", "").endswith("RetryInfo"):
                return float(str(d.get("retryDelay", "")).rstrip("s"))
    except (AttributeError, TypeError, ValueError):
        pass
    return None
