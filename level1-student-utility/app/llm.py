"""Thin wrapper around the Gemini API.

Responsibilities:
- build the request with a system prompt + a JSON response schema
- retry transient failures (rate limits, 5xx, timeouts) with exponential backoff
- translate every failure into an `LLMError` with a user-friendly message
- validate the returned JSON against the Pydantic model before it reaches the UI
"""
import logging
import time
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .config import get_settings
from .prompts import SYSTEM_PROMPT

log = logging.getLogger("studypilot.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """An error that is safe to show to the user."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    settings = get_settings()
    if not settings.gemini_api_key:
        raise LLMError("The server is missing GEMINI_API_KEY. Add it to the .env file and restart.", 500)
    if _client is None:
        _client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.request_timeout_s * 1000)),
        )
    return _client


def _friendly(err: errors.APIError) -> LLMError:
    code = err.code or 0
    if code in (401, 403):
        return LLMError("The AI service rejected the API key. Check GEMINI_API_KEY.", 500)
    if code == 404:
        return LLMError("The configured AI model is not available. Check GEMINI_MODEL.", 500)
    if code == 429:
        return LLMError("The AI service is busy (rate limit reached). Please wait a moment and try again.", 429)
    if code == 400:
        return LLMError("The AI service could not process this input. Try shortening or rephrasing it.", 400)
    return LLMError("The AI service is temporarily unavailable. Please try again.", 503)


def generate_structured(prompt: str, schema: type[T], temperature: float = 0.4) -> T:
    """Call the model and return a validated instance of `schema`."""
    settings = get_settings()
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )

    # Attempt plan: the primary model gets `max_retries + 1` tries, then each fallback model
    # gets one try. Free-tier models sometimes return 503 "high demand", so falling back to a
    # sibling model keeps the app usable instead of failing.
    plan = [settings.gemini_model] * (settings.max_retries + 1) + settings.fallback_models
    last_error: LLMError | None = None
    for attempt, model in enumerate(plan):
        try:
            started = time.perf_counter()
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            log.info("model=%s attempt=%d latency=%.2fs", model, attempt + 1, time.perf_counter() - started)

            if not response.text:
                # Empty text usually means the response was blocked by safety filters.
                raise LLMError("The AI returned no content for this input (it may have been filtered). "
                               "Try rephrasing.", 422)
            return schema.model_validate_json(response.text)

        except errors.APIError as e:
            log.warning("Gemini API error model=%s attempt=%d: %s", model, attempt + 1, str(e)[:120])
            last_error = _friendly(e)
            retryable = (e.code or 0) in (404, 429, 500, 502, 503, 504)
        except ValidationError as e:
            log.warning("Malformed model output (attempt %d): %s", attempt + 1, e)
            last_error = LLMError("The AI returned an unexpected format. Please try again.", 502)
            retryable = True
        except LLMError:
            raise
        except Exception as e:  # network errors, timeouts, etc.
            log.exception("Unexpected error calling Gemini")
            last_error = LLMError("Could not reach the AI service. Check your internet connection and try again.", 503)
            retryable = True

        if not retryable or attempt == len(plan) - 1:
            break
        if plan[attempt + 1] == model:
            time.sleep(1.0 * (2 ** attempt))  # backoff only when retrying the same model

    assert last_error is not None
    raise last_error
