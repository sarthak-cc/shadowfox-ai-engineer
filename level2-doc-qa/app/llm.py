"""Gemini client: embeddings + structured generation, with retries and friendly errors."""
import logging
import time
from typing import TypeVar

import numpy as np
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .config import get_settings

log = logging.getLogger("doclens.llm")
T = TypeVar("T", bound=BaseModel)

EMBED_BATCH = 100  # API limit per request
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
        raise LLMError("The server is missing GEMINI_API_KEY. Add it to the .env file and restart.", 500)
    if _client is None:
        _client = genai.Client(api_key=s.gemini_api_key,
                               http_options=types.HttpOptions(timeout=int(s.request_timeout_s * 1000)))
    return _client


def _friendly(err: errors.APIError) -> LLMError:
    code = err.code or 0
    if code in (401, 403):
        return LLMError("The AI service rejected the API key. Check GEMINI_API_KEY.", 500)
    if code == 404:
        return LLMError("The configured AI model is not available. Check the model names in .env.", 500)
    if code == 429:
        return LLMError("The AI service is busy (rate limit reached). Please wait a moment and try again.", 429)
    if code == 400:
        return LLMError("The AI service could not process this request.", 400)
    return LLMError("The AI service is temporarily unavailable. Please try again.", 503)


# ---------------- embeddings ----------------

def embed(texts: list[str], task: str) -> np.ndarray:
    """Embed texts and return an L2-normalised float32 matrix (n, dim).

    task: "RETRIEVAL_DOCUMENT" for chunks, "RETRIEVAL_QUERY" for questions. Gemini embeds
    the two asymmetrically, which improves question -> passage matching.
    """
    s = get_settings()
    client = get_client()
    config = types.EmbedContentConfig(task_type=task, output_dimensionality=s.embedding_dim)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        for attempt in range(4):
            try:
                result = client.models.embed_content(model=s.embedding_model, contents=batch, config=config)
                vectors.extend(e.values for e in result.embeddings)
                break
            except errors.APIError as e:
                log.warning("embedding error (attempt %d): %s", attempt + 1, str(e)[:120])
                if (e.code or 0) not in RETRYABLE or attempt == 3:
                    raise _friendly(e) from e
                time.sleep(1.5 * (2 ** attempt))
            except Exception as e:
                log.exception("embedding request failed")
                if attempt == 3:
                    raise LLMError("Could not reach the AI service to create embeddings.", 503) from e
                time.sleep(1.5 * (2 ** attempt))
    matrix = np.asarray(vectors, dtype=np.float32)
    # Truncated (768-d) Gemini embeddings are not unit length, so normalise for cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


# ---------------- generation ----------------

def generate_structured(prompt: str, schema: type[T], system: str, temperature: float = 0.2) -> T:
    s = get_settings()
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    plan = [s.gemini_model] * (s.max_retries + 1) + s.fallback_models
    last_error: LLMError | None = None
    for attempt, model in enumerate(plan):
        try:
            started = time.perf_counter()
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            log.info("model=%s attempt=%d latency=%.2fs", model, attempt + 1, time.perf_counter() - started)
            if not response.text:
                raise LLMError("The AI returned no content (it may have been filtered). Try rephrasing.", 422)
            return schema.model_validate_json(response.text)
        except errors.APIError as e:
            log.warning("generation error model=%s attempt=%d: %s", model, attempt + 1, str(e)[:120])
            last_error = _friendly(e)
            retryable = (e.code or 0) in RETRYABLE | {404}
        except ValidationError:
            log.warning("malformed model output (attempt %d)", attempt + 1)
            last_error = LLMError("The AI returned an unexpected format. Please try again.", 502)
            retryable = True
        except LLMError:
            raise
        except Exception:
            log.exception("unexpected generation error")
            last_error = LLMError("Could not reach the AI service. Check your connection and try again.", 503)
            retryable = True
        if not retryable or attempt == len(plan) - 1:
            break
        if plan[attempt + 1] == model:
            time.sleep(1.0 * (2 ** attempt))
    assert last_error is not None
    raise last_error
