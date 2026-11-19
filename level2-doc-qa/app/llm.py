"""Gemini client: embeddings + structured generation, with retries and friendly errors."""
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from typing import TypeVar

import numpy as np
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .config import get_settings

log = logging.getLogger("doclens.llm")
T = TypeVar("T", bound=BaseModel)

EMBED_BATCH = 32      # texts per request (API limit is 100; smaller batches return faster)
EMBED_WORKERS = 4     # batches in flight at once
EMBED_ATTEMPTS = 4
RETRYABLE = {429, 500, 502, 503, 504}


class RateLimiter:
    """Sliding one-minute window over the number of texts embedded.

    On the Gemini free tier every text in a batch counts as one request, and the limit is
    100 per minute. Waiting *before* sending keeps us under the limit instead of hitting
    429 errors and retrying.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    def acquire(self, n: int, on_wait: Callable[[float], None] | None = None, reserve: int = 0) -> None:
        """Block until `n` texts fit in the window. `reserve` lets urgent calls (a user's question)
        use a few extra slots above the normal limit, so they don't queue behind a big upload."""
        limit = self.per_minute + reserve
        n = min(n, limit)
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] >= 60:
                    self.events.popleft()
                used = sum(k for _, k in self.events)
                if used + n <= limit:
                    self.events.append((now, n))
                    return
                # wait until enough of the oldest usage expires
                freed, wait = 0, 0.0
                for ts, k in self.events:
                    freed += k
                    if used - freed + n <= limit:
                        wait = 60 - (now - ts) + 0.2
                        break
            if on_wait:
                on_wait(wait)
            time.sleep(min(wait, 5))


_limiter: RateLimiter | None = None


def _get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().embed_per_minute)
    return _limiter


def _retry_delay(err: errors.APIError) -> float | None:
    """Seconds the API asks us to wait (RetryInfo.retryDelay, e.g. "55s"), if given."""
    try:
        for d in (err.details or {}).get("error", {}).get("details", []):
            if d.get("@type", "").endswith("RetryInfo"):
                return float(str(d.get("retryDelay", "")).rstrip("s"))
    except (AttributeError, TypeError, ValueError):
        pass
    return None


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
        # The SDK's own retry layer is disabled (attempts=1): retries are handled here, so a
        # slow request fails after one timeout instead of silently retrying for minutes.
        _client = genai.Client(api_key=s.gemini_api_key, http_options=types.HttpOptions(
            timeout=int(s.request_timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)))
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

def _embed_batch(batch: list[str], config: types.EmbedContentConfig,
                 on_wait: Callable[[float], None] | None = None) -> list[list[float]]:
    s = get_settings()
    client = get_client()
    for attempt in range(EMBED_ATTEMPTS):
        reserve = 8 if config.task_type == "RETRIEVAL_QUERY" else 0
        _get_limiter().acquire(len(batch), on_wait, reserve)
        delay = 1.0 * (2 ** attempt)
        try:
            started = time.perf_counter()
            result = client.models.embed_content(model=s.embedding_model, contents=batch, config=config)
            log.info("embedded %d texts in %.1fs", len(batch), time.perf_counter() - started)
            return [e.values for e in result.embeddings]
        except errors.APIError as e:
            log.warning("embedding error (attempt %d): %s", attempt + 1, str(e)[:120])
            if (e.code or 0) not in RETRYABLE or attempt == EMBED_ATTEMPTS - 1:
                if e.code == 429:
                    raise LLMError("The free-tier embedding limit (100 passages per minute) is used up. "
                                   "Wait a minute and try again.", 429) from e
                raise _friendly(e) from e
            if e.code == 429:
                # Quota hit anyway (e.g. other requests on the same key): wait as long as the API asks.
                delay = min(_retry_delay(e) or 30.0, 65.0) + 1
        except Exception as e:  # timeouts, connection resets
            log.warning("embedding request failed (attempt %d): %s", attempt + 1, type(e).__name__)
            if attempt == EMBED_ATTEMPTS - 1:
                raise LLMError("The AI service is responding too slowly right now. Please try again.", 503) from e
        if on_wait and delay > 3:
            on_wait(delay)
        time.sleep(delay)
    raise AssertionError("unreachable")


def embed(texts: list[str], task: str, on_progress: Callable[[int], None] | None = None,
          on_wait: Callable[[float], None] | None = None) -> np.ndarray:
    """Embed texts and return an L2-normalised float32 matrix (n, dim).

    task: "RETRIEVAL_DOCUMENT" for chunks, "RETRIEVAL_QUERY" for questions. Gemini embeds
    the two asymmetrically, which improves question -> passage matching.

    Texts are sent in small batches, several at a time, so one slow request doesn't hold
    up the whole document. `on_progress(n_done)` is called as batches finish, and
    `on_wait(seconds)` when we pause for the rate limit.
    """
    s = get_settings()
    if not texts:
        return np.zeros((0, s.embedding_dim), dtype=np.float32)
    config = types.EmbedContentConfig(task_type=task, output_dimensionality=s.embedding_dim)
    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    results: list[list[list[float]] | None] = [None] * len(batches)
    done = 0
    with ThreadPoolExecutor(max_workers=min(EMBED_WORKERS, len(batches))) as pool:
        futures = {pool.submit(_embed_batch, b, config, on_wait): i for i, b in enumerate(batches)}
        try:
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                done += len(batches[i])
                if on_progress:
                    on_progress(done)
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    matrix = np.asarray([v for batch in results for v in batch], dtype=np.float32)
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

    def call(model: str) -> T:
        started = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
        except errors.APIError as e:
            log.warning("generation error model=%s: %s", model, str(e)[:120])
            raise _Attempt(_friendly(e), retryable=(e.code or 0) in RETRYABLE | {404}) from e
        except Exception as e:  # timeout / connection
            log.warning("generation failed model=%s: %s", model, type(e).__name__)
            raise _Attempt(LLMError("The AI service is responding too slowly right now. Please try again.", 503),
                           retryable=True) from e
        log.info("model=%s latency=%.2fs", model, time.perf_counter() - started)
        if not response.text:
            raise _Attempt(LLMError("The AI returned no content (it may have been filtered). Try rephrasing.", 422),
                           retryable=False)
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as e:
            raise _Attempt(LLMError("The AI returned an unexpected format. Please try again.", 502),
                           retryable=True) from e

    # Hedged requests: start with the primary model; if it hasn't answered within HEDGE_AFTER_S
    # (or it fails), also start the next model, and use whichever valid answer arrives first.
    # Free-tier models are sometimes overloaded for minutes - this bounds the cost of a stuck
    # model to a few seconds instead of a full timeout.
    plan = list(dict.fromkeys([s.gemini_model, *s.fallback_models]))
    pending: set[Future] = set()
    last_error: LLMError | None = None
    next_model = 0

    def launch() -> None:
        nonlocal next_model
        pending.add(_POOL.submit(call, plan[next_model]))
        next_model += 1

    launch()
    while pending:
        done, _ = wait(pending, timeout=s.hedge_after_s, return_when=FIRST_COMPLETED)
        if not done:                                   # primary is slow: hedge with the next model
            if next_model < len(plan):
                log.info("no answer after %.0fs - also asking %s", s.hedge_after_s, plan[next_model])
                launch()
            continue
        for fut in done:
            pending.discard(fut)
            try:
                return fut.result()
            except _Attempt as a:
                last_error = a.error
                if not a.retryable:
                    raise a.error from None
        if next_model < len(plan):                     # a model failed: try the next one now
            launch()
    assert last_error is not None
    raise last_error


class _Attempt(Exception):
    def __init__(self, error: LLMError, retryable: bool) -> None:
        super().__init__(error.message)
        self.error = error
        self.retryable = retryable


_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gen")
