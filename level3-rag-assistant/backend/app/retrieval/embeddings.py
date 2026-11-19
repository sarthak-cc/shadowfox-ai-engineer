"""Gemini embeddings: batched, parallel, rate-limited and L2-normalised."""
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from google.genai import errors, types

from ..core.config import get_settings
from ..generation.gemini import RETRYABLE, LLMError, friendly, get_client, retry_delay

log = logging.getLogger("rag.embeddings")

BATCH = 32
WORKERS = 4
ATTEMPTS = 4

DOC = "RETRIEVAL_DOCUMENT"
QUERY = "RETRIEVAL_QUERY"


class RateLimiter:
    """Sliding one-minute window over embedded texts (free tier: 100/min, each text counts).

    Waiting before sending keeps us under the quota instead of collecting 429s. `reserve`
    lets a user's question use a few slots above the normal budget so it never queues
    behind a large upload.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    def acquire(self, n: int, on_wait: Callable[[float], None] | None = None, reserve: int = 0) -> None:
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
                freed, wait = 0, 1.0
                for ts, k in self.events:
                    freed += k
                    if used - freed + n <= limit:
                        wait = 60 - (now - ts) + 0.2
                        break
            if on_wait:
                on_wait(wait)
            time.sleep(min(wait, 5))


_limiter: RateLimiter | None = None


def limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().embed_per_minute)
    return _limiter


def _embed_batch(batch: list[str], task: str, on_wait: Callable[[float], None] | None) -> list[list[float]]:
    s = get_settings()
    config = types.EmbedContentConfig(task_type=task, output_dimensionality=s.embedding_dim)
    for attempt in range(ATTEMPTS):
        limiter().acquire(len(batch), on_wait, reserve=10 if task == QUERY else 0)
        delay = 1.0 * (2 ** attempt)
        try:
            t = time.perf_counter()
            result = get_client().models.embed_content(model=s.embedding_model, contents=batch, config=config)
            log.info("embedded %d texts in %.1fs", len(batch), time.perf_counter() - t)
            return [e.values for e in result.embeddings]
        except errors.APIError as e:
            log.warning("embedding error (attempt %d): %s", attempt + 1, str(e)[:120])
            if e.code == 429 and "PerDay" in str(e):
                # Daily cap (1,000 texts/day on the free tier): retrying is pointless until it resets.
                raise LLMError("The daily free-tier embedding limit (1,000 passages per day) has been reached. "
                               "It resets at midnight Pacific time - or set EMBEDDING_PROVIDER=local to embed "
                               "on this machine without limits.", 429) from e
            if (e.code or 0) not in RETRYABLE or attempt == ATTEMPTS - 1:
                if e.code == 429:
                    raise LLMError("The free-tier embedding limit (100 passages per minute) is used up. "
                                   "Wait a minute and try again.", 429) from e
                raise friendly(e) from e
            if e.code == 429:
                delay = min(retry_delay(e) or 30.0, 65.0) + 1
        except LLMError:
            raise
        except Exception as e:
            log.warning("embedding request failed (attempt %d): %s", attempt + 1, type(e).__name__)
            if attempt == ATTEMPTS - 1:
                raise LLMError("The AI service is responding too slowly right now. Please try again.", 503) from e
        if on_wait and delay > 3:
            on_wait(delay)
        time.sleep(delay)
    raise AssertionError("unreachable")


# ---------------- local provider (fastembed, ONNX - no PyTorch, no quota) ----------------

_local_model = None
_local_lock = threading.Lock()


def _local_embed(texts: list[str], task: str, on_progress: Callable[[int], None] | None) -> np.ndarray:
    global _local_model
    s = get_settings()
    with _local_lock:
        if _local_model is None:
            try:
                from fastembed import TextEmbedding
                _local_model = TextEmbedding(s.local_embedding_model, cache_dir=str(s.model_cache_dir))
            except Exception as e:
                raise LLMError(f"Could not load the local embedding model ({type(e).__name__}).", 503) from e
    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        # bge models expect an instruction prefix on queries; fastembed's query_embed adds it.
        it = _local_model.query_embed(batch) if task == QUERY else _local_model.passage_embed(batch)
        vectors.extend(it)
        if on_progress:
            on_progress(min(i + 64, len(texts)))
    return np.asarray(vectors, dtype=np.float32)


def embed(texts: list[str], task: str, on_progress: Callable[[int], None] | None = None,
          on_wait: Callable[[float], None] | None = None) -> np.ndarray:
    """Return an L2-normalised float32 matrix (len(texts), dim).

    `task` is RETRIEVAL_DOCUMENT for chunks and RETRIEVAL_QUERY for questions - both
    providers embed them asymmetrically, which helps short questions match longer passages.
    """
    s = get_settings()
    if not texts:
        return np.zeros((0, s.vector_dim), dtype=np.float32)
    if s.embedding_provider == "local":
        m = _local_embed(texts, task, on_progress)
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
    batches =[texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
    results: list[list[list[float]] | None] = [None] * len(batches)
    done = 0
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(batches))) as pool:
        futures = {pool.submit(_embed_batch, b, task, on_wait): i for i, b in enumerate(batches)}
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
    m = np.asarray([v for batch in results for v in batch], dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
