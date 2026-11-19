"""Structured and streaming generation with hedged requests.

Hedging: free-tier models are sometimes overloaded for minutes (503s or 20-40 s
responses). Each call starts on the preferred model; if it has not responded within
`hedge_after_s` (or fails), the next model is started too and the first good result
wins. For streaming, "responded" means "produced its first token" - once a stream has
started, it is the one the user sees.
"""
import logging
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from ..core.config import get_settings
from .gemini import RETRYABLE, LLMError, friendly, get_client

log = logging.getLogger("rag.llm")
T = TypeVar("T", bound=BaseModel)

_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="llm")


class _Attempt(Exception):
    def __init__(self, error: LLMError, retryable: bool) -> None:
        super().__init__(error.message)
        self.error, self.retryable = error, retryable


def _max_wait() -> float:
    """Hard ceiling for one hedged call: every model gets its start slot plus one full timeout."""
    s = get_settings()
    return s.hedge_after_s * (len(_models()) + 1) + s.request_timeout_s


def _models() -> list[str]:
    s = get_settings()
    return list(dict.fromkeys([s.gemini_model, *s.gemini_fallback_models]))


def _as_attempt(e: Exception, model: str) -> _Attempt:
    if isinstance(e, _Attempt):
        return e
    if isinstance(e, errors.APIError):
        log.warning("model=%s error: %s", model, str(e)[:120])
        return _Attempt(friendly(e), retryable=(e.code or 0) in RETRYABLE | {404})
    log.warning("model=%s failed: %s", model, type(e).__name__)
    return _Attempt(LLMError("The AI service is responding too slowly right now. Please try again.", 503), True)


# ---------------- structured ----------------

def generate_structured(prompt: str, schema: type[T], system: str, temperature: float = 0.1) -> T:
    s = get_settings()
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                         response_mime_type="application/json", response_schema=schema)

    def call(model: str) -> T:
        t = time.perf_counter()
        try:
            r = get_client().models.generate_content(model=model, contents=prompt, config=config)
        except Exception as e:
            raise _as_attempt(e, model) from e
        log.info("structured model=%s %.2fs", model, time.perf_counter() - t)
        if not r.text:
            raise _Attempt(LLMError("The AI returned no content (it may have been filtered).", 422), False)
        try:
            return schema.model_validate_json(r.text)
        except ValidationError as e:
            raise _Attempt(LLMError("The AI returned an unexpected format.", 502), True) from e

    plan, nxt = _models(), 0
    pending: set[Future] = set()
    last: LLMError | None = None

    def launch() -> None:
        nonlocal nxt
        pending.add(_POOL.submit(call, plan[nxt]))
        nxt += 1

    launch()
    give_up = time.monotonic() + _max_wait()
    while pending:
        done, _ = wait(pending, timeout=s.hedge_after_s, return_when=FIRST_COMPLETED)
        if not done:
            if nxt < len(plan):
                log.info("hedging: also asking %s", plan[nxt])
                launch()
            elif time.monotonic() > give_up:      # every model is hung - don't wait forever
                raise LLMError("The AI service is too slow right now. Please try again in a minute.", 503)
            continue
        for fut in done:
            pending.discard(fut)
            try:
                return fut.result()
            except _Attempt as a:
                last = a.error
                if not a.retryable:
                    raise a.error from None
        if nxt < len(plan):
            launch()
    raise last or LLMError("The AI service is unavailable.", 503)


# ---------------- streaming ----------------

_END = object()


def stream_text(prompt: str, system: str, temperature: float = 0.2) -> Iterator[str]:
    """Yield answer text chunks. Hedges on time-to-first-token across models."""
    s = get_settings()
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    plan, nxt = _models(), 0
    out: queue.Queue = queue.Queue()
    winner: dict[str, str | None] = {"model": None}
    lock = threading.Lock()
    failures: list[_Attempt] = []
    started: list[str] = []

    def run(model: str) -> None:
        try:
            for chunk in get_client().models.generate_content_stream(model=model, contents=prompt, config=config):
                text = chunk.text or ""
                if not text:
                    continue
                with lock:
                    if winner["model"] is None:
                        winner["model"] = model
                        log.info("stream won by %s", model)
                    if winner["model"] != model:
                        return               # another model is already streaming
                out.put((model, text))
            out.put((model, _END))
        except Exception as e:
            out.put((model, _as_attempt(e, model)))

    def launch() -> None:
        nonlocal nxt
        started.append(plan[nxt])
        _POOL.submit(run, plan[nxt])
        nxt += 1

    launch()
    deadline = time.monotonic() + s.hedge_after_s
    give_up = time.monotonic() + _max_wait()
    while True:
        timeout = None if winner["model"] else max(0.05, deadline - time.monotonic())
        try:
            model, item = out.get(timeout=timeout if winner["model"] is None else s.request_timeout_s)
        except queue.Empty:
            if winner["model"] is None:
                if nxt < len(plan):
                    log.info("no first token after %.0fs - also starting %s", s.hedge_after_s, plan[nxt])
                    launch()
                    deadline = time.monotonic() + s.hedge_after_s
                elif time.monotonic() > give_up:  # no model has produced a token: stop waiting
                    raise LLMError("The AI service is too slow right now. Please try again in a minute.", 503)
                else:
                    deadline = time.monotonic() + s.hedge_after_s
                continue
            raise LLMError("The answer stream stalled. Please try again.", 503)

        if isinstance(item, _Attempt):
            if winner["model"] == model:          # failed mid-stream: nothing sensible to recover
                raise item.error
            failures.append(item)
            if not item.retryable:
                raise item.error
            if nxt < len(plan):
                launch()
                deadline = time.monotonic() + s.hedge_after_s
            elif len(failures) >= len(started):
                raise failures[-1].error
            continue
        if model != winner["model"]:
            continue
        if item is _END:
            return
        yield item
