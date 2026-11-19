"""Thin typed client for the backend API (used by the Streamlit app)."""
import json
import os
from collections.abc import Iterator

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8003").rstrip("/")


class ApiError(Exception):
    pass


def _check(r: httpx.Response):
    if r.status_code >= 400:
        try:
            msg = r.json().get("error") or r.text
        except ValueError:
            msg = r.text or f"HTTP {r.status_code}"
        raise ApiError(msg)
    return None if r.status_code == 204 else r.json()


def _req(method: str, path: str, **kw):
    try:
        with httpx.Client(base_url=BACKEND_URL, timeout=kw.pop("timeout", 30)) as c:
            return _check(c.request(method, path, **kw))
    except httpx.HTTPError as e:
        raise ApiError(f"Can't reach the backend at {BACKEND_URL} ({type(e).__name__}).") from e


def health() -> dict:
    return _req("GET", "/api/health", timeout=5)


def documents() -> list[dict]:
    return _req("GET", "/api/documents")


def upload(name: str, data: bytes) -> dict:
    return _req("POST", "/api/documents", files={"file": (name, data)}, timeout=120)


def job(job_id: str) -> dict:
    return _req("GET", f"/api/jobs/{job_id}")


def delete(doc_id: str) -> None:
    _req("DELETE", f"/api/documents/{doc_id}")


def load_samples() -> list[dict]:
    return _req("POST", "/api/samples", timeout=300)


def chat_stream(question: str, history: list[dict], document_ids: list[str] | None, top_k: int) -> Iterator[dict]:
    """Yield server-sent events as dicts."""
    body = {"question": question, "history": history, "document_ids": document_ids, "top_k": top_k}
    try:
        with httpx.stream("POST", f"{BACKEND_URL}/api/chat/stream", json=body,
                          timeout=httpx.Timeout(300, connect=5)) as r:
            if r.status_code >= 400:
                r.read()
                _check(r)
            for line in r.iter_lines():
                if line.startswith("data:"):
                    yield json.loads(line[5:])
    except httpx.HTTPError as e:
        raise ApiError(f"Lost connection to the backend ({type(e).__name__}).") from e
