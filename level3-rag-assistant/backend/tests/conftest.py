"""Test fixtures: an isolated app with Gemini and the reranker replaced by deterministic fakes."""
import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings

SAMPLES = Path(__file__).resolve().parents[2] / "samples"
DIM = get_settings().vector_dim


def fake_embed(texts, task, on_progress=None, on_wait=None):
    """Bag-of-words hashing embedding: texts sharing words get similar vectors."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) > 3:
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
    out += 1e-6
    if on_progress:
        on_progress(len(texts))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def fake_rerank(query, passages):
    """Word-overlap relevance in 0-1."""
    q = {w for w in query.lower().split() if len(w) > 3}
    return [min(1.0, len(q & set(p.lower().split())) / max(1, len(q))) for p in passages]


class FakeLLM:
    """Scriptable stand-in for structured + streaming generation."""

    def __init__(self):
        self.answer = "Late work loses 10% per day [1]."
        self.revised = "Late work loses 10% per day after the grace days [1]."
        self.claims = [{"claim": "Late work loses 10% per day", "supported": True, "sources": [1]}]
        self.rewrite = {"standalone_question": None, "search_queries": ["late penalty"]}
        self.fail_rewrite = False
        self.calls: list[str] = []

    def structured(self, prompt, schema, system, temperature=0.1):
        from app.generation.gemini import LLMError
        name = schema.__name__
        self.calls.append(name)
        if name == "QueryRewrite":
            if self.fail_rewrite:
                raise LLMError("busy", 503)
            q = self.rewrite["standalone_question"] or prompt.split("Latest user message: ")[1].split("\n")[0]
            return schema(standalone_question=q, search_queries=self.rewrite["search_queries"])
        if name == "ClaimVerdicts":
            claims = self.claims if self.calls.count("ClaimVerdicts") == 1 else \
                [{"claim": "fixed", "supported": True, "sources": [1]}]
            return schema(claims=claims)
        raise AssertionError(name)

    def stream(self, prompt, system, temperature=0.2):
        text = self.revised if "previous draft" in prompt else self.answer
        self.calls.append("stream")
        for i in range(0, len(text), 7):
            yield text[i:i + 7]


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(tmp_path, monkeypatch, llm):
    from app import main
    from app.graph import pipeline
    from app.retrieval import reranker as rr
    from app.services import documents

    s = get_settings()
    monkeypatch.setattr(s, "data_dir", tmp_path)
    monkeypatch.setattr(s, "rerank_min_score", 0.2)
    monkeypatch.setattr(pipeline, "embed", fake_embed)
    monkeypatch.setattr(documents, "embed", fake_embed)
    monkeypatch.setattr(pipeline, "generate_structured", llm.structured)
    monkeypatch.setattr(pipeline, "stream_text", llm.stream)
    monkeypatch.setattr(rr.reranker, "score", fake_rerank)
    monkeypatch.setattr(rr.reranker, "load", lambda: True)

    app = main.create_app()
    app.state.container = main.build_container()
    with TestClient(app) as c:
        yield c


def upload(client, name: str, data: bytes) -> dict:
    r = client.post("/api/documents", files={"file": (name, data)})
    assert r.status_code == 202, r.text
    job = r.json()
    for _ in range(300):
        if job["status"] != "processing":
            return job
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    raise AssertionError("ingest job did not finish")


@pytest.fixture
def handbook(client):
    return upload(client, "course-handbook.md", (SAMPLES / "course-handbook.md").read_bytes())["document"]
