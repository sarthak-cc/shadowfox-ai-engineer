"""Tests for ingestion, retrieval and the API. Gemini calls are mocked."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import ingest, main, qa
from app.schemas import GroundedAnswer

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
DIM = 768


def fake_embed(texts, task):
    """Deterministic bag-of-words embedding: similar words -> similar vectors."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) > 3:
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
    out += 1e-6
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "data_dir", tmp_path)
    monkeypatch.setattr(main.settings, "min_score", 0.15)
    monkeypatch.setattr(main, "embed", fake_embed)
    monkeypatch.setattr(qa, "embed", fake_embed)
    main.store.clear()
    yield TestClient(main.app)
    main.store.clear()


def upload(client, name, data: bytes):
    return client.post("/api/documents", files={"file": (name, data)})


# ---------------- ingestion ----------------

def test_pdf_extraction_keeps_pages():
    pages = ingest.extract_pages((SAMPLES / "os-lecture-cpu-scheduling.pdf").read_bytes(), "pdf")
    assert [p.number for p in pages] == [1, 2, 3]
    assert "convoy effect" in pages[1].text


def test_chunks_respect_size_and_pages():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(200))
    chunks = ingest.chunk_pages([ingest.Page(1, text), ingest.Page(2, text)], size=500, overlap=100)
    assert all(len(c.text) <= 500 * 1.3 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    # consecutive chunks on a page overlap
    first, second = chunks[0].text, chunks[1].text
    assert first.split(". ")[-1].rstrip(".") in second


def test_markdown_is_cleaned():
    pages = ingest.extract_pages(b"# Title\n\nSome **bold** text with a [link](http://x.y) here.", "md")
    assert pages[0].text.startswith("Title")
    assert "**" not in pages[0].text and "http" not in pages[0].text


@pytest.mark.parametrize("name, data, fragment", [
    ("notes.docx", b"hello", "Unsupported file type"),
    ("empty.txt", b"", "empty"),
    ("fake.pdf", b"not a pdf at all", "not a valid PDF"),
    ("tiny.txt", b"hi", "too little text"),
])
def test_bad_files_rejected(client, name, data, fragment):
    r = upload(client, name, data)
    assert r.status_code == 422
    assert fragment in r.json()["error"]


# ---------------- API flow ----------------

def test_upload_list_duplicate_delete(client):
    data = (SAMPLES / "course-handbook.md").read_bytes()
    r = upload(client, "course-handbook.md", data)
    assert r.status_code == 201
    doc = r.json()
    assert doc["kind"] == "md" and doc["chunks"] >= 1

    assert [d["id"] for d in client.get("/api/documents").json()] == [doc["id"]]
    assert upload(client, "copy.md", data).status_code == 409          # duplicate content

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get("/api/documents").json() == []


def test_ask_requires_documents_and_question(client):
    assert client.post("/api/ask", json={"question": "What is the late policy?"}).status_code == 400
    upload(client, "course-handbook.md", (SAMPLES / "course-handbook.md").read_bytes())
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 422 and "empty" in r.json()["error"]


def test_grounded_answer_with_citation_cleanup(client, monkeypatch):
    upload(client, "course-handbook.md", (SAMPLES / "course-handbook.md").read_bytes())
    captured = {}

    def fake_generate(prompt, schema, system, temperature=0.2):
        captured["prompt"] = prompt
        return GroundedAnswer(found=True, answer="Late work loses 10% per day [1]. Also [9].",
                              cited=[1, 9], confidence="high")

    monkeypatch.setattr(qa, "generate_structured", fake_generate)
    r = client.post("/api/ask", json={"question": "What is the penalty for late submission grace days?"})
    body = r.json()
    assert r.status_code == 200 and body["found"] is True
    assert "[9]" not in body["answer"]                 # invalid citation removed
    assert body["sources"][0]["cited"] is True
    assert "grace days" in captured["prompt"].lower()  # retrieved text was put in the prompt


def test_irrelevant_question_skips_llm(client, monkeypatch):
    upload(client, "course-handbook.md", (SAMPLES / "course-handbook.md").read_bytes())
    monkeypatch.setattr(main.settings, "min_score", 0.99)

    def boom(*a, **k):
        raise AssertionError("LLM must not be called when nothing relevant is retrieved")

    monkeypatch.setattr(qa, "generate_structured", boom)
    body = client.post("/api/ask", json={"question": "Who won the football world cup?"}).json()
    assert body["found"] is False and body["confidence"] == "none"


def test_document_scope_filter(client, monkeypatch):
    a = upload(client, "course-handbook.md", (SAMPLES / "course-handbook.md").read_bytes()).json()
    b = upload(client, "os.pdf", (SAMPLES / "os-lecture-cpu-scheduling.pdf").read_bytes()).json()
    monkeypatch.setattr(qa, "generate_structured",
                        lambda *a, **k: GroundedAnswer(found=True, answer="x [1]", cited=[1], confidence="medium"))
    body = client.post("/api/ask", json={"question": "scheduling quantum round robin",
                                         "document_ids": [a["id"]]}).json()
    assert {s["document_id"] for s in body["sources"]} <= {a["id"]}
    assert b["id"] not in {s["document_id"] for s in body["sources"]}
