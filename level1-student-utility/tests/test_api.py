"""API tests. The LLM call is mocked so tests are fast, free and deterministic."""
import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from app import llm, main
from app.schemas import QuizQuestion, QuizResult, SummaryResult, KeyTerm

client = TestClient(main.app)

NOTES = " ".join(["Photosynthesis converts light energy into chemical energy in plants."] * 8)


@pytest.fixture
def fake_llm(monkeypatch):
    def install(result=None, exc=None):
        def _fake(prompt, schema, temperature=0.4):
            if exc:
                raise exc
            return result
        monkeypatch.setattr(main, "generate_structured", _fake)
    return install


# ---------- validation ----------

@pytest.mark.parametrize("payload, fragment", [
    ({"notes": ""}, "cannot be empty"),
    ({"notes": "   "}, "cannot be empty"),
    ({"notes": "too short"}, "too short"),
    ({"notes": "1234 5678 " * 40}, "readable text"),
    ({"notes": "a " * 7000}, "too long"),
])
def test_summarize_rejects_bad_input(payload, fragment):
    r = client.post("/api/summarize", json=payload)
    assert r.status_code == 422
    assert fragment in r.json()["error"]


def test_quiz_rejects_out_of_range_count():
    r = client.post("/api/quiz", json={"notes": NOTES, "num_questions": 50})
    assert r.status_code == 422


def test_improve_requires_both_fields():
    r = client.post("/api/improve", json={"question": "Why is the sky blue?", "answer": ""})
    assert r.status_code == 422
    assert "Answer" in r.json()["error"]


# ---------- happy paths ----------

def test_summarize_ok(fake_llm):
    fake_llm(SummaryResult(title="T", tldr="x", key_points=["a"], key_terms=[KeyTerm(term="t", definition="d")],
                           revision_tip="tip"))
    r = client.post("/api/summarize", json={"notes": NOTES})
    assert r.status_code == 200
    assert r.json()["title"] == "T"


def test_quiz_drops_malformed_questions(fake_llm):
    good = QuizQuestion(question="q", options=["a", "b", "c", "d"], correct_index=1, explanation="e")
    bad = QuizQuestion(question="q", options=["a", "b"], correct_index=5, explanation="e")
    fake_llm(QuizResult(title="Quiz", questions=[good, bad, good]))
    r = client.post("/api/quiz", json={"notes": NOTES, "num_questions": 3})
    assert r.status_code == 200
    assert len(r.json()["questions"]) == 2


# ---------- error handling ----------

def test_llm_error_is_returned_as_friendly_json(fake_llm):
    fake_llm(exc=llm.LLMError("The AI service is busy.", 429))
    r = client.post("/api/summarize", json={"notes": NOTES})
    assert r.status_code == 429
    assert r.json() == {"error": "The AI service is busy."}


@pytest.mark.parametrize("code, status", [(401, 500), (404, 500), (429, 429), (503, 503)])
def test_api_errors_are_mapped(code, status):
    err = errors.APIError(code, {"error": {"message": "x", "status": "X"}})
    assert llm._friendly(err).status_code == status


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
