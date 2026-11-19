import json

from .conftest import SAMPLES, upload

LATE_Q = "What is the penalty for late submission after grace days?"


def ask(client, question, **kw):
    r = client.post("/api/chat", json={"question": question, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_grounded_answer_end_to_end(client, handbook, llm):
    r = ask(client, LATE_Q)
    assert r["status"] == "answered"
    assert r["answer"].startswith("Late work loses 10% per day")
    assert r["groundedness"]["score"] == 1.0
    assert r["sources"][0]["cited"] and r["sources"][0]["document_name"] == "course-handbook.md"
    assert r["confidence"] in {"high", "medium"}
    assert [t["node"] for t in r["trace"]] == ["rewrite", "retrieve", "rerank", "select", "generate", "check", "finalize"]
    assert "QueryRewrite" not in llm.calls          # no history -> rewrite skipped


def test_follow_up_is_rewritten_with_history(client, handbook, llm):
    llm.rewrite = {"standalone_question": "Can grace days be used for quizzes?", "search_queries": ["grace days quizzes"]}
    r = ask(client, "Can I use them for quizzes?", history=[
        {"role": "user", "content": "What are grace days?"}, {"role": "assistant", "content": "Three free late days [1]."}])
    assert r["standalone_question"] == "Can grace days be used for quizzes?"
    assert "grace days quizzes" in r["search_queries"]


def test_rewrite_failure_falls_back_to_original_question(client, handbook, llm):
    llm.fail_rewrite = True
    r = ask(client, LATE_Q, history=[{"role": "user", "content": "hi there"}])
    assert r["standalone_question"] == LATE_Q and r["status"] == "answered"


def test_irrelevant_question_never_reaches_the_generator(client, handbook, llm):
    r = ask(client, "Who won the football world cup in Qatar?")
    assert r["status"] == "not_found" and r["confidence"] == "none"
    assert "stream" not in llm.calls
    assert [t["node"] for t in r["trace"]][-2:] == ["no_answer", "finalize"]


def test_model_not_found_reply(client, handbook, llm):
    llm.answer = "NOT_FOUND: The documents cover grading, not cafeteria menus."
    r = ask(client, LATE_Q)
    assert r["status"] == "not_found" and "cafeteria" in r["answer"]
    assert not any(s["cited"] for s in r["sources"])


def test_unsupported_claims_trigger_one_revision(client, handbook, llm):
    llm.claims = [{"claim": "Late work loses 10% per day", "supported": True, "sources": [1]},
                  {"claim": "Late work is accepted for 2 weeks", "supported": False, "sources": []}]
    r = ask(client, LATE_Q)
    assert r["status"] == "revised"
    assert r["answer"].endswith("after the grace days [1].")
    assert [t["node"] for t in r["trace"]].count("check") == 2
    assert r["confidence"] != "high"


def test_invalid_citations_lower_groundedness(client, handbook, llm):
    llm.answer = "Late work loses 10% per day [7]."
    r = ask(client, LATE_Q)
    first_check = r["trace"][[t["node"] for t in r["trace"]].index("check")]
    assert first_check and r["status"] == "revised"          # score capped -> revision


def test_stream_event_order(client, handbook):
    with client.stream("POST", "/api/chat/stream", json={"question": LATE_Q}) as r:
        events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    types = [e["type"] for e in events]
    assert types.index("sources") < types.index("token") < types.index("check") < types.index("done")
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == events[-1]["response"]["answer"]


def test_document_scope(client, handbook):
    pdf = upload(client, "os.pdf", (SAMPLES / "os-lecture-cpu-scheduling.pdf").read_bytes())["document"]
    r = ask(client, LATE_Q, document_ids=[pdf["id"]])
    assert all(s["document_id"] == pdf["id"] for s in r["sources"])


def test_api_validation_and_errors(client, handbook):
    assert client.post("/api/chat", json={"question": "  "}).json()["error"] == "Question cannot be empty."
    assert client.post("/api/chat", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/api/chat", json={"question": "valid question", "document_ids": ["nope"]}).status_code == 400
    dup = client.post("/api/documents", files={"file": ("copy.md", (SAMPLES / "course-handbook.md").read_bytes())})
    assert dup.status_code == 409 and "already indexed" in dup.json()["error"]
    assert client.post("/api/documents", files={"file": ("x.exe", b"MZ")}).status_code == 422
    assert client.get("/api/jobs/missing").status_code == 404


def test_no_documents(client):
    r = client.post("/api/chat", json={"question": "anything at all?"})
    assert r.status_code == 400 and "Upload at least one document" in r.json()["error"]


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["index"]["type"] == "FAISS IndexFlatIP"
