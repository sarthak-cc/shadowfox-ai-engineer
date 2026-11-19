import numpy as np

from app.retrieval.embeddings import RateLimiter
from app.retrieval.hybrid import hybrid_search
from app.retrieval.keyword import KeywordIndex, tokenize
from app.retrieval.vector_store import VectorStore
from app.schemas import DocumentInfo

from .conftest import DIM, fake_embed


def _doc(i: str) -> DocumentInfo:
    return DocumentInfo(id=i, name=f"{i}.md", kind="md", pages=1, chunks=0, characters=0, uploaded_at="2026-01-01")


def _store(tmp_path) -> VectorStore:
    s = VectorStore(DIM, tmp_path)
    a = ["Round robin uses a time quantum", "The convoy effect slows FCFS scheduling"]
    b = ["Late work loses ten percent per day", "Grace days cannot be used for quizzes"]
    s.add_document(_doc("a"), "ha", [(1, None, t) for t in a], fake_embed(a, "D"))
    s.add_document(_doc("b"), "hb", [(1, None, t) for t in b], fake_embed(b, "D"))
    return s


def test_faiss_search_scope_remove_and_persist(tmp_path):
    s = _store(tmp_path)
    q = fake_embed(["convoy effect scheduling"], "Q")
    assert s.chunks[s.search(q, 1)[0][0][0]].text.startswith("The convoy")
    scoped = s.search(q, 4, ["b"])[0]
    assert scoped and all(s.chunks[cid].document_id == "b" for cid, _ in scoped)

    assert s.remove_document("a") and s.index.ntotal == 2
    reloaded = VectorStore(DIM, tmp_path)
    reloaded.load()
    assert reloaded.index.ntotal == 2 and set(reloaded.documents) == {"b"}


def test_bm25_finds_exact_terms_and_respects_scope(tmp_path):
    s = _store(tmp_path)
    kw = KeywordIndex(s)
    hits = kw.search("quizzes grace", 3)
    assert s.chunks[hits[0][0]].text.startswith("Grace days")
    assert kw.search("quizzes grace", 3, ["a"]) == []
    assert "explain" not in tokenize("Explain the CI/CD pipeline")


def test_hybrid_fusion_rewards_agreement(tmp_path):
    s = _store(tmp_path)
    queries = ["grace days quizzes", "late work penalty"]
    cands = hybrid_search(s, KeywordIndex(s), queries, fake_embed(queries, "Q"), k_each=4, pool=4)
    top = cands[0]
    assert top.chunk.document_id == "b"
    assert top.dense_rank and top.bm25_rank          # found by both retrievers
    assert [c.fused for c in cands] == sorted([c.fused for c in cands], reverse=True)


def test_rate_limiter_reserve_and_wait(monkeypatch):
    from app.retrieval import embeddings
    clock = {"t": 0.0}
    monkeypatch.setattr(embeddings.time, "monotonic", lambda: clock["t"])
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", lambda x: (slept.append(x), clock.__setitem__("t", clock["t"] + x)))
    rl = RateLimiter(90)
    rl.acquire(90)
    rl.acquire(5, reserve=10)          # a question still fits
    assert slept == []
    rl.acquire(32)                      # an upload batch must wait for the window
    assert sum(slept) >= 59


def test_store_rejects_wrong_shape(tmp_path):
    s = VectorStore(DIM, tmp_path)
    try:
        s.add_document(_doc("x"), "h", [(1, None, "t")], np.zeros((1, 3), dtype=np.float32))
    except AssertionError:
        return
    raise AssertionError("shape mismatch was accepted")


def test_index_from_another_embedding_model_is_not_reused(tmp_path):
    s = VectorStore(DIM, tmp_path, "gemini:gemini-embedding-001:768")
    s.add_document(_doc("a"), "h", [(1, None, "some text")], fake_embed(["some text"], "D"))
    other = VectorStore(DIM, tmp_path, "local:BAAI/bge-small-en-v1.5")
    other.load()
    assert other.index.ntotal == 0 and not other.documents
