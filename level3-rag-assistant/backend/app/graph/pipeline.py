"""The question-answering workflow, built with LangGraph.

    rewrite ─► retrieve ─► rerank ─► select ─┬─► generate ─► check ─┬─► finalize ─► END
                                             │                ▲     │
                                             │                └─────┴── revise (≤ max_revisions)
                                             └─► no_answer ─► finalize

rewrite    resolve follow-ups using chat history; produce 2-3 search queries
retrieve   hybrid search (FAISS dense + BM25) for every query, fused with RRF
rerank     local cross-encoder scores (question, passage) pairs
select     keep passages above an absolute and a relative relevance floor; nothing
           left -> "not found" WITHOUT calling the generator (it never gets a chance
           to answer from memory)
generate   streamed, cited Markdown answer from the selected passages only
check      groundedness: deterministic citation checks + an LLM claim-by-claim audit
revise     if too many claims are unsupported, rewrite once with the problems listed
finalize   mark cited sources, compute a confidence score, assemble the response

Every node is timed and recorded in `trace`; nodes also emit progress events through
LangGraph's stream writer, which the API forwards to the UI as server-sent events.
"""
import logging
import operator
import re
import time
from collections.abc import Callable, Iterator
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from ..core.config import get_settings
from ..generation import prompts
from ..generation.gemini import LLMError
from ..generation.llm import generate_structured, stream_text
from ..retrieval.embeddings import QUERY, embed
from ..retrieval.hybrid import Candidate, hybrid_search
from ..retrieval.keyword import KeywordIndex
from ..retrieval.reranker import reranker
from ..retrieval.vector_store import VectorStore
from ..schemas import (ChatRequest, ChatResponse, ClaimVerdicts, Groundedness, QueryRewrite, Source,
                       TraceStep)

log = logging.getLogger("rag.pipeline")

NOT_FOUND_MSG = ("I couldn't find this in your documents. Try rephrasing, or check that the right "
                 "documents are selected.")
_CITE = re.compile(r"\[(\d+)\]")


class State(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    document_ids: list[str] | None
    top_k: int
    standalone_question: str
    search_queries: list[str]
    candidates: list[Candidate]
    context: list[Candidate]
    answer: str
    status: str
    revisions: int
    groundedness: Groundedness | None
    problems: list[str]
    trace: Annotated[list[TraceStep], operator.add]
    confidence: tuple[str, float]
    sources: list[Source]


def _emit(event: dict) -> None:
    try:
        get_stream_writer()(event)
    except Exception:  # not running inside a streaming graph (e.g. direct unit test)
        pass


def _node(name: str):
    """Decorator: emit start/end events and append a timed TraceStep."""
    def wrap(fn: Callable[["Pipeline", State], dict]):
        def run(self: "Pipeline", state: State) -> dict:
            _emit({"type": "step", "node": name, "status": "start"})
            t = time.perf_counter()
            update = fn(self, state) or {}
            ms = int((time.perf_counter() - t) * 1000)
            detail = update.pop("_detail", "")
            _emit({"type": "step", "node": name, "status": "end", "ms": ms, "detail": detail})
            update["trace"] = [TraceStep(node=name, ms=ms, detail=detail)]
            return update
        return run
    return wrap


def to_sources(context: list[Candidate], cited: set[int] | None = None) -> list[Source]:
    cited = cited or set()
    return [Source(n=i, chunk_id=c.chunk.id, document_id=c.chunk.document_id, document_name=c.document_name,
                   page=c.chunk.page, section=c.chunk.section, text=c.chunk.text,
                   rerank_score=None if c.rerank_score is None else round(c.rerank_score, 4),
                   dense_score=None if c.dense_score is None else round(c.dense_score, 4),
                   dense_rank=c.dense_rank, bm25_rank=c.bm25_rank, cited=i in cited)
            for i, c in enumerate(context, start=1)]


def _passages(context: list[Candidate]) -> str:
    return prompts.format_passages([(i, c.document_name, c.chunk.page, c.chunk.section, c.chunk.text)
                                    for i, c in enumerate(context, start=1)])


class Pipeline:
    def __init__(self, store: VectorStore, keyword: KeywordIndex) -> None:
        self.store = store
        self.keyword = keyword
        g = StateGraph(State)
        for name in ("rewrite", "retrieve", "rerank", "select", "generate", "check", "revise",
                     "no_answer", "finalize"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "select")
        g.add_conditional_edges("select", lambda s: "generate" if s.get("context") else "no_answer",
                                {"generate": "generate", "no_answer": "no_answer"})
        g.add_conditional_edges("generate", lambda s: "check" if s["status"] == "answered" else "finalize",
                                {"check": "check", "finalize": "finalize"})
        g.add_conditional_edges("check", self._after_check, {"revise": "revise", "finalize": "finalize"})
        g.add_edge("revise", "check")
        g.add_edge("no_answer", "finalize")
        g.add_edge("finalize", END)
        self.graph = g.compile()

    # ---------------- nodes ----------------

    @_node("rewrite")
    def rewrite(self, state: State) -> dict:
        q, history = state["question"], state.get("history") or []
        if not history:
            # Nothing to resolve: a first question is searched as asked. Hybrid search (dense +
            # BM25) already covers paraphrases and exact terms, so an LLM call here would add
            # seconds of latency (and extra embedding quota) for little gain.
            return {"standalone_question": q, "search_queries": [q], "_detail": "skipped (no conversation history)"}
        try:
            r = generate_structured(
                prompts.REWRITE_PROMPT.format(history=prompts.format_history(history), question=q),
                QueryRewrite, prompts.REWRITE_SYSTEM)
            standalone = r.standalone_question.strip() or q
            extra = [x.strip() for x in r.search_queries if x.strip()]
        except LLMError as e:          # rewriting is an optimisation - never fail the question on it
            log.warning("query rewrite failed, using the original question: %s", e.message)
            standalone, extra = q, []
        queries = list(dict.fromkeys([standalone, *extra]))[:3]
        return {"standalone_question": standalone, "search_queries": queries,
                "_detail": " | ".join(queries)}

    @_node("retrieve")
    def retrieve(self, state: State) -> dict:
        s = get_settings()
        queries = state["search_queries"]
        vectors = embed(queries, QUERY)
        cands = hybrid_search(self.store, self.keyword, queries, vectors, s.candidates_per_query,
                              s.rerank_pool, state.get("document_ids"))
        both = sum(1 for c in cands if c.dense_rank and c.bm25_rank)
        return {"candidates": cands,
                "_detail": f"{len(cands)} candidates from {len(queries)} queries ({both} found by both)"}

    @_node("rerank")
    def rerank(self, state: State) -> dict:
        cands = state["candidates"]
        scores = reranker.score(state["standalone_question"], [c.chunk.text for c in cands])
        if scores is None:
            return {"_detail": f"reranker {reranker.status}; kept fusion order"}
        for c, sc in zip(cands, scores):
            c.rerank_score = sc
        cands = sorted(cands, key=lambda c: c.rerank_score, reverse=True)
        best = cands[0].rerank_score if cands else 0
        return {"candidates": cands, "_detail": f"best relevance {best:.2f}"}

    @_node("select")
    def select(self, state: State) -> dict:
        s = get_settings()
        k = state.get("top_k") or s.top_k
        cands = state["candidates"]
        if cands and cands[0].rerank_score is not None:
            floor = max(s.rerank_min_score, cands[0].rerank_score * s.rerank_relative_floor)
            keep = [c for c in cands if c.rerank_score >= floor][:k]
            rule = f"relevance ≥ {floor:.3f}"
        else:
            keep = [c for c in cands if (c.dense_score or 0) >= s.min_dense_score or (c.bm25_rank or 99) <= 2][:k]
            rule = f"dense ≥ {s.min_dense_score} or top BM25"
        _emit({"type": "sources", "sources": [x.model_dump() for x in to_sources(keep)]})
        return {"context": keep, "_detail": f"{len(keep)} of {len(cands)} passages kept ({rule})"}

    @_node("generate")
    def generate(self, state: State) -> dict:
        prompt = prompts.ANSWER_PROMPT.format(passages=_passages(state["context"]),
                                              question=state["standalone_question"])
        buf, streaming = "", False
        for piece in stream_text(prompt, prompts.ANSWER_SYSTEM):
            buf += piece
            if not streaming and len(buf) >= 12:
                if buf.lstrip().startswith("NOT_FOUND"):
                    continue                     # keep reading silently; handled below
                streaming = True
                _emit({"type": "token", "text": buf})
            elif streaming:
                _emit({"type": "token", "text": piece})
        text = buf.strip()
        if text.startswith("NOT_FOUND"):
            reason = text.split(":", 1)[-1].strip()
            msg = "The documents don't answer this question." + (f" {reason}" if reason else "")
            _emit({"type": "token", "text": msg})
            return {"answer": msg, "status": "not_found", "_detail": "model reported not found"}
        if not streaming and text:
            _emit({"type": "token", "text": text})
        return {"answer": text, "status": "answered", "revisions": state.get("revisions", 0),
                "_detail": f"{len(text)} chars"}

    @_node("check")
    def check(self, state: State) -> dict:
        n = len(state["context"])
        answer = state["answer"]
        cited = [int(x) for x in _CITE.findall(answer)]
        invalid = sorted({x for x in cited if not 1 <= x <= n})
        sentences = [x for x in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(x.split()) >= 6]
        uncited = sum(1 for x in sentences if not _CITE.search(x))
        try:
            verdicts = generate_structured(
                prompts.CHECK_PROMPT.format(passages=_passages(state["context"]), answer=answer),
                ClaimVerdicts, prompts.CHECK_SYSTEM)
        except LLMError as e:
            log.warning("groundedness check unavailable: %s", e.message)
            return {"groundedness": None, "status": "unverified", "_detail": "check unavailable"}
        claims = verdicts.claims
        supported = sum(1 for c in claims if c.supported)
        score = supported / len(claims) if claims else 1.0
        if invalid:
            score = min(score, 0.5)
        g = Groundedness(score=round(score, 3), claims=claims, uncited_sentences=uncited, invalid_citations=invalid)
        problems = [c.claim for c in claims if not c.supported] + [f"citation [{x}] does not exist" for x in invalid]
        _emit({"type": "check", "groundedness": g.model_dump()})
        return {"groundedness": g, "problems": problems,
                "_detail": f"{supported}/{len(claims)} claims supported"}

    def _after_check(self, state: State) -> str:
        s = get_settings()
        g = state.get("groundedness")
        if g is not None and g.score < s.groundedness_threshold and state.get("revisions", 0) < s.max_revisions:
            return "revise"
        return "finalize"

    @_node("revise")
    def revise(self, state: State) -> dict:
        prompt = prompts.REVISE_PROMPT.format(
            passages=_passages(state["context"]), question=state["standalone_question"],
            problems="\n".join(f"- {p}" for p in state.get("problems", [])))
        try:
            text = "".join(stream_text(prompt, prompts.ANSWER_SYSTEM, temperature=0.0)).strip()
        except LLMError as e:
            log.warning("revision failed, keeping the draft: %s", e.message)
            return {"revisions": state.get("revisions", 0) + 1, "_detail": "revision failed; draft kept"}
        _emit({"type": "revision", "answer": text})
        return {"answer": text, "status": "revised", "revisions": state.get("revisions", 0) + 1,
                "_detail": f"{len(state.get('problems', []))} problems fixed"}

    @_node("no_answer")
    def no_answer(self, state: State) -> dict:
        _emit({"type": "token", "text": NOT_FOUND_MSG})
        return {"answer": NOT_FOUND_MSG, "status": "not_found", "_detail": "no passage passed the relevance floor"}

    @_node("finalize")
    def finalize(self, state: State) -> dict:
        context = state.get("context") or []
        status = state["status"]
        g = state.get("groundedness")
        cited = {int(x) for x in _CITE.findall(state["answer"])} if status != "not_found" else set()
        sources = to_sources(context, cited) if status != "not_found" else \
            to_sources(state.get("candidates", [])[:3])

        if status == "not_found":
            score, level = 0.0, "none"
        else:
            # retrieval strength: best reranker relevance (or a dense-score proxy without a reranker)
            top = context[0]
            if top.rerank_score is not None:
                retrieval = min(1.0, top.rerank_score / 0.5)
            else:
                retrieval = max(0.0, min(1.0, ((top.dense_score or 0) - 0.45) / 0.35))
            grounded = g.score if g is not None else 0.5
            score = round(0.6 * grounded + 0.4 * retrieval, 3)
            if not cited:
                score = min(score, 0.3)
            level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
            if status == "revised" and level == "high":
                level = "medium"
        return {"_detail": f"{level} ({score:.2f})", "confidence": (level, score), "sources": sources}

    # ---------------- entry points ----------------

    def _initial(self, req: ChatRequest) -> State:
        s = get_settings()
        hist = [(t.role, t.content) for t in req.history][-2 * s.history_turns:] if s.history_turns else []
        return {"question": req.question, "history": hist, "document_ids": req.document_ids,
                "top_k": req.top_k or s.top_k, "revisions": 0, "trace": []}

    def _response(self, req: ChatRequest, state: dict, total_ms: int) -> ChatResponse:
        level, score = state.get("confidence", ("none", 0.0))
        return ChatResponse(
            question=req.question,
            standalone_question=state.get("standalone_question", req.question),
            search_queries=state.get("search_queries", [req.question]),
            answer=state.get("answer", ""), status=state.get("status", "not_found"),
            confidence=level, confidence_score=score, groundedness=state.get("groundedness"),
            sources=state.get("sources", []), trace=state.get("trace", []), total_ms=total_ms)

    def run(self, req: ChatRequest) -> ChatResponse:
        t = time.perf_counter()
        state = self.graph.invoke(self._initial(req))
        return self._response(req, state, int((time.perf_counter() - t) * 1000))

    def stream(self, req: ChatRequest) -> Iterator[dict[str, Any]]:
        """Yield progress events, then {"type": "done", "response": ChatResponse}."""
        t = time.perf_counter()
        final: dict = {}
        for mode, chunk in self.graph.stream(self._initial(req), stream_mode=["custom", "values"]):
            if mode == "custom":
                yield chunk
            else:
                final = chunk
        resp = self._response(req, final, int((time.perf_counter() - t) * 1000))
        yield {"type": "done", "response": resp.model_dump()}
