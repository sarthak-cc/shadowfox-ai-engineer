"""Request / response models shared by the API, the pipeline and the frontend."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DocKind = Literal["pdf", "txt", "md"]


# ---------------- documents ----------------

class DocumentInfo(BaseModel):
    id: str
    name: str
    kind: DocKind
    pages: int
    chunks: int
    characters: int
    sections: int = Field(0, description="Distinct headings found (Markdown / numbered questions)")
    uploaded_at: str


class IngestJob(BaseModel):
    id: str
    name: str
    status: Literal["processing", "done", "error"]
    stage: Literal["parsing", "embedding", "waiting", "indexing", "done", "error"]
    pages: int = 0
    done: int = 0
    total: int = 0
    wait_s: float | None = None
    elapsed_s: float | None = None
    document: DocumentInfo | None = None
    error: str | None = None


# ---------------- chat ----------------

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8_000)


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=1_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    document_ids: list[str] | None = Field(None, description="Limit retrieval to these documents")
    top_k: int | None = Field(None, ge=1, le=12)

    @field_validator("question")
    @classmethod
    def check_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) < 3 or not any(ch.isalpha() for ch in v):
            raise ValueError("Please type a real question.")
        return v


class Source(BaseModel):
    n: int = Field(description="Citation number [n] used in the answer")
    chunk_id: int
    document_id: str
    document_name: str
    page: int
    section: str | None = None
    text: str
    rerank_score: float | None = Field(None, description="Cross-encoder relevance, 0-1")
    dense_score: float | None = Field(None, description="Cosine similarity of the best query")
    dense_rank: int | None = None
    bm25_rank: int | None = None
    cited: bool = False


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    sources: list[int] = Field(default_factory=list)


class Groundedness(BaseModel):
    score: float = Field(description="Share of claims supported by the cited passages, 0-1")
    claims: list[ClaimCheck] = Field(default_factory=list)
    uncited_sentences: int = 0
    invalid_citations: list[int] = Field(default_factory=list)


class TraceStep(BaseModel):
    node: str
    ms: int
    detail: str = ""


class ChatResponse(BaseModel):
    question: str
    standalone_question: str
    search_queries: list[str]
    answer: str
    status: Literal["answered", "not_found", "revised", "unverified"]
    confidence: Literal["high", "medium", "low", "none"]
    confidence_score: float
    groundedness: Groundedness | None
    sources: list[Source]
    trace: list[TraceStep]
    total_ms: int


# ---------------- LLM structured outputs ----------------

class QueryRewrite(BaseModel):
    standalone_question: str = Field(description="The question rewritten to be self-contained")
    search_queries: list[str] = Field(description="2-3 short keyword-style search queries")


class ClaimVerdicts(BaseModel):
    claims: list[ClaimCheck]
