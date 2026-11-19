"""Typed request / response models."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocumentInfo(BaseModel):
    id: str
    name: str
    kind: Literal["pdf", "txt", "md"]
    pages: int
    chunks: int
    characters: int
    uploaded_at: str


class IngestJob(BaseModel):
    """Progress of a background upload."""
    id: str
    name: str
    status: Literal["processing", "done", "error"]
    stage: Literal["reading", "embedding", "waiting", "saving", "done", "error"]
    pages: int = 0
    done: int = Field(0, description="Chunks embedded so far")
    total: int = Field(0, description="Chunks to embed")
    document: DocumentInfo | None = None
    error: str | None = None
    elapsed_s: float | None = None
    wait_s: float | None = Field(None, description="Seconds until embedding resumes (rate limit)")


class AskRequest(BaseModel):
    question: str = Field(..., max_length=1_000)
    document_ids: list[str] | None = Field(
        None, description="Restrict the search to these documents. None = search all.")
    top_k: int = Field(5, ge=1, le=10)

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
    n: int = Field(description="Citation number used in the answer, e.g. [1]")
    document_id: str
    document_name: str
    page: int
    score: float = Field(description="Cosine similarity between question and chunk, 0-1")
    text: str
    cited: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool = Field(description="False when the documents do not contain the answer")
    confidence: Literal["high", "medium", "low", "none"]
    sources: list[Source]
    retrieval_ms: int
    generation_ms: int


# Shape the LLM must return (Gemini structured output).
class GroundedAnswer(BaseModel):
    found: bool = Field(description="True only if the context contains the information needed to answer")
    answer: str = Field(description="Markdown answer with [n] citations after each claim, or a short "
                                    "explanation of what is missing when found is false")
    cited: list[int] = Field(description="Numbers of the context passages actually used")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high = directly and fully stated in context; medium = needs light inference or "
                    "is partially covered; low = only loosely supported")
