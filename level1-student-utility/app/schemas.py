"""Typed request / response models.

Requests are validated by Pydantic before any API call is made, so empty or
nonsensical input never reaches the model. Response models double as the
JSON schema we ask Gemini to follow (structured output), which makes the
output predictable and easy to render.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT = 12_000  # ~3k tokens: plenty for a page of notes, keeps cost/latency low


def _require_real_text(value: str, min_words: int, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} cannot be empty.")
    words = stripped.split()
    if len(words) < min_words:
        raise ValueError(f"{label} is too short - please provide at least {min_words} words.")
    letters = sum(ch.isalpha() for ch in stripped)
    if letters / max(len(stripped), 1) < 0.4:
        raise ValueError(f"{label} doesn't look like readable text.")
    return stripped


# ---------- Requests ----------

class SummarizeRequest(BaseModel):
    notes: str = Field(..., max_length=MAX_TEXT)
    length: Literal["brief", "standard", "detailed"] = "standard"

    @field_validator("notes")
    @classmethod
    def check_notes(cls, v: str) -> str:
        return _require_real_text(v, 30, "Notes")


class QuizRequest(BaseModel):
    notes: str = Field(..., max_length=MAX_TEXT)
    num_questions: int = Field(5, ge=3, le=10)
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    @field_validator("notes")
    @classmethod
    def check_notes(cls, v: str) -> str:
        return _require_real_text(v, 40, "Notes")


class ImproveRequest(BaseModel):
    question: str = Field(..., max_length=1_000)
    answer: str = Field(..., max_length=6_000)

    @field_validator("question")
    @classmethod
    def check_question(cls, v: str) -> str:
        return _require_real_text(v, 3, "Question")

    @field_validator("answer")
    @classmethod
    def check_answer(cls, v: str) -> str:
        return _require_real_text(v, 5, "Answer")


class ExplainRequest(BaseModel):
    topic: str = Field(..., max_length=200)
    level: Literal["beginner", "exam", "advanced"] = "exam"
    subject: str | None = Field(None, max_length=80)

    @field_validator("topic")
    @classmethod
    def check_topic(cls, v: str) -> str:
        return _require_real_text(v, 1, "Topic")


# ---------- Responses (also used as Gemini response schemas) ----------

class KeyTerm(BaseModel):
    term: str
    definition: str


class SummaryResult(BaseModel):
    title: str = Field(description="Short title describing the notes")
    tldr: str = Field(description="One or two sentence overview")
    key_points: list[str] = Field(description="Main ideas as concise bullet points")
    key_terms: list[KeyTerm] = Field(description="Important terms with one-line definitions")
    revision_tip: str = Field(description="One practical tip for revising this material")


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly four answer options")
    correct_index: int = Field(description="0-based index of the correct option")
    explanation: str = Field(description="Why the correct option is right, citing the notes")


class QuizResult(BaseModel):
    title: str
    questions: list[QuizQuestion]


class Feedback(BaseModel):
    strengths: list[str]
    gaps: list[str] = Field(description="Missing points, errors or weak areas")


class ImproveResult(BaseModel):
    score_before: int = Field(description="Quality score of the original answer, 1-10")
    score_after: int = Field(description="Expected score of the improved answer, 1-10")
    feedback: Feedback
    improved_answer: str = Field(description="Rewritten, exam-ready answer in Markdown")
    changes: list[str] = Field(description="What was changed and why")


class ExplainResult(BaseModel):
    title: str
    one_liner: str = Field(description="The concept in one plain sentence")
    explanation: str = Field(description="Main explanation in Markdown")
    analogy: str = Field(description="A relatable everyday analogy")
    example: str = Field(description="A worked or concrete example in Markdown")
    common_mistakes: list[str]
    check_questions: list[str] = Field(description="2-3 self-check questions")
