"""StudyPilot - FastAPI backend serving four AI study tools and the web UI."""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import prompts
from .config import get_settings
from .llm import LLMError, generate_structured
from .schemas import (
    ExplainRequest, ExplainResult, ImproveRequest, ImproveResult, QuizRequest,
    QuizResult, SummarizeRequest, SummaryResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="StudyPilot", version="1.0.0",
              description="AI-powered study utilities: summaries, quizzes, answer improvement, explanations.")


# ---------- Error handling: every error becomes {"error": "..."} ----------

@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0]
    msg = first.get("msg", "Invalid input.")
    msg = msg.removeprefix("Value error, ")
    if first.get("type") == "string_too_long":
        field = first.get("loc", ["", "input"])[-1]
        msg = f"The {field} is too long (limit {first.get('ctx', {}).get('max_length')} characters)."
    return JSONResponse(status_code=422, content={"error": msg})


# ---------- API ----------

@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "model": s.gemini_model, "api_key_configured": bool(s.gemini_api_key)}


@app.post("/api/summarize", response_model=SummaryResult)
def summarize(req: SummarizeRequest) -> SummaryResult:
    return generate_structured(prompts.summarize_prompt(req.notes, req.length), SummaryResult, 0.3)


@app.post("/api/quiz", response_model=QuizResult)
def quiz(req: QuizRequest) -> QuizResult:
    result = generate_structured(
        prompts.quiz_prompt(req.notes, req.num_questions, req.difficulty), QuizResult, 0.6)
    # Defensive post-processing: drop malformed questions instead of showing broken UI.
    result.questions = [
        q for q in result.questions
        if len(q.options) == 4 and 0 <= q.correct_index < 4
    ][: req.num_questions]
    if not result.questions:
        raise LLMError("The AI could not build a valid quiz from these notes. Try adding more detail.", 422)
    return result


@app.post("/api/improve", response_model=ImproveResult)
def improve(req: ImproveRequest) -> ImproveResult:
    result = generate_structured(prompts.improve_prompt(req.question, req.answer), ImproveResult, 0.4)
    result.score_before = max(1, min(10, result.score_before))
    result.score_after = max(1, min(10, result.score_after))
    return result


@app.post("/api/explain", response_model=ExplainResult)
def explain(req: ExplainRequest) -> ExplainResult:
    return generate_structured(prompts.explain_prompt(req.topic, req.level, req.subject), ExplainResult, 0.5)


# ---------- Frontend ----------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
