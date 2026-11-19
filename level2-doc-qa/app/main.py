"""DocLens - FastAPI backend: upload documents, ask questions, get grounded answers."""
import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .ingest import IngestError, chunk_pages, detect_kind, extract_pages
from .llm import LLMError, embed
from .qa import answer_question
from .schemas import AskRequest, AskResponse, DocumentInfo, IngestJob
from .store import ChunkRecord, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("doclens")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

settings = get_settings()
store = VectorStore(settings.embedding_dim)
store.load()

_jobs: dict[str, IngestJob] = {}   # background upload jobs, polled by the UI
_pending: set[str] = set()         # sha256 of files currently being indexed

app = FastAPI(title="DocLens", version="1.0.0",
              description="Upload PDF/TXT/Markdown files and ask questions answered from their content.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"],
                   allow_headers=["Content-Type"])


# ---------- errors -> {"error": "..."} ----------

@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(IngestError)
async def ingest_error_handler(_: Request, exc: IngestError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0]
    msg = first.get("msg", "Invalid input.").removeprefix("Value error, ")
    if first.get("type") == "string_too_long":
        msg = "The question is too long (limit 1,000 characters)."
    return JSONResponse(status_code=422, content={"error": msg})


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


# ---------- API ----------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": settings.gemini_model, "embedding_model": settings.embedding_model,
            "api_key_configured": bool(settings.gemini_api_key),
            "documents": len(store.documents), "chunks": len(store.chunks)}


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return sorted(store.documents.values(), key=lambda d: d.uploaded_at, reverse=True)


@app.post("/api/documents", response_model=IngestJob, status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """Validate the file, then index it in the background. Poll /api/jobs/{id} for progress."""
    name = (file.filename or "").strip() or "untitled"
    kind = detect_kind(name)

    if len(store.documents) + len(_pending) >= settings.max_documents:
        return _error(409, f"You can index up to {settings.max_documents} documents. Remove one first.")

    max_bytes = int(settings.max_file_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        return _error(413, f"File is too large (limit {settings.max_file_mb:g} MB).")

    sha = hashlib.sha256(data).hexdigest()
    if sha in store.hashes:
        existing = store.documents[store.hashes[sha]].name
        return _error(409, f"This file is already indexed as \"{existing}\".")
    if sha in _pending:
        return _error(409, "This file is already being indexed.")

    job = IngestJob(id=uuid.uuid4().hex[:12], name=name, status="processing", stage="reading")
    _jobs[job.id] = job
    _pending.add(sha)
    threading.Thread(target=_run_job, args=(job, kind, data, sha), daemon=True).start()
    return job


@app.get("/api/jobs/{job_id}", response_model=IngestJob)
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return _error(404, "Upload job not found.")
    return job


def _run_job(job: IngestJob, kind: str, data: bytes, sha: str) -> None:
    started = time.perf_counter()
    try:
        job.document = _index_document(job.name, kind, data, sha, job)
        job.status, job.stage = "done", "done"
    except (IngestError, LLMError) as e:
        job.status, job.stage, job.error = "error", "error", str(getattr(e, "message", e))
    except Exception:
        log.exception("indexing failed for %s", job.name)
        job.status, job.stage, job.error = "error", "error", "Something went wrong while indexing this file."
    finally:
        _pending.discard(sha)
        job.elapsed_s = round(time.perf_counter() - started, 1)
        # keep the registry small
        if len(_jobs) > 200:
            for old in list(_jobs)[:100]:
                _jobs.pop(old, None)


def _index_document(name: str, kind: str, data: bytes, sha: str, job: IngestJob | None = None) -> DocumentInfo:
    pages = extract_pages(data, kind)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise IngestError("No usable text was found in this file.")

    if job:
        job.stage, job.total, job.pages = "embedding", len(chunks), len(pages)

    def progress(n: int) -> None:
        if job:
            job.done, job.stage, job.wait_s = n, "embedding", None

    def waiting(seconds: float) -> None:
        if job:
            job.stage, job.wait_s = "waiting", round(seconds)

    vectors = embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT", on_progress=progress, on_wait=waiting)
    if job:
        job.stage = "saving"

    doc = DocumentInfo(
        id=uuid.uuid4().hex[:12], name=name, kind=kind, pages=len(pages), chunks=len(chunks),
        characters=sum(len(p.text) for p in pages),
        uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    store.add(doc, sha, [ChunkRecord(doc.id, c.page, c.text) for c in chunks], vectors)
    log.info("indexed %s: %d pages, %d chunks", name, len(pages), len(chunks))
    return doc


@app.post("/api/samples", response_model=list[DocumentInfo])
async def load_samples():
    """Index the bundled sample documents (skips any that are already indexed)."""
    added: list[DocumentInfo] = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        try:
            kind = detect_kind(path.name)
        except IngestError:
            continue
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha not in store.hashes:
            added.append(await run_in_threadpool(_index_document, path.name, kind, data, sha))
    return added


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    if not store.remove(doc_id):
        return _error(404, "Document not found.")


@app.delete("/api/documents", status_code=204)
def delete_all_documents():
    store.clear()


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not store.documents:
        return _error(400, "Upload at least one document before asking a question.")
    if req.document_ids is not None:
        unknown = [d for d in req.document_ids if d not in store.documents]
        if unknown:
            return _error(400, "Some selected documents no longer exist. Refresh the page.")
        if not req.document_ids:
            return _error(400, "Select at least one document to search.")
    return answer_question(req, store)


# ---------- UI (mounted last so /api routes win) ----------
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
