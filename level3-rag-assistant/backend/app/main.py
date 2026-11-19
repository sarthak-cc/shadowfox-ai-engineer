"""FastAPI application factory."""
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.deps import Container
from .api.routes import chat, documents, system
from .core.config import get_settings
from .generation.gemini import LLMError
from .graph.pipeline import Pipeline
from .ingestion.parsers import IngestError
from .retrieval.keyword import KeywordIndex
from .retrieval.reranker import reranker
from .retrieval.vector_store import VectorStore
from .services.documents import Conflict, DocumentService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "google_genai.models", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("rag")


def build_container() -> Container:
    s = get_settings()
    store = VectorStore(s.vector_dim, s.data_dir, s.embedding_id)
    store.load()
    keyword = KeywordIndex(store)
    return Container(store=store, keyword=keyword, documents=DocumentService(store),
                     pipeline=Pipeline(store, keyword))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "container"):
        app.state.container = build_container()
    # Load the reranker in the background so the API is available immediately.
    threading.Thread(target=reranker.load, daemon=True).start()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="RAG Assistant API", version="1.0.0", lifespan=lifespan,
                  description="Grounded question answering over PDF, TXT and Markdown files.")
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        t = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        if request.url.path.startswith("/api") and not request.url.path.startswith("/api/jobs"):
            log.info("%s %s %s -> %d (%.0f ms)", rid, request.method, request.url.path, response.status_code,
                     (time.perf_counter() - t) * 1000)
        return response

    def err(status: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": message})

    @app.exception_handler(LLMError)
    async def _llm(_: Request, e: LLMError):
        return err(e.status_code, e.message)

    @app.exception_handler(IngestError)
    async def _ingest(_: Request, e: IngestError):
        return err(422, str(e))

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, e: Conflict):
        return err(409, str(e))

    @app.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException):
        return err(e.status_code, str(e.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        first = e.errors()[0]
        msg = str(first.get("msg", "Invalid input.")).removeprefix("Value error, ")
        if first.get("type") == "string_too_long":
            msg = f"{first['loc'][-1]} is too long."
        return err(422, msg)

    for r in (documents.router, chat.router, system.router):
        app.include_router(r)
    return app


app = create_app()
