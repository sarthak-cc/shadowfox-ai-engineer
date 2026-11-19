"""Document lifecycle: validate -> background ingest job (parse, chunk, embed, index) -> list / delete."""
import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from ..core.config import get_settings
from ..generation.gemini import LLMError
from ..ingestion.chunking import chunk_blocks
from ..ingestion.parsers import IngestError, detect_kind, parse
from ..retrieval.embeddings import DOC, embed
from ..retrieval.vector_store import VectorStore
from ..schemas import DocumentInfo, IngestJob

log = logging.getLogger("rag.documents")


class Conflict(Exception):
    """409 - duplicate file or library full."""


class DocumentService:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self.jobs: dict[str, IngestJob] = {}
        self.pending: set[str] = set()
        self._lock = threading.Lock()

    def all(self) -> list[DocumentInfo]:
        return sorted(self.store.documents.values(), key=lambda d: d.uploaded_at, reverse=True)

    def start_ingest(self, name: str, data: bytes, background: bool = True) -> IngestJob:
        s = get_settings()
        kind = detect_kind(name)
        if not data:
            raise IngestError("The file is empty.")
        if len(data) > s.max_file_mb * 1024 * 1024:
            raise IngestError(f"File is too large (limit {s.max_file_mb:g} MB).")
        sha = hashlib.sha256(data).hexdigest()
        with self._lock:
            if sha in self.store.hashes:
                raise Conflict(f'This file is already indexed as "{self.store.documents[self.store.hashes[sha]].name}".')
            if sha in self.pending:
                raise Conflict("This file is already being indexed.")
            if len(self.store.documents) + len(self.pending) >= s.max_documents:
                raise Conflict(f"The library is full ({s.max_documents} documents). Remove one first.")
            self.pending.add(sha)
            job = IngestJob(id=uuid.uuid4().hex[:12], name=name, status="processing", stage="parsing")
            self.jobs[job.id] = job
        if background:
            threading.Thread(target=self._run, args=(job, kind, data, sha), daemon=True).start()
        else:
            self._run(job, kind, data, sha)
        return job

    def _run(self, job: IngestJob, kind: str, data: bytes, sha: str) -> None:
        s = get_settings()
        t = time.perf_counter()
        try:
            blocks, pages = parse(data, kind)
            chunks = chunk_blocks(blocks, s.chunk_size, s.chunk_overlap)
            if not chunks:
                raise IngestError("No usable text was found in this file.")
            job.stage, job.pages, job.total = "embedding", pages, len(chunks)

            def progress(n: int) -> None:
                job.done, job.stage, job.wait_s = n, "embedding", None

            def waiting(sec: float) -> None:
                job.stage, job.wait_s = "waiting", round(sec)

            vectors = embed([c.embed_text(job.name) for c in chunks], DOC, on_progress=progress, on_wait=waiting)
            job.stage = "indexing"
            doc = DocumentInfo(
                id=uuid.uuid4().hex[:12], name=job.name, kind=kind, pages=pages, chunks=len(chunks),
                characters=sum(len(c.text) for c in chunks),
                sections=len({c.section for c in chunks if c.section}),
                uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            self.store.add_document(doc, sha, [(c.page, c.section, c.text) for c in chunks], vectors)
            job.document, job.status, job.stage = doc, "done", "done"
            log.info("indexed %s: %d pages, %d chunks in %.1fs", job.name, pages, len(chunks), time.perf_counter() - t)
        except (IngestError, LLMError) as e:
            job.status, job.stage, job.error = "error", "error", getattr(e, "message", str(e))
        except Exception:
            log.exception("ingest failed for %s", job.name)
            job.status, job.stage, job.error = "error", "error", "Something went wrong while indexing this file."
        finally:
            job.elapsed_s = round(time.perf_counter() - t, 1)
            with self._lock:
                self.pending.discard(sha)
                if len(self.jobs) > 200:
                    for old in list(self.jobs)[:100]:
                        self.jobs.pop(old, None)

    def load_samples(self) -> list[DocumentInfo]:
        added = []
        for path in sorted(get_settings().samples_dir.glob("*")):
            try:
                detect_kind(path.name)
            except IngestError:
                continue
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() in self.store.hashes:
                continue
            job = self.start_ingest(path.name, data, background=False)
            if job.status == "error":
                raise LLMError(job.error or "Could not index the samples.", 503)
            added.append(job.document)
        return added
