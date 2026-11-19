from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ...core.config import get_settings
from ...schemas import DocumentInfo, IngestJob
from ..deps import Container, get_container

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(c: Container = Depends(get_container)):
    return c.documents.all()


@router.post("/documents", response_model=IngestJob, status_code=202,
             summary="Upload a PDF/TXT/MD file; indexing runs in the background")
async def upload_document(file: UploadFile = File(...), c: Container = Depends(get_container)):
    limit = int(get_settings().max_file_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    return c.documents.start_ingest((file.filename or "untitled").strip(), data)


@router.get("/jobs/{job_id}", response_model=IngestJob, summary="Poll indexing progress")
def get_job(job_id: str, c: Container = Depends(get_container)):
    job = c.documents.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Upload job not found.")
    return job


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, c: Container = Depends(get_container)):
    if not c.store.remove_document(doc_id):
        raise HTTPException(404, "Document not found.")


@router.delete("/documents", status_code=204)
def delete_all(c: Container = Depends(get_container)):
    c.store.clear()


@router.post("/samples", response_model=list[DocumentInfo], summary="Index the bundled sample documents")
async def load_samples(c: Container = Depends(get_container)):
    return await run_in_threadpool(c.documents.load_samples)
