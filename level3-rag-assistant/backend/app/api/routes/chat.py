import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from ...generation.gemini import LLMError
from ...schemas import ChatRequest, ChatResponse
from ..deps import Container, get_container

router = APIRouter(prefix="/api", tags=["chat"])
log = logging.getLogger("rag.api.chat")


def _validate(req: ChatRequest, c: Container) -> None:
    if not c.store.documents:
        raise HTTPException(400, "Upload at least one document before asking a question.")
    if req.document_ids is not None:
        if not req.document_ids:
            raise HTTPException(400, "Select at least one document to search.")
        unknown = [d for d in req.document_ids if d not in c.store.documents]
        if unknown:
            raise HTTPException(400, "Some selected documents no longer exist. Refresh the page.")


@router.post("/chat", response_model=ChatResponse, summary="Answer a question (single JSON response)")
async def chat(req: ChatRequest, c: Container = Depends(get_container)):
    _validate(req, c)
    return await run_in_threadpool(c.pipeline.run, req)


@router.post("/chat/stream", summary="Answer a question as server-sent events",
             description="Events: `step` (pipeline progress), `sources` (selected passages), `token` "
                         "(answer text), `check` (groundedness), `revision` (corrected answer), "
                         "`done` (final ChatResponse) or `error`.")
def chat_stream(req: ChatRequest, c: Container = Depends(get_container)):
    _validate(req, c)

    def events():
        try:
            for event in c.pipeline.stream(req):
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        except LLMError as e:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': e.message, 'status': e.status_code})}\n\n"
        except Exception:
            log.exception("stream failed")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'Something went wrong while answering.'})}\n\n"

    # A sync generator: Starlette iterates it in a worker thread, so the event loop stays free.
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
