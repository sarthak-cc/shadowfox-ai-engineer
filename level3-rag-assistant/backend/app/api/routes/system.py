from fastapi import APIRouter, Depends

from ...core.config import get_settings
from ...retrieval.reranker import reranker
from ..deps import Container, get_container

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", summary="Liveness and component status")
def health(c: Container = Depends(get_container)):
    s = get_settings()
    return {
        "status": "ok",
        "api_key_configured": bool(s.gemini_api_key),
        "models": {"generation": [s.gemini_model, *s.gemini_fallback_models],
                   "embedding": f"{s.embedding_id} ({s.vector_dim}-d)", "reranker": s.rerank_model},
        "reranker": reranker.status,
        "index": {"documents": len(c.store.documents), "chunks": c.store.index.ntotal, "type": "FAISS IndexFlatIP"},
    }


@router.get("/config", summary="Effective retrieval / generation settings (no secrets)")
def config():
    s = get_settings()
    return s.model_dump(include={"chunk_size", "chunk_overlap", "candidates_per_query", "rerank_pool", "top_k",
                                 "rerank_min_score", "rerank_relative_floor", "groundedness_threshold",
                                 "max_revisions", "history_turns", "hedge_after_s", "embed_per_minute"})
