"""Application container: one instance of each component, created at startup."""
from dataclasses import dataclass

from fastapi import Request

from ..graph.pipeline import Pipeline
from ..retrieval.keyword import KeywordIndex
from ..retrieval.vector_store import VectorStore
from ..services.documents import DocumentService


@dataclass
class Container:
    store: VectorStore
    keyword: KeywordIndex
    documents: DocumentService
    pipeline: Pipeline


def get_container(request: Request) -> Container:
    return request.app.state.container
