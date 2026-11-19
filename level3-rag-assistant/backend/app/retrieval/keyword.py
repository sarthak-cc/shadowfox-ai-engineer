"""BM25 keyword search over the same chunks.

Dense embeddings capture meaning but can miss exact tokens: acronyms ("DORA"), code
names ("Jenkinsfile"), numbers, rare terms. BM25 catches those. The index is rebuilt
lazily when the vector store changes (cheap at this scale: tens of ms for thousands
of chunks).
"""
import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi

from .vector_store import VectorStore

_STOP = set("""a an the and or of to in on for is are was were be been being what which who whom how why
when where does do did can could should would will with from that this these those it its as by at about
into than then there their my your our me i you we they he she them his her if not no yes explain describe
define discuss write give list short note notes""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()) if t not in _STOP and len(t) > 1]


class KeywordIndex:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._version = -1
        self._ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._bm25: BM25Okapi | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if self._version == self.store.version:
                return
            with self.store.lock:
                items = [(c.id, f"{c.section or ''} {c.text}") for c in self.store.chunks.values()]
                version = self.store.version
            self._ids = np.array([i for i, _ in items], dtype=np.int64)
            corpus = [tokenize(t) or ["_"] for _, t in items]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._version = version

    def search(self, query: str, k: int, document_ids: list[str] | None = None) -> list[tuple[int, float]]:
        self._ensure()
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        if document_ids:
            allowed = set(document_ids)
            chunks = self.store.chunks
            mask = np.array([chunks.get(int(i)) is not None and chunks[int(i)].document_id in allowed
                             for i in self._ids], dtype=bool)
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores)[:k]
        return [(int(self._ids[i]), float(scores[i])) for i in order if scores[i] > 0]
