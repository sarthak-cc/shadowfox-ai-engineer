"""A small in-memory vector store with disk persistence.

All chunk embeddings live in one (n, dim) float32 matrix. Because rows are
L2-normalised, cosine similarity for a query is a single matrix-vector product:
`scores = matrix @ query`. For a few thousand chunks (dozens of documents)
this exact search takes well under a millisecond, so an approximate index
(FAISS, HNSW) is unnecessary at this scale.

The store is saved to `data/` (embeddings as .npy, metadata as JSON) so uploaded
documents survive a server restart.
"""
import json
import logging
import threading
from dataclasses import asdict, dataclass

import numpy as np

from .config import get_settings
from .schemas import DocumentInfo

log = logging.getLogger("doclens.store")


@dataclass
class ChunkRecord:
    document_id: str
    page: int
    text: str


@dataclass
class Hit:
    chunk: ChunkRecord
    document_name: str
    score: float


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self.chunks: list[ChunkRecord] = []
        self.documents: dict[str, DocumentInfo] = {}
        self.hashes: dict[str, str] = {}          # sha256 -> document id (duplicate detection)
        self._lock = threading.Lock()

    # ---------- mutation ----------

    def add(self, doc: DocumentInfo, sha256: str, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
        assert vectors.shape == (len(chunks), self.dim)
        with self._lock:
            self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])
            self.chunks.extend(chunks)
            self.documents[doc.id] = doc
            self.hashes[sha256] = doc.id
            self._save()

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            keep = np.array([c.document_id != doc_id for c in self.chunks], dtype=bool)
            self.matrix = self.matrix[keep] if len(keep) else self.matrix
            self.chunks = [c for c, k in zip(self.chunks, keep) if k]
            del self.documents[doc_id]
            self.hashes = {h: d for h, d in self.hashes.items() if d != doc_id}
            self._save()
            return True

    def clear(self) -> None:
        with self._lock:
            self.matrix = np.zeros((0, self.dim), dtype=np.float32)
            self.chunks, self.documents, self.hashes = [], {}, {}
            self._save()

    # ---------- search ----------

    def search(self, query: np.ndarray, top_k: int, document_ids: list[str] | None = None) -> list[Hit]:
        """Exact cosine-similarity search, optionally restricted to some documents."""
        with self._lock:
            if not self.chunks:
                return []
            scores = self.matrix @ query.astype(np.float32)
            if document_ids:
                allowed = set(document_ids)
                mask = np.array([c.document_id in allowed for c in self.chunks], dtype=bool)
                scores = np.where(mask, scores, -np.inf)
            k = min(top_k, int(np.isfinite(scores).sum()))
            if k == 0:
                return []
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            return [Hit(self.chunks[i], self.documents[self.chunks[i].document_id].name, float(scores[i]))
                    for i in idx]

    # ---------- persistence ----------

    def _save(self) -> None:
        d = get_settings().data_dir
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.matrix)
        meta = {
            "dim": self.dim,
            "chunks": [asdict(c) for c in self.chunks],
            "documents": [doc.model_dump() for doc in self.documents.values()],
            "hashes": self.hashes,
        }
        (d / "store.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self) -> None:
        d = get_settings().data_dir
        try:
            meta = json.loads((d / "store.json").read_text(encoding="utf-8"))
            matrix = np.load(d / "embeddings.npy")
            if meta["dim"] != self.dim or matrix.shape[0] != len(meta["chunks"]):
                log.warning("Stored index does not match current settings - starting empty.")
                return
            self.matrix = matrix.astype(np.float32)
            self.chunks = [ChunkRecord(**c) for c in meta["chunks"]]
            self.documents = {doc["id"]: DocumentInfo(**doc) for doc in meta["documents"]}
            self.hashes = meta.get("hashes", {})
            log.info("Loaded %d documents / %d chunks from disk", len(self.documents), len(self.chunks))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("Could not load stored index - starting empty.")
