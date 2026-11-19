"""FAISS vector index + chunk metadata, with document-scoped search and persistence.

* Index: `IndexIDMap2(IndexFlatIP)` - exact inner-product search over L2-normalised
  vectors (= cosine similarity), addressed by stable int64 chunk ids so documents can
  be removed without rebuilding.
* Document scope: FAISS `IDSelectorBatch` restricts a search to the chunk ids of the
  selected documents *inside* the index scan, so a filtered search still returns a full
  top-k (no "search then filter" shortfall).
* Why exact (Flat) and not IVF/HNSW? A student library is thousands of chunks; exact
  search over that takes well under a millisecond and needs no training. The class
  keeps the index behind one interface, so swapping in `IndexHNSWFlat` for millions of
  chunks would not touch the rest of the app.
* Persistence: `index.faiss` + `meta.json` in the data directory, written atomically.
"""
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from ..schemas import DocumentInfo

log = logging.getLogger("rag.store")


@dataclass
class ChunkMeta:
    id: int
    document_id: str
    page: int
    section: str | None
    text: str


class VectorStore:
    def __init__(self, dim: int, data_dir: Path, embedding_id: str = "") -> None:
        self.dim = dim
        self.embedding_id = embedding_id
        self.data_dir = data_dir
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.chunks: dict[int, ChunkMeta] = {}
        self.documents: dict[str, DocumentInfo] = {}
        self.hashes: dict[str, str] = {}
        self.next_id = 0
        self.lock = threading.RLock()
        self.version = 0          # bumped on every change (BM25 uses it to know when to rebuild)

    # ---------- mutation ----------

    def add_document(self, doc: DocumentInfo, sha256: str, chunks: list[tuple[int, str | None, str]],
                     vectors: np.ndarray) -> None:
        assert vectors.shape == (len(chunks), self.dim), vectors.shape
        with self.lock:
            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype=np.int64)
            self.index.add_with_ids(vectors.astype(np.float32), ids)
            for cid, (page, section, text) in zip(ids.tolist(), chunks):
                self.chunks[cid] = ChunkMeta(cid, doc.id, page, section, text)
            self.next_id += len(chunks)
            self.documents[doc.id] = doc
            self.hashes[sha256] = doc.id
            self.version += 1
            self._save()

    def remove_document(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.documents:
                return False
            ids = np.array([c.id for c in self.chunks.values() if c.document_id == doc_id], dtype=np.int64)
            if len(ids):
                self.index.remove_ids(ids)
            for cid in ids.tolist():
                del self.chunks[cid]
            del self.documents[doc_id]
            self.hashes = {h: d for h, d in self.hashes.items() if d != doc_id}
            self.version += 1
            self._save()
            return True

    def clear(self) -> None:
        with self.lock:
            self.index.reset()
            self.chunks, self.documents, self.hashes = {}, {}, {}
            self.version += 1
            self._save()

    # ---------- search ----------

    def chunk_ids_for(self, document_ids: list[str] | None) -> np.ndarray | None:
        if not document_ids:
            return None
        allowed = set(document_ids)
        return np.array([c.id for c in self.chunks.values() if c.document_id in allowed], dtype=np.int64)

    def search(self, queries: np.ndarray, k: int, document_ids: list[str] | None = None
               ) -> list[list[tuple[int, float]]]:
        """For each query vector, return [(chunk_id, cosine)] best first."""
        with self.lock:
            if self.index.ntotal == 0:
                return [[] for _ in range(len(queries))]
            ids = self.chunk_ids_for(document_ids)
            if ids is not None and len(ids) == 0:
                return [[] for _ in range(len(queries))]
            k = min(k, self.index.ntotal if ids is None else len(ids))
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids)) if ids is not None else None
            scores, found = self.index.search(queries.astype(np.float32), k, params=params)
        return [[(int(i), float(s)) for i, s in zip(row_i, row_s) if i != -1]
                for row_i, row_s in zip(found, scores)]

    # ---------- persistence ----------

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp_index = self.data_dir / "index.faiss.tmp"
        tmp_meta = self.data_dir / "meta.json.tmp"
        faiss.write_index(self.index, str(tmp_index))
        tmp_meta.write_text(json.dumps({
            "dim": self.dim, "embedding_id": self.embedding_id, "next_id": self.next_id,
            "chunks": [asdict(c) for c in self.chunks.values()],
            "documents": [d.model_dump() for d in self.documents.values()],
            "hashes": self.hashes,
        }), encoding="utf-8")
        os.replace(tmp_index, self.data_dir / "index.faiss")
        os.replace(tmp_meta, self.data_dir / "meta.json")

    def load(self) -> None:
        index_path, meta_path = self.data_dir / "index.faiss", self.data_dir / "meta.json"
        if not (index_path.exists() and meta_path.exists()):
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))
            if meta.get("embedding_id", self.embedding_id) != self.embedding_id:
                log.warning("Stored index was built with %s, but the app now uses %s. Vectors from different "
                            "models are not comparable - starting with an empty library (re-upload documents).",
                            meta.get("embedding_id"), self.embedding_id)
                return
            if meta["dim"] != self.dim or index.ntotal != len(meta["chunks"]):
                log.warning("Stored index does not match settings (dim/size) - starting empty.")
                return
            with self.lock:
                self.index = index
                self.chunks = {c["id"]: ChunkMeta(**c) for c in meta["chunks"]}
                self.documents = {d["id"]: DocumentInfo(**d) for d in meta["documents"]}
                self.hashes = meta.get("hashes", {})
                self.next_id = meta.get("next_id", len(self.chunks))
                self.version += 1
            log.info("Loaded %d documents / %d chunks", len(self.documents), len(self.chunks))
        except Exception:
            log.exception("Could not load the stored index - starting empty.")
