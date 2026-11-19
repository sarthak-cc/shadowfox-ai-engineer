"""Hybrid retrieval: several queries x (dense + BM25), merged with Reciprocal Rank Fusion.

RRF score(chunk) = sum over result lists of 1 / (k + rank). It only uses ranks, so it
combines cosine similarities and BM25 scores (different, incomparable scales) without
any tuning, and rewards chunks that several queries / both retrievers agree on.
"""
from dataclasses import dataclass, field

import numpy as np

from .keyword import KeywordIndex
from .vector_store import ChunkMeta, VectorStore

RRF_K = 60


@dataclass
class Candidate:
    chunk: ChunkMeta
    document_name: str
    fused: float = 0.0
    dense_score: float | None = None      # best cosine across queries
    dense_rank: int | None = None         # best rank across queries (1 = top)
    bm25_rank: int | None = None
    rerank_score: float | None = None
    matched_queries: set[int] = field(default_factory=set)


def hybrid_search(store: VectorStore, keyword: KeywordIndex, queries: list[str], query_vectors: np.ndarray,
                  k_each: int, pool: int, document_ids: list[str] | None = None) -> list[Candidate]:
    cands: dict[int, Candidate] = {}

    def get(cid: int) -> Candidate | None:
        if cid not in cands:
            meta = store.chunks.get(cid)
            if meta is None:
                return None
            cands[cid] = Candidate(meta, store.documents[meta.document_id].name)
        return cands[cid]

    for qi, hits in enumerate(store.search(query_vectors, k_each, document_ids)):
        for rank, (cid, score) in enumerate(hits, start=1):
            c = get(cid)
            if c is None:
                continue
            c.fused += 1 / (RRF_K + rank)
            c.dense_score = max(score, c.dense_score if c.dense_score is not None else -1)
            c.dense_rank = min(rank, c.dense_rank or rank)
            c.matched_queries.add(qi)

    for qi, q in enumerate(queries):
        for rank, (cid, _) in enumerate(keyword.search(q, k_each, document_ids), start=1):
            c = get(cid)
            if c is None:
                continue
            c.fused += 1 / (RRF_K + rank)
            c.bm25_rank = min(rank, c.bm25_rank or rank)
            c.matched_queries.add(qi)

    return sorted(cands.values(), key=lambda c: c.fused, reverse=True)[:pool]
