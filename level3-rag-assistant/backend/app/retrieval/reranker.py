"""Cross-encoder reranking (runs locally, no API calls, no quota).

Embedding search compares a question vector with passage vectors computed separately.
A cross-encoder reads the question and the passage *together* and scores how well the
passage answers it - much more precise, but too slow to run on every chunk. So:
retrieve ~20 candidates cheaply (hybrid search), then rerank only those.

Model: `ms-marco-MiniLM-L-6-v2` via fastembed (ONNX runtime, ~80 MB, no PyTorch). It
scores 20 passages in ~0.3 s on a laptop CPU. Raw logits are mapped to 0-1 with a
sigmoid so thresholds are readable.

If the model can't be loaded (offline first run, blocked download), the pipeline keeps
working: candidates keep their fusion order and filtering falls back to dense scores.
"""
import logging
import math
import threading

from ..core.config import get_settings

log = logging.getLogger("rag.reranker")


class Reranker:
    def __init__(self) -> None:
        self._model = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        if self._model is not None:
            return "ready"
        return f"unavailable: {self._error}" if self._error else "loading"

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self._error:
                return False
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                s = get_settings()
                self._model = TextCrossEncoder(s.rerank_model, cache_dir=str(s.model_cache_dir))
                log.info("reranker %s loaded", s.rerank_model)
                return True
            except Exception as e:  # network, disk, unsupported platform...
                self._error = type(e).__name__
                log.warning("reranker unavailable (%s) - falling back to fusion order", e)
                return False

    def score(self, query: str, passages: list[str]) -> list[float] | None:
        """Relevance in 0-1 for each passage, or None if the model is unavailable."""
        if not passages or not self.load():
            return None
        logits = list(self._model.rerank(query, passages, batch_size=32))
        return [1 / (1 + math.exp(-x)) for x in logits]


reranker = Reranker()
