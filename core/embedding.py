"""Local embedding provider: EmbeddingGemma as a GGUF file run by llama.cpp
(``llama-cpp-python``), fully offline. Asymmetric prompts per the model card:
``task: search result | query: …`` for queries, ``title: none | text: …`` for
documents. Vectors are returned L2-normalised as float32 rows."""
from __future__ import annotations

import os
import time

import numpy as np

from config import ROOT

MODEL_DIR = os.path.join(ROOT, "model")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0.0, 1.0, norms)


class EmbeddingProvider:
    def __init__(self, model_path: str, n_threads: int | None = None):
        from llama_cpp import Llama

        if not os.path.isabs(model_path):
            model_path = os.path.join(MODEL_DIR, model_path)
        self.model_path = model_path
        self.model_id = os.path.basename(model_path)
        self.total_ms = 0.0
        self._model = Llama(model_path=model_path, embedding=True, n_ctx=2048, verbose=False,
                            n_threads=n_threads or os.cpu_count())
        self.dimension = int(self._model.n_embd())

    def pop_ms(self) -> float:
        ms, self.total_ms = self.total_ms, 0.0
        return ms

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        t0 = time.perf_counter()
        out = np.stack([np.asarray(self._model.embed(prefix + t), dtype=np.float32) for t in texts])
        self.total_ms += (time.perf_counter() - t0) * 1000.0
        return l2_normalize(out)

    def encode_query(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts), "task: search result | query: ")

    def encode_document(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts), "title: none | text: ")

    @property
    def status(self) -> str:
        return f"OK — {self.model_id} (dim={self.dimension}, llama.cpp)"


_PROVIDER: EmbeddingProvider | None = None


def get_provider(model_path: str) -> EmbeddingProvider:
    """One model per process."""
    global _PROVIDER
    if _PROVIDER is None or os.path.basename(_PROVIDER.model_path) != os.path.basename(model_path):
        _PROVIDER = EmbeddingProvider(model_path)
    return _PROVIDER
