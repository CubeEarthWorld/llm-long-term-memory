"""Local embedding provider: any GGUF embedding model run by llama.cpp
(``llama-cpp-python``), fully offline. The asymmetric retrieval prompts come from
the model card and are configurable (EmbeddingGemma's by default). Vectors are
returned L2-normalised as float32 rows; the engine imposes no dimension."""
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
    def __init__(self, model_path: str, query_prefix: str = "", document_prefix: str = ""):
        from llama_cpp import Llama

        if not os.path.isabs(model_path):
            model_path = os.path.join(MODEL_DIR, model_path)
        self.model_path = model_path
        self.model_id = os.path.basename(model_path)
        self.query_prefix, self.document_prefix = query_prefix, document_prefix
        self.total_ms = 0.0
        # n_ctx=0: the model's own training context (e5 trains at 512, Gemma at 2048).
        self._model = Llama(model_path=model_path, embedding=True, n_ctx=0, verbose=False,
                            n_threads=os.cpu_count())
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
        return self._encode(list(texts), self.query_prefix)

    def encode_document(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts), self.document_prefix)

    @property
    def status(self) -> str:
        return f"OK — {self.model_id} (dim={self.dimension}, llama.cpp)"


_PROVIDER: EmbeddingProvider | None = None


def get_provider(glob) -> EmbeddingProvider:
    """One model per process (``glob`` is the GlobalConfig: path + retrieval prompts)."""
    global _PROVIDER
    want = (os.path.basename(glob.embedding_model), glob.embed_query_prefix, glob.embed_document_prefix)
    if _PROVIDER is None or (os.path.basename(_PROVIDER.model_path),
                             _PROVIDER.query_prefix, _PROVIDER.document_prefix) != want:
        _PROVIDER = EmbeddingProvider(glob.embedding_model, want[1], want[2])
    return _PROVIDER
