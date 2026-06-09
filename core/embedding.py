"""Local embedding provider — EmbeddingGemma only, no fallback.

The model (``google/embeddinggemma-300m``) is downloaded into the project-local
``./model`` directory so the whole folder is portable across machines (copy it and
no re-download is needed). Asymmetric query/document prompts and Matryoshka (MRL)
truncation to 512/256/128 are supported. If the model cannot load, we raise a
clear error — the experiment is only meaningful with the real embeddings.
"""
from __future__ import annotations

import os
import time

import numpy as np

from core import BASE_DIR

# Project-local model cache so ./model is portable across PCs.
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
# Route all HF downloads into ./model BEFORE huggingface_hub / transformers import.
os.environ.setdefault("HF_HOME", MODEL_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(MODEL_DIR, "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODEL_DIR)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D array, avoiding division by zero."""
    if vec.ndim == 1:
        n = np.linalg.norm(vec)
        return vec / n if n else vec
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vec / norms


def truncate_normalize(vec: np.ndarray, dim: int) -> np.ndarray:
    """MRL truncation: keep the first ``dim`` components then renormalize."""
    if vec.ndim == 1:
        v = vec[:dim]
        n = np.linalg.norm(v)
        return v / n if n else v
    v = vec[:, :dim]
    return l2_normalize(v)


class EmbeddingProvider:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        dim_full: int = 768,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.dim_full = dim_full
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.total_ms = 0.0          # accumulated embed time (read+reset per turn)
        self._model = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {"cache_folder": MODEL_DIR}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        self._model = SentenceTransformer(self.model_name, **kwargs)
        try:  # method renamed across sentence-transformers versions
            self.dim_full = int(self._model.get_embedding_dimension())
        except AttributeError:
            self.dim_full = int(self._model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------ #
    def pop_ms(self) -> float:
        ms, self.total_ms = self.total_ms, 0.0
        return ms

    def _encode(self, texts: List[str], kind: str) -> np.ndarray:
        t0 = time.perf_counter()
        m = self._model
        if kind == "query" and hasattr(m, "encode_query"):
            arr = m.encode_query(texts, normalize_embeddings=True)
        elif kind == "document" and hasattr(m, "encode_document"):
            arr = m.encode_document(texts, normalize_embeddings=True)
        else:
            prompt_name = {"query": "query", "document": "document"}.get(kind)
            try:
                arr = m.encode(texts, prompt_name=prompt_name, normalize_embeddings=True)
            except Exception:
                arr = m.encode(texts, normalize_embeddings=True)
        self.total_ms += (time.perf_counter() - t0) * 1000.0
        return np.asarray(arr, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def encode_plain(self, texts, dim: int | None = None) -> np.ndarray:
        return self._encode_typed(texts, "plain", dim)

    def encode_query(self, texts, dim: int | None = None) -> np.ndarray:
        return self._encode_typed(texts, "query", dim)

    def encode_document(self, texts, dim: int | None = None) -> np.ndarray:
        return self._encode_typed(texts, "document", dim)

    def _encode_typed(self, texts, kind: str, dim: int | None = None) -> np.ndarray:
        return self._maybe_truncate(self._encode(_as_list(texts), kind), dim)

    def _maybe_truncate(self, mat: np.ndarray, dim: int | None) -> np.ndarray:
        if dim is not None and dim < mat.shape[1]:
            return truncate_normalize(mat, dim)
        return mat

    @property
    def status(self) -> str:
        return f"OK — {self.model_name} (dim={self.dim_full}) @ ./model"


def _as_list(texts) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


# Module-level singleton cache (one heavy model per process) -------------------
_PROVIDER: EmbeddingProvider | None = None
_PROVIDER_KEY: str | None = None


def get_provider(model_name: str, dim_full: int) -> EmbeddingProvider:
    global _PROVIDER, _PROVIDER_KEY
    if _PROVIDER is None or _PROVIDER_KEY != model_name:
        _PROVIDER = EmbeddingProvider(model_name=model_name, dim_full=dim_full)
        _PROVIDER_KEY = model_name
    return _PROVIDER
