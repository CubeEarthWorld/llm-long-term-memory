"""Deterministic mocks so the ENGRAM engine can be evaluated without the real
embedding model or an LLM API key.

* ``FakeEmbeddingProvider`` builds vectors from token overlap, so texts that share
  words / CJK characters are similar (cosine ≈ shared/√(n1·n2)), while unrelated
  texts are near-orthogonal. Identical tokens -> identical vector.
* ``FakeLLM`` only needs ``dream_cluster`` for the eval (save_memory is called
  directly): it merges a cluster into a single gist.
* ``VirtualClock`` lets scenarios accelerate time arbitrarily (activation decay,
  spacing) at zero real cost.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_DAY = 24 * 60 * 60


def _tokens(text: str) -> List[str]:
    text = str(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    cjk = [c for c in text if "぀" <= c <= "鿿" or "ｦ" <= c <= "ﾝ"]
    return words + cjk


def _seed(tok: str) -> int:
    return int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)


class FakeEmbeddingProvider:
    def __init__(self, dim_full: int = 768):
        self.dim_full = dim_full
        self.total_ms = 0.0
        self._cache: dict[str, np.ndarray] = {}

    def _tok_vec(self, tok: str) -> np.ndarray:
        v = self._cache.get(tok)
        if v is None:
            rng = np.random.default_rng(_seed(tok))
            v = rng.standard_normal(self.dim_full).astype(np.float32)
            self._cache[tok] = v
        return v

    def _embed(self, text: str) -> np.ndarray:
        toks = _tokens(text)
        if not toks:
            v = np.ones(self.dim_full, dtype=np.float32)
        else:
            v = np.sum([self._tok_vec(t) for t in toks], axis=0)
        n = np.linalg.norm(v)
        return (v / n if n else v).astype(np.float32)

    def _encode(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([self._embed(t) for t in texts]).astype(np.float32)

    def encode_document(self, texts, dim=None) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, texts, dim=None) -> np.ndarray:
        return self._encode(texts)

    def encode_plain(self, texts, dim=None) -> np.ndarray:
        return self._encode(texts)

    def pop_ms(self) -> float:
        ms, self.total_ms = self.total_ms, 0.0
        return ms


class FakeLLM:
    """Merge-to-gist dreaming for ENGRAM eval. save_memory is called directly,
    so converse/extract are not exercised here."""

    def dream_cluster(self, members: list[dict], current_time: str = "") -> dict:
        if not members:
            return {"action": "none", "memories": []}
        gist = " / ".join(m["text"] for m in members)[:160]
        return {"action": "merge",
                "memories": [{"text": gist, "timezone": members[0].get("timezone", "Asia/Tokyo")}]}

    def converse(self, memory_pack, user_text, tools, current_time=""):  # not used in eval
        raise NotImplementedError

    def extract_save_candidates(self, user_text, assistant_text, current_time=""):  # not used in eval
        return []


class VirtualClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)

    def advance_days(self, days: float) -> None:
        self.t += float(days) * _DAY
