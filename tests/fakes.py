"""Deterministic mocks so the engine can be evaluated without the real embedding
model or an LLM API key.

``FakeEmbeddingProvider`` builds vectors from token overlap and is bit-for-bit
identical to the Dart test double (FNV-1a token seeds → splitmix64 streams), which
is what makes the cross-language conformance test possible."""
from __future__ import annotations

import re

import numpy as np

_DAY = 24 * 60 * 60
_M64 = (1 << 64) - 1


def _tokens(text: str) -> list[str]:
    text = str(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    cjk = [c for c in text if "぀" <= c <= "鿿" or "ｦ" <= c <= "ﾝ"]
    return words + cjk


def _fnv1a(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & _M64
    return h


def _splitmix(x: int) -> int:
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


class FakeEmbeddingProvider:
    def __init__(self, dimension: int = 64, model_id: str = "fake/token-overlap"):
        self.dimension = dimension
        self.model_id = model_id
        self.total_ms = 0.0
        self._cache: dict[str, np.ndarray] = {}

    def _tok_vec(self, tok: str) -> np.ndarray:
        v = self._cache.get(tok)
        if v is None:
            state = _fnv1a(tok)
            out = np.empty(self.dimension, dtype=np.float64)
            for i in range(self.dimension):
                state = (state + 0x9E3779B97F4A7C15) & _M64
                out[i] = (_splitmix(state) >> 11) / 9007199254740992.0 * 2 - 1
            v = self._cache[tok] = out
        return v

    def embed(self, text: str) -> np.ndarray:
        toks = _tokens(text)
        v = np.ones(self.dimension) if not toks else np.sum([self._tok_vec(t) for t in toks], axis=0)
        n = float(np.linalg.norm(v))
        return (v / n if n else v).astype(np.float32)

    def encode_document(self, texts) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]).astype(np.float32)

    encode_query = encode_document

    def pop_ms(self) -> float:
        ms, self.total_ms = self.total_ms, 0.0
        return ms


class BrokenEmbeddingProvider(FakeEmbeddingProvider):
    """Simulates an unavailable model."""

    def __init__(self):
        super().__init__(model_id="fake/broken")

    def encode_document(self, texts):
        raise RuntimeError("embedder offline")

    encode_query = encode_document


class FakeLLM:
    """Merge-to-gist dreaming (joins member texts); only used as the dream adjudicator."""

    def __init__(self, action: str = "replace"):
        self.action = action
        self.dream_calls = 0

    def dream_cluster(self, members: list[dict], current_time: str = "") -> dict:
        self.dream_calls += 1
        if self.action == "keep" or not members:
            return {"action": "keep", "ids": [], "memories": []}
        if self.action == "error":
            raise RuntimeError("llm down")
        return {"action": "replace", "ids": [m["id"] for m in members[1:]],
                "memories": [" / ".join(m["text"] for m in members)]}


class VirtualClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)

    def advance_days(self, days: float) -> None:
        self.t += float(days) * _DAY
