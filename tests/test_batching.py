"""Cue embeddings are batched: clusters() and dream() embed the distinct non-empty cues of
their whole seed list in one ``encode_query`` call, and clusters() still returns exactly
what the per-seed 1.0.1 algorithm returned (mirrors the Dart vector_index_test)."""
from __future__ import annotations

import numpy as np

from fakes import FakeEmbeddingProvider

_WORDS = ["kyoto", "trip", "hotel", "train", "temple", "autumn"]


class CountingProvider(FakeEmbeddingProvider):
    """Records every ``encode_query`` batch; can fail them or mis-size one cue."""

    def __init__(self, offline_queries: bool = False, bad_cue: str | None = None):
        super().__init__()
        self.offline_queries = offline_queries
        self.bad_cue = bad_cue
        self.query_batches: list[list[str]] = []

    def encode_query(self, texts):
        self.query_batches.append(list(texts))
        if self.offline_queries:
            raise RuntimeError("embedder offline")
        return [np.zeros(self.dimension // 2, dtype=np.float32) if t == self.bad_cue else self.embed(t)
                for t in texts]


def _related(make_system, provider):
    s, clock = make_system(provider=provider)
    for i in range(40):
        clock.advance(3600)
        cue = "" if i % 4 == 0 else f"where {_WORDS[i % 3]} {_WORDS[(i + 3) % 6]}"
        s.remember(f"{_WORDS[i % 6]} {_WORDS[(i + 1) % 6]} {_WORDS[(i + 2) % 6]} n{i}", cue=cue)
    return s


def _reference_clusters(s, cue_vector, budget):
    """clusters() as 1.0.1 computed it: one embedding and one full scan per seed."""
    rows = [m for m in s.memories() if s._is_indexed(m)]
    M = np.stack([m.vector for m in rows])
    seeds = [m for m in rows if not m.consolidated][: 8 * budget]
    out = []
    for seed in seeds:
        sims = M @ cue_vector(seed)
        near = sorted(((float(sims[i]), i) for i in range(len(rows))
                       if float(sims[i]) >= s.cfg.theta_related
                       and rows[i].id != seed.id and rows[i].created_at <= seed.created_at),
                      key=lambda t: (-t[0], t[1]))
        cluster = [seed.text] + [rows[i].text for _, i in near[: s.cfg.dream_max_members - 1]]
        if len(cluster) >= 2:
            out.append(cluster)
    return out


def _texts(clusters):
    return [[m.text for m in c] for c in clusters]


def test_clusters_embed_every_cue_in_one_call(make_system):
    provider = CountingProvider()
    s = _related(make_system, provider)
    seeds = [m for m in s.memories() if not m.consolidated][:80]
    assert len(seeds) > 10
    provider.query_batches.clear()
    got = s.clusters(budget=10)
    cues: list[str] = []
    for m in seeds:
        if m.cue and m.cue not in cues:
            cues.append(m.cue)
    assert provider.query_batches == [cues]
    fake = FakeEmbeddingProvider()
    assert got and _texts(got) == _reference_clusters(
        s, lambda m: m.vector if not m.cue else fake.embed(m.cue), 10)


def test_dream_embeds_every_cue_in_one_call(make_system):
    provider = CountingProvider()
    s = _related(make_system, provider)
    provider.query_batches.clear()
    assert s.dream(budget=10)
    assert len(provider.query_batches) == 1
    batch = provider.query_batches[0]
    assert len(set(batch)) == len(batch)


def test_no_cues_no_query_call(make_system):
    provider = CountingProvider()
    s, _ = make_system(provider=provider)
    for t in ("alpha beta gamma", "alpha beta delta", "alpha beta"):
        s.remember(t)
    provider.query_batches.clear()
    assert s.clusters()
    s.dream()
    assert provider.query_batches == []


def test_offline_queries_fall_back_to_own_vectors(make_system):
    provider = CountingProvider(offline_queries=True)
    s = _related(make_system, provider)
    provider.query_batches.clear()
    got = s.clusters(budget=10)
    assert len(provider.query_batches) == 1
    assert _texts(got) == _reference_clusters(s, lambda m: m.vector, 10)


def test_wrong_dimension_cue_falls_back_for_that_cue_only(make_system):
    bad = "where kyoto train"
    provider = CountingProvider(bad_cue=bad)
    s = _related(make_system, provider)
    assert any(m.cue == bad and not m.consolidated for m in s.memories())
    fake = FakeEmbeddingProvider()
    assert _texts(s.clusters(budget=10)) == _reference_clusters(
        s, lambda m: m.vector if not m.cue or m.cue == bad else fake.embed(m.cue), 10)
