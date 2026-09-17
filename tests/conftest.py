"""Shared fixtures: deterministic embedder + fake LLM + virtual clock, no model or key needed."""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import default_config  # noqa: E402
from core.storage import Store  # noqa: E402
from eval.mocks import FakeEmbeddingProvider, FakeLLM, VirtualClock  # noqa: E402
from memory.engine import LongTermMemory  # noqa: E402


@pytest.fixture
def cfg():
    c = default_config()
    c.memory.cosine_floor = 0.0          # the fake embedder is centred at zero
    return c


@pytest.fixture
def make_system(tmp_path, cfg):
    """Factory: build an isolated LongTermMemory with a virtual clock.

    Usage:  system, clock = make_system()
            system, clock = make_system(llm=FakeLLM("keep"), config=cfg, provider=..., store=...)
    """
    created = []

    def _factory(llm=None, config=None, provider=None, store=None, start=1_700_000_000.0):
        c = config or cfg
        store = store or Store(str(tmp_path / f"engram_{len(created)}.db"))
        provider = provider or FakeEmbeddingProvider()
        clock = VirtualClock(start=start)
        system = LongTermMemory.__new__(LongTermMemory)
        system.store, system.provider, system.llm, system.cfg, system.glob = store, provider, llm or FakeLLM(), c.memory, c.glob
        system._clock, system._traces, system._write_times, system._last_recall = clock, {}, [], {}
        system.initialize()
        created.append(system)
        return system, clock

    yield _factory
    for s in created:
        s.store.close()


def tokens(n, prefix="t"):
    """n distinct tokens; disjoint prefixes give near-orthogonal vectors."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def by_text(system, text):
    return next((m for m in system.memories() if m.text == text), None)
