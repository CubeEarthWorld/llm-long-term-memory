"""Shared fixtures: deterministic embedder + fake LLM + virtual clock, no model or key needed."""
from __future__ import annotations

import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_TESTS), _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import Config  # noqa: E402
from core.storage import Store  # noqa: E402
from fakes import FakeEmbeddingProvider, FakeLLM, VirtualClock  # noqa: E402
from memory.engine import LongTermMemory  # noqa: E402


@pytest.fixture
def cfg():
    c = Config()
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
        system = LongTermMemory(store, provider, llm or FakeLLM(), c.memory, c.glob.default_timezone, clock)
        created.append(system)
        return system, clock

    yield _factory
    for s in created:
        s.store.close()
