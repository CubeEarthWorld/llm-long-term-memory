"""Shared pytest fixtures for the LLM Long-Term Memory (ENGRAM v1.1) test suite.

All tests run with deterministic mocks (token-overlap embeddings + a scriptable
fake LLM + a virtual clock) so no model download or API key is required.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make the repository root importable regardless of where pytest is invoked.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import default_config  # noqa: E402
from core.storage import Store  # noqa: E402
from eval.mocks import FakeEmbeddingProvider, VirtualClock  # noqa: E402
from memory.long_term_memory import LongTermMemory  # noqa: E402


class _ConvResult:
    """Minimal stand-in for core.llm_client.ConverseResult."""

    def __init__(self, text: str = "ok", prompt: str = ""):
        self.text = text
        self.prompt = prompt
        self.invocations: list = []
        self.ok = True
        self.error = None
        self.rounds = 1


class ScriptableLLM:
    """Configurable fake LLM exercising the three generation points.

    * ``save_on_converse``: when True, converse() calls save_memory(user_text)
      so a full turn writes a memory (used to drive TurnRunner end-to-end).
    * ``dream_action``: 'merge' | 'split' | 'none' returned by dream_cluster.
    * ``extract``: list of strings returned by extract_save_candidates.
    """

    def __init__(self, save_on_converse: bool = True, dream_action: str = "merge",
                 extract: list[str] | None = None):
        self.save_on_converse = save_on_converse
        self.dream_action = dream_action
        self.extract = extract or []
        self.converse_calls = 0
        self.dream_calls = 0
        self.converse_current_time: str | None = None
        self.extract_current_time: str | None = None
        self.dream_current_time: str | None = None

    def converse(self, memory_pack, user_text, tools, current_time=""):
        self.converse_calls += 1
        self.converse_current_time = current_time
        if self.save_on_converse and "save_memory" in tools:
            tools["save_memory"](text=user_text[:170])
        return _ConvResult(text="ok", prompt=f"[pack]{memory_pack}[/pack]{user_text}")

    def extract_save_candidates(self, user_text, assistant_text, current_time=""):
        self.extract_current_time = current_time
        return list(self.extract)

    def dream_cluster(self, members, current_time=""):
        self.dream_calls += 1
        self.dream_current_time = current_time
        if not members or self.dream_action == "none":
            return {"action": "none", "memories": []}
        if self.dream_action == "split":
            return {"action": "split",
                    "memories": [{"text": m["text"][:160], "timezone": "Asia/Tokyo"} for m in members[:2]]}
        gist = " / ".join(m["text"] for m in members)[:160]
        return {"action": "merge", "memories": [{"text": gist, "timezone": "Asia/Tokyo"}]}


@pytest.fixture
def cfg():
    return default_config()


@pytest.fixture
def make_system(tmp_path):
    """Factory: build an isolated LongTermMemory with a virtual clock.

    Usage:  system, clock, cfg = make_system()
            system, clock, cfg = make_system(llm=ScriptableLLM(dream_action='none'))
    """
    created = []

    def _factory(llm=None, config=None, start=1_700_000_000.0):
        c = config or default_config()
        db = tmp_path / f"engram_{len(created)}.db"
        store = Store(str(db))
        provider = FakeEmbeddingProvider(c.glob.dim_full)
        system = LongTermMemory(store, provider, llm or ScriptableLLM(), c.memory, c.glob)
        system.reset()
        clock = VirtualClock(start=start)
        system.set_clock(clock)
        created.append(system)
        return system, clock, c

    yield _factory
    for s in created:
        try:
            s.store.close()
        except Exception:
            pass


@pytest.fixture
def system(make_system):
    s, clock, c = make_system()
    return s


def tokens(n, prefix="t"):
    """n space-separated distinct tokens; disjoint prefixes give near-orthogonal vecs."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def by_text(system, text):
    return system.store.one(
        "SELECT * FROM memory WHERE text=? AND superseded_by IS NULL", (text,))
