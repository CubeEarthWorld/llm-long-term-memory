"""server.py — the polling endpoint must stay responsive, and read-only
endpoints must not raise.

The reported symptom — 'startup takes very long / no response after seed or
dream' — is reproduced as a deterministic concurrency test: while a long
mutating job holds the global LOCK (model load, seed turn, dream), the
frontend's /api/state poll must still return promptly.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("fastapi")

import server  # noqa: E402
from config import default_config  # noqa: E402
from core.base import TurnRunner  # noqa: E402
from core.metrics import MetricsRecorder  # noqa: E402
from core.storage import Store  # noqa: E402
from eval.mocks import FakeEmbeddingProvider  # noqa: E402
from memory.long_term_memory import LongTermMemory  # noqa: E402
from conftest import ScriptableLLM  # noqa: E402


def _install_fake_engine(tmp_path):
    cfg = default_config()
    provider = FakeEmbeddingProvider(cfg.glob.dim_full)
    provider.status = "OK — fake embedding"      # real provider exposes .status
    llm = ScriptableLLM(dream_action="merge")
    llm.status = "OK — fake llm"                  # real LLMClient exposes .status
    store = Store(str(tmp_path / "srv.db"))
    system = LongTermMemory(store, provider, llm, cfg.memory, cfg.glob)
    system.reset()
    recorder = MetricsRecorder(max_history=cfg.glob.max_metrics_history)
    engine = {
        "provider": provider, "llm": llm, "store": store, "system": system,
        "recorder": recorder, "runner": TurnRunner(provider, llm, recorder, system),
        "turn": 0, "log": [], "start_time": None, "seeded": False,
        "last_dream": [], "cfg": cfg, "seed": [],
    }
    server.ENGINE["e"] = engine
    server.ENGINE["cfg"] = cfg
    server.STATE.update(ready=True, running=False, progress="", error=None, init_error=None)
    return engine


def test_get_state_stays_responsive_while_lock_held():
    """/api/state must NOT block on the heavy mutation LOCK.

    Holding LOCK simulates init_engine's model load or an in-flight seed/dream
    job. If get_state() blocks on LOCK, the UI shows 'no response'.
    """
    server.ENGINE["e"] = None
    server.STATE.update(ready=False, running=True, progress="loading", error=None, init_error=None)

    release = threading.Event()
    completed = threading.Event()
    result = {}

    def hold_lock():
        with server.LOCK:
            release.wait(timeout=10)

    def poll():
        result["state"] = server.get_state()
        completed.set()

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    time.sleep(0.15)                      # ensure LOCK is held

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    responsive = completed.wait(timeout=2.0)

    release.set()
    holder.join(timeout=10)
    poller.join(timeout=10)

    assert responsive, "/api/state blocked while LOCK was held — UI cannot poll during load/seed/dream"
    assert result["state"]["running"] is True


def test_read_only_endpoints_do_not_raise(tmp_path):
    _install_fake_engine(tmp_path)
    sysm = server.ENGINE["e"]["system"]
    for tail in ("plan", "hotel", "food"):
        sysm.save_memory(f"trip kyoto {tail} schedule note")

    assert isinstance(server.get_state(), dict)
    assert isinstance(server.get_config(), dict)
    assert "stats" in server.db()
    assert "rows" in server.metrics()
    assert "clusters" in server.clusters()
    assert "results" in server.dream_log()
    assert "turns" in server.turns_detail()
    assert "utterances" in server.seed_utts()


def test_db_endpoint_snapshot_shape(tmp_path):
    _install_fake_engine(tmp_path)
    sysm = server.ENGINE["e"]["system"]
    sysm.save_memory("snapshot shape probe memory")
    payload = server.db()
    assert set(payload["tables"]).issuperset({"memory", "vec", "spec"})
    assert payload["stats"]["total_records"] >= 1
