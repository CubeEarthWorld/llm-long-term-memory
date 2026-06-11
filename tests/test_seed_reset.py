"""Seed start must reset the DB (the UI's default and forced path).

/api/seed -> _start_seed_run(do_reset=True) -> reset_state(engine) ->
system.reset(): every memory/vec/conflict/dream_log/turn_log row is deleted and
the self-describing spec is reinstalled.
"""
from __future__ import annotations

from core.engine import reset_state
from core.metrics import MetricsRecorder


def test_system_reset_clears_all_rows(system):
    system.save_memory("alpha one fact about the user")
    system.save_memory("beta two fact about the user")
    assert system.total_records() == 2
    assert system.store.count("vec") == 2

    system.reset()

    assert system.total_records() == 0
    assert system.store.count("memory") == 0
    assert system.store.count("vec") == 0
    assert system.store.count("conflict") == 0
    assert system.store.count("dream_log") == 0
    assert system.store.count("spec") > 0   # spec reinstalled (self-describing DB)


def test_reset_state_resets_engine_and_db(make_system):
    """reset_state — invoked by the seed run when reset=True — wipes DB + session."""
    s, clock, _ = make_system()
    s.save_memory("memory present before the seed reset")
    assert s.total_records() == 1

    engine = {
        "system": s,
        "recorder": MetricsRecorder(),
        "turn": 7,
        "log": [{"turn": 7, "utterance": "x"}],
        "seeded": True,
        "last_dream": [{"action": "merge"}],
    }
    reset_state(engine)

    assert s.total_records() == 0          # DB wiped
    assert engine["turn"] == 0
    assert engine["log"] == []
    assert engine["seeded"] is False
    assert engine["last_dream"] == []


def test_reset_then_resave_works(system):
    """After a reset the engine is usable again (fresh seed can write)."""
    system.save_memory("first generation memory")
    system.reset()
    ev = system.save_memory("post-reset memory writes fine")
    assert ev["action"] == "inserted"
    assert system.total_records() == 1
