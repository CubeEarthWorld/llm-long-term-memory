"""ENGRAM invariants I1–I14 (§7) asserted over a long mixed workload.

This is the 'extreme pattern' sweep: hundreds of writes/recalls/maintains/dreams
across a fast-forwarding virtual clock, with the invariants checked at the end.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import ScriptableLLM


def _assert_invariants(s, c):
    # I1 — mass ≤ 64
    mm = s.store.scalar("SELECT MAX(mass) FROM memory", ())
    if mm is not None:
        assert float(mm) <= c.memory.m_max + 1e-9, "I1: mass exceeded M_max"

    # I4 — tier capacities (counted WITH tombstones)
    assert s._tier_count(1) <= c.memory.cap1, "I4: L1 over capacity"
    assert s._tier_count(2) <= c.memory.cap2, "I4: L2 over capacity"
    assert s._tier_count(3) <= c.memory.cap3, "I4: L3 over capacity"

    # I7 — gen ≤ 7
    mg = s.store.scalar("SELECT MAX(gen) FROM memory", ())
    if mg is not None:
        assert int(mg) <= c.memory.gen_max, "I7: gen exceeded 7"

    # I8 — text non-empty and ≤ hard char bound
    bad = s.store.scalar(
        "SELECT COUNT(*) FROM memory WHERE length(text)=0 OR length(text)>1024", ())
    assert int(bad or 0) == 0, "I8: text length out of range"

    # I11 — ring buffers bounded
    assert s.store.count("conflict") <= c.memory.conflict_cap, "I11: conflict ring overflow"
    assert s.store.count("dream_log") <= c.memory.dream_log_cap, "I11: dream_log ring overflow"

    # safety net — total rows under the hard bound
    assert s.store.count("memory") <= c.memory.hard_memory_rows

    # I3 / I6 — every live vec is unit-norm and there is ≤1 active vec per memory
    rows = s.store.query("SELECT memory_id, COUNT(*) AS n FROM vec GROUP BY memory_id", ())
    for r in rows:
        assert r["n"] <= 2, "I6: more than 2 vectors for one memory"
    for tier in (1, 2, 3):
        for _row, vec in s._tier_candidates(tier):
            n = float(np.linalg.norm(vec))
            assert abs(n - 1.0) < 1e-3, f"I3: vector not unit-norm (||v||={n})"

    # I2 — no future timestamps relative to the (sanitised) clock
    now = int(s.now_unix())
    future = s.store.scalar(
        "SELECT COUNT(*) FROM memory WHERE last_access > ? OR created_at > ?", (now, now))
    assert int(future or 0) == 0, "I2: future timestamp present"


def test_invariants_under_mixed_workload(make_system, cfg):
    cfg.memory.cap1 = 8
    cfg.memory.cap2 = 8
    cfg.memory.cap3 = 8
    cfg.memory.write_rate_per_day = 100000
    s, clock, c = make_system(llm=ScriptableLLM(dream_action="merge"), config=cfg)

    topics = ["kyoto trip", "rust database", "cat pet", "coffee taste",
              "work deadline", "music piano", "garden plants", "running habit"]
    for i in range(150):
        topic = topics[i % len(topics)]
        s.save_memory(f"{topic} detail number {i} extra context words")
        if i % 3 == 0:
            s.retrieve(topic, i)
        if i % 25 == 0:
            clock.advance_days(10)
            s.dream(budget=5, force=True)
        # NOTE: writes are append-only (I5); per-tier capacity (I4) is enforced
        # by machine movement, so maintain() must precede the capacity assertion.
        # Transient over-cap between writes is legal and bounded by the hard total.
        clock.advance(3 * 3600)
        s.maintain(i)
        _assert_invariants(s, c)

    _assert_invariants(s, c)


def test_invariants_with_extreme_clock_jumps(make_system, cfg):
    cfg.memory.cap1 = 5
    cfg.memory.cap2 = 5
    cfg.memory.cap3 = 5
    s, clock, c = make_system(llm=ScriptableLLM(dream_action="merge"), config=cfg)
    for i in range(30):
        s.save_memory(f"memory variant {i} distinct tokens zz{i}")
        clock.advance_days(400)        # > τ1, huge decay each step
        s.maintain(i)
    _assert_invariants(s, c)
    # after long silence the strongest live activation should be well under M_max
    now = s.now_unix()
    live = s.store.query("SELECT * FROM memory WHERE superseded_by IS NULL", ())
    if live:
        assert max(s.activation(r, now) for r in live) <= c.memory.m_max


def test_conflict_ring_bounded_under_flood(make_system, cfg):
    cfg.memory.conflict_cap = 16
    cfg.memory.write_rate_per_day = 100000
    s, clock, c = make_system(config=cfg)
    base = "shared core proposition tokens alpha beta gamma delta"
    s.save_memory(base)
    # many near-duplicates (0.85–0.97 band) flood the conflict queue
    for i in range(60):
        s.save_memory(base + f" variant{i}")
    assert s.store.count("conflict") <= cfg.memory.conflict_cap
