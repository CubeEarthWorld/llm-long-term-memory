"""The forgetting curve and spacing effect (SPEC §3)."""
from __future__ import annotations


def test_retrievability_halves_every_stability(make_system):
    s, _ = make_system()
    m = s.memory(s.remember("half life")["id"])
    t0 = m.last_recall
    assert s.retrievability(m, t0) == 1.0
    assert abs(s.retrievability(m, t0 + 86400) - 0.5) < 1e-9
    assert abs(s.retrievability(m, t0 + 2 * 86400) - 0.25) < 1e-9
    assert s.retrievability(m, t0 - 1000) == 1.0          # clock rollback
    assert s.retrievability(m, t0 + 86400 * 100000) == 0.0  # underflow is exactly 0


def test_spaced_beats_massed(make_system):
    s, clock = make_system()
    massed = s.remember("massed alpha")["id"]
    spaced = s.remember("spaced beta")["id"]
    for _ in range(3):
        clock.advance(60)
        s.recall("massed alpha")
    clock.advance_days(3)
    s.recall("spaced beta")
    assert s.memory(massed).stability < 86400 * 1.05
    assert s.memory(spaced).stability > 86400 * 2


def test_no_immortal_memory(make_system, cfg):
    s, clock = make_system()
    mid = s.remember("rehearsed daily")["id"]
    for _ in range(400):
        clock.advance_days(30)
        s.recall("rehearsed daily")
    m = s.memory(mid)
    assert m.stability == cfg.memory.max_stability
    assert s.retrievability(m, clock.t + 100 * 365 * 86400) < 0.001


def test_strength_orders_stable_above_fresh_junk(make_system):
    s, clock = make_system()
    fact = s.remember("stable fact", salience=10)["id"]
    clock.advance_days(5)
    junk = s.memory(s.remember("junk")["id"])
    assert s.strength(s.memory(fact), clock.t) > s.strength(junk, clock.t)


def test_dormant_but_relevant_still_competes(make_system):
    s, clock = make_system()
    s.remember("rust database project")
    clock.advance_days(3650)
    r = s.recall("rust database")["recalled"][0]
    assert r["R"] < 1e-6 and r["score"] > 0.25
