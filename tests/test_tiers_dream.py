"""Tier movement (§5.4) and dream consolidation (§6) — including extreme caps."""
from __future__ import annotations

import pytest

from conftest import ScriptableLLM, tokens


# ====================================================================== #
# machine movement
# ====================================================================== #
def test_l1_overflow_demotes_to_l2(make_system, cfg):
    cfg.memory.cap1 = 3
    s, clock, _ = make_system(config=cfg)
    for i in range(6):
        s.save_memory(tokens(6, f"g{i}_"))
    s.maintain(1)
    st = s.stats()
    assert st["L1"] <= 3
    assert st["L2"] >= 3
    assert s.total_records() == 6


def test_promote_on_high_activation(make_system, cfg):
    cfg.memory.cap1 = 1
    s, clock, _ = make_system(config=cfg)
    s.save_memory("anchor keep one")
    s.save_memory(tokens(6, "p"))
    s.maintain(1)
    demoted = s.store.one("SELECT * FROM memory WHERE tier=2 AND superseded_by IS NULL", ())
    assert demoted is not None
    s.store.exec("UPDATE memory SET mass=?, last_access=? WHERE id=?",
                 (20.0, int(s.now_unix()), demoted["id"]))   # A ≥ θ_up
    s._promote(s.now_unix())
    assert s.store.scalar("SELECT tier FROM memory WHERE id=?", (demoted["id"],)) == 1


def test_l3_overflow_eviction_is_death(make_system, cfg):
    cfg.memory.cap1 = cfg.memory.cap2 = cfg.memory.cap3 = 1
    s, clock, _ = make_system(config=cfg)
    for i in range(4):
        s.save_memory(tokens(5, f"d{i}_"))
    before = s.total_records()
    s.maintain(1)
    assert before == 4
    assert s.total_records() == 3   # one physically evicted from L3


def test_demoted_vector_is_coarsened(make_system, cfg):
    cfg.memory.cap1 = 2
    s, clock, _ = make_system(config=cfg)
    for i in range(5):
        s.save_memory(tokens(6, f"v{i}_"))
    s.maintain(1)
    # demoted rows should carry int8 256-d vectors (L2 MRL form)
    l2vecs = s.store.query(
        "SELECT v.dim, v.dtype FROM vec v JOIN memory m ON m.id=v.memory_id WHERE m.tier=2", ())
    assert l2vecs
    for r in l2vecs:
        assert r["dim"] == cfg.memory.dim2
        assert r["dtype"] == "int8"


# ====================================================================== #
# dream
# ====================================================================== #
def test_dream_no_candidates_below_cluster_min(make_system):
    s, clock, _ = make_system()
    s.save_memory("alpha one")
    s.save_memory("beta two")     # only 2 memories < cluster_min(3)
    assert s.dream(force=True) == []


def test_dream_merge_reduces_rows_and_bumps_gen(make_system):
    s, clock, _ = make_system(llm=ScriptableLLM(dream_action="merge"))
    for tail in ("plan", "hotel", "food"):
        s.save_memory(f"trip kyoto {tail} schedule note")
    before = s.total_records()
    results = s.dream(budget=5, force=True)
    merged = [r for r in results if r["action"] == "merge"]
    assert before == 3
    assert s.total_records() < before
    assert merged
    assert max(a["gen"] for r in merged for a in r["after"]) == 1


def test_dream_idempotent_via_fingerprint(make_system):
    s, clock, _ = make_system(llm=ScriptableLLM(dream_action="merge"))
    for tail in ("plan", "hotel", "food"):
        s.save_memory(f"trip kyoto {tail} schedule note")
    s.dream(budget=5, force=True)
    n_after_first = s.total_records()
    second = s.dream(budget=5, force=True)
    # nothing left to merge / fingerprint guard → no further consolidation
    assert all(r["action"] == "none" for r in second) or second == []
    assert s.total_records() == n_after_first


def test_dream_none_action_records_unchanged(make_system):
    s, clock, _ = make_system(llm=ScriptableLLM(dream_action="none"))
    for tail in ("plan", "hotel", "food"):
        s.save_memory(f"trip kyoto {tail} schedule note")
    before = s.total_records()
    results = s.dream(budget=5, force=True)
    assert s.total_records() == before   # 'none' leaves bodies intact
    assert all(r["action"] == "none" for r in results)
    assert s.store.count("dream_log") >= 1


def test_dream_budget_zero_runs_only_chores(make_system):
    s, clock, _ = make_system(llm=ScriptableLLM(dream_action="merge"))
    for tail in ("plan", "hotel", "food"):
        s.save_memory(f"trip kyoto {tail} schedule note")
    before = s.total_records()
    results = s.dream(budget=0, force=True)
    assert results == []
    assert s.total_records() == before


def test_dream_merged_mass_capped(make_system, cfg):
    s, clock, _ = make_system(llm=ScriptableLLM(dream_action="merge"), config=cfg)
    for tail in ("plan", "hotel", "food", "temple", "garden"):
        s.save_memory(f"trip kyoto {tail} schedule note")
        # pump each member's mass high so Σ would exceed M_max without the cap
        s.store.exec("UPDATE memory SET mass=60 WHERE text=?",
                     (f"trip kyoto {tail} schedule note",))
    s.dream(budget=5, force=True)
    maxmass = s.store.scalar("SELECT MAX(mass) FROM memory", ())
    assert float(maxmass) <= cfg.memory.m_max + 1e-9


def test_dream_passes_dream_time_not_creation_time(make_system):
    """Consolidation must see 'now' so relative dates anchored to each memory's
    local_time can be rewritten as absolute dates (the merged row gets a fresh
    created_at, so a surviving『再来週』would silently shift its meaning)."""
    s, clock, _ = make_system(llm=ScriptableLLM(dream_action="merge"))
    for tail in ("plan", "hotel", "food"):
        s.save_memory(f"trip kyoto {tail} schedule note")
    created_local = s.now_local()
    clock.advance_days(21)            # dream runs 3 weeks after the memories were written
    s.dream(budget=5, force=True)
    assert s.llm.dream_current_time == s.now_local()
    assert s.llm.dream_current_time != created_local


def test_dream_snapshot_created(make_system):
    s, clock, _ = make_system(llm=ScriptableLLM(dream_action="merge"))
    for tail in ("plan", "hotel", "food"):
        s.save_memory(f"trip kyoto {tail} schedule note")
    s.dream(budget=5, force=True)
    import os
    snaps = [f for f in os.listdir(s._snap_dir) if f.endswith(".db")] if os.path.isdir(s._snap_dir) else []
    assert snaps   # 8-gen snapshot ring writes at least one file
