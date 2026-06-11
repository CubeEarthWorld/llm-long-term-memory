"""save_memory (§5.1) and retrieve (§5.2) — normal, edge and extreme inputs."""
from __future__ import annotations

import pytest

from conftest import by_text, tokens


# ====================================================================== #
# WRITE — validation & identity thresholds
# ====================================================================== #
def test_save_empty_rejected(system):
    assert system.save_memory("").get("action") == "rejected"
    assert system.save_memory("    ").get("action") == "rejected"
    assert system.save_memory("\n\t  \n").get("action") == "rejected"


def test_save_oversize_bytes_rejected(system):
    ev = system.save_memory("a" * 2000)   # 2000 bytes > 1024 hard bound
    assert ev.get("action") == "rejected"
    assert system.total_records() == 0


def test_save_long_text_is_shortened_to_text_max(system):
    ev = system.save_memory("b" * 171)
    assert ev["action"] == "inserted"
    row = by_text(system, "b" * 170) or system.store.one(
        "SELECT * FROM memory WHERE text LIKE 'b%'", ())
    assert len(row["text"]) <= 170


def test_save_exactly_170_chars_kept(system):
    text = "c" * 170
    system.save_memory(text)
    assert by_text(system, text) is not None


def test_save_surrogate_text_does_not_crash(system):
    """Un-encodable text (lone surrogate) must be rejected, not raise (I14).

    §8.1 mandates byte-length validation; a corrupt write should be refused
    gracefully rather than crash the turn with UnicodeEncodeError.
    """
    ev = system.save_memory("\ud800 broken surrogate")
    assert ev.get("action") in ("rejected", "inserted")  # never raises


def test_new_insert_disjoint_tokens(system):
    system.save_memory(tokens(10, "a"))
    ev = system.save_memory(tokens(10, "b"))
    assert ev["action"] == "inserted"
    assert system.total_records() == 2


def test_exact_duplicate_is_rehearsal_no_new_row(system, make_system):
    s, clock, _ = make_system()
    s.save_memory(tokens(8))
    clock.advance(2 * 3600)
    ev = s.save_memory(tokens(8))
    assert ev["action"] == "reinforced"
    assert s.total_records() == 1
    assert float(by_text(s, tokens(8))["mass"]) > 1.0


def test_theta_same_supersedes(system):
    system.save_memory("cat dog bird")
    ev = system.save_memory("cat dog bird!")   # ≥0.97 → supersede
    assert ev["action"] == "updated"
    assert system.total_records() == 1
    assert system.stats()["tombstones"] == 1


def test_theta_conflict_enqueues(system):
    system.save_memory(tokens(20))
    ev = system.save_memory(tokens(18) + " z0 z1")   # ~0.9 → conflict band
    assert ev["action"] == "conflict"
    assert system.stats()["conflict"] == 1
    assert system.total_records() == 2


def test_write_rate_limit(make_system, cfg):
    cfg.memory.write_rate_per_day = 3
    s, clock, _ = make_system(config=cfg)
    actions = [s.save_memory(f"distinct memory {i} qqq{i}")["action"] for i in range(6)]
    assert "rate_limited" in actions
    # at most the rate cap of genuinely-new rows landed
    assert s.total_records() <= 3


def test_mass_never_exceeds_m_max(make_system, cfg):
    s, clock, _ = make_system(config=cfg)
    s.save_memory("rehearse me please")
    for _ in range(200):
        clock.advance(2 * 3600)               # always past refractory
        s.save_memory("rehearse me please")   # exact-text rehearsal +1
    row = by_text(s, "rehearse me please")
    assert float(row["mass"]) <= cfg.memory.m_max + 1e-9


# ====================================================================== #
# READ — retrieve
# ====================================================================== #
def test_retrieve_empty_query(system):
    res = system.retrieve("", 1)
    assert res.pack_text == ""
    assert res.recalled == []


def test_retrieve_whitespace_query(system):
    assert system.retrieve("\n\n\n", 1).pack_text == ""


def test_retrieve_returns_relevant_memory(system):
    system.save_memory("user lives in kyoto city")
    res = system.retrieve("where does the user live kyoto", 2)
    assert "kyoto" in res.pack_text


def test_retrieve_respects_budget_chars(make_system, cfg):
    s, clock, c = make_system(config=cfg)
    for i in range(40):
        s.save_memory(f"shared topic memory variant {i} aaa bbb ccc")
    res = s.retrieve("shared topic memory aaa bbb ccc", 2)
    assert len(res.pack_text) <= c.glob.budget_chars
    assert len(res.recalled) <= c.memory.inject_n   # ≤5 injected (§4.2)


def test_retrieve_huge_query_does_not_crash(system):
    system.save_memory("token alpha beta gamma")
    res = system.retrieve("token " * 5000, 2)
    assert isinstance(res.pack_text, str)


def test_retrieve_recall_updates_last_access(make_system):
    s, clock, _ = make_system()
    s.save_memory("recall target memory zeta")
    before = by_text(s, "recall target memory zeta")["last_access"]
    clock.advance(3 * 3600)
    s.retrieve("recall target memory zeta", 2)
    after = by_text(s, "recall target memory zeta")["last_access"]
    assert after > before


def test_filter_by_score_relaxes_strict_first(system):
    """_filter_by_score keeps the strictest non-empty threshold, then relaxes.

    Regression: with thresholds declared loosest-first, the stricter threshold
    was dead code and never tightened the candidate set.
    """
    system.memory.score_thresholds = (0.2, 0.1)
    items = [{"score": 0.25}, {"score": 0.15}, {"score": 0.05}]
    kept = system._filter_by_score(items)
    assert kept == [{"score": 0.25}]   # 0.2 applied first (strictest non-empty)

    # if nothing clears the strict bar, relax to the looser one
    items2 = [{"score": 0.15}, {"score": 0.05}]
    kept2 = system._filter_by_score(items2)
    assert {round(it["score"], 2) for it in kept2} == {0.15}

    # if even the loosest bar is missed, inject nothing — unrelated memories
    # must not be recalled (and thereby reinforced) just because they exist
    items3 = [{"score": 0.05}, {"score": 0.01}]
    assert system._filter_by_score(items3) == []


def test_injected_pack_includes_id_for_delete(system):
    system.save_memory("user owns a vintage motorcycle")
    res = system.retrieve("tell me about the user motorcycle", 2)
    assert "《id:" in res.pack_text   # id must be present so delete_memory works


def test_injected_pack_uses_unix_and_timezone(make_system):
    """Injection header is raw Unix seconds + timezone, not a formatted datetime."""
    import re
    s, clock, _ = make_system(start=1_700_000_000.0)
    s.save_memory("user enjoys hiking in the mountains")
    res = s.retrieve("tell me about user hiking mountains", 2)
    assert "Asia/Tokyo;+09:00" in res.pack_text    # timezone present
    assert "1700000000" in res.pack_text           # raw unix seconds present
    header = res.pack_text.split("]", 1)[0]
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", header)  # no formatted clock time


# ====================================================================== #
# DELETE — §5.3
# ====================================================================== #
def test_delete_tombstones(system):
    ev = system.save_memory("forgettable fact xyz")
    mid = ev["id"]
    out = system.delete_memory(mid)
    assert out["action"] == "tombstoned"
    assert system.total_records() == 0                 # excluded from live count
    assert system.stats()["tombstones"] == 1


def test_delete_hard_physical(system):
    ev = system.save_memory("hard delete me now")
    out = system.delete_memory(ev["id"], hard=True)
    assert out["action"] == "deleted"
    assert system.store.count("memory") == 0
    assert system.store.count("vec") == 0              # vec cascades


def test_delete_not_found(system):
    assert system.delete_memory("01NONEXISTENTULID0000000000")["action"] == "not_found"
