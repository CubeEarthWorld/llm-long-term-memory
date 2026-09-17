"""remember / recall / cite / forget / capacity (SPEC §4)."""
from __future__ import annotations

import pytest

from conftest import by_text, tokens
from eval.mocks import BrokenEmbeddingProvider, FakeEmbeddingProvider


def test_insert_rehearse_reject(make_system):
    s, clock = make_system()
    a = s.remember("user lives in kyoto")
    assert a["action"] == "inserted" and a["consolidated"] is True and a["stability"] == 86400
    clock.advance_days(2)
    b = s.remember("  user  lives in kyoto ")
    assert b["action"] == "reinforced" and b["id"] == a["id"] and b["stability"] > 86400
    assert s.remember("   ")["action"] == "rejected"
    assert s.total_records() == 1


def test_text_hygiene(make_system, cfg):
    cfg.memory.text_max = 20
    s, _ = make_system(config=cfg)
    r = s.remember("a《id:x》b\n\n c " + "x" * 40)
    assert "《" not in r["text"] and "\n" not in r["text"] and len(r["text"]) <= 20


def test_related_write_is_labile_and_reactivates_neighbour(make_system):
    s, _ = make_system()
    old = s.remember("cat dog bird fish")
    assert old["consolidated"]
    r = s.remember("cat dog bird fish!")
    assert r["cos"] > 0.9 and r["consolidated"] is False
    assert s.memory(old["id"]).consolidated is False
    assert s.total_records() == 2


def test_salience_scales_and_clamps(make_system, cfg):
    s, _ = make_system()
    assert s.remember("alpha", salience=3)["stability"] == 3 * 86400
    huge = by_text(s, "beta") if False else s.memory(s.remember("beta", salience=1e9)["id"])
    assert huge.stability == cfg.memory.max_stability / 365
    zero = s.memory(s.remember("gamma", salience=0)["id"])
    assert zero.stability == 1.0 and s.retrievability(zero, zero.last_recall + 100) < 1e-20


def test_daily_write_limit(make_system, cfg):
    cfg.memory.writes_per_day = 2
    s, clock = make_system(config=cfg)
    s.remember("a1 b1"); s.remember("a2 b2")
    assert s.remember("a3 b3")["action"] == "rate_limited"
    clock.advance_days(1.01)
    assert s.remember("a3 b3")["action"] == "inserted"


def test_embedder_offline_keeps_text(make_system, tmp_path):
    from core.storage import Store
    store = Store(str(tmp_path / "shared.db"))
    broken, _ = make_system(provider=BrokenEmbeddingProvider(), store=store)
    assert broken.remember("kept while offline")["action"] == "inserted"
    assert broken.recall("kept while offline")["recalled"] == []
    healed, _ = make_system(provider=FakeEmbeddingProvider(model_id="fake/broken"), store=store)
    assert healed.recall("kept while offline")["recalled"][0]["text"] == "kept while offline"


def test_eviction_prefers_weak_old_traces(make_system, cfg):
    cfg.memory.capacity, cfg.memory.grace_period = 3, 86400
    s, clock = make_system(config=cfg)
    s.remember("old weak one")
    strong = s.remember("old strong two", salience=10)["id"]
    clock.advance_days(2)
    s.remember("fresh three")
    assert s.remember("fresh four")["evicted"] == "old weak one"
    assert s.memory(strong) is not None


def test_flood_evicts_its_own_members(make_system, cfg):
    cfg.memory.capacity, cfg.memory.grace_period = 20, 1e9
    s, _ = make_system(config=cfg)
    keep = s.remember("precious", salience=10)["id"]
    for i in range(40):
        s.remember(f"junk {i} x{i}")
    assert s.total_records() == 20 and s.memory(keep) is not None


def test_recall_packs_and_strengthens(make_system):
    s, clock = make_system()
    cat = s.memory(s.remember("user has a cat named tama")["id"])
    s.remember("user works at a bank")
    clock.advance_days(1)
    r = s.recall("tell me about the cat tama")
    assert r["recalled"][0]["id"] == cat.id
    assert r["pack_text"].startswith(f"[{cat.created_at} Asia/Tokyo;+09:00] user has a cat named tama　《id:{cat.id}》\n")
    after = s.memory(cat.id)
    assert after.stability > cat.stability and after.last_recall == int(clock.t)


def test_recall_nothing_when_nothing_matches(make_system, cfg):
    cfg.memory.min_score = 0.25
    s, _ = make_system(config=cfg)
    s.remember("alpha beta gamma")
    assert s.recall("zzz yyy")["recalled"] == []


def test_relative_cut_and_multi_cue(make_system):
    s, _ = make_system()
    s.remember("kyoto trip hotel booking"); s.remember("kyoto weather"); s.remember("random unrelated note")
    texts = [c["text"] for c in s.recall("kyoto trip hotel booking")["recalled"]]
    assert "random unrelated note" not in texts
    s.remember("user likes matcha ice cream"); s.remember("user plays tennis on sunday")
    r = s.recall("matcha ice cream is great.\nalso tennis on sunday?")
    assert {c["text"] for c in r["recalled"]} >= {"user likes matcha ice cream", "user plays tennis on sunday"}


def test_budget_after_first_line(make_system, cfg):
    cfg.memory.budget_chars = 120
    s, _ = make_system(config=cfg)
    for i in range(4):
        s.remember(f"kyoto note number {i} about kyoto")
    assert len(s.recall("kyoto note")["recalled"]) == 1


def test_cite_completes_strengthening(make_system):
    s, clock = make_system()
    mid = s.remember("user birthday is 1990-05-03")["id"]
    clock.advance_days(3)
    s.recall("when is the birthday?")
    half = s.memory(mid).stability
    assert s.cite(f"It is on 1990-05-03 《id:{mid}》") == [mid]
    assert s.memory(mid).stability > half
    assert s.cite(f"again 《id:{mid}》") == []


def test_forget(make_system):
    s, _ = make_system()
    mid = s.remember("to be forgotten")["id"]
    assert s.forget(mid)["action"] == "deleted"
    assert s.forget(mid)["action"] == "not_found"
    assert s.total_records() == 0


def test_initialize_clamps_and_persists(make_system, tmp_path):
    from core.storage import Store
    store = Store(str(tmp_path / "persist.db"))
    s, clock = make_system(store=store)
    m = s.memory(s.remember("future")["id"])
    store.put(m.with_(last_recall=int(clock.t) + 1_000_000, created_at=int(clock.t) + 1_000_000, stability=1e12))
    s2, _ = make_system(store=store)
    loaded = s2.memories()[0]
    assert loaded.last_recall <= clock.t and loaded.stability == s2.cfg.max_stability


@pytest.mark.parametrize("n", [1, 7])
def test_tokens_helper(n):
    assert len(tokens(n).split()) == n
