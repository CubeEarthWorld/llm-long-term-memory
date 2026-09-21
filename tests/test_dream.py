"""The sleep phase (SPEC §5) and the LLM verdict parser."""
from __future__ import annotations

from core.llm_client import _parse_dream, _parse_texts
from fakes import BrokenEmbeddingProvider, FakeEmbeddingProvider, FakeLLM


def seeded(make_system, **kw):
    s, clock = make_system(**kw)
    for t in ("trip kyoto plan schedule note", "trip kyoto hotel schedule note", "trip kyoto food schedule note",
              "completely different topic zebra"):
        s.remember(t)
    return s, clock


def test_replace_merges_into_gist_with_live_evidence(make_system):
    s, _ = seeded(make_system)
    before = s.memories()
    reports = s.dream()
    assert len(reports) == 1 and reports[0]["action"] == "replace" and len(reports[0]["before"]) == 3
    gist = s.memory(reports[0]["after"][0]["id"])
    assert gist.consolidated and abs(gist.stability - 3 * 86400) < 1 and gist.last_recall == before[0].last_recall
    assert s.total_records() == 2 and all(m.consolidated for m in s.memories())
    assert s.dream() == []                                   # settled store ⇒ no LLM calls


def test_keep_marks_consolidated_without_touching_strength(make_system):
    s, _ = seeded(make_system, llm=FakeLLM("keep"))
    before = {m.id: m for m in s.memories()}
    assert s.dream()[0]["action"] == "keep"
    for m in s.memories():
        assert m.consolidated and m.stability == before[m.id].stability and m.last_recall == before[m.id].last_recall


def test_error_leaves_seed_labile(make_system):
    s, _ = seeded(make_system, llm=FakeLLM("error"))
    reports = s.dream()
    assert reports[0]["action"] == "error"
    assert s.clusters()
    s.llm = FakeLLM()
    assert s.dream()[0]["action"] == "replace"


def test_confabulation_guard(make_system):
    class Halluc(FakeLLM):
        def dream_cluster(self, members, current_time=""):
            return {"action": "replace", "memories": ["zzz qqq unrelated hallucination"]}

    class TooMany(FakeLLM):
        def dream_cluster(self, members, current_time=""):
            # Names every candidate, so the gist count is the only thing left to reject.
            return {"action": "replace", "ids": [m["id"] for m in members[1:]],
                    "memories": [f"trip kyoto {i}" for i in range(len(members) + 1)]}

    s, _ = seeded(make_system, llm=Halluc())
    assert s.dream()[0]["action"] == "keep"
    s2, _ = seeded(make_system, llm=TooMany())
    assert s2.dream()[0]["action"] == "keep"


def test_budget_and_fifo_seeds(make_system):
    s, _ = make_system()
    s.remember("weak pair one alpha x"); s.remember("weak pair one alpha y")
    s.remember("strong pair two beta x", salience=5); s.remember("strong pair two beta y", salience=5)
    reports = s.dream(budget=1)
    assert len(reports) == 1 and reports[0]["before"][0]["text"].startswith("weak")
    assert len(s.clusters()) == 1


def test_correction_is_seeded_by_the_new_evidence(make_system):
    seen = {}

    class Spy(FakeLLM):
        def dream_cluster(self, members, current_time=""):
            seen["members"] = members
            return {"action": "keep", "ids": [], "memories": []}

    s, clock = make_system(llm=Spy())
    old = s.remember("user lives in tokyo city", salience=5)["id"]
    clock.advance_days(30)
    fresh = s.remember("user lives in osaka city")["id"]
    s.dream()
    assert seen["members"][0]["id"] == fresh          # the newest evidence seeds
    assert old in {m["id"] for m in seen["members"]}
    assert s.memory(old).consolidated                 # only offered, not rewritten


def test_only_named_candidates_are_absorbed(make_system):
    class Picky(FakeLLM):
        def dream_cluster(self, members, current_time=""):
            if members[0]["text"] != "user lives in osaka city":
                return {"action": "keep", "ids": [], "memories": []}
            return {"action": "replace",
                    "ids": [m["id"] for m in members[1:] if m["text"] == "user lives in tokyo city"],
                    "memories": ["user moved to osaka city"]}

    s, clock = make_system(llm=Picky())
    s.remember("user lives in tokyo city")
    bystander = s.remember("user lives in tokyo city with a cat")["id"]
    clock.advance_days(30)
    s.remember("user lives in osaka city")
    assert any(r["action"] == "replace" for r in s.dream())
    assert s.memory(bystander) is not None            # not named ⇒ not rewritten
    texts = {m.text for m in s.memories()}
    assert "user moved to osaka city" in texts and "user lives in tokyo city" not in texts


def test_model_switch_reindexes_from_text(make_system, tmp_path):
    from core.storage import Store
    store = Store(str(tmp_path / "switch.db"))
    a, _ = make_system(store=store, provider=FakeEmbeddingProvider(model_id="m1"))
    a.remember("persisted across models")
    b, _ = make_system(store=store, provider=FakeEmbeddingProvider(model_id="m2"))
    assert b.memories()[0].model_id == "m2"
    assert len(b.recall("persisted across models")["recalled"]) == 1
    c, _ = make_system(store=store, provider=BrokenEmbeddingProvider())
    assert c.dream() == [] and c.memories()[0].model_id == "m2"


def test_cue_reaches_the_version_it_supersedes(make_system):
    s, clock = make_system()
    s.remember("user lives in tokyo city")
    clock.advance_days(30)
    # Text far from the old fact (cos ≈ -0.11 < θ_related); only the cue reaches
    # back to it (cos ≈ 0.90).
    s.remember("resident of osaka prefecture now", cue="user lives in city")
    clusters = [{m.text for m in c} for c in s.clusters()]
    assert clusters == [{"user lives in tokyo city", "resident of osaka prefecture now"}]


def test_parsers_are_lenient():
    assert _parse_dream('```json\n{"action":"keep"}\n```') == {"action": "keep", "ids": [], "memories": []}
    verdict = _parse_dream('x {"action":"replace","ids":["A","B"],"memories":["a",{"text":"b"},""]} y')
    assert verdict["memories"] == ["a", "b"] and verdict["ids"] == ["A", "B"]
    assert _parse_dream('{"action":"replace","memories":["a"]}')["ids"] is None
    assert _parse_dream("garbage") is None            # truncated answers retry, never "keep"
    assert _parse_dream(None) is None
    assert _parse_texts('{"memories":["a"," ","b"]}') == ["a", "b"]
