"""End-to-end turn: recall → converse (tools) → cite, with a scriptable fake LLM."""
from __future__ import annotations

import pytest

from core.llm_client import ConverseResult
from core.metrics import MetricsRecorder
from core.turn import TurnRunner


class ScriptedLLM:
    """Saves the utterance with a salience, deletes on request, cites what it was shown."""

    def __init__(self, saves: int = 1, extract: list[str] | None = None):
        self.saves = saves
        self.extract = extract or []
        self.known = None

    def converse(self, memory_pack, user_text, tools, current_time=""):
        for i in range(self.saves):
            tools["save_memory"](text=f"{user_text} {i}" if i else user_text, salience=2,
                                 cue=f"why {user_text}")
        if user_text.startswith("forget:"):
            tools["delete_memory"](id=user_text.split(":", 1)[1])
        cited = " ".join(line.split("《")[1].rstrip("》\n").join("《》") for line in memory_pack.splitlines())
        return ConverseResult(f"ok {cited}", f"[pack]{memory_pack}[/pack]")

    def extract_save_candidates(self, user_text, assistant_text, current_time="", known=""):
        self.known = known
        return list(self.extract)

    def dream_cluster(self, members, current_time=""):
        return {"action": "keep", "memories": []}


def test_turn_saves_recalls_and_cites(make_system, cfg):
    llm = ScriptedLLM()
    s, clock = make_system(llm=llm)
    runner = TurnRunner(s.provider, llm, MetricsRecorder(), s, cfg.glob)
    m1 = runner.run_turn(1, "user likes matcha ice cream")
    assert m1.written_rows[0]["action"] == "inserted" and m1.written_rows[0]["stability"] == 2 * 86400
    clock.advance_days(1)
    m2 = runner.run_turn(2, "matcha ice cream again?")
    assert m2.pack_n >= 1 and "user likes matcha ice cream" in m2.pack_text
    assert m2.cited and s.memory(m2.cited[0]).stability > 2 * 86400
    assert m2.counts["records"] == 2 and m2.total_ms >= 0
    assert s.memory(m1.written_rows[0]["id"]).cue == "why user likes matcha ice cream"


def test_per_turn_cap_and_fallback(make_system, cfg):
    cfg.glob.max_writes_per_turn = 2
    llm = ScriptedLLM(saves=5)
    s, _ = make_system(llm=llm, config=cfg)
    m = TurnRunner(s.provider, llm, MetricsRecorder(), s, cfg.glob).run_turn(1, "hello there")
    assert sum(1 for e in m.written_rows if e["action"] == "inserted") == 2
    assert sum(1 for e in m.written_rows if e["action"] == "rate_limited") == 3
    llm2 = ScriptedLLM(saves=0, extract=["extracted fact one", "extracted fact two"])
    s2, _ = make_system(llm=llm2, config=cfg)
    s2.remember("user likes matcha ice cream")                 # so the pack is not empty
    m2 = TurnRunner(s2.provider, llm2, MetricsRecorder(), s2, cfg.glob).run_turn(1, "matcha ice cream again?")
    assert [e["text"] for e in m2.written_rows] == ["extracted fact one", "extracted fact two"]
    assert llm2.known == m2.pack_text            # the fallback is shown what was recalled


def test_retry_backs_off_on_transient_failures_only(monkeypatch):
    """_retry: EmptyResponse is transient, anything unmatched propagates at once."""
    import core.llm_client as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise llm_mod.EmptyResponse("empty body (finish_reason=length)")
        return "answer"

    assert llm_mod.LLMClient._retry(None, flaky, "t") == "answer" and len(calls) == 3

    def fatal():
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        llm_mod.LLMClient._retry(None, fatal, "t")
    assert len(calls) == 4                       # not retriable ⇒ one attempt


def test_delete_tool(make_system, cfg):
    llm = ScriptedLLM(saves=0)
    s, _ = make_system(llm=llm)
    mid = s.remember("delete me later")["id"]
    m = TurnRunner(s.provider, llm, MetricsRecorder(), s, cfg.glob).run_turn(1, f"forget:{mid}")
    assert m.written_rows[0]["action"] == "deleted" and s.total_records() == 0
