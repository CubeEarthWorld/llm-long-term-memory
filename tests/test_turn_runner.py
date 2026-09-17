"""End-to-end turn: recall → converse (tools) → cite, with a scriptable fake LLM."""
from __future__ import annotations

from core.base import TurnRunner
from core.metrics import MetricsRecorder


class _Conv:
    def __init__(self, text: str, prompt: str = ""):
        self.text, self.prompt, self.invocations, self.ok, self.error, self.rounds = text, prompt, [], True, None, 1


class ScriptedLLM:
    """Saves the utterance with a salience, deletes on request, cites what it was shown."""

    def __init__(self, saves: int = 1, extract: list[str] | None = None):
        self.saves = saves
        self.extract = extract or []
        self.last_pack = ""

    def converse(self, memory_pack, user_text, tools, current_time=""):
        self.last_pack = memory_pack
        for i in range(self.saves):
            tools["save_memory"](text=f"{user_text} {i}" if i else user_text, salience=2)
        if user_text.startswith("forget:"):
            tools["delete_memory"](id=user_text.split(":", 1)[1])
        cited = " ".join(line.split("《")[1].rstrip("》\n").join("《》") for line in memory_pack.splitlines())
        return _Conv(f"ok {cited}", prompt=f"[pack]{memory_pack}[/pack]")

    def extract_save_candidates(self, user_text, assistant_text, current_time="", known=""):
        self.known = known
        return list(self.extract)

    def dream_cluster(self, members, current_time=""):
        return {"action": "keep", "memories": []}


def test_turn_saves_recalls_and_cites(make_system):
    llm = ScriptedLLM()
    s, clock = make_system(llm=llm)
    runner = TurnRunner(s.provider, llm, MetricsRecorder(), s)
    m1 = runner.run_turn(1, "user likes matcha ice cream")
    assert m1.written_rows[0]["action"] == "inserted" and m1.written_rows[0]["stability"] == 2 * 86400
    clock.advance_days(1)
    m2 = runner.run_turn(2, "matcha ice cream again?")
    assert m2.pack_n >= 1 and "user likes matcha ice cream" in m2.pack_text
    assert m2.cited and s.memory(m2.cited[0]).stability > 2 * 86400
    assert m2.counts["records"] == 2 and m2.total_ms >= 0


def test_per_turn_cap_and_fallback(make_system, cfg):
    cfg.glob.max_writes_per_turn = 2
    llm = ScriptedLLM(saves=5)
    s, _ = make_system(llm=llm, config=cfg)
    m = TurnRunner(s.provider, llm, MetricsRecorder(), s).run_turn(1, "hello there")
    assert sum(1 for e in m.written_rows if e["action"] == "inserted") == 2
    assert sum(1 for e in m.written_rows if e["action"] == "rate_limited") == 3
    llm2 = ScriptedLLM(saves=0, extract=["extracted fact one", "extracted fact two"])
    s2, _ = make_system(llm=llm2, config=cfg)
    m2 = TurnRunner(s2.provider, llm2, MetricsRecorder(), s2).run_turn(1, "chit chat")
    assert [e["text"] for e in m2.written_rows] == ["extracted fact one", "extracted fact two"]


def test_delete_tool(make_system):
    llm = ScriptedLLM(saves=0)
    s, _ = make_system(llm=llm)
    mid = s.remember("delete me later")["id"]
    m = TurnRunner(s.provider, llm, MetricsRecorder(), s).run_turn(1, f"forget:{mid}")
    assert m.written_rows[0]["action"] == "deleted" and s.total_records() == 0
