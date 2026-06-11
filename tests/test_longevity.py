"""Long-term usage bounds: turn_log persistence cap and per-turn write cap.

Regressions guarded here:
* turn_log grew without bound in the DB (the only unbounded table) and was
  loaded in full at startup, so months of conversation inflated the DB and
  the engine's startup time.
* max_writes_per_turn was enforced only on the extraction fallback, so the
  LLM could apply unlimited save_memory tool calls within a single turn.
"""
from __future__ import annotations

from conftest import _ConvResult, ScriptableLLM, tokens
from core.base import _SAVE_ACTIONS, TurnRunner
from core.metrics import MetricsRecorder


# ====================================================================== #
# turn_log — bounded persistence
# ====================================================================== #
def test_turn_log_capped_in_db(make_system, cfg):
    cfg.glob.max_turn_log = 5
    s, clock, c = make_system(config=cfg)
    for turn in range(1, 13):
        s.save_turn_log(turn, f"utterance {turn}", "", float(turn), "{}")
    assert s.store.count("turn_log") == 5
    loaded = s.load_turn_log()
    assert [r["turn"] for r in loaded] == [8, 9, 10, 11, 12]   # newest, ascending


def test_load_turn_log_limits_preexisting_rows(make_system, cfg):
    """Rows persisted before a cap change are still limited (and ordered) on load."""
    s, clock, c = make_system(config=cfg)
    for turn in range(1, 8):
        s.store.exec(
            "INSERT OR REPLACE INTO turn_log(turn,utterance,note,timestamp,system_json) "
            "VALUES(?,?,?,?,?)",
            (turn, f"u{turn}", "", float(turn), "{}"),
        )
    s.glob.max_turn_log = 3
    assert [r["turn"] for r in s.load_turn_log()] == [5, 6, 7]


# ====================================================================== #
# max_writes_per_turn — enforced on the tool path
# ====================================================================== #
class _GreedyLLM(ScriptableLLM):
    """Calls save_memory once per supplied text within a single converse call."""

    def __init__(self, texts: list[str]):
        super().__init__(save_on_converse=False)
        self.texts = list(texts)

    def converse(self, memory_pack, user_text, tools, current_time=""):
        self.converse_calls += 1
        for t in self.texts:
            tools["save_memory"](text=t)
        return _ConvResult(text="ok", prompt=f"[pack]{memory_pack}[/pack]{user_text}")


def test_tool_saves_capped_per_turn(make_system, cfg):
    cfg.memory.max_writes_per_turn = 3
    texts = [tokens(8, f"p{i}x") for i in range(7)]    # disjoint tokens → all new rows
    llm = _GreedyLLM(texts)
    s, clock, c = make_system(llm=llm, config=cfg)
    runner = TurnRunner(s.provider, llm, MetricsRecorder(max_history=10), s)

    metrics = runner.run_turn(1, "hello there")
    applied = [e for e in metrics.written_rows if e.get("action") in _SAVE_ACTIONS]
    limited = [e for e in metrics.written_rows if e.get("action") == "rate_limited"]
    assert len(applied) == 3
    assert len(limited) == 4
    assert s.total_records() == 3
