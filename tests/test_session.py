"""Seed CSV handling and the app session (turn log, seed replay), with fakes."""
from __future__ import annotations

import pytest

import core.session as session_mod
from config import Config
from core import seed
from core.session import Session
from fakes import FakeEmbeddingProvider
from test_turn_runner import ScriptedLLM


def test_parse_duration():
    assert [seed.parse_duration(x) for x in ("0", "", None, "30m", "2h", "1d", "3", "5y", "junk")] == \
        [0, 0, 0, 1800, 7200, 86400, 3 * 86400, 5 * 31536000, 0]


def test_csv_roundtrip_keeps_notes(tmp_path):
    path = str(tmp_path / "seed.csv")
    assert seed.load(path) == seed.clean(seed.DEFAULT_SEED)            # no file → built-in scenario
    items = seed.clean([{"text": " a,b ", "note": "重要", "advance": "5y"}, {"text": " "}, {"text": "c"}])
    assert items == [{"text": "a,b", "note": "重要", "advance": "5y"}, {"text": "c", "note": "", "advance": "0"}]
    seed.save(path, items)
    assert seed.load(path) == items
    assert seed.parse_csv("﻿text,note,advance\nx,n,1d\n") == [{"text": "x", "note": "n", "advance": "1d"}]
    assert seed.parse_csv("x,2h\ny\n") == [{"text": "x", "note": "", "advance": "2h"},
                                           {"text": "y", "note": "", "advance": "0"}]


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    llm = ScriptedLLM()
    monkeypatch.setattr(session_mod, "get_provider", lambda _path: FakeEmbeddingProvider())
    monkeypatch.setattr(session_mod, "LLMClient", lambda _glob: llm)
    monkeypatch.setattr(session_mod, "DB_PATH", str(tmp_path / "app.db"))
    cfg = Config()
    cfg.memory.cosine_floor = 0.0
    opened = []

    def _make(wipe=False):
        opened.append(Session(cfg, wipe=wipe))
        return opened[-1]

    yield _make
    for s in opened:
        s.close()


def test_replay_runs_on_a_virtual_clock_and_restores_it(make_session):
    s = make_session(wipe=True)
    seen = []
    s.replay([{"text": "user likes tea", "note": "重要", "advance": "0"},
              {"text": "user moved to osaka", "advance": "5y"}], lambda turn, item: seen.append(turn))
    assert seen == [1, 2] and s.turn == 2 and s.seeded and s.memory._clock is None
    assert [e["note"] for e in s.log] == ["重要", ""]
    assert s.log[1]["timestamp"] - s.log[0]["timestamp"] >= 5 * 31536000
    t0 = s.start_time

    def boom(turn, item):
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        s.replay([{"text": "x y z", "advance": "1d"}], boom)
    assert s.memory._clock is None                                      # restored even on failure
    s.replay([{"text": "kept virtual", "advance": "1d"}], restore_clock=False)
    assert s.memory._clock is not None
    s.close()
    assert make_session().start_time == t0                              # survives a restart


def test_turn_log_window_drops_the_oldest(make_session):
    s = make_session(wipe=True)
    s.cfg.glob.max_turn_log = 2
    for i in range(3):
        s.run_turn(f"turn {i} fact about cats")
    assert [e["turn"] for e in s.log] == [2, 3]                          # RAM window
    assert [r["turn"] for r in s.store.load_turn_log(1000)] == [2, 3]    # DB window


def test_turn_log_survives_restart_and_reset_clears_it(make_session):
    s = make_session(wipe=True)
    s.run_turn("user has a cat", note="n1")
    s.close()
    s2 = make_session()
    assert s2.turn == 1 and s2.log[0]["note"] == "n1" and s2.log[0]["system"]["records"] == 1
    assert s2.memory.total_records() == 1
    s2.reset()
    assert (s2.turn, s2.log, s2.start_time, s2.memory.total_records()) == (0, [], None, 0)
    s2.close()
    assert make_session().log == []
