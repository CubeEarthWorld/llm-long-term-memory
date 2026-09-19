"""The app session: assembles the runtime (embedding provider, LLM client, SQLite
store, memory engine, metrics, turn runner) and owns the turn log. The web server
and the CLI both drive the app through this one object."""
from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, ContextManager

from config import ROOT, Config
from core import seed
from core.embedding import get_provider
from core.llm_client import LLMClient
from core.metrics import MetricsRecorder
from core.storage import Store
from core.turn import TurnRunner
from memory.engine import LongTermMemory

DATA_DIR = os.environ.get("MEMORY_DATA_DIR", os.path.join(ROOT, "data"))  # runtime SQLite + CSV/JSON output
DB_PATH = os.path.join(DATA_DIR, os.path.basename(os.environ.get("MEMORY_DB_NAME", "llm_long_term_memory.db")))
SEED_CSV_PATH = os.path.join(DATA_DIR, "seed.csv")


class Session:
    def __init__(self, cfg: Config, wipe: bool = False):
        """Assemble the runtime. ``wipe`` starts from an empty DB (file delete, then a
        clear in case the file was locked); otherwise the persisted turn log is restored."""
        self.cfg = cfg
        self.provider = get_provider(cfg.glob.embedding_model)
        self.llm = LLMClient(cfg.glob)
        if wipe:
            Store.delete_files(DB_PATH)
        self.store = Store(DB_PATH)
        self.memory = LongTermMemory(self.store, self.provider, self.llm, cfg.memory, cfg.glob.default_timezone)
        self.recorder = MetricsRecorder(max_history=cfg.glob.max_metrics_history)
        self.runner = TurnRunner(self.provider, self.llm, self.recorder, self.memory, cfg.glob)
        self.log: list[dict[str, Any]] = []
        self.seeded = False
        self.last_dream: list[dict[str, Any]] = []
        if wipe:
            self.memory.reset()
        else:
            self.log = [{"turn": int(r["turn"]), "utterance": str(r["utterance"]), "note": str(r["note"]),
                         "timestamp": float(r["timestamp"]), "system": _loads(r["system_json"])}
                        for r in self.store.load_turn_log(cfg.glob.max_turn_log)]
        self.turn = max((e["turn"] for e in self.log), default=0)

    @property
    def start_time(self) -> float | None:
        """Timestamp of the oldest turn still in the log."""
        return self.log[0]["timestamp"] if self.log else None

    def close(self) -> None:
        self.store.close()

    def reset(self) -> None:
        """Erase memories, turn log and metrics while keeping the DB open."""
        self.memory.reset()
        self.memory.set_clock(None)  # drop any virtual seed clock -> back to real time
        self.recorder.reset()
        self.turn = 0
        self.log = []
        self.seeded = False
        self.last_dream = []

    def run_turn(self, utterance: str, note: str = "") -> int:
        """Execute one user turn (recall → respond → write → cite), log and persist it.
        ``note`` is a display-only annotation. Returns the turn number."""
        turn = self.turn + 1
        detail = self.runner.run_turn(turn, utterance).to_detail_dict()
        self.turn = turn
        entry = {"turn": turn, "utterance": utterance, "note": note,
                 "timestamp": self.memory.now_unix(), "system": detail}
        keep = self.cfg.glob.max_turn_log   # sliding window: bounded RAM and DB
        self.log = (self.log + [entry])[-keep:]
        self.store.save_turn_log(turn, utterance, note, entry["timestamp"],
                                 json.dumps(detail, ensure_ascii=False), keep)
        return turn

    def dream(self, budget: int | None = None) -> list[dict[str, Any]]:
        """Memory consolidation (dreaming): ≤ budget LLM adjudications (None = cfg.dream_budget)."""
        self.last_dream = self.memory.dream(budget=budget)
        return self.last_dream

    def replay(self, items: list[dict[str, str]], on_turn: Callable[[int, dict], None] | None = None, *,
               guard: ContextManager = nullcontext(), restore_clock: bool = True) -> None:
        """Replay seed utterances along a virtual timeline to exercise forgetting.

        The first utterance is anchored at the wall clock; each item's ``advance``
        (e.g. "5y") cumulatively pushes the virtual clock forward. ``guard`` is
        entered once per turn (the web server passes its lock, so readers can poll
        between turns); ``on_turn(turn, item)`` runs inside it. Headless runs pass
        ``restore_clock=False`` so a following dream sees the seeded memories in its
        past rather than its future."""
        offset = [0.0]
        with guard:
            self.memory.set_clock(lambda: time.time() + offset[0])
        try:
            for item in items:
                offset[0] += seed.parse_duration(item["advance"])
                with guard:
                    turn = self.run_turn(item["text"], item.get("note", ""))
                    if on_turn:
                        on_turn(turn, item)
                time.sleep(0.02)   # let readers waiting on the guard observe the finished turn
        finally:
            with guard:
                if restore_clock:
                    self.memory.set_clock(None)
                self.seeded = True


def _loads(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
