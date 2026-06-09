"""Engine construction and turn-level orchestration helpers.

The "engine" is a plain dict that holds all runtime objects (embedding provider,
LLM client, SQLite store, memory system, metrics recorder, turn runner).  This
module factory-assembles the engine and provides thin wrappers for turns,
dreaming, and seed replay.
"""
from __future__ import annotations

import os
import time
from typing import Callable, List, Optional

from config import Config
from core.base import TurnRunner
from core.embedding import get_provider
from core.llm_client import LLMClient
from core.metrics import MetricsRecorder
from core.storage import Store
from memory.long_term_memory import LongTermMemory
from seed_utterances import SEED_ADVANCE, SEED_NOTES, SEED_UTTERANCES

# Virtual-clock advance units used by seed utterances (e.g. "5y", "8d", "12h").
# The "y" unit is treated as exactly 365 days; there is no "month" unit to avoid
# variable-length ambiguity.
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}


def _parse_duration(spec) -> float:
    """Parse a seed 'advance' value into seconds. '5y' '8d' '12h' '30m' -> seconds;
    a bare number means days; '0'/''/garbage -> 0."""
    s = str(spec or "0").strip().lower()
    if not s or s == "0":
        return 0.0
    unit = _DURATION_UNITS.get(s[-1])
    body = s[:-1] if unit else s
    try:
        return float(body) * (unit if unit else _DURATION_UNITS["d"])
    except ValueError:
        return 0.0

# Project layout constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DATA_DIR = os.path.join(BASE_DIR, "data")                               # runtime SQLite + JSON output
SYSTEM_ID = "llm_long_term_memory"
SYSTEM_TITLE = "LLM Long-Term Memory"
DB_FILENAME = "llm_long_term_memory.db"


def default_seed() -> List[dict]:
    csv_path = os.path.join(DATA_DIR, "seed_utterances.csv")
    if os.path.exists(csv_path):
        try:
            import csv
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                rows = [
                    {
                        "text": (r.get("text") or "").strip(),
                        "note": (r.get("note") or "").strip(),
                        "advance": ((r.get("advance") or "0").strip() or "0"),
                    }
                    for r in csv.DictReader(f)
                ]
            rows = [r for r in rows if r["text"]]
            if rows:
                return rows
        except Exception:
            pass
    return [
        {
            "text": text,
            "note": (SEED_NOTES[i] if i < len(SEED_NOTES) else ""),
            "advance": (SEED_ADVANCE[i] if i < len(SEED_ADVANCE) else "0"),
        }
        for i, text in enumerate(SEED_UTTERANCES)
    ]


def build_engine(cfg: Config, wipe: bool = False, seed: Optional[List[dict]] = None) -> dict:
    """Assemble a full engine dict from configuration.

    Args:
        cfg: Global + memory-system configuration.
        wipe: If True, delete any existing SQLite DB before opening.
        seed: Optional list of seed items; falls back to the built-in scenario.
    """
    provider = get_provider(cfg.glob.embedding_model, cfg.glob.dim_full)
    llm = LLMClient(
        provider=cfg.glob.llm_provider,
        deepseek_model=cfg.glob.deepseek_model,
        deepseek_base_url=cfg.glob.deepseek_base_url,
        gemini_model=cfg.glob.gemini_model,
        temperature=cfg.glob.temperature,
        max_output_tokens=cfg.glob.max_output_tokens,
    )
    store = _open_store(wipe)
    system = LongTermMemory(store, provider, llm, cfg.memory, cfg.glob)
    if wipe:
        system.reset()

    recorder = MetricsRecorder(max_history=cfg.glob.max_metrics_history)
    return {
        "provider": provider,
        "llm": llm,
        "store": store,
        "system": system,
        "recorder": recorder,
        "runner": TurnRunner(provider, llm, recorder, system),
        "turn": 0,
        "log": [],
        "start_time": None,
        "seeded": False,
        "last_dream": [],
        "cfg": cfg,
        "seed": list(seed) if seed is not None else default_seed(),
    }


def _open_store(wipe: bool) -> Store:
    path = os.path.join(DATA_DIR, DB_FILENAME)
    store = Store(path)
    if not wipe:
        return store
    store.wipe_file()
    return Store(path)


def dispose_engine(engine: dict) -> None:
    """Close the underlying SQLite store and release resources."""
    store = engine.get("store")
    if store:
        store.close()


def reset_state(engine: dict) -> None:
    """Reset the memory system, metrics, and turn log while keeping the DB open."""
    engine["system"].reset()
    engine["system"].set_clock(None)  # drop any virtual seed clock -> back to real time
    engine["recorder"].reset()
    engine["turn"] = 0
    engine["log"] = []
    engine["seeded"] = False
    engine["last_dream"] = []


def run_turn(engine: dict, utterance: str, note: str = "") -> int:
    """Execute one user turn (retrieve → respond → write → maintain) and append to the log.

    Args:
        utterance: Raw user text.
        note: Human-readable annotation (display-only, not stored in memory).

    Returns:
        The turn number that was just executed.
    """
    import time as _time
    turn = engine["turn"] + 1
    engine["runner"].run_turn(turn, utterance)
    engine["turn"] = turn
    now = _time.time()
    if engine.get("start_time") is None:
        engine["start_time"] = now
    engine["log"].append({"turn": turn, "utterance": utterance, "note": note, "timestamp": now})
    # Sliding window to prevent unbounded in-memory growth during very long sessions.
    cfg = engine.get("cfg") or Config()
    max_log = cfg.glob.max_turn_log
    if len(engine["log"]) > max_log:
        engine["log"] = engine["log"][-max_log:]
    return turn


def run_dream(engine: dict, max_clusters: int = 1, force: bool = False) -> List[dict]:
    """Trigger memory consolidation (dreaming) on the engine's memory system."""
    results = engine["system"].dream(max_clusters=max_clusters, force=force)
    engine["last_dream"] = results
    return results


def run_seed(engine: dict, on_progress: Optional[Callable[[int, str], None]] = None) -> None:
    """Replay seed utterances along a virtual timeline to exercise forgetting.

    The first utterance is anchored at the current wall-clock time.  Each item's
    "advance" field (e.g. "5y") cumulatively pushes the virtual clock forward,
    so memories genuinely decay and may be archived between widely-spaced turns.
    This is useful for deterministic demonstration of long-term forgetting.
    """
    state = {"offset": 0.0}
    engine["system"].set_clock(lambda: time.time() + state["offset"])
    for item in engine.get("seed") or default_seed():
        state["offset"] += _parse_duration(item.get("advance", "0"))
        turn = run_turn(engine, item["text"], item.get("note", ""))
        if on_progress:
            on_progress(turn, item["text"])
    engine["seeded"] = True
