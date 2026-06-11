"""Engine construction and turn-level orchestration helpers.

The "engine" is a plain dict that holds all runtime objects (embedding provider,
LLM client, SQLite store, memory system, metrics recorder, turn runner).  This
module factory-assembles the engine and provides thin wrappers for turns,
dreaming, and seed replay.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from typing import Any, Callable, Dict, List, TypedDict

from config import Config
from core.base import TurnRunner
from core.embedding import EmbeddingProvider, get_provider
from core.llm_client import LLMClient
from core.metrics import MetricsRecorder
from core.storage import Store
from memory.long_term_memory import LongTermMemory
from seed_utterances import SEED_ADVANCE, SEED_UTTERANCES

# Virtual-clock advance units used by seed utterances (e.g. "5y", "8d", "12h").
# The "y" unit is treated as exactly 365 days; there is no "month" unit to avoid
# variable-length ambiguity.
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}


def _parse_duration(spec: object) -> float:
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
DATA_DIR = os.environ.get("MEMORY_DATA_DIR", os.path.join(BASE_DIR, "data"))  # runtime SQLite + JSON output
SYSTEM_ID = "llm_long_term_memory"
SYSTEM_TITLE = "LLM Long-Term Memory"
DB_FILENAME = os.path.basename(os.environ.get("MEMORY_DB_NAME", "llm_long_term_memory.db"))

SEED_CSV_PATH = os.path.join(DATA_DIR, "seed.csv")


class Engine(TypedDict, total=False):
    """Typed structure for the runtime engine dict."""
    provider: EmbeddingProvider
    llm: LLMClient
    store: Store
    system: LongTermMemory
    recorder: MetricsRecorder
    runner: TurnRunner
    turn: int
    log: list[dict[str, Any]]
    start_time: float | None
    seeded: bool
    last_dream: list[dict[str, Any]]
    cfg: Config
    seed: list[dict[str, str]]


def load_seed_csv(path: str) -> list[dict[str, str]]:
    """Load seed items from a CSV file. Returns empty list on missing/unparseable file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return [
                {
                    "text": (r.get("text") or "").strip(),
                    "advance": ((r.get("advance") or "0").strip() or "0"),
                }
                for r in csv.DictReader(f)
                if (r.get("text") or "").strip()
            ]
    except Exception:
        return []


def default_seed() -> list[dict[str, str]]:
    """Return seed utterances: CSV if available, otherwise built-in scenario."""
    rows = load_seed_csv(SEED_CSV_PATH)
    if rows:
        return rows
    return [
        {"text": t, "advance": SEED_ADVANCE[i] if i < len(SEED_ADVANCE) else "0"}
        for i, t in enumerate(SEED_UTTERANCES)
    ]


def clean_seed_items(raw) -> list[dict[str, str]]:
    """Normalise a list of raw dicts into valid seed items (text/advance)."""
    items = []
    for item in raw or []:
        text = str((item or {}).get("text", "")).strip()
        if text:
            items.append({
                "text": text,
                "advance": (str((item or {}).get("advance", "0")).strip() or "0"),
            })
    return items


def parse_seed_csv(text: str) -> list[dict[str, str]]:
    """Tolerant CSV parser that accepts both header and header-less CSV for seeds."""
    reader = csv.DictReader(io.StringIO(text))
    items: List[Dict[str, str]] = []
    if reader.fieldnames and any((h or "").strip().lower() == "text" for h in reader.fieldnames):
        for row in reader:
            items.append({
                "text": (row.get("text") or "").strip(),
                "advance": ((row.get("advance") or "0").strip() or "0"),
            })
    else:
        for row in csv.reader(io.StringIO(text)):
            if row:
                items.append({
                    "text": (row[0] or "").strip(),
                    "advance": ((row[1].strip() if len(row) > 1 else "0") or "0"),
                })
    return clean_seed_items(items)


def build_engine(cfg: Config, wipe: bool = False, seed: list[dict[str, str]] | None = None) -> Engine:
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

    # Load persisted turn log from DB (if not wiping).
    log: list[dict[str, Any]] = []
    start_time: float | None = None
    max_turn = 0
    if not wipe:
        persisted = system.load_turn_log()
        for row in persisted:
            system_detail = {}
            try:
                system_detail = json.loads(row["system_json"])
            except (json.JSONDecodeError, TypeError):
                pass
            entry = {
                "turn": int(row["turn"]),
                "utterance": str(row["utterance"]),
                "note": str(row.get("note", "")),
                "timestamp": float(row["timestamp"]),
                "system": system_detail,
            }
            log.append(entry)
            if entry["turn"] > max_turn:
                max_turn = entry["turn"]
            ts = entry["timestamp"]
            if start_time is None or ts < start_time:
                start_time = ts

    # Respect max_turn_log sliding window.
    max_log = cfg.glob.max_turn_log
    if len(log) > max_log:
        log = log[-max_log:]

    return {
        "provider": provider,
        "llm": llm,
        "store": store,
        "system": system,
        "recorder": recorder,
        "runner": TurnRunner(provider, llm, recorder, system),
        "turn": max_turn,
        "log": log,
        "start_time": start_time,
        "seeded": False,
        "last_dream": [],
        "cfg": cfg,
        "seed": list(seed) if seed is not None else default_seed(),
    }


def _open_store(wipe: bool) -> Store:
    path = os.path.join(DATA_DIR, DB_FILENAME)
    if wipe:
        Store(path).wipe_file()
    return Store(path)


def dispose_engine(engine: Engine | None) -> None:
    """Close the underlying SQLite store and release resources (no-op if no engine yet)."""
    if not engine:
        return
    store = engine.get("store")
    if store:
        store.close()


def reset_state(engine: Engine) -> None:
    """Reset the memory system, metrics, and turn log while keeping the DB open."""
    engine["system"].reset()
    engine["system"].set_clock(None)  # drop any virtual seed clock -> back to real time
    engine["recorder"].reset()
    engine["turn"] = 0
    engine["log"] = []
    engine["seeded"] = False
    engine["last_dream"] = []


def run_turn(engine: Engine, utterance: str, note: str = "") -> int:
    """Execute one user turn (retrieve → respond → write → maintain) and append to the log.

    Args:
        utterance: Raw user text.
        note: Human-readable annotation (display-only, not stored in memory).

    Returns:
        The turn number that was just executed.
    """
    turn = engine["turn"] + 1
    engine["runner"].run_turn(turn, utterance)
    engine["turn"] = turn
    now = engine["system"].now_unix()
    if engine.get("start_time") is None:
        engine["start_time"] = now

    # Build system detail for this turn (from recorder) and persist to DB.
    cfg = engine.get("cfg") or Config()
    system_detail = {}
    metrics = engine["recorder"].for_turn(turn, SYSTEM_ID)
    if metrics:
        system_detail = metrics.to_detail_dict(SYSTEM_TITLE)

    entry = {"turn": turn, "utterance": utterance, "note": note, "timestamp": now, "system": system_detail}
    engine["log"].append(entry)

    # Persist to DB so turn log survives restarts.
    system = engine["system"]
    system.save_turn_log(turn, utterance, note, now, json.dumps(system_detail, ensure_ascii=False))

    # Sliding window to prevent unbounded in-memory growth during very long sessions.
    max_log = cfg.glob.max_turn_log
    if len(engine["log"]) > max_log:
        engine["log"] = engine["log"][-max_log:]
    return turn


def run_dream(engine: Engine, max_clusters: int = 1, force: bool = False) -> list[dict[str, Any]]:
    """Trigger memory consolidation (dreaming) on the engine's memory system."""
    results = engine["system"].dream(max_clusters=max_clusters, force=force)
    engine["last_dream"] = results
    return results


def run_seed(engine: Engine, on_progress: Callable[[int, str], None] | None = None) -> None:
    """Replay seed utterances along a virtual timeline to exercise forgetting.

    The first utterance is anchored at the current wall-clock time.  Each item's
    "advance" field (e.g. "5y") cumulatively pushes the virtual clock forward,
    so memories genuinely lose activation and demote across tiers between widely-spaced turns.
    This is useful for deterministic demonstration of long-term forgetting.
    """
    state = {"offset": 0.0}
    engine["system"].set_clock(lambda: time.time() + state["offset"])
    try:
        for item in engine.get("seed") or default_seed():
            state["offset"] += _parse_duration(item.get("advance", "0"))
            turn = run_turn(engine, item["text"])
            if on_progress:
                on_progress(turn, item["text"])
    finally:
        # Always restore real clock, even if seed replay is interrupted.
        engine["system"].set_clock(None)
    engine["seeded"] = True
