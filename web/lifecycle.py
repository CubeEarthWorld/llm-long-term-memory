"""Engine lifecycle jobs: build/rebuild, seed replay and seed CSV persistence."""
from __future__ import annotations

import csv
import os
import threading
import time
import traceback

from config import Config, default_config
from core.engine import (
    DATA_DIR,
    SEED_CSV_PATH,
    Engine,
    _parse_duration,
    build_engine,
    dispose_engine,
    reset_state,
    run_turn,
)
from web.state import ENGINE, LOCK, SEED, STATE


def init_engine(cfg: Config | None = None, wipe: bool = False) -> None:
    """Build or rebuild the engine in the current thread.

    Disposes any existing engine first.  If wipe=False and a previous DB exists,
    maintain() is called once to gracefully catch up on decayed memories.
    State flags (ready / init_error) are updated for the UI polling loop.
    """
    with LOCK:
        STATE["ready"] = False
        cfg = cfg or default_config()
        try:
            dispose_engine(ENGINE["e"])
            ENGINE["e"] = build_engine(cfg, wipe=wipe, seed=SEED["items"])
            ENGINE["cfg"] = cfg
            # After a long downtime, let maintenance catch up on capacity (tombstone
            # sweep + demotion/eviction) over a few bounded passes instead of doing it
            # all at the first turn.
            if not wipe:
                try:
                    system = ENGINE["e"]["system"]
                    for _ in range(20):
                        before = system.total_records()
                        system.maintain(ENGINE["e"]["turn"])
                        if before == system.total_records():
                            break
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
            STATE["init_error"] = None
            STATE["ready"] = True
        except Exception as exc:  # noqa: BLE001
            STATE["init_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            STATE["progress"] = ""


def _start_seed_run(engine: Engine, do_reset: bool, seed_items: list[dict[str, str]]) -> None:
    """Run seed replay turn-by-turn, releasing LOCK between turns so the UI
    can poll /api/state and /api/turns-detail in near real-time.
    """
    total = len(seed_items)

    def worker():
        with LOCK:
            STATE["running"] = True
            STATE["error"] = None
        try:
            if do_reset:
                with LOCK:
                    reset_state(engine)

            offset_state = {"offset": 0.0}
            with LOCK:
                engine["system"].set_clock(lambda: time.time() + offset_state["offset"])

            try:
                for item in seed_items:
                    offset_state["offset"] += _parse_duration(item.get("advance", "0"))
                    with LOCK:
                        run_turn(engine, item["text"])
                        STATE.update(progress=f"turn {engine['turn']}/{total}: {item['text'][:16]}...")
                    # Yield the GIL briefly so pending readers can acquire LOCK
                    # and observe the newly completed turn.
                    time.sleep(0.02)
            finally:
                with LOCK:
                    engine["system"].set_clock(None)
                    engine["seeded"] = True
        except Exception as exc:  # noqa: BLE001
            with LOCK:
                STATE["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            with LOCK:
                STATE["running"] = False
                STATE["progress"] = ""

    threading.Thread(target=worker, daemon=True).start()


def _save_seed(items: list[dict]) -> None:
    """Persist seed utterances to CSV so edits survive server restarts."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SEED_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "note", "advance"])
        for item in items:
            writer.writerow([item["text"], item.get("note", ""), item["advance"]])
