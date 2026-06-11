"""Shared mutable server state.

All mutating jobs are serialised through ``run_job`` (which holds LOCK), so the
SQLite store and the in-memory engine are never touched concurrently. The dicts
below are only ever mutated in place (never rebound), so ``from web.state
import STATE`` aliases stay valid everywhere — including ``server.STATE`` after
the re-export in server.py.
"""
from __future__ import annotations

import threading
import traceback

from core.engine import default_seed

STATE = {"ready": False, "running": False, "progress": "", "error": None, "init_error": None}
ENGINE: dict = {"e": None, "cfg": None}   # "e" holds the Engine from core.engine
LOCK = threading.RLock()
SEED: dict = {"items": default_seed()}


def run_job(fn) -> None:
    """Wrap a function in a daemon thread that holds LOCK and updates STATE.

    This serialises all mutating operations (turn, seed, dream, reset) so that
    the SQLite store and the in-memory engine are never touched concurrently.

    NOTE: STATE["progress"] is managed by each job function, NOT cleared here,
    so that dream results (e.g. "dreamed 3 cluster(s), 2 consolidated") can
    persist and be visible to the frontend after the job completes.
    """
    def worker():
        with LOCK:
            STATE["running"] = True
            STATE["error"] = None
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                STATE["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                STATE["running"] = False
                # Do NOT clear STATE["progress"] — let each job manage its own.

    threading.Thread(target=worker, daemon=True).start()
