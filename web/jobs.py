"""Server runtime state and background jobs.

Every mutation of the session runs under ``LOCK``, so the SQLite store and the
in-memory engine are never touched concurrently. ``STATE`` / ``APP`` are mutated
in place (never rebound), so ``from web.jobs import STATE`` stays valid everywhere.
Only this module writes ``STATE`` flags and ``APP["session"]``.
"""
from __future__ import annotations

import threading
import traceback

from fastapi import HTTPException

from config import Config
from core import seed
from core.session import SEED_CSV_PATH, Session

STATE = {"ready": False, "running": False, "progress": "", "error": None, "init_error": None}
APP: dict = {"session": None, "seed": seed.load(SEED_CSV_PATH)}   # the live Session + the editable seed list
LOCK = threading.RLock()
_CLAIM = threading.Lock()   # guards the running flag only — never held across a job


def session(*, idle: bool = False) -> Session:
    """The ready session, or 503; with ``idle`` also 409 while a job is running."""
    s = APP["session"]
    if not s or not STATE["ready"]:
        raise HTTPException(503, "Engine is still starting.")
    if idle:
        require_idle()
    return s


def require_idle() -> None:
    if STATE["running"]:
        raise HTTPException(409, "A job is already running.")


def init_session(cfg: Config | None = None, wipe: bool = False) -> None:
    """Build or rebuild the session (closing any existing one); sets ready / init_error."""
    with LOCK:
        STATE["ready"] = False
        try:
            if APP["session"]:
                APP["session"].close()
            APP["session"] = Session(cfg or Config(), wipe=wipe)
            STATE["init_error"] = None
            STATE["ready"] = True
        except Exception as exc:  # noqa: BLE001
            STATE["init_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            STATE["progress"] = ""


def run_job(fn, *, hold_lock: bool = True, clear_progress: bool = False) -> None:
    """Run ``fn`` in a daemon thread as *the* running job (409 if one is already running).

    ``running`` is claimed before the thread starts (under its own short lock, not
    LOCK, which a job may hold for minutes), so two requests cannot both pass.
    ``hold_lock=False`` is for jobs that take LOCK per step themselves (seed
    replay) so readers can poll between steps. Progress is left in place unless
    ``clear_progress`` — a dream's result line stays visible."""
    with _CLAIM:
        require_idle()
        STATE["running"] = True
    STATE["error"] = None

    def worker():
        try:
            if hold_lock:
                with LOCK:
                    fn()
            else:
                fn()
        except Exception as exc:  # noqa: BLE001
            STATE["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            STATE["running"] = False
            if clear_progress:
                STATE["progress"] = ""

    threading.Thread(target=worker, daemon=True).start()


def run_seed_job(s: Session, do_reset: bool) -> None:
    """Replay the seed turn by turn, releasing LOCK between turns so the UI can poll."""
    items = list(APP["seed"])

    def on_turn(turn: int, item: dict) -> None:
        STATE["progress"] = f"turn {turn}/{len(items)}: {item['text'][:16]}..."

    def job():
        if do_reset:
            with LOCK:
                s.reset()
        s.replay(items, on_turn, guard=LOCK)

    run_job(job, hold_lock=False, clear_progress=True)
