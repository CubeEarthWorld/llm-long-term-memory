"""Server runtime state and background jobs.

Every mutation of the session runs under ``LOCK``, so the SQLite store and the
in-memory engine are never touched concurrently. ``STATE`` / ``APP`` are mutated
in place (never rebound), so ``from jobs import STATE`` stays valid everywhere.
Only this module writes ``STATE`` and ``APP``; server.py only reads them.
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


def session() -> Session:
    """The ready session, or 503."""
    s = APP["session"]
    if not s or not STATE["ready"]:
        raise HTTPException(503, "Engine is still starting.")
    return s


def require_idle() -> None:
    if STATE["running"]:
        raise HTTPException(409, "A job is already running.")


def init_session(cfg: Config | None = None, wipe: bool = False) -> None:
    """Build or rebuild the session (closing any existing one); sets ready / init_error.

    The startup call (``cfg is None``) is a no-op once a session exists: a reset is
    admitted while not ready, and it must not be undone by a slow startup finishing
    second and rebuilding with the default configuration."""
    with LOCK:
        if cfg is None and APP["session"] is not None:
            return
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


def run_job(fn, *, per_step: bool = False) -> None:
    """Run ``fn`` in a daemon thread as *the* running job (409 if one is already running).

    ``running`` is claimed before the thread starts (under its own short lock, not
    LOCK, which a job may hold for minutes), so two requests cannot both pass; the
    previous job's progress line is cleared there, so no job ever shows another's.
    ``per_step=True`` is for jobs that take LOCK per step themselves (seed replay)
    so readers can poll between steps."""
    with _CLAIM:
        require_idle()
        STATE["running"] = True
        STATE["progress"] = ""
    STATE["error"] = None

    def worker():
        try:
            if per_step:
                fn()
            else:
                with LOCK:
                    fn()
        except Exception as exc:  # noqa: BLE001
            STATE["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            STATE["running"] = False

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

    run_job(job, per_step=True)


def run_dream_job(s: Session, budget: int | None) -> dict:
    """Consolidate in the background; the response reports the labile traces the pass
    will scan (0 = nothing to settle, so no job is started). The count of clusters the
    pass actually adjudicated is only known when it ends, and lands in ``progress``."""
    require_idle()              # the read below takes LOCK, which a running job may hold for minutes
    with LOCK:
        n = s.memory.stats()["labile"]
    if not n:
        return {"ok": True, "n": 0,
                "message": "Dream 対象のクラスタがありません（不安定な記憶に近傍がない＝整理済み）"}

    def job():
        results = s.dream(budget)
        replaced = sum(1 for r in results if r.get("action") == "replace")
        STATE["progress"] = f"dreamed {len(results)} cluster(s), {replaced} consolidated"

    run_job(job)
    return {"ok": True, "n": n}


def set_seed(items: list[dict[str, str]]) -> dict:
    """Replace the editable seed list and persist it under LOCK — ``seed.save`` truncates
    data/seed.csv, so two concurrent saves would interleave into a mixed file."""
    with LOCK:
        APP["seed"] = items        # rebound atomically; the CSV is independent of the DB
        seed.save(SEED_CSV_PATH, items)
    return {"ok": True, "n": len(items)}
