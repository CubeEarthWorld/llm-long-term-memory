"""FastAPI backend for the LLM Long-Term Memory prototype.

Serves a JSON REST API plus the no-build React frontend in ./frontend.
The engine is built lazily on startup (or on reset) in a background thread
so that the UI can poll /api/state while heavy models load.

Thread-safety is handled by a single lock in web.jobs; FastAPI itself runs
handlers concurrently, but all mutating jobs go through run_job(), which admits
one job at a time and serialises it via the lock.
"""
from __future__ import annotations

import os
import threading
import time
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ROOT, SYSTEM_TITLE, Config
from core import seed
from core.session import SEED_CSV_PATH
from web.jobs import APP, LOCK, STATE, init_session, require_idle, run_job, run_seed_job, session

app = FastAPI(title=SYSTEM_TITLE)


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles subclass that disables browser caching for HTML/JS/CSS.

    This avoids stale frontend code during rapid iterative development.
    """
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path in ("", ".", "index.html") or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# --------------------------------------------------------------------------- #
# Pydantic request models
# --------------------------------------------------------------------------- #

class ConfigBody(BaseModel):
    """POST /api/reset payload — full Config dict serialised as JSON."""
    config: dict


class TurnBody(BaseModel):
    """POST /api/turn payload — a single user utterance."""
    text: str


class DreamBody(BaseModel):
    """POST /api/dream payload — LLM adjudications to spend (omitted = cfg.dream_budget)."""
    budget: int | None = None


class SeedRunBody(BaseModel):
    """POST /api/seed payload — replay options.

    reset=True (default) wipes the memory store before replaying, so each seed
    run starts from an empty DB; reset=False appends onto the current memories.
    """
    reset: bool = True


class SeedBody(BaseModel):
    """POST /api/seed-utterances payload — replace the entire seed list."""
    items: list


class CsvBody(BaseModel):
    """POST /api/seed-utterances/import payload — raw CSV text."""
    csv: str


@app.on_event("startup")
def _startup() -> None:
    """Start session initialisation in a background thread so uvicorn boots quickly."""
    threading.Thread(target=init_session, daemon=True).start()


@app.get("/api/state")
def get_state():
    """Polling endpoint used by the frontend to show loading / error / ready state.

    Deliberately LOCK-free: this is the liveness poll the UI hits continuously,
    and it must stay responsive while a long-held LOCK (model load on startup, a
    seed replay, or a dream pass) is in flight. It only reads atomic flag/ref
    values and never touches the SQLite connection, so no lock is required.
    """
    s = APP["session"]
    return {**STATE, "turn": s.turn if s else 0, "seeded": s.seeded if s else False,
            "embedding": s.provider.status if s else "", "llm": s.llm.status if s else ""}


@app.get("/api/config")
def get_config():
    """Return the currently active configuration as a serialisable dict."""
    with LOCK:
        s = APP["session"]
        return (s.cfg if s else Config()).to_dict()


@app.post("/api/reset")
def reset(body: ConfigBody):
    """Wipe the DB and rebuild the session with the supplied configuration.
    Allowed while not ready, so a bad configuration (init_error) can be corrected."""
    require_idle()
    try:
        cfg = Config.from_dict(body.config)
    except ValueError as e:
        raise HTTPException(400, str(e))
    run_job(lambda: init_session(cfg, wipe=True))
    return {"ok": True}


@app.post("/api/reset-db")
def reset_db():
    """Soft-reset: erase memories, turn log and metrics without rebuilding the session."""
    s = session()
    with LOCK:                 # idle check under LOCK: a seed replay admits between turns
        require_idle()
        s.reset()
    return {"ok": True}


@app.post("/api/seed")
def run_seed(body: SeedRunBody | None = None):
    """Replay the current seed utterances along a virtual timeline.

    By default the memory store is wiped first (body.reset); pass reset=False to
    append the seed onto whatever memories already exist.
    """
    run_seed_job(session(), do_reset=body is None or body.reset)
    return {"ok": True}


@app.post("/api/turn")
def turn(body: TurnBody):
    """Run one user turn (recall → respond → write → cite) asynchronously."""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "空の発話です。")
    s = session()
    run_job(lambda: s.run_turn(text))
    return {"ok": True}


@app.post("/api/dream")
def dream(body: DreamBody):
    """Trigger memory consolidation (dreaming): ≤ budget LLM adjudications."""
    s = session(idle=True)
    with LOCK:
        n = len(s.memory.clusters())
    if not n:
        return {"ok": True, "n": 0, "message": "Dream 対象のクラスタがありません（不安定な記憶に近傍がない＝整理済み）"}
    budget = None if body.budget is None else max(1, body.budget)

    def job():
        results = s.dream(budget)
        replaced = sum(1 for r in results if r.get("action") == "replace")
        STATE["progress"] = f"dreamed {len(results)} cluster(s), {replaced} consolidated"
    run_job(job)
    return {"ok": True, "n": n}


@app.get("/api/dream-log")
def dream_log():
    """Return the results of the most recent dreaming pass."""
    with LOCK:
        s = APP["session"]
        return {"results": list(s.last_dream) if s else []}


@app.get("/api/turns-detail")
def turns_detail():
    """Return the full turn log with per-turn metrics and recalled memories."""
    with LOCK:
        s = APP["session"]
        return {"start_time": s.start_time if s else None, "turns": list(s.log) if s else []}


@app.get("/api/db")
def db():
    """Introspection endpoint: DB stats plus raw table snapshots (for the UI DB tab)."""
    with LOCK:
        s = APP["session"]
        if not s:
            return {"stats": {}, "tables": {}}
        m = s.memory
        return {
            "stats": {**m.stats(), "vector_mb": round(m.vector_mb(), 3), "db_kb": round(m.db_size_bytes() / 1024, 1)},
            "tables": {"memory": m.snapshot()},
        }


@app.get("/api/clusters")
def clusters():
    """Return current dream cluster candidates with member details (preview, no execution)."""
    with LOCK:
        s = APP["session"]
        if not s or not STATE["ready"]:
            return {"clusters": [], "message": "エンジン未起動"}
        m = s.memory
        now = m.now_unix()
        result = [{
            "seed": members[0].id[:10],
            "member_count": len(members),
            "members": [{"id": x.id, "text": x.text, "R": round(m.retrievability(x, now), 2),
                         "stability_days": round(x.stability / 86400, 1)} for x in members],
        } for members in m.clusters()]
    return {"clusters": result, "total_clusters": len(result)}


@app.get("/api/metrics")
def metrics():
    """Return all recorded turn metrics as rows, plus invariant checks."""
    with LOCK:
        s = APP["session"]
        if not s:
            return {"rows": [], "invariants": {}}
        rows = s.recorder.rows()
        mem = s.cfg.memory
    return {
        "budget": mem.budget_chars,
        "cap": mem.capacity,
        "invariants": {
            f"全 pack <= {mem.budget_chars}字 (注入予算)": all(r["pack_chars"] <= mem.budget_chars for r in rows),
            f"全 records <= {mem.capacity}件 (capacity)": all(r["records"] <= mem.capacity for r in rows),
        },
        "rows": rows,
    }


@app.get("/api/seed-utterances")
def seed_utts():
    """Return the currently configured seed utterances."""
    return {"utterances": [{"i": i + 1, **item} for i, item in enumerate(list(APP["seed"]))]}


def _set_seed(items: list[dict[str, str]]) -> dict:
    APP["seed"] = items        # rebound atomically; the CSV is independent of the DB
    seed.save(SEED_CSV_PATH, items)
    return {"ok": True, "n": len(items)}


@app.post("/api/seed-utterances")
def save_seed_utts(body: SeedBody):
    """Replace the seed utterances list and persist it to CSV."""
    items = seed.clean(body.items)
    if not items:
        raise HTTPException(400, "少なくとも1件の発話が必要です。")
    return _set_seed(items)


@app.post("/api/seed-utterances/reset")
def reset_seed_utts():
    """Restore the built-in default seed scenario."""
    return _set_seed(seed.clean(seed.DEFAULT_SEED))


@app.post("/api/seed-utterances/import")
def import_seed_utts(body: CsvBody):
    """Parse raw CSV text and return cleaned seed items (preview before save)."""
    items = seed.parse_csv(body.csv or "")
    if not items:
        raise HTTPException(400, "CSVから有効な発話を読み取れませんでした（text列が必要です）。")
    return {"items": items, "n": len(items)}


app.mount("/", NoCacheStaticFiles(directory=os.path.join(ROOT, "frontend"), html=True), name="frontend")


def _open_browser():
    """Open the default web browser after a short delay (convenience for local use)."""
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=8501)
    except OSError as exc:
        # Most common cause: a previous (possibly hung) instance still holds 8501.
        print(
            "\n[error] could not start the web server on http://localhost:8501\n"
            f"        {type(exc).__name__}: {exc}\n"
            "        Port 8501 is likely held by a previous instance. Close the other\n"
            "        server window, or run start.bat again (it now frees a stale port).\n"
        )
        raise SystemExit(1)
