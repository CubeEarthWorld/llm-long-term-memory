"""FastAPI backend for the LLM Long-Term Memory prototype.

Serves a JSON REST API plus the no-build React frontend in ./frontend.
The engine is built lazily on startup (or on reset) in a background thread
so that the UI can poll /api/state while heavy models load.

Thread-safety is handled by a single module-level threading.Lock; FastAPI
itself runs handlers concurrently, but all mutating jobs are wrapped in
run_job() which serialises them via the lock.
"""
from __future__ import annotations

import csv
import io
import os
import threading
import time
import traceback
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config, default_config
from core.engine import (
    BASE_DIR,
    DATA_DIR,
    SEED_CSV_PATH,
    SYSTEM_ID,
    SYSTEM_TITLE,
    Engine,
    _parse_duration,
    build_engine,
    clean_seed_items,
    default_seed,
    dispose_engine,
    parse_seed_csv,
    reset_state,
    run_dream,
    run_turn,
)

app = FastAPI(title=SYSTEM_TITLE)

# Shared mutable state — accessed only while LOCK is held by run_job(),
# or read-only in GET handlers after the engine is ready.
STATE = {"ready": False, "running": False, "progress": "", "error": None, "init_error": None}
ENGINE: dict = {"e": None, "cfg": None}   # "e" holds the Engine from core.engine
LOCK = threading.RLock()


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
    """POST /api/dream payload — how many clusters to consolidate."""
    max_clusters: int = 1
    force: bool = True


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


# --------------------------------------------------------------------------- #
# Seed CSV persistence helpers
# --------------------------------------------------------------------------- #

def _save_seed(items: list[dict]) -> None:
    """Persist seed utterances to CSV so edits survive server restarts."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SEED_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "note", "advance"])
        for item in items:
            writer.writerow([item["text"], item.get("note", ""), item["advance"]])


SEED: dict = {"items": default_seed()}


# --------------------------------------------------------------------------- #
# Engine lifecycle
# --------------------------------------------------------------------------- #

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


def _active_engine() -> Engine:
    """Return the current engine, or raise HTTPException(503/409) if unavailable/busy."""
    with LOCK:
        engine = ENGINE["e"]
        if not engine or not STATE["ready"]:
            raise HTTPException(503, "Engine is still starting.")
        if STATE["running"]:
            raise HTTPException(409, "A job is already running.")
        return engine


def _system_detail(engine: Engine, turn: int) -> dict:
    """Build the per-turn detail object returned by /api/turns-detail."""
    metrics = engine["recorder"].for_turn(turn, SYSTEM_ID)
    if not metrics:
        return {}
    return metrics.to_detail_dict(SYSTEM_TITLE)


@app.on_event("startup")
def _startup() -> None:
    """Start engine initialisation in a background thread so uvicorn boots quickly."""
    threading.Thread(target=lambda: init_engine(wipe=False), daemon=True).start()


@app.get("/api/state")
def get_state():
    """Polling endpoint used by the frontend to show loading / error / ready state.

    Deliberately LOCK-free: this is the liveness poll the UI hits continuously,
    and it must stay responsive while a long-held LOCK (model load on startup, a
    seed replay, or a dream pass) is in flight. It only reads atomic flag/ref
    values and never touches the SQLite connection, so no lock is required.
    Acquiring LOCK here was the cause of the 'no response during load/seed/dream'
    hang: the poll blocked behind the very job whose progress it was meant to show.
    """
    e = ENGINE["e"]
    state = {k: STATE.get(k) for k in ("ready", "running", "progress", "error", "init_error")}
    turn, seeded = (e["turn"], e["seeded"]) if e else (0, False)
    emb = e["provider"].status if e else ""
    llm = e["llm"].status if e else ""
    return {**state, "turn": turn, "seeded": seeded, "embedding": emb, "llm": llm, "n_seed": len(SEED["items"]), "system_title": SYSTEM_TITLE}


@app.get("/api/config")
def get_config():
    """Return the currently active configuration as a serialisable dict."""
    with LOCK:
        cfg = ENGINE["cfg"] or default_config()
        return cfg.to_dict()


@app.post("/api/reset")
def reset(body: ConfigBody):
    """Wipe the DB and rebuild the engine with the supplied configuration."""
    with LOCK:
        if STATE["running"]:
            raise HTTPException(409, "A job is already running.")
    cfg = Config.from_dict(body.config)
    def job():
        init_engine(cfg, wipe=True)
    run_job(job)
    return {"ok": True}


@app.post("/api/reset-db")
def reset_db():
    """Soft-reset: clear in-memory state (turn log, metrics) without deleting the DB."""
    with LOCK:
        engine = ENGINE["e"]
        if not engine or not STATE["ready"]:
            raise HTTPException(503, "Engine is still starting.")
        if STATE["running"]:
            raise HTTPException(409, "A job is already running.")
        reset_state(engine)
    return {"ok": True}


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


@app.post("/api/seed")
def seed(body: SeedRunBody | None = None):
    """Replay the current seed utterances along a virtual timeline.

    By default the memory store is wiped first (body.reset); pass reset=False to
    append the seed onto whatever memories already exist.
    """
    with LOCK:
        if STATE["running"]:
            raise HTTPException(409, "A job is already running.")
        engine = ENGINE["e"]
        if not engine or not STATE["ready"]:
            raise HTTPException(503, "Engine is still starting.")
        engine["seed"] = SEED["items"]
        do_reset = True if body is None else bool(body.reset)
        seed_items = list(engine["seed"]) if isinstance(engine["seed"], list) else default_seed()
    _start_seed_run(engine, do_reset, seed_items)
    return {"ok": True}


@app.post("/api/turn")
def turn(body: TurnBody):
    """Run one user turn (retrieve → respond → write → maintain) asynchronously."""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "空の発話です。")
    def job():
        engine = ENGINE["e"]
        if not engine or not STATE["ready"]:
            raise HTTPException(503, "Engine is still starting.")
        run_turn(engine, text)
    run_job(job)
    return {"ok": True}


@app.post("/api/dream")
def dream(body: DreamBody):
    """Trigger memory consolidation (dreaming) on the top-N priority clusters."""
    with LOCK:
        engine = ENGINE["e"]
        if not engine or not STATE["ready"]:
            raise HTTPException(503, "Engine is still starting.")
        if STATE["running"]:
            raise HTTPException(409, "A job is already running.")
        system = engine["system"]
        candidates = system._dream_candidates(system.now_unix(), force=bool(body.force))
    if not candidates:
        return {"ok": True, "n": 0, "message": "Dream 対象の適格クラスタがありません（メンバー3件未満、または「変更なし」指紋でスキップ）"}
    n = max(1, int(body.max_clusters))
    force = bool(body.force)
    def job():
        engine = ENGINE["e"]
        if not engine or not STATE["ready"]:
            return
        results = run_dream(engine, max_clusters=n, force=force)
        merged = sum(1 for r in results if r.get("action") != "none")
        STATE["progress"] = f"dreamed {len(results)} cluster(s), {merged} consolidated"
    run_job(job)
    return {"ok": True, "n": len(candidates)}


@app.get("/api/dream-log")
def dream_log():
    """Return the results of the most recent dreaming pass."""
    with LOCK:
        engine = ENGINE["e"]
        return {"results": list(engine.get("last_dream", [])) if engine else []}


@app.get("/api/turns-detail")
def turns_detail():
    """Return the full turn log enriched with per-turn metrics and recalled memories."""
    with LOCK:
        engine = ENGINE["e"]
        if not engine:
            return {"turns": [], "start_time": None}
        log = list(engine["log"])
        st = engine.get("start_time")
    return {
        "start_time": st,
        "turns": [{"turn": r["turn"], "utterance": r["utterance"], "note": r.get("note", ""), "timestamp": r.get("timestamp"), "system": r.get("system") or _system_detail(engine, r["turn"])} for r in log],
    }


@app.get("/api/db")
def db():
    """Introspection endpoint: DB stats plus raw table snapshots (for the UI DB tab)."""
    with LOCK:
        engine = ENGINE["e"]
        if not engine:
            return {"stats": {}, "tables": {}}
        system = engine["system"]
        return {
            "stats": {
                "total_records": system.total_records(),
                **system.stats(),
                "vector_mb": round(system.vector_mb(), 3),
                "db_kb": round(system.db_size_bytes() / 1024, 1),
            },
            "tables": system.snapshot(),
        }


@app.get("/api/clusters")
def clusters():
    """Return current dream cluster candidates with member details (preview, no execution)."""
    with LOCK:
        engine = ENGINE["e"]
        if not engine or not STATE["ready"]:
            return {"clusters": [], "message": "エンジン未起動"}
        system = engine["system"]
        now = system.now_unix()
        candidates = system._dream_candidates(now, force=True)
    result = []
    for fp, members, prio in candidates:
        member_list = []
        for m in members:
            member_list.append({
                "id": m["id"],
                "text": m["text"],
                "tier": int(m["tier"]),
                "gen": int(m["gen"]),
                "A": round(system.activation(m, now), 2),
            })
        result.append({
            "cluster_fp": fp[:12],
            "priority": round(prio, 3),
            "member_count": len(members),
            "members": member_list,
        })
    # Also include any clusters that were already adjudicated (from dream_log)
    return {"clusters": result, "total_clusters": len(result)}


@app.get("/api/metrics")
def metrics():
    """Return all recorded turn metrics as rows, plus invariant checks."""
    with LOCK:
        engine = ENGINE["e"]
        if not engine:
            return {"rows": [], "invariants": {}}
        hist = list(engine["recorder"].history)
        mem = ENGINE["cfg"].memory
        budget = ENGINE["cfg"].glob.budget_chars
        cap = mem.cap1 + mem.cap2 + mem.cap3
    rows = [m.row() for m in hist]
    return {
        "budget": budget,
        "cap": cap,
        "invariants": {
            f"全 pack <= {budget}字 (注入予算)": all(r["pack_chars"] <= budget for r in rows),
            f"全 records <= {cap}件 (L1+L2+L3 容量)": all(r["records"] <= cap for r in rows),
            f"L1 <= {mem.cap1}件": all((r.get("L1") or 0) <= mem.cap1 for r in rows),
            f"L2 <= {mem.cap2}件": all((r.get("L2") or 0) <= mem.cap2 for r in rows),
            f"L3 <= {mem.cap3}件": all((r.get("L3") or 0) <= mem.cap3 for r in rows),
        },
        "rows": rows,
    }


@app.get("/api/seed-utterances")
def seed_utts():
    """Return the currently configured seed utterances."""
    with LOCK:
        items = list(SEED["items"])
    return {"utterances": [{"i": i + 1, **item} for i, item in enumerate(items)]}


@app.post("/api/seed-utterances")
def save_seed_utts(body: SeedBody):
    """Replace the seed utterances list and persist it to CSV."""
    items = clean_seed_items(body.items)
    if not items:
        raise HTTPException(400, "少なくとも1件の発話が必要です。")
    SEED["items"] = items
    _save_seed(items)
    if ENGINE["e"]:
        ENGINE["e"]["seed"] = items
    return {"ok": True, "n": len(items)}


@app.post("/api/seed-utterances/reset")
def reset_seed_utts():
    """Restore the built-in default seed scenario."""
    SEED["items"] = default_seed()
    _save_seed(SEED["items"])
    if ENGINE["e"]:
        ENGINE["e"]["seed"] = SEED["items"]
    return {"ok": True, "n": len(SEED["items"])}


@app.get("/api/seed-utterances/export")
def export_seed_utts():
    """Download the current seed utterances as a UTF-8 CSV with BOM."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["text", "note", "advance"])
    for item in SEED["items"]:
        writer.writerow([item["text"], item.get("note", ""), item["advance"]])
    return Response(
        content="\ufeff" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="seed.csv"'},
    )


@app.post("/api/seed-utterances/import")
def import_seed_utts(body: CsvBody):
    """Parse raw CSV text and return cleaned seed items (preview before save)."""
    items = parse_seed_csv(body.csv or "")
    if not items:
        raise HTTPException(400, "CSVから有効な発話を読み取れませんでした（text列が必要です）。")
    return {"items": items, "n": len(items)}


app.mount("/", NoCacheStaticFiles(directory=os.path.join(BASE_DIR, "frontend"), html=True), name="frontend")


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
