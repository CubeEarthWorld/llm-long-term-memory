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
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config, default_config
from core.engine import (
    BASE_DIR,
    DATA_DIR,
    SYSTEM_ID,
    SYSTEM_TITLE,
    build_engine,
    default_seed,
    dispose_engine,
    reset_state,
    run_dream,
    run_seed,
    run_turn,
)

app = FastAPI(title=SYSTEM_TITLE)

# Shared mutable state — accessed only while LOCK is held by run_job(),
# or read-only in GET handlers after the engine is ready.
STATE = {"ready": False, "running": False, "progress": "", "error": None, "init_error": None}
ENGINE = {"e": None, "cfg": None}   # "e" holds the engine dict from core.engine
LOCK = threading.Lock()
SEED_CSV = os.path.join(DATA_DIR, "seed_utterances.csv")  # persisted seed utterances


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


class SeedBody(BaseModel):
    """POST /api/seed-utterances payload — replace the entire seed list."""
    items: list


class CsvBody(BaseModel):
    """POST /api/seed-utterances/import payload — raw CSV text."""
    csv: str


# --------------------------------------------------------------------------- #
# Seed CSV persistence helpers
# --------------------------------------------------------------------------- #

def _load_seed() -> List[dict]:
    """Load seed utterances from CSV, or fall back to the built-in scenario."""
    if not os.path.exists(SEED_CSV):
        return default_seed()
    try:
        with open(SEED_CSV, encoding="utf-8-sig", newline="") as f:
            rows = [
                {
                    "text": (r.get("text") or "").strip(),
                    "note": (r.get("note") or "").strip(),
                    "advance": ((r.get("advance") or "0").strip() or "0"),
                }
                for r in csv.DictReader(f)
            ]
        rows = [r for r in rows if r["text"]]
        return rows or default_seed()
    except Exception:
        traceback.print_exc()
        return default_seed()


def _save_seed(items: List[dict]) -> None:
    """Persist seed utterances to CSV so edits survive server restarts."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SEED_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "note", "advance"])
        for item in items:
            writer.writerow([item["text"], item.get("note", ""), (item.get("advance", "0") or "0")])


def _clean_seed_items(raw) -> List[dict]:
    """Normalise a list of raw dicts into valid seed items (text/note/advance)."""
    items = []
    for item in raw or []:
        text = str((item or {}).get("text", "")).strip()
        if text:
            items.append({
                "text": text,
                "note": str((item or {}).get("note", "")).strip(),
                "advance": (str((item or {}).get("advance", "0")).strip() or "0"),
            })
    return items


SEED = {"items": _load_seed()}


# --------------------------------------------------------------------------- #
# Engine lifecycle
# --------------------------------------------------------------------------- #

def init_engine(cfg: Config | None = None, wipe: bool = False) -> None:
    """Build or rebuild the engine in the current thread.

    Disposes any existing engine first.  If wipe=False and a previous DB exists,
    maintain() is called once to gracefully catch up on decayed memories.
    State flags (ready / init_error) are updated for the UI polling loop.
    """
    STATE["ready"] = False
    cfg = cfg or default_config()
    try:
        if ENGINE["e"]:
            dispose_engine(ENGINE["e"])
        ENGINE["e"] = build_engine(cfg, wipe=wipe, seed=SEED["items"])
        ENGINE["cfg"] = cfg
        # After a long downtime, gradually clean up stale memories instead of
        # archiving everything at the first write().
        if not wipe and ENGINE["e"]:
            try:
                ENGINE["e"]["system"].maintain(ENGINE["e"]["turn"])
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
                STATE["progress"] = ""

    threading.Thread(target=worker, daemon=True).start()


def _active_engine() -> dict:
    """Return the current engine, or raise HTTPException(503/409) if unavailable/busy."""
    engine = ENGINE["e"]
    if not engine or not STATE["ready"]:
        raise HTTPException(503, "Engine is still starting.")
    if STATE["running"]:
        raise HTTPException(409, "A job is already running.")
    return engine


def _system_detail(engine: dict, turn: int) -> dict:
    """Build the per-turn detail object returned by /api/turns-detail."""
    metrics = engine["recorder"].for_turn(turn, SYSTEM_ID)
    if not metrics:
        return {}
    return {
        "title": SYSTEM_TITLE,
        "response": metrics.response,
        "write_note": metrics.write_note,
        "records": metrics.total_records,
        "pack_chars": metrics.pack_chars,
        "pack_n": metrics.pack_n,
        "pack_text": metrics.pack_text,
        "prompt": metrics.prompt,
        "written": metrics.written_rows,
        "times": {
            "total": round(metrics.total_ms, 1),
            "llm": round(metrics.llm_ms, 1),
            "retrieve": round(metrics.retrieve_ms, 1),
            "write": round(metrics.write_ms, 1),
            "maintain": round(metrics.maintain_ms, 1),
            "embed": round(metrics.embed_ms, 1),
        },
        "recalled": [
            {"id": item.mem_id, "text": item.text, "score": round(item.score, 3), **item.extra}
            for item in metrics.recalled
        ],
    }


@app.on_event("startup")
def _startup() -> None:
    """Start engine initialisation in a background thread so uvicorn boots quickly."""
    threading.Thread(target=lambda: init_engine(wipe=False), daemon=True).start()


@app.get("/api/state")
def get_state():
    """Polling endpoint used by the frontend to show loading / error / ready state."""
    engine = ENGINE["e"]
    return {
        **{k: STATE[k] for k in ("ready", "running", "progress", "error", "init_error")},
        "turn": engine["turn"] if engine else 0,
        "seeded": engine["seeded"] if engine else False,
        "embedding": engine["provider"].status if engine else "",
        "llm": engine["llm"].status if engine else "",
        "n_seed": len(SEED["items"]),
        "system_title": SYSTEM_TITLE,
    }


@app.get("/api/config")
def get_config():
    """Return the currently active configuration as a serialisable dict."""
    cfg = ENGINE["cfg"] or default_config()
    return cfg.to_dict()


@app.post("/api/reset")
def reset(body: ConfigBody):
    """Wipe the DB and rebuild the engine with the supplied configuration."""
    if STATE["running"]:
        raise HTTPException(409, "A job is already running.")
    cfg = Config.from_dict(body.config)
    threading.Thread(target=lambda: init_engine(cfg, wipe=True), daemon=True).start()
    return {"ok": True}


@app.post("/api/reset-db")
def reset_db():
    """Soft-reset: clear in-memory state (turn log, metrics) without deleting the DB."""
    reset_state(_active_engine())
    return {"ok": True}


@app.post("/api/seed")
def seed():
    """Replay the current seed utterances along a virtual timeline."""
    engine = _active_engine()
    engine["seed"] = SEED["items"]
    total = len(SEED["items"])

    def job():
        reset_state(engine)
        run_seed(engine, on_progress=lambda t, u: STATE.update(progress=f"turn {t}/{total}: {u[:16]}..."))

    run_job(job)
    return {"ok": True}


@app.post("/api/turn")
def turn(body: TurnBody):
    """Run one user turn (retrieve → respond → write → maintain) asynchronously."""
    engine = _active_engine()
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "空の発話です。")
    run_job(lambda: run_turn(engine, text))
    return {"ok": True}


@app.post("/api/dream")
def dream(body: DreamBody):
    """Trigger memory consolidation (dreaming) on the top-N priority clusters."""
    engine = _active_engine()
    n = max(1, int(body.max_clusters))
    force = bool(body.force)

    # Immediate candidate check so the UI can show "nothing to dream" right away.
    system = engine["system"]
    candidates = system._dream_candidates(system.now_unix(), force=force)
    if not candidates:
        return {"ok": True, "n": 0, "message": "Dream 対象のクラスタがありません（クラスタが小さすぎるか、クールダウン中です）"}

    def job():
        results = run_dream(engine, max_clusters=n, force=force)
        merged = sum(1 for r in results if r.get("action") != "none")
        STATE["progress"] = f"dreamed {len(results)} cluster(s), {merged} consolidated"

    run_job(job)
    return {"ok": True, "n": len(candidates)}


@app.get("/api/dream-log")
def dream_log():
    """Return the results of the most recent dreaming pass."""
    engine = ENGINE["e"]
    return {"results": engine.get("last_dream", []) if engine else []}


@app.get("/api/turns-detail")
def turns_detail():
    """Return the full turn log enriched with per-turn metrics and recalled memories."""
    engine = ENGINE["e"]
    if not engine:
        return {"turns": [], "start_time": None}
    return {
        "start_time": engine.get("start_time"),
        "turns": [
            {
                "turn": row["turn"],
                "utterance": row["utterance"],
                "note": row.get("note", ""),
                "timestamp": row.get("timestamp"),
                "system": _system_detail(engine, row["turn"]),
            }
            for row in engine["log"]
        ]
    }


@app.get("/api/db")
def db():
    """Introspection endpoint: DB stats plus raw table snapshots (for the UI DB tab)."""
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


@app.get("/api/metrics")
def metrics():
    """Return all recorded turn metrics as rows, plus invariant checks."""
    engine = ENGINE["e"]
    if not engine:
        return {"rows": [], "invariants": {}}
    rows = [metrics.row() for metrics in engine["recorder"].history]
    cfg = ENGINE["cfg"]
    budget = cfg.glob.budget_chars
    cap = cfg.glob.total_cap
    return {
        "budget": budget,
        "cap": cap,
        "invariants": {
            f"全 pack <= {budget}字": all(row["pack_chars"] <= budget for row in rows) if rows else True,
            f"全 records <= {cap}件": all(row["records"] <= cap for row in rows) if rows else True,
        },
        "rows": rows,
    }


@app.get("/api/seed-utterances")
def seed_utts():
    """Return the currently configured seed utterances."""
    return {
        "utterances": [
            {
                "i": index + 1,
                "text": item["text"],
                "note": item.get("note", ""),
                "advance": (item.get("advance", "0") or "0"),
            }
            for index, item in enumerate(SEED["items"])
        ]
    }


@app.post("/api/seed-utterances")
def save_seed_utts(body: SeedBody):
    """Replace the seed utterances list and persist it to CSV."""
    items = _clean_seed_items(body.items)
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
        writer.writerow([item["text"], item.get("note", ""), (item.get("advance", "0") or "0")])
    return Response(
        content="\ufeff" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="seed_utterances.csv"'},
    )


@app.post("/api/seed-utterances/import")
def import_seed_utts(body: CsvBody):
    """Parse raw CSV text and return cleaned seed items (preview before save)."""
    items = _parse_seed_csv(body.csv or "")
    if not items:
        raise HTTPException(400, "CSVから有効な発話を読み取れませんでした（text列が必要です）。")
    return {"items": items, "n": len(items)}


def _parse_seed_csv(text: str) -> List[dict]:
    """Tolerant CSV parser that accepts both header and header-less CSV for seeds."""
    reader = csv.DictReader(io.StringIO(text))
    items: List[dict] = []
    if reader.fieldnames and any((h or "").strip().lower() == "text" for h in reader.fieldnames):
        for row in reader:
            items.append({
                "text": (row.get("text") or "").strip(),
                "note": (row.get("note") or row.get("memo") or "").strip(),
                "advance": ((row.get("advance") or "0").strip() or "0"),
            })
    else:
        for row in csv.reader(io.StringIO(text)):
            if row:
                items.append({
                    "text": (row[0] or "").strip(),
                    "note": (row[1].strip() if len(row) > 1 else ""),
                    "advance": ((row[2].strip() if len(row) > 2 else "0") or "0"),
                })
    return _clean_seed_items(items)


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
    uvicorn.run(app, host="127.0.0.1", port=8501)
