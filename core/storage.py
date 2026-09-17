"""SQLite substrate for the engine: one ``memory`` table (SPEC §2) plus the app's
turn log. The engine keeps every trace in RAM; the store only persists."""
from __future__ import annotations

import glob as _glob
import os
import sqlite3
import time
from contextlib import contextmanager

import numpy as np

from memory.model import Memory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  tz TEXT NOT NULL,
  last_recall INTEGER NOT NULL,
  stability REAL NOT NULL,
  consolidated INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS turn_log (
  turn INTEGER PRIMARY KEY,
  utterance TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  timestamp REAL NOT NULL,
  system_json TEXT NOT NULL DEFAULT '{}'
);
"""


class Store:
    def __init__(self, path: str, snapshot_gens: int = 8):
        self.path = path
        self.snapshot_gens = snapshot_gens
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")   # WAL-safe; one fsync per checkpoint, not per commit
        self.conn.executescript(_SCHEMA)
        self._in_txn = False

    # -- MemoryStore contract ------------------------------------------- #
    def load_all(self) -> list[Memory]:
        return [Memory(r["id"], r["text"], int(r["created_at"]), r["tz"], int(r["last_recall"]),
                       float(r["stability"]), bool(r["consolidated"]), r["model_id"],
                       np.frombuffer(r["vector"], dtype=np.float32).copy())
                for r in self.conn.execute("SELECT * FROM memory ORDER BY rowid")]

    def put(self, m: Memory) -> None:
        self._exec("INSERT OR REPLACE INTO memory(id,text,created_at,tz,last_recall,stability,consolidated,model_id,vector) "
                   "VALUES(?,?,?,?,?,?,?,?,?)",
                   (m.id, m.text, m.created_at, m.tz, m.last_recall, m.stability, int(m.consolidated),
                    m.model_id, np.asarray(m.vector, dtype=np.float32).tobytes()))

    def remove(self, mid: str) -> None:
        self._exec("DELETE FROM memory WHERE id=?", (mid,))

    def clear(self) -> None:
        self._exec("DELETE FROM memory")
        self._exec("DELETE FROM turn_log")

    @contextmanager
    def transaction(self):
        """Group several writes into one atomic commit (rollback on error)."""
        if self._in_txn:
            yield
            return
        self._in_txn = True
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self._in_txn = False

    def backup(self) -> str | None:
        """Copy the DB into a rotating ring of ``snapshot_gens`` files (before each dream)."""
        if self.path == ":memory:":
            return None
        try:
            snap_dir = os.path.join(os.path.dirname(self.path), "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            dst = os.path.join(snap_dir, f"snap_{int(time.time() * 1000)}.db")
            bck = sqlite3.connect(dst)
            try:
                with bck:
                    self.conn.backup(bck)
            finally:
                bck.close()
            for old in sorted(_glob.glob(os.path.join(snap_dir, "snap_*.db")))[:-self.snapshot_gens]:
                try:
                    os.remove(old)
                except OSError:
                    pass
            return dst
        except Exception:
            return None

    # -- app-level turn log ---------------------------------------------- #
    def save_turn_log(self, turn: int, utterance: str, note: str, timestamp: float, system_json: str,
                      keep: int) -> None:
        with self.transaction():
            self._exec("INSERT OR REPLACE INTO turn_log(turn,utterance,note,timestamp,system_json) VALUES(?,?,?,?,?)",
                       (turn, utterance, note, timestamp, system_json))
            self._exec("DELETE FROM turn_log WHERE turn NOT IN (SELECT turn FROM turn_log ORDER BY turn DESC LIMIT ?)",
                       (keep,))

    def load_turn_log(self, keep: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM turn_log ORDER BY turn DESC LIMIT ?", (keep,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- misc ------------------------------------------------------------ #
    def _exec(self, sql: str, params=()) -> None:
        self.conn.execute(sql, params)
        if not self._in_txn:
            self.conn.commit()

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0])

    def db_size_bytes(self) -> int:
        return sum(os.path.getsize(self.path + s) for s in ("", "-wal", "-shm") if os.path.exists(self.path + s))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def wipe_file(self) -> None:
        self.close()
        for s in ("", "-wal", "-shm"):
            if os.path.exists(self.path + s):
                try:
                    os.remove(self.path + s)
                except Exception:
                    pass
