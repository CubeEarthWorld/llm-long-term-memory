u"""SQLite-backed storage with numpy vector helpers.

The memory system owns one SQLite file (the inspectable DB shown in the UI).
Vectors are stored as float32 BLOBs and searched by brute-force cosine in numpy —
at <=3000 rows x 768 dims this is a fraction of a millisecond, so no FAISS/HNSW
native dependency is needed.
"""
from __future__ import annotations

import glob as _glob
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Sequence

import numpy as np

_VALID_TABLES = {"memory", "vec", "conflict", "dream_log", "spec"}


def pack_vec(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_vec(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def quantize_int8(vec: np.ndarray) -> tuple[bytes, float]:
    """Symmetric int8 quantization (ENGRAM §8: L2/L3 store int8 + scale).

    Returns (blob, scale) so the original can be approximately recovered as
    ``q * scale``. The input is expected unit-norm; the scale captures its peak.
    """
    v = np.asarray(vec, dtype=np.float32)
    peak = float(np.max(np.abs(v))) if v.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    q = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
    return q.tobytes(), scale


def dequantize_int8(blob: bytes, scale: float) -> np.ndarray:
    """Inverse of :func:`quantize_int8`. Caller re-normalizes before cosine."""
    q = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
    return q * float(scale)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def vec_label(blob: bytes | None) -> str:
    """Human-readable label for a stored vector blob."""
    return "" if blob is None else f"<{len(blob)}B vec>"


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        # Incremental auto-vacuum so the dream housekeeping can reclaim freed pages
        # without a full VACUUM (§6). Effective from creation on a fresh (wiped) DB.
        self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._in_txn = False

    # -- basic helpers -------------------------------------------------- #
    def exec(self, sql: str, params: Sequence = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        if not self._in_txn:
            self.conn.commit()
        return cur

    @contextmanager
    def transaction(self):
        """Group several exec() calls into one atomic commit (rollback on error).

        Nested use is a no-op: the outermost transaction() owns the commit.
        """
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

    def execscript(self, sql: str) -> None:
        self.conn.executescript(sql)
        self.conn.commit()

    def query(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def one(self, sql: str, params: Sequence = ()) -> sqlite3.Row | None:
        row = self.conn.execute(sql, params).fetchone()
        return row

    def scalar(self, sql: str, params: Sequence = ()):
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    # -- maintenance ---------------------------------------------------- #
    def count(self, table: str, where: str = "", params: Sequence = ()) -> int:
        if table not in _VALID_TABLES:
            raise ValueError(f"Invalid table: {table}")
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.scalar(sql, params) or 0)

    def db_size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total

    def vector_storage_mb(self, tables_cols: Sequence[tuple[str, str]]) -> float:
        total = 0
        for table, col in tables_cols:
            if table not in _VALID_TABLES:
                raise ValueError(f"Invalid table: {table}")
            for r in self.query(f"SELECT {col} AS v FROM {table} WHERE {col} IS NOT NULL"):
                total += len(r["v"])
        return total / (1024 * 1024)

    def rows_as_dicts(self, table: str, drop_blobs: bool = True, limit: int = 5000) -> list[dict]:
        if table not in _VALID_TABLES:
            raise ValueError(f"Invalid table: {table}")
        def _clean(row: sqlite3.Row) -> dict:
            d = dict(row)
            if drop_blobs:
                for k, v in list(d.items()):
                    if isinstance(v, (bytes, bytearray)):
                        d[k] = vec_label(v)
            return d
        return [_clean(r) for r in self.query(f"SELECT * FROM {table} LIMIT ?", (limit,))]

    # -- durability ----------------------------------------------------- #
    def wal_checkpoint(self) -> None:
        """Truncate the WAL so the -wal file cannot grow without bound (§6, §8.1)."""
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass

    def incremental_vacuum(self) -> None:
        try:
            self.conn.execute("PRAGMA incremental_vacuum;")
            self.conn.commit()
        except Exception:
            pass

    def snapshot(self, snap_dir: str, gens: int = 8) -> str | None:
        """Copy the DB into a rotating ring of ``gens`` snapshots (ENGRAM §6 snapshot()).

        Uses the SQLite online backup API for a consistent copy, then prunes the
        oldest files so the保険 stays bounded (~80MB for 8 generations).
        """
        try:
            os.makedirs(snap_dir, exist_ok=True)
            dst = os.path.join(snap_dir, f"snap_{int(time.time() * 1000)}.db")
            bck = sqlite3.connect(dst)
            try:
                with bck:
                    self.conn.backup(bck)
            finally:
                bck.close()
            snaps = sorted(_glob.glob(os.path.join(snap_dir, "snap_*.db")))
            for old in snaps[:-gens]:
                try:
                    os.remove(old)
                except OSError:
                    pass
            return dst
        except Exception:
            return None

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def wipe_file(self) -> None:
        self.close()
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
