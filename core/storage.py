u"""SQLite-backed storage with numpy vector helpers.

The memory system owns one SQLite file (the inspectable DB shown in the UI).
Vectors are stored as float32 BLOBs and searched by brute-force cosine in numpy —
at <=3000 rows x 768 dims this is a fraction of a millisecond, so no FAISS/HNSW
native dependency is needed.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def pack_vec(vec: Optional[np.ndarray]) -> Optional[bytes]:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_vec(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")

    # -- basic helpers -------------------------------------------------- #
    def exec(self, sql: str, params: Sequence = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def execscript(self, sql: str) -> None:
        self.conn.executescript(sql)
        self.conn.commit()

    def query(self, sql: str, params: Sequence = ()) -> List[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def one(self, sql: str, params: Sequence = ()) -> Optional[sqlite3.Row]:
        row = self.conn.execute(sql, params).fetchone()
        return row

    def scalar(self, sql: str, params: Sequence = ()):
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    # -- vector search -------------------------------------------------- #
    def cosine_search(
        self,
        table: str,
        vec_col: str,
        query_vec: np.ndarray,
        k: int,
        where: str = "",
        params: Sequence = (),
        id_col: str = "id",
    ) -> List[Tuple]:
        """Return [(id, score, row), ...] top-k by cosine over rows matching `where`.

        Vectors are assumed L2-normalized, so cosine == dot product.
        """
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        rows = self.query(sql, params)
        if not rows:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn:
            q = q / qn
        scored = []
        for r in rows:
            blob = r[vec_col]
            if blob is None:
                continue
            v = unpack_vec(blob)
            if v.shape[0] != q.shape[0]:
                continue  # all stored vectors share one dimension; skip any stray mismatch
            score = float(np.dot(v, q))
            scored.append((r[id_col], score, r))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # -- maintenance ---------------------------------------------------- #
    def count(self, table: str, where: str = "", params: Sequence = ()) -> int:
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

    def vector_storage_mb(self, tables_cols: Sequence[Tuple[str, str]]) -> float:
        total = 0
        for table, col in tables_cols:
            for r in self.query(f"SELECT {col} AS v FROM {table} WHERE {col} IS NOT NULL"):
                total += len(r["v"])
        return total / (1024 * 1024)

    def rows_as_dicts(self, table: str, drop_blobs: bool = True, limit: int = 5000) -> List[Dict]:
        rows = self.query(f"SELECT * FROM {table} LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            if drop_blobs:
                for k, v in list(d.items()):
                    if isinstance(v, (bytes, bytearray)):
                        d[k] = f"<{len(v)}B vec>"
            out.append(d)
        return out

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
