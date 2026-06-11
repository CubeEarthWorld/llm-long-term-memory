"""LLM Long-Term Memory — internal engine implementing ENGRAM v1.1.

Spec: ``ENGRAM_spec_v1_1.md`` (embedded verbatim into the ``spec`` table at init,
making the DB self-describing). The whole system compresses to four sentences:

    生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。破壊はすべて夢の中。
    (Generation only at the moment of verbalization. All judgement is distance.
     All forgetting is arithmetic. All destruction happens inside the dream.)

* **Text is canonical, vectors are an index.** Memories are short self-contained
  propositions (≤170 chars). The ``vec`` table is a derived, regenerable cache.
* **Three tiers** (§2): L1 episodic (768d f32, τ=7d), L2 semantic (256d int8,
  τ=90d), L3 schema (128d int8, τ=3y). MRL truncation = forgetting resolution.
* **Activation** ``A = mass·2^(−Δt/τ)`` (§4.1) gates nothing but reweights the
  cosine score (§4.2). Identity is pure cosine thresholds (§4.3).
* **Append-only during conversation.** Physical deletion / consolidation happens
  only in :meth:`dream` (§5–§6), 1 adjudication = 1 transaction.

The class is named ``LongTermMemory`` and the DB filename is unchanged; only the
algorithm/schema/parameters are ENGRAM (the previous FSRS model is fully removed).
"""
from __future__ import annotations

import os
import time

import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[misc,assignment]

from config import GlobalConfig, LongTermMemoryConfig
from core.base import MemorySystem
from core.embedding import EmbeddingProvider
from core.llm_client import LLMClient
from core.storage import Store
from memory.dream_ops import DreamOpsMixin
from memory.helpers import _fmt_local, _utc_datetime
from memory.maintenance_ops import MaintenanceOpsMixin
from memory.read_ops import ReadOpsMixin
from memory.vector_ops import VectorOpsMixin
from memory.write_ops import WriteOpsMixin

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC_PATH = os.path.join(_REPO_ROOT, "ENGRAM_spec_v1_1.md")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  tier INTEGER NOT NULL,
  gen INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  tz TEXT NOT NULL,
  last_access INTEGER NOT NULL,
  last_bonus_at INTEGER NOT NULL DEFAULT 0,
  mass REAL NOT NULL DEFAULT 1.0,
  superseded_by TEXT,
  CHECK (tier IN (1,2,3)),
  CHECK (gen BETWEEN 0 AND 7),
  CHECK (mass <= 64),
  CHECK (length(text) BETWEEN 1 AND 1024)
);
CREATE TABLE IF NOT EXISTS vec (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,
  dim INTEGER NOT NULL,
  dtype TEXT NOT NULL,
  scale REAL,
  v BLOB NOT NULL,
  UNIQUE(memory_id, model_id)
);
CREATE TABLE IF NOT EXISTS conflict (a TEXT, b TEXT, at INTEGER);
CREATE TABLE IF NOT EXISTS dream_log (fp TEXT PRIMARY KEY, verdict TEXT, at INTEGER);
CREATE TABLE IF NOT EXISTS turn_log (
  turn INTEGER PRIMARY KEY,
  utterance TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  timestamp REAL NOT NULL,
  system_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS spec (k TEXT, v TEXT);
CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory(tier);
CREATE INDEX IF NOT EXISTS idx_memory_super ON memory(superseded_by);
CREATE INDEX IF NOT EXISTS idx_memory_text ON memory(text);
CREATE INDEX IF NOT EXISTS idx_vec_memory ON vec(memory_id);
"""


class LongTermMemory(VectorOpsMixin, WriteOpsMixin, ReadOpsMixin,
                     MaintenanceOpsMixin, DreamOpsMixin, MemorySystem):
    """ENGRAM engine facade: state, clock and introspection live here; the
    write/read/tier/dream pipelines are provided by the operation mixins."""

    system_id = "llm_long_term_memory"
    system_name = "LLM Long-Term Memory"

    def __init__(
        self,
        store: Store,
        provider: EmbeddingProvider,
        llm: LLMClient,
        memory: LongTermMemoryConfig,
        glob: GlobalConfig,
    ):
        self.store = store
        self.provider = provider
        self.llm = llm
        self.memory = memory
        self.glob = glob
        self._clock = None                       # optional virtual clock (experiments)
        self.model_id = glob.embedding_model
        self._write_times: list[float] = []       # soft write-rate window
        self._doc_vec_cache: dict[str, np.ndarray] = {}  # id → 768d doc vector (text is immutable)
        self._maint_count = 0
        self._snap_dir = os.path.join(os.path.dirname(store.path), "snapshots")
        self.store.execscript(SCHEMA)
        self._install_spec()

    # ================================================================== #
    # time / clock (I2)
    # ================================================================== #
    def now_unix(self) -> float:
        return float(self._clock()) if self._clock else time.time()

    def now_local(self, now: float | None = None) -> str:
        """Current local datetime string ('2026-06-11 21:30 +09:00') for LLM prompts."""
        now = self.now_unix() if now is None else now
        return _fmt_local(now, self._tz_for(now))

    def set_clock(self, fn) -> None:
        self._clock = fn

    def _sanitize_timestamps(self, now: float) -> None:
        """Clamp future timestamps to now so a clock rollback cannot mint immortal mass (I2)."""
        n = int(now)
        self.store.exec(
            "UPDATE memory SET last_access=? WHERE last_access>?", (n, n))
        self.store.exec(
            "UPDATE memory SET last_bonus_at=? WHERE last_bonus_at>?", (n, n))
        self.store.exec(
            "UPDATE memory SET created_at=? WHERE created_at>?", (n, n))

    def _tz_for(self, now: float) -> str:
        """Return 'IANA;+offset' for the configured timezone at ``now`` (§3.2)."""
        name = self.glob.default_timezone
        try:
            dt = _utc_datetime(now).astimezone(ZoneInfo(name))
            off = dt.strftime("%z") or "+0000"
            return f"{name};{off[:3]}:{off[3:]}"
        except Exception:
            return f"{name};+00:00"

    # ================================================================== #
    # spec self-description (§0, §13.5)
    # ================================================================== #
    def _install_spec(self) -> None:
        try:
            with open(_SPEC_PATH, encoding="utf-8") as f:
                spec_text = f.read()
        except Exception:
            spec_text = "ENGRAM v1.1 (spec file not found at install time)"
        self.store.exec("DELETE FROM spec")
        with self.store.transaction():
            self.store.exec("INSERT INTO spec(k,v) VALUES('active_model',?)", (self.model_id,))
            self.store.exec("INSERT INTO spec(k,v) VALUES('spec_version',?)", ("ENGRAM v1.1",))
            self.store.exec("INSERT INTO spec(k,v) VALUES('four_sentences',?)",
                            ("生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。破壊はすべて夢の中。",))
            self.store.exec("INSERT INTO spec(k,v) VALUES('spec_full',?)", (spec_text,))

    # ================================================================== #
    # introspection
    # ================================================================== #
    def _get(self, mid: str):
        return self.store.one("SELECT * FROM memory WHERE id=?", (mid,))

    def _event(self, action: str, row, now: float, **extra) -> dict:
        return {
            "action": action, "id": row["id"], "text": row["text"],
            "tier": int(row["tier"]), "gen": int(row["gen"]),
            "mass": round(float(row["mass"]), 2),
            "A": round(self.activation(row, now), 2), "tz": row["tz"], **extra,
        }

    def stats(self) -> dict[str, int]:
        return {
            "L1": self._tier_count(1), "L2": self._tier_count(2), "L3": self._tier_count(3),
            "tombstones": self.store.count("memory", "superseded_by IS NOT NULL"),
            "conflict": self.store.count("conflict"),
            "dream_log": self.store.count("dream_log"),
        }

    def total_records(self) -> int:
        return self.store.count("memory", "superseded_by IS NULL")

    def snapshot(self) -> dict[str, list[dict]]:
        now = self.now_unix()
        mem = []
        for r in self.store.query("SELECT * FROM memory ORDER BY tier, created_at DESC LIMIT 5000"):
            d = {k: r[k] for k in r.keys()}
            d["mass"] = round(float(r["mass"]), 3)
            d["A"] = round(self.activation(r, now), 3)
            d["local"] = _fmt_local(r["created_at"], r["tz"])
            d["tombstone"] = r["superseded_by"] is not None
            mem.append(d)
        spec = []
        for r in self.store.query("SELECT k, v FROM spec"):
            val = r["v"]
            spec.append({"k": r["k"], "v": (val[:200] + "…") if len(val) > 200 else val})
        return {
            "memory": mem,
            "vec": self.store.rows_as_dicts("vec", limit=5000),
            "conflict": self.store.rows_as_dicts("conflict", limit=512),
            "dream_log": self.store.rows_as_dicts("dream_log", limit=512),
            "spec": spec,
        }

    def vector_mb(self) -> float:
        return self.store.vector_storage_mb([("vec", "v")])

    def db_size_bytes(self) -> int:
        return self.store.db_size_bytes()

    def save_turn_log(self, turn: int, utterance: str, note: str, timestamp: float, system_json: str) -> None:
        """Persist a turn log entry with its system detail to the DB.

        The table is kept to the newest ``max_turn_log`` rows so the DB cannot
        grow without bound over months of conversation.
        """
        with self.store.transaction():
            self.store.exec(
                "INSERT OR REPLACE INTO turn_log(turn,utterance,note,timestamp,system_json) VALUES(?,?,?,?,?)",
                (turn, utterance, note, timestamp, system_json),
            )
            self.store.exec(
                "DELETE FROM turn_log WHERE turn NOT IN "
                "(SELECT turn FROM turn_log ORDER BY turn DESC LIMIT ?)",
                (self.glob.max_turn_log,),
            )

    def load_turn_log(self) -> list[dict]:
        """Load the newest ``max_turn_log`` turn log entries, ordered by turn."""
        rows = self.store.query(
            "SELECT turn, utterance, note, timestamp, system_json FROM turn_log "
            "ORDER BY turn DESC LIMIT ?",
            (self.glob.max_turn_log,),
        )
        return [dict(r) for r in reversed(rows)]

    def reset(self) -> None:
        for table in ("vec", "memory", "conflict", "dream_log", "turn_log", "spec"):
            self.store.exec(f"DELETE FROM {table}")
        self._write_times = []
        self._doc_vec_cache.clear()
        self._install_spec()


