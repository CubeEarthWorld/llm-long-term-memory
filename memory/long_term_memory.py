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

import hashlib
import json
import math
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone as dt_timezone

import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[misc,assignment]

from config import GlobalConfig, LongTermMemoryConfig
from core.base import MemorySystem, RetrieveResult
from core.embedding import EmbeddingProvider, l2_normalize, truncate_normalize
from core.llm_client import LLMClient
from core.metrics import RecalledItem
from core.storage import (
    Store,
    dequantize_int8,
    pack_vec,
    quantize_int8,
    unpack_vec,
)

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

# Max entries in the per-id 768d re-embed cache used by _identity_scan (§5.1).
_DOC_VEC_CACHE_MAX = 4096


class LongTermMemory(MemorySystem):
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
            utc_dt = datetime(1970, 1, 1, tzinfo=dt_timezone.utc) + timedelta(seconds=float(now))
            if utc_dt.year > 9999:
                raise OverflowError("year > 9999")
            dt = utc_dt.astimezone(ZoneInfo(name))
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
    # activation A (§4.1)
    # ================================================================== #
    def _tau(self, tier: int) -> float:
        return (self.memory.tau1, self.memory.tau2, self.memory.tau3)[int(tier) - 1]

    def _dim(self, tier: int) -> int:
        return (self.memory.dim1, self.memory.dim2, self.memory.dim3)[int(tier) - 1]

    def activation(self, row, now: float) -> float:
        """A(now) = mass · 2^(−max(0, now−last_access)/τ_tier). Underflow → 0 (§4.1, §8.1)."""
        mass = float(row["mass"])
        dt = max(0.0, now - float(row["last_access"]))
        exp = dt / self._tau(int(row["tier"]))
        if exp > self.memory.decay_exp_cap:
            return 0.0
        return min(mass * (2.0 ** (-exp)), self.memory.m_max)

    def _apply_recall(self, row, now: float) -> None:
        """Recall update with the §4.1 ordering: fold decay first, then refractory bonus."""
        mass = self.activation(row, now)              # 1) fold decay into mass
        lb = float(row["last_bonus_at"])
        if now - lb >= self.memory.refractory_seconds:  # 2) refractory gate
            mass = min(mass + 1.0, self.memory.m_max)
            lb = now
        self.store.exec(
            "UPDATE memory SET mass=?, last_access=?, last_bonus_at=? WHERE id=?",
            (mass, int(now), int(lb), row["id"]),
        )

    def _reinforce_write(self, row, now: float) -> None:
        """Exact-text rehearsal (§5.1.2): definite +1, bypassing the refractory gate."""
        mass = min(self.activation(row, now) + 1.0, self.memory.m_max)
        self.store.exec(
            "UPDATE memory SET mass=?, last_access=?, last_bonus_at=? WHERE id=?",
            (mass, int(now), int(now), row["id"]),
        )

    # ================================================================== #
    # vectors (§3.3, §8 MRL+int8)
    # ================================================================== #
    def _embed_doc(self, text: str) -> np.ndarray:
        return np.asarray(self.provider.encode_document(text, dim=self.glob.dim_full)[0], dtype=np.float32)

    def _store_vec(self, mid: str, v_full: np.ndarray, tier: int) -> None:
        dim = self._dim(tier)
        vt = truncate_normalize(np.asarray(v_full, dtype=np.float32), dim)
        if tier == 1:
            blob, dtype, scale = pack_vec(vt), "f32", None
        else:
            blob, scale = quantize_int8(vt)
            dtype = "int8"
        self.store.exec(
            "INSERT OR REPLACE INTO vec(id,memory_id,model_id,dim,dtype,scale,v) VALUES(?,?,?,?,?,?,?)",
            (ulid(), mid, self.model_id, dim, dtype, scale, blob),
        )

    @staticmethod
    def _vec_array(vrow) -> np.ndarray:
        """Decode a stored vec row into a unit-norm float32 vector."""
        if vrow["dtype"] == "f32":
            v = unpack_vec(vrow["v"])
        else:
            v = dequantize_int8(vrow["v"], vrow["scale"])
        return l2_normalize(np.asarray(v, dtype=np.float32))

    def _tier_candidates(self, tier: int) -> tuple[list, np.ndarray | None]:
        """Return (rows, unit-norm vector matrix) for live memories in ``tier`` (joins vec).

        The matrix form lets callers score a whole tier with one ``M @ q``
        instead of a per-row Python loop of ``np.dot`` calls.
        """
        rows = self.store.query(
            "SELECT m.*, v.dim AS _dim, v.dtype AS _dtype, v.scale AS _scale, v.v AS _v "
            "FROM memory m JOIN vec v ON v.memory_id=m.id AND v.model_id=? "
            "WHERE m.tier=? AND m.superseded_by IS NULL",
            (self.model_id, tier),
        )
        if not rows:
            return [], None
        vecs = []
        for r in rows:
            if r["_dtype"] == "f32":
                v = unpack_vec(r["_v"])
            else:
                v = dequantize_int8(r["_v"], r["_scale"])
            vecs.append(l2_normalize(np.asarray(v, dtype=np.float32)))
        return list(rows), np.stack(vecs)

    @staticmethod
    def _coarse128(vec: np.ndarray) -> np.ndarray:
        return truncate_normalize(np.asarray(vec, dtype=np.float32), 128)

    # ================================================================== #
    # WRITE — save_memory tool (§5.1)
    # ================================================================== #
    def save_memory(self, text: str, now: float | None = None) -> dict:
        """LLM tool: store one self-contained proposition. Append-only (I5)."""
        now = self.now_unix() if now is None else now
        text = (text or "").strip()
        if not text:
            return {"action": "rejected", "error": "空のテキスト", "text": ""}
        try:
            nbytes = len(text.encode("utf-8"))
        except UnicodeEncodeError:
            # Lone surrogates / un-encodable input is corrupt; reject gracefully (I14)
            # instead of crashing the turn. Echo a sanitized snippet for the log.
            safe = text.encode("utf-8", "replace").decode("utf-8")
            return {"action": "rejected", "error": "不正な文字コード(UTF-8エンコード不可)", "text": safe[:60]}
        if nbytes > self.memory.text_hard_max:
            return {"action": "rejected", "error": "テキストが長すぎます(>1024B)", "text": text[:60]}
        if len(text) > self.memory.text_max:
            text = _shorten(text, self.memory.text_max)
        if not self._write_rate_ok(now):
            return {"action": "rate_limited", "error": "書込みレート上限", "text": text}

        # 1) exact-text rehearsal (SHA-256 equivalence via stored text) — no new row.
        dup = self.store.one(
            "SELECT * FROM memory WHERE text=? AND superseded_by IS NULL", (text,))
        if dup is not None:
            self._reinforce_write(dup, now)
            return self._event("reinforced", self._get(dup["id"]), now)

        # 2) embed (document prompt) and scan ALL tiers (§5.1.3).
        v = self._embed_doc(text)
        best = self._identity_scan(v)
        self._write_times.append(now)

        if best is not None and best[1] >= self.memory.theta_same:
            # ≥0.97: same proposition update → tombstone old, insert new (再固定化).
            carry = min(self.activation(best[2], now) + 1.0, self.memory.m_max)
            nid, row = self._insert(text, v, carry, now)
            self._supersede(best[0], nid)
            return self._event("updated", row, now, cos=round(best[1], 3), superseded=best[0])

        if best is not None and best[1] >= self.memory.theta_conflict:
            # 0.85–0.97: conflict / related → keep both, enqueue for dream adjudication.
            nid, row = self._insert(text, v, 1.0, now)
            self._push_conflict(best[0], nid, now)
            return self._event("conflict", row, now, cos=round(best[1], 3), conflict_with=best[0])

        # <0.85: new memory.
        nid, row = self._insert(text, v, 1.0, now)
        return self._event("inserted", row, now, cos=round(best[1], 3) if best else 0.0)

    def _identity_scan(self, v_full: np.ndarray):
        """Best (id, precise_cos, row) across all tiers; in-band L2/L3 refined at 768d (§5.1)."""
        cands: dict = {}
        for tier in (1, 2, 3):
            rows, M = self._tier_candidates(tier)
            if not rows:
                continue
            qt = truncate_normalize(v_full, self._dim(tier))
            sims = M @ qt
            for row, c in zip(rows, sims):
                c = float(c)
                prev = cands.get(row["id"])
                if prev is None or c > prev[0]:
                    cands[row["id"]] = (c, row, tier)
        best = None
        band = self.memory.theta_conflict - self.memory.precise_margin
        for mid, (c, row, tier) in cands.items():
            cos = c
            if tier != 1 and c >= band:
                vs = self._doc_vec_cached(row)             # text is canonical → re-embed at 768d
                cos = float(np.dot(v_full, vs))
            if best is None or cos > best[1]:
                best = (mid, cos, row)
        return best

    def _doc_vec_cached(self, row) -> np.ndarray:
        """768d document vector for precision refinement, cached per memory id.

        Safe because text is immutable for a given id (edits create a new row);
        without this, every write re-embeds the same L2/L3 neighbours. Cleared
        wholesale at the size cap so the cache itself stays bounded.
        """
        mid = row["id"]
        v = self._doc_vec_cache.get(mid)
        if v is None:
            if len(self._doc_vec_cache) >= _DOC_VEC_CACHE_MAX:
                self._doc_vec_cache.clear()
            v = self._embed_doc(row["text"])
            self._doc_vec_cache[mid] = v
        return v

    def _insert(self, text: str, v_full: np.ndarray, mass: float, now: float,
                gen: int = 0, tier: int = 1):
        mid = ulid()
        self.store.exec(
            "INSERT INTO memory(id,text,tier,gen,created_at,tz,last_access,last_bonus_at,mass,superseded_by) "
            "VALUES(?,?,?,?,?,?,?,?,?,NULL)",
            (mid, text, tier, int(gen), int(now), self._tz_for(now),
             int(now), int(now), float(mass)),
        )
        self._store_vec(mid, v_full, tier)
        return mid, self._get(mid)

    def _supersede(self, old_id: str, new_id: str) -> None:
        self.store.exec("UPDATE memory SET superseded_by=? WHERE id=?", (new_id, old_id))

    def _push_conflict(self, a: str, b: str, now: float) -> None:
        self.store.exec("INSERT INTO conflict(a,b,at) VALUES(?,?,?)", (a, b, int(now)))
        self._trim_ring("conflict", self.memory.conflict_cap)

    def _write_rate_ok(self, now: float) -> bool:
        day = now - 86400
        self._write_times = [t for t in self._write_times if t >= day]
        return len(self._write_times) < self.memory.write_rate_per_day

    # ================================================================== #
    # DELETE — delete_memory tool (§5.3)
    # ================================================================== #
    def delete_memory(self, mid: str, hard: bool = False) -> dict:
        """Tombstone (conversation) or immediate physical delete (soft side). id-only (I10)."""
        row = self.store.one("SELECT * FROM memory WHERE id=?", (mid,))
        if row is None:
            return {"action": "not_found", "id": mid}
        if hard:
            with self.store.transaction():
                self._physical_delete(mid)
            return {"action": "deleted", "id": mid, "text": row["text"]}
        # Conversation-time delete = tombstone (self-referential), purged in next dream.
        self.store.exec("UPDATE memory SET superseded_by=? WHERE id=?", (mid, mid))
        return {"action": "tombstoned", "id": mid, "text": row["text"]}

    # ================================================================== #
    # READ — retrieve (§5.2, §4.2)
    # ================================================================== #
    def retrieve(self, query: str, turn: int) -> RetrieveResult:
        now = self.now_unix()
        self._sanitize_timestamps(now)
        chunks = _split_paragraphs(query)[:16]
        if not chunks:
            return RetrieveResult(pack_text="", recalled=[])
        q_mat = self.provider.encode_query(chunks, dim=self.glob.dim_full)

        cands: dict = {}
        for tier in (1, 2, 3):
            rows, M = self._tier_candidates(tier)
            if not rows:
                continue
            qts = np.stack([truncate_normalize(np.asarray(q, dtype=np.float32), self._dim(tier))
                            for q in q_mat])
            sims = (M @ qts.T).max(axis=1)        # best cosine over query chunks, per row
            for row, vec, c in zip(rows, M, sims):
                c = float(c)
                prev = cands.get(row["id"])
                if prev is None or c > prev["cos"]:
                    cands[row["id"]] = {"id": row["id"], "cos": c, "row": row,
                                        "mmr": self._coarse128(vec)}

        for c in cands.values():
            A = self.activation(c["row"], now)
            A_abs = math.log1p(A) / math.log1p(self.memory.m_max)
            c["A"] = A
            c["score"] = max(0.0, c["cos"]) * (self.memory.alpha + (1.0 - self.memory.alpha) * A_abs)

        ranked = self._mmr(self._filter_by_score(list(cands.values())))
        result, packed_ids = self._pack(ranked, now)
        for mid in packed_ids:
            r = self._get(mid)
            if r is not None:
                self._apply_recall(r, now)
        return result

    def _filter_by_score(self, items: list[dict]) -> list[dict]:
        """Drop low-score memories, relaxing the threshold only if nothing remains.

        Thresholds are applied strictest-first (descending) so the relaxation is
        monotone regardless of the order they are declared in config; otherwise a
        looser threshold listed first would mask every stricter one. If even the
        loosest threshold matches nothing, inject nothing: returning unrelated
        memories would mark them recalled and spuriously reinforce their mass.
        """
        for threshold in sorted(self.memory.score_thresholds, reverse=True):
            filtered = [it for it in items if it["score"] >= threshold]
            if filtered:
                return filtered
        return []

    def _mmr(self, items: list[dict]) -> list[dict]:
        """MMR selection (§4.2): next = argmax[score − λ·max cos(m, selected)]."""
        if not items:
            return []
        lam = self.memory.mmr_lambda
        pool = sorted(items, key=lambda x: x["score"], reverse=True)[:50]
        selected: list[dict] = []
        while pool and len(selected) < self.memory.inject_n:
            best, best_val = None, -1e9
            for it in pool:
                if not selected:
                    val = it["score"]
                else:
                    max_sim = max(float(np.dot(it["mmr"], s["mmr"])) for s in selected)
                    val = lam_mmr(it["score"], lam, max_sim)
                if val > best_val:
                    best, best_val = it, val
            selected.append(best)
            pool.remove(best)
        return selected

    def _pack(self, ranked: list[dict], now: float):
        """Format injected memories ≤ budget_chars with a unix+TZ header + id.

        Header is the raw 64-bit Unix seconds and the stored 'IANA;+offset'
        timezone (e.g. ``[1749641400 Asia/Tokyo;+09:00]``) rather than a
        human-formatted local datetime, per project preference.
        """
        used, lines, recalled, ids = 0, [], [], []
        for it in ranked:
            row = it["row"]
            line = f"[{int(row['created_at'])} {row['tz']}] {row['text']}　《id:{row['id']}》\n"
            if used + len(line) > self.glob.budget_chars:
                continue
            lines.append(line)
            used += len(line)
            ids.append(row["id"])
            recalled.append(RecalledItem(
                mem_id=row["id"], text=row["text"], score=round(it["score"], 3),
                extra={
                    "cos": round(it["cos"], 3), "A": round(it["A"], 2),
                    "mass": round(float(row["mass"]), 2), "tier": int(row["tier"]),
                    "gen": int(row["gen"]), "tz": row["tz"],
                },
            ))
            if used >= self.glob.budget_chars:
                break
        return RetrieveResult(pack_text="".join(lines), recalled=recalled), ids

    # ================================================================== #
    # machine movement (§5.4) — no LLM
    # ================================================================== #
    def _promote(self, now: float) -> None:
        """A ≥ θ_up L2/L3 → re-embed text at 768d → L1 (re-fixation; text is canonical)."""
        rows = self.store.query(
            "SELECT * FROM memory WHERE tier IN (2,3) AND superseded_by IS NULL")
        for row in rows:
            if self.activation(row, now) >= self.memory.theta_up:
                v = self._doc_vec_cached(row)
                self.store.exec("UPDATE memory SET tier=1 WHERE id=?", (row["id"],))
                self._store_vec(row["id"], v, 1)

    def _tier_count(self, tier: int) -> int:
        return self.store.count("memory", "tier=?", (tier,))   # includes tombstones (I4)

    def _enforce_capacity(self, now: float) -> None:
        """Keep each tier within capacity: sweep tombstones, then demote / evict (§5.4, I4)."""
        for tier, cap, to in ((1, self.memory.cap1, 2), (2, self.memory.cap2, 3)):
            guard = 0
            while self._tier_count(tier) > cap and guard < self.memory.hard_memory_rows:
                guard += 1
                tomb = self.store.one(
                    "SELECT id FROM memory WHERE tier=? AND superseded_by IS NOT NULL LIMIT 1", (tier,))
                if tomb is not None:
                    self._physical_delete(tomb["id"])
                    continue
                vid = self._demote_victim(tier, now)
                if vid is None:
                    break
                self._demote(vid, to)
        guard = 0
        while self._tier_count(3) > self.memory.cap3 and guard < self.memory.hard_memory_rows:
            guard += 1
            tomb = self.store.one(
                "SELECT id FROM memory WHERE tier=3 AND superseded_by IS NOT NULL LIMIT 1")
            if tomb is not None:
                self._physical_delete(tomb["id"])
                continue
            vid = self._evict_victim(now)
            if vid is None:
                break
            self._physical_delete(vid)   # this is death in this system (§5.4)

    def _demote_victim(self, tier: int, now: float) -> str | None:
        rows = self.store.query(
            "SELECT * FROM memory WHERE tier=? AND superseded_by IS NULL", (tier,))
        if not rows:
            return None
        under = [r for r in rows if self.activation(r, now) < self.memory.theta_down]
        pool = under or rows                                   # A<θ_down first, else lowest A
        return min(pool, key=lambda r: self.activation(r, now))["id"]

    def _evict_victim(self, now: float) -> str | None:
        rows = self.store.query(
            "SELECT * FROM memory WHERE tier=3 AND superseded_by IS NULL")
        if not rows:
            return None
        return min(rows, key=lambda r: self.activation(r, now))["id"]

    def _demote(self, mid: str, to_tier: int) -> None:
        vr = self.store.one(
            "SELECT * FROM vec WHERE memory_id=? AND model_id=?", (mid, self.model_id))
        if vr is None:
            return
        v = self._vec_array(vr)
        vt = truncate_normalize(v, self._dim(to_tier))         # MRL: 768→256→128 prefixes
        blob, scale = quantize_int8(vt)
        self.store.exec(
            "UPDATE vec SET dim=?, dtype='int8', scale=?, v=? WHERE memory_id=? AND model_id=?",
            (self._dim(to_tier), scale, blob, mid, self.model_id),
        )
        self.store.exec("UPDATE memory SET tier=? WHERE id=?", (to_tier, mid))

    def _physical_delete(self, mid: str) -> None:
        self.store.exec("DELETE FROM memory WHERE id=?", (mid,))   # vec cascades (ON DELETE CASCADE)

    # ================================================================== #
    # DREAM (§6) — offline, LLM + embedding
    # ================================================================== #
    def dream(self, now_unix: float | None = None, max_clusters: int | None = None,
              budget: int | None = None, force: bool = False) -> list[dict]:
        now = self.now_unix() if now_unix is None else now_unix
        self._sanitize_timestamps(now)
        b = budget if budget is not None else (max_clusters if max_clusters is not None else self.memory.dream_budget)
        b = max(0, min(int(b), self.memory.dream_budget_hard))

        self.store.snapshot(self._snap_dir, self.memory.snapshot_gens)   # 8-gen ring
        self._housekeeping(now)

        results: list[dict] = []
        for fp, members, prio in self._dream_candidates(now, force=force)[:b]:
            results.append(self._adjudicate(fp, members, prio, now))

        self._promote(now)
        self._enforce_capacity(now)
        self.store.wal_checkpoint()
        self.store.incremental_vacuum()
        return results

    def _housekeeping(self, now: float) -> None:
        """Cheap, LLM-free chores (§6): tombstone sweep, ring trims, vec orphan GC, WAL."""
        self._sweep_tombstones(now)
        self._trim_ring("conflict", self.memory.conflict_cap)
        self._trim_dream_log()
        self.store.exec("DELETE FROM vec WHERE model_id<>?", (self.model_id,))
        self.store.exec("DELETE FROM vec WHERE memory_id NOT IN (SELECT id FROM memory)")
        self.store.wal_checkpoint()

    def _sweep_tombstones(self, now: float) -> None:
        cutoff = int(now - self.memory.tombstone_sweep_age)
        self.store.exec(
            "DELETE FROM memory WHERE superseded_by IS NOT NULL AND last_access < ?", (cutoff,))
        for tier in (1, 2, 3):
            total = self._tier_count(tier)
            if total <= 0:
                continue
            tomb = self.store.count("memory", "tier=? AND superseded_by IS NOT NULL", (tier,))
            if tomb > self.memory.tombstone_sweep_pct * total:
                self.store.exec(
                    "DELETE FROM memory WHERE tier=? AND superseded_by IS NOT NULL", (tier,))

    def _trim_ring(self, table: str, cap: int) -> None:
        n = self.store.count(table)
        if n > cap:
            self.store.exec(
                f"DELETE FROM {table} WHERE rowid IN "
                f"(SELECT rowid FROM {table} ORDER BY rowid ASC LIMIT ?)", (n - cap,))

    def _trim_dream_log(self) -> None:
        n = self.store.count("dream_log")
        if n > self.memory.dream_log_cap:
            self.store.exec(
                "DELETE FROM dream_log WHERE fp IN "
                "(SELECT fp FROM dream_log ORDER BY at ASC LIMIT ?)",
                (n - self.memory.dream_log_cap,))

    def _live_rows_coarse(self, tiers) -> list[tuple]:
        out = []
        for tier in tiers:
            rows, M = self._tier_candidates(tier)
            if not rows:
                continue
            out.extend((row, self._coarse128(vec)) for row, vec in zip(rows, M))
        return out

    def _dream_candidates(self, now: float, force: bool = False) -> list[tuple]:
        """Cluster live L1/L2 by coarse vector; rank eligible clusters by priority (§6)."""
        rows = self._live_rows_coarse((1, 2))
        if len(rows) < self.memory.cluster_min:
            return []
        X = np.stack([c for _, c in rows]).astype(np.float32)
        k = max(1, round(len(rows) / 4))
        labels = _kmeans(X, k)
        groups: dict = {}
        for (row, coarse), lab in zip(rows, labels):
            groups.setdefault(int(lab), []).append((row, coarse))

        conflict_pairs = {tuple(sorted((r["a"], r["b"])))
                          for r in self.store.query("SELECT a,b FROM conflict")}
        l1_overflow = max(0.0, self._tier_count(1) - self.memory.cap1) / max(1, self.memory.cap1)

        out = []
        for members in groups.values():
            if len(members) < self.memory.cluster_min:
                continue
            rows_m = [m[0] for m in members]
            ids = sorted(m["id"] for m in rows_m)
            fp = _fp(ids)
            if self._dream_skip(fp):
                continue
            coarses = [m[1] for m in members]
            coh = _cohesion(coarses)
            max_gen = max(int(m["gen"]) for m in rows_m)
            idset = set(ids)
            n_conf = sum(1 for a, b in conflict_pairs if a in idset and b in idset)
            prio = (len(rows_m) * coh * (2.0 ** (-max_gen))
                    * (1.0 + l1_overflow) * (1.0 + n_conf))
            out.append((fp, rows_m, prio))
        out.sort(key=lambda t: t[2], reverse=True)
        return out

    def _dream_skip(self, fp: str) -> bool:
        v = self.store.scalar("SELECT verdict FROM dream_log WHERE fp=?", (fp,))
        return v == "変更なし"

    def _record_dream(self, fp: str, verdict: str, now: float) -> None:
        self.store.exec(
            "INSERT OR REPLACE INTO dream_log(fp,verdict,at) VALUES(?,?,?)", (fp, verdict, int(now)))

    def _adjudicate(self, fp: str, members: list, prio: float, now: float) -> dict:
        member_rows = [self._get(m["id"]) for m in members]
        member_rows = [r for r in member_rows if r is not None]
        before = [{"id": r["id"], "text": r["text"], "tier": int(r["tier"]),
                   "gen": int(r["gen"])} for r in member_rows]
        if len(member_rows) < self.memory.cluster_min:
            return {"cluster_fp": fp[:12], "action": "none", "priority": round(prio, 3),
                    "before": before, "after": [], "deleted_ids": []}

        payload = [{
            "id": r["id"], "text": r["text"], "gen": int(r["gen"]),
            "A": round(self.activation(r, now), 2),
            "local_time": _fmt_local(r["created_at"], r["tz"]), "timezone": r["tz"],
        } for r in member_rows[: self.memory.dream_max_members]]

        decision = self.llm.dream_cluster(payload) or {}
        action = decision.get("action", "none")
        new_mems = [nm for nm in (decision.get("memories") or []) if str(nm.get("text", "")).strip()]

        if action == "none" or not new_mems:
            self._record_dream(fp, "変更なし", now)
            return {"cluster_fp": fp[:12], "action": "none", "priority": round(prio, 3),
                    "before": before, "after": [], "deleted_ids": []}

        sum_A = sum(self.activation(r, now) for r in member_rows)
        max_gen = max(int(r["gen"]) for r in member_rows)
        is_merge = action != "split"
        gen = min(max_gen + (1 if is_merge else 0), self.memory.gen_max)
        deleted = [r["id"] for r in member_rows]
        after = []
        with self.store.transaction():
            for r in member_rows:
                self._physical_delete(r["id"])
            picks = new_mems[: self.memory.dream_max_members]
            per_mass = (min(sum_A, self.memory.m_max) if is_merge
                        else min(sum_A / max(1, len(picks)), self.memory.m_max))
            for nm in picks:
                text = str(nm.get("text", "")).strip()
                if not text:
                    continue
                if len(text) > self.memory.text_max:
                    text = _shorten(text, self.memory.text_max)
                v = self._embed_doc(text)
                _, row = self._insert(text, v, per_mass, now, gen=gen, tier=1)
                after.append({"id": row["id"], "text": row["text"], "gen": int(row["gen"])})
        self._record_dream(fp, action, now)
        return {"cluster_fp": fp[:12], "action": "merge" if is_merge else "split",
                "priority": round(prio, 3), "before": before, "after": after, "deleted_ids": deleted}

    # ================================================================== #
    # maintenance (every turn, no LLM)
    # ================================================================== #
    def maintain(self, turn: int) -> None:
        now = self.now_unix()
        self._sanitize_timestamps(now)
        self._sweep_tombstones(now)
        self._enforce_capacity(now)       # demotion/eviction only (no re-embedding)
        self._maint_count += 1
        if self._maint_count % 20 == 0:
            self.store.wal_checkpoint()

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


# ====================================================================== #
# module helpers
# ====================================================================== #
def lam_mmr(score: float, lam: float, max_sim: float) -> float:
    return score - lam * max_sim


def _fp(ids: list[str]) -> str:
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()


def _cohesion(coarses: list[np.ndarray]) -> float:
    """Mean pairwise cosine of a cluster's coarse vectors (∈ (0,1])."""
    n = len(coarses)
    if n < 2:
        return 1.0
    M = np.stack(coarses)
    sims = M @ M.T
    total = float(sims.sum() - np.trace(sims))
    return max(0.0, total / (n * (n - 1)))


def _kmeans(X: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> np.ndarray:
    """Tiny spherical k-means on unit-norm rows (cosine = dot). Method is free per §6."""
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(iters):
        new = (X @ cent.T).argmax(axis=1)
        if it > 0 and np.array_equal(new, labels):
            labels = new
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                cent[j] = l2_normalize(members.mean(axis=0))
    return labels


def _split_paragraphs(text: str) -> list[str]:
    """Split a turn into chunks so a memory relevant to any part can surface (§5.2)."""
    import re
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(blocks) <= 2 and len(text) > 80:
        sents = [s.strip() for s in re.split(r"(?<=[。！？…\n])", text) if s.strip()]
        if len(sents) > len(blocks):
            blocks = sents
    return blocks or []


def _fmt_local(unix, tzfield) -> str:
    """'2026-06-11 21:30 +09:00' from a UNIX time and a stored 'IANA;+offset' field (§5.2)."""
    name = str(tzfield).split(";")[0]
    parts = str(tzfield).split(";")
    off = parts[1] if len(parts) > 1 else "+00:00"
    try:
        utc_dt = datetime(1970, 1, 1, tzinfo=dt_timezone.utc) + timedelta(seconds=float(unix))
        if utc_dt.year > 9999:
            raise OverflowError("year > 9999")
        dt = utc_dt.astimezone(ZoneInfo(name))
        off = dt.strftime("%z") or "+0000"
        return f"{dt:%Y-%m-%d %H:%M} {off[:3]}:{off[3:]}"
    except Exception:
        try:
            utc_dt = datetime(1970, 1, 1, tzinfo=dt_timezone.utc) + timedelta(seconds=float(unix))
            if utc_dt.year > 9999:
                raise OverflowError("year > 9999")
            dt = utc_dt.astimezone(dt_timezone.utc)
            return f"{dt:%Y-%m-%d %H:%M} {off}"
        except Exception:
            # Beyond datetime.MAXYEAR (9999) — format manually
            ts = float(unix)
            secs = int(ts)
            mins, sec = divmod(secs, 60)
            hrs, min_ = divmod(mins, 60)
            days, hr = divmod(hrs, 24)
            # Gregorian calendar: days since 1970-01-01
            year, month, day = _ymd_from_ordinal(719163 + days)  # 719163 = date(1970,1,1).toordinal()
            return f"{year:04d}-{month:02d}-{day:02d} {hr:02d}:{min_:02d} {off}"


def _ymd_from_ordinal(n: int) -> tuple[int, int, int]:
    """Convert proleptic Gregorian ordinal (1 = 0001-01-01) to (year, month, day).
    Based on the algorithm from `datetime._ymd2ord` reversed. Works for any year."""
    # Adjust for 0001-01-01 being ordinal 1
    n -= 1
    # 400-year cycles: 146097 days
    n400, n = divmod(n, 146097)
    y400 = n400 * 400
    # 100-year cycles within 400: 36524 days (except every 400th)
    n100, n = divmod(n, 36524)
    if n100 > 3:
        n100 = 3
        n += 36524
    y100 = n100 * 100
    # 4-year cycles: 1461 days (except every 100th)
    n4, n = divmod(n, 1461)
    y4 = n4 * 4
    # 1-year cycles: 365 days (except every 4th)
    n1, n = divmod(n, 365)
    if n1 > 3:
        n1 = 3
        n += 365
    y1 = n1
    year = y400 + y100 + y4 + y1 + 1
    # Month/day within the year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_month = [31, 28 + is_leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = 1
    for dim in days_in_month:
        if n < dim:
            break
        n -= dim
        month += 1
    day = n + 1
    return year, month, day


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("。", "．", ".", "、", " "):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.5:
            return cut[: idx + 1]
    return cut


# Crockford base32 alphabet (RFC 9562 / ULID): excludes I, L, O, U.
_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid_fallback() -> str:
    """Self-contained ULID: 48-bit ms timestamp + 80 random bits, 26 Crockford chars.

    Keeps the engine running even if the optional ``python-ulid`` package is not
    installed (it lives only here, in a hot path). Sort order still equals time
    order because the timestamp occupies the high bits (§3.2).
    """
    ts = int(time.time() * 1000) & ((1 << 48) - 1)
    val = (ts << 80) | secrets.randbits(80)          # 128-bit ULID value
    chars = []
    for _ in range(26):                               # 26 × 5 bits = 130 bits (top 2 are 0)
        chars.append(_CROCKFORD32[val & 0x1F])
        val >>= 5
    return "".join(reversed(chars))


def ulid() -> str:
    """ULID as string: first 48 bits = ms Unix time → sort order = time order (§3.2).

    Uses ``python-ulid`` when available, otherwise the self-contained fallback so a
    missing/optional dependency never hard-crashes the write/dream path.
    """
    try:
        from ulid import ULID
        return str(ULID())
    except Exception:
        return _ulid_fallback()
