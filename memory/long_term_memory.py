"""LLM Long-Term Memory implementation.

Human-memory-inspired model. Each memory separates three axes that the previous
single ``freq`` conflated:

* ``stability`` (S, seconds) — durability. Grows on every successful recall, more
  when recalled near forgetting (spacing / testing effect, Jost's law, LTP).
* ``freq`` — cumulative access count, a small ranking signal only.
* ``w`` — importance / salience; ``confidence`` — reliability (from provenance).

Retrievability follows a **power law** ``r = (1 + dt/S)^(-beta)`` (closer to the
empirical forgetting curve than a single exponential). Recall is *gated* by ``r``
so decay causes real, cue-dependent forgetting — not just lower ranking.

Forgetting is two-stage and bounded: a memory that decays below ``r_archive_floor``
(after a grace period) is moved to a light ``archive`` table (256-d coarse vector
only, no 768-d) — inaccessible to normal recall but recoverable. When the same
information reappears it is **restored with a stability head-start** (relearning
savings). The archive itself is capped, so total storage stays bounded.

Clusters are stable identities consolidated by ``dream()`` (sleep-like). Dreamed
memories become more durable (episodic -> semantic gist) and keep lineage
(``source_ids`` / ``created_by_dream_id``) for auditability.

The 256-d coarse vector (MRL truncation) used for clustering / diversity /
spreading is derived from the full vector on the fly for live memories; for the
archive it is the only vector stored.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import secrets
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from config import GlobalConfig, LongTermMemoryConfig
from core.base import MemorySystem, RetrieveResult, WriteResult
from core.embedding import EmbeddingProvider, truncate_normalize
from core.llm_client import LLMClient
from core.metrics import RecalledItem
from core.storage import Store, cosine, pack_vec, unpack_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  text TEXT, v_full BLOB,
  w REAL, provenance TEXT, confidence REAL, freq REAL, stability REAL,
  accessed_at_unix REAL, updated_at_unix REAL, created_at_unix REAL,
  timezone TEXT, cluster_id TEXT,
  source_ids TEXT, summary_of TEXT, dream_action TEXT, created_by_dream_id TEXT
);
CREATE TABLE IF NOT EXISTS archive (
  id TEXT PRIMARY KEY,
  text TEXT, v_coarse BLOB,
  w REAL, provenance TEXT, confidence REAL,
  last_r REAL, archived_at_unix REAL, text_hash TEXT, timezone TEXT
);
CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY,
  last_dreaming_unix REAL
);
CREATE INDEX IF NOT EXISTS idx_archive_hash ON archive(text_hash);
CREATE INDEX IF NOT EXISTS idx_archive_evict ON archive(last_r, archived_at_unix);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at_unix);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at_unix);
"""


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
        self._clock = None  # optional callable -> float (virtual clock for experiments)
        self.store.execscript(SCHEMA)
        # Migration: add created_at_unix to existing DBs
        existing = {c["name"] for c in self.store.query("PRAGMA table_info(memories)")}
        if "created_at_unix" not in existing:
            self.store.exec("ALTER TABLE memories ADD COLUMN created_at_unix REAL")

    # -- time / clock -------------------------------------------------- #
    def now_unix(self) -> float:
        return float(self._clock()) if self._clock else time.time()

    def set_clock(self, fn) -> None:
        """Inject a clock (callable -> unix seconds) so experiments can accelerate time."""
        self._clock = fn

    # -- stability / retrievability ------------------------------------ #
    def _confidence(self, provenance: str) -> float:
        return self.memory.confidence_user if provenance == "user" else self.memory.confidence_inferred

    def initial_stability(self, w: float, confidence: float) -> float:
        """Initial S (seconds): importance extends it; low confidence starts more labile."""
        base = self.memory.stab_base_seconds * (1.0 + self.memory.kappa * w)
        return base * (self.memory.labile_frac + (1.0 - self.memory.labile_frac) * confidence)

    def _stability(self, row) -> float:
        return float(row["stability"])

    def retrievability(self, row, now_unix: float) -> float:
        """Power-law retrievability in (0, 1]: r = (1 + dt/S)^(-beta)."""
        S = max(1e-6, self._stability(row))
        elapsed = max(0.0, now_unix - float(row["accessed_at_unix"]))
        return (1.0 + elapsed / S) ** (-self.memory.forget_beta)

    def _coarse(self, v_full) -> np.ndarray:
        return truncate_normalize(np.asarray(v_full, dtype=np.float32), self.glob.dim_coarse)

    # -- write --------------------------------------------------------- #
    def write(self, turn: int, user_text: str, assistant_text: str) -> WriteResult:
        t0 = time.perf_counter()
        cands = self.llm.extract_memory(user_text, assistant_text)
        extract_ms = (time.perf_counter() - t0) * 1000.0
        if not cands:
            return WriteResult(written_ids=[], note="保存なし（重要でない発話）", extract_ms=extract_ms)

        now_unix = self.now_unix()
        tz = self.glob.default_timezone
        written, written_rows = [], []
        for cand in cands:
            mid, events = self._write_one(
                cand.get("text", user_text),
                _clamp01(float(cand.get("w", 0.6))),
                cand.get("provenance", "user"),
                tz,
                now_unix,
            )
            written.append(mid)
            written_rows.extend(events)
        self._enforce_capacity(now_unix)
        return WriteResult(written_ids=written, written_rows=written_rows, note=f"{len(written_rows)}件", extract_ms=extract_ms)

    def _write_one(self, text: str, w: float, provenance: str, timezone: str, now_unix: float):
        if len(text) > self.memory.mem_max_chars:
            text = _shorten(text, self.memory.mem_max_chars)
        v = self.provider.encode_document(text, dim=self.glob.dim_full)[0]
        vc = self._coarse(v)
        confidence = self._confidence(provenance)

        # 1) Near-duplicate of a live memory: reinforce + refresh, do not copy.
        nn = self._nearest(v)
        if nn and nn[1] > self.memory.tau_dup:
            self._reinforce(nn[2], now_unix)
            self.store.exec(
                "UPDATE memories SET text=?, w=?, updated_at_unix=?, timezone=? WHERE id=?",
                (text, max(w, float(nn[2]["w"])), now_unix, timezone, nn[0]),
            )
            row = self.store.one("SELECT * FROM memories WHERE id=?", (nn[0],))
            return nn[0], [_memory_event("updated", row, self.retrievability(row, now_unix))]

        # 2) Reappearance of an archived memory: restore with a stability head-start (savings).
        arc = self._savings_lookup(vc, text)
        if arc is not None:
            mid = self._restore_from_archive(arc, text, w, v, vc, timezone, now_unix)
            row = self.store.one("SELECT * FROM memories WHERE id=?", (mid,))
            return mid, [_memory_event("restored", row, self.retrievability(row, now_unix))]

        # 3) Fresh insert.
        S0 = self.initial_stability(w, confidence)
        mid, row = self._insert_memory(text, v, w, provenance, confidence, S0, timezone, now_unix, vc=vc)
        return mid, [_memory_event("inserted", row, self.retrievability(row, now_unix))]

    def _nearest(self, v):
        """Return the single nearest live memory to vector ``v`` as (id, cos, row), or None."""
        nearest = self.store.cosine_search("memories", "v_full", v, k=1)
        return nearest[0] if nearest else None

    def _reinforce(self, row, now_unix: float) -> None:
        """Successful access: grow stability (more when nearly forgotten) and bump access count.

        Both stability and freq are capped to prevent unbounded growth over very long use.
        """
        r = self.retrievability(row, now_unix)
        S = self._stability(row)
        new_S = min(S * (1.0 + self.memory.stab_growth_c * (1.0 - r)), self.memory.max_stability_seconds)
        new_freq = min(float(row["freq"]) + self.memory.reinforce_inc, self.memory.max_freq)
        self.store.exec(
            "UPDATE memories SET stability=?, freq=?, accessed_at_unix=? WHERE id=?",
            (new_S, new_freq, now_unix, row["id"]),
        )

    def _insert_memory(self, text, v, w, provenance, confidence, stability, timezone, now_unix,
                       vc=None, lineage=None):
        """Insert one memory (lineage cols NULL unless dreamed), assign its cluster, return (id, row)."""
        mid = uuid7()
        lin = lineage or {}
        self.store.exec(
            "INSERT INTO memories("
            "id,text,v_full,w,provenance,confidence,freq,stability,"
            "accessed_at_unix,updated_at_unix,created_at_unix,timezone,"
            "source_ids,summary_of,dream_action,created_by_dream_id"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, text, pack_vec(v), w, provenance, confidence,
             self.memory.freq_seed, stability, now_unix, now_unix, now_unix, timezone,
             lin.get("source_ids"), lin.get("summary_of"),
             lin.get("dream_action"), lin.get("created_by_dream_id")),
        )
        self._assign_cluster(mid, vc if vc is not None else self._coarse(v), now_unix)
        return mid, self.store.one("SELECT * FROM memories WHERE id=?", (mid,))

    # -- savings (archive reappearance) -------------------------------- #
    def _savings_lookup(self, vc: np.ndarray, text: str):
        """Check whether ``text`` (or a coarse-vector neighbour) already exists in archive.

        Returns the matching archive row so the caller can restore it with a stability head-start.
        """
        if self.store.count("archive") == 0:
            return None
        exact = self.store.one("SELECT * FROM archive WHERE text_hash=?", (_text_hash(text),))
        if exact is not None:
            return exact
        hits = self.store.cosine_search("archive", "v_coarse", vc, k=1)
        if hits and hits[0][1] >= self.memory.tau_savings:
            return hits[0][2]
        return None

    def _restore_from_archive(self, arc, text, w, v, vc, timezone, now_unix) -> str:
        """Move an archived memory back to active store with a relearning savings bonus."""
        self.store.exec("DELETE FROM archive WHERE id=?", (arc["id"],))
        confidence = float(arc["confidence"])
        w_new = max(w, float(arc["w"]))
        S0 = self.initial_stability(w_new, confidence) * self.memory.savings_gain
        mid, _ = self._insert_memory(text, v, w_new, arc["provenance"], confidence, S0, timezone, now_unix, vc=vc)
        return mid

    # -- clustering (centroid/medoid derived from members) ------------- #
    def _cluster_centroids(self) -> Dict[str, np.ndarray]:
        """Derive the centroid (mean coarse vector, L2-normalised) for every cluster."""
        rows = self.store.query("SELECT cluster_id, v_full FROM memories WHERE cluster_id IS NOT NULL")
        acc: Dict[str, List[np.ndarray]] = {}
        for r in rows:
            acc.setdefault(r["cluster_id"], []).append(self._coarse(unpack_vec(r["v_full"])))
        out: Dict[str, np.ndarray] = {}
        for cid, vecs in acc.items():
            out[cid] = _normalize(np.mean(np.stack(vecs), axis=0))
        return out

    def _cluster_medoid(self, cid: str, members=None) -> Optional[str]:
        """Return the member ID whose coarse vector is closest to the cluster centroid."""
        if members is None:
            members = self.store.query("SELECT id, v_full FROM memories WHERE cluster_id=?", (cid,))
        if not members:
            return None
        coarse = [(m["id"], self._coarse(unpack_vec(m["v_full"]))) for m in members]
        cen = _normalize(np.mean(np.stack([c for _, c in coarse]), axis=0))
        return max(coarse, key=lambda t: float(np.dot(t[1], cen)))[0]

    def _assign_cluster(self, mid: str, vc: np.ndarray, now_unix: float) -> str:
        """Assign a memory to its nearest existing cluster, or create a new cluster if none is close enough."""
        best_cid, best_sim = None, -1.0
        for cid, cen in self._cluster_centroids().items():
            sim = float(np.dot(vc, cen))
            if sim > best_sim:
                best_cid, best_sim = cid, sim
        if best_cid is not None and best_sim >= self.memory.tau_link:
            self.store.exec("UPDATE memories SET cluster_id=? WHERE id=?", (best_cid, mid))
            return best_cid
        cid = uuid7()
        self.store.exec("INSERT INTO clusters(id,last_dreaming_unix) VALUES(?,?)", (cid, now_unix))
        self.store.exec("UPDATE memories SET cluster_id=? WHERE id=?", (cid, mid))
        return cid

    # -- forgetting: two-stage (active -> archive -> bounded delete) --- #
    def _is_protected(self, row, r: float, now_unix: float) -> bool:
        """High-importance + high-confidence memories resist eviction down to r_hard_floor."""
        created = row["created_at_unix"] if "created_at_unix" in row.keys() else None
        age = max(0.0, now_unix - float(created or row["accessed_at_unix"]))
        return (
            float(row["confidence"] or 0.0) >= self.memory.protect_confidence
            and float(row["w"]) >= self.memory.protect_w
            and r >= self.memory.r_hard_floor
            and age < self.memory.protect_max_seconds
        )

    def _enforce_capacity(self, now_unix: float) -> None:
        """Ensure active store stays within total_cap by archiving decayed and then evicting lowest-r memories."""
        self._archive_decayed(now_unix)
        while self.store.count("memories") > self.glob.total_cap:
            victim = self._evict_target(now_unix)
            if victim is None:
                break
            self._archive_memory(victim, now_unix)
        self._enforce_archive_cap()

    def _archive_decayed(self, now_unix: float) -> None:
        """Move memories that have decayed below the floor (after grace) into the archive.

        To avoid mass-forgetting after a very long downtime, only a limited number
        of memories are archived per maintenance pass (lowest-r first).
        """
        # Query rewritten so an index on accessed_at_unix can be used.
        rows = self.store.query(
            "SELECT * FROM memories WHERE accessed_at_unix < ?",
            (now_unix - self.memory.archive_grace_seconds,),
        )
        victims = []
        for row in rows:
            r = self.retrievability(row, now_unix)
            if r < self.memory.r_archive_floor and not self._is_protected(row, r, now_unix):
                victims.append((row["id"], r))
        # Archive lowest-r memories first, capped per turn.
        victims.sort(key=lambda x: x[1])
        for mid, _ in victims[: self.memory.max_decay_per_turn]:
            if self.store.one("SELECT 1 FROM memories WHERE id=?", (mid,)):
                self._archive_memory(mid, now_unix)

    def _evict_target(self, now_unix: float) -> Optional[str]:
        """Lowest-retrievability unprotected memory (with cluster-medoid protection)."""
        # Query rewritten so an index on updated_at_unix can be used.
        rows = self.store.query(
            "SELECT * FROM memories WHERE updated_at_unix < ?",
            (now_unix - self.memory.min_residency_seconds,),
        )
        if not rows:
            rows = self.store.query("SELECT * FROM memories")
        if not rows:
            return None
        rmap = {r["id"]: self.retrievability(r, now_unix) for r in rows}
        pool = [r for r in rows if not self._is_protected(r, rmap[r["id"]], now_unix)] or rows
        victim = min(pool, key=lambda r: rmap[r["id"]])
        victim_id = victim["id"]
        cid = victim["cluster_id"]
        if cid is not None:
            members = self.store.query("SELECT * FROM memories WHERE cluster_id=?", (cid,))
            if len(members) >= 2 and victim_id == self._cluster_medoid(cid, members):
                others = [m for m in members if m["id"] != victim_id]
                victim_id = min(others, key=lambda m: self.retrievability(m, now_unix))["id"]
        return victim_id

    def _archive_memory(self, mid: str, now_unix: float) -> None:
        """Move a single live memory into the archive (coarse vector only) and delete it from active."""
        row = self.store.one("SELECT * FROM memories WHERE id=?", (mid,))
        if row is None:
            return
        vc = self._coarse(unpack_vec(row["v_full"]))
        self.store.exec(
            "INSERT OR REPLACE INTO archive("
            "id,text,v_coarse,w,provenance,confidence,last_r,archived_at_unix,text_hash,timezone"
            ") VALUES(?,?,?,?,?,?,?,?,?,?)",
            (row["id"], row["text"], pack_vec(vc), row["w"], row["provenance"],
             float(row["confidence"] or self.memory.confidence_inferred),
             self.retrievability(row, now_unix), now_unix, _text_hash(row["text"]), row["timezone"]),
        )
        cid = row["cluster_id"]
        self.store.exec("DELETE FROM memories WHERE id=?", (mid,))
        if cid is not None and self.store.count("memories", "cluster_id=?", (cid,)) == 0:
            self.store.exec("DELETE FROM clusters WHERE id=?", (cid,))

    def _enforce_archive_cap(self) -> None:
        """Permanently delete oldest/lowest-r archived memories when archive_cap is exceeded."""
        while self.store.count("archive") > self.glob.archive_cap:
            victim = self.store.one(
                "SELECT id FROM archive ORDER BY last_r ASC, archived_at_unix ASC LIMIT 1"
            )
            if victim is None:
                break
            self.store.exec("DELETE FROM archive WHERE id=?", (victim["id"],))

    # -- retrieve ------------------------------------------------------ #
    def retrieve(self, query: str, turn: int) -> RetrieveResult:
        now_unix = self.now_unix()
        candidates = self._gather_candidates(query, now_unix)
        self._apply_spreading(candidates)
        for c in candidates.values():
            c["score"] = self._retrieval_score(c["row"], c["cos"], c["r"]) + c.get("spread", 0.0)

        ranked = self._mmr(list(candidates.values()))
        result, packed_ids = self._pack_retrieved(ranked, now_unix)
        self._reinforce_ids(packed_ids, now_unix)
        self._apply_interference(candidates, set(packed_ids), now_unix)
        return result

    def _gather_candidates(self, query: str, now_unix: float) -> Dict[str, Dict]:
        """Recall per paragraph and union the results.

        The turn text is split into paragraphs; each is embedded and searched
        independently so a memory relevant to *any* paragraph can surface (a
        single whole-text embedding averages distinct topics together and dilutes
        them). A memory found by several paragraphs keeps its best (max) cosine,
        so the gate and downstream score reflect its strongest cue. From here the
        pipeline is unchanged: spreading -> score -> MMR -> pack by score."""
        paras = self._split_paragraphs(query)
        q_mat = self.provider.encode_query(paras, dim=self.glob.dim_full)
        gated: Dict[str, Dict] = {}
        fallback: Dict[str, Dict] = {}
        for q in q_mat:
            for mid, cos, row in self.store.cosine_search(
                "memories", "v_full", q, k=self.memory.k_retrieve
            ):
                if cos < self.glob.sim_floor:
                    continue
                existing = fallback.get(mid)
                if existing is not None and cos <= existing["cos"]:
                    continue  # already kept this memory with an equal/stronger paragraph cue
                r = self.retrievability(row, now_unix)
                cand = {
                    "id": mid,
                    "cos": cos,
                    "r": r,
                    "row": row,
                    "cluster_id": row["cluster_id"],
                    "v": self._coarse(unpack_vec(row["v_full"])),
                }
                fallback[mid] = cand
                if self._recall_gate(cos, r):
                    gated[mid] = cand
        candidates = gated if gated else fallback
        self._recall_from_archive(q_mat, candidates, now_unix)
        return candidates

    def _recall_from_archive(self, q_mat, candidates: Dict[str, Dict], now_unix: float) -> None:
        """思い出し (recall): search the archive too and restore high-relevance matches mid-retrieve.

        A memory that decayed below ``r_archive_floor`` lives only in the light ``archive``
        (256-d coarse vector) and is invisible to normal recall. Here a cue whose coarse cosine
        to an archived memory clears ``tau_recall`` brings it back: the text is re-embedded to a
        full vector, restored with a relearning head-start (``_restore_from_archive``), and added
        to the candidate set so it flows through the unchanged score -> MMR -> pack pipeline. This
        mirrors human reconsolidation — recalling a faded memory returns it to active store."""
        if self.store.count("archive") == 0:
            return
        best: Dict[str, tuple] = {}  # archived id -> (best coarse cos, row)
        for q in q_mat:
            qc = self._coarse(q)
            for aid, cos, arow in self.store.cosine_search(
                "archive", "v_coarse", qc, k=self.memory.k_retrieve
            ):
                if cos < self.memory.tau_recall:
                    continue
                if aid not in best or cos > best[aid][0]:
                    best[aid] = (cos, arow)
        restored = 0
        for aid, (cos, arow) in sorted(best.items(), key=lambda kv: kv[1][0], reverse=True):
            if restored >= self.memory.max_recall_per_turn:
                break
            text = arow["text"]
            v = self.provider.encode_document(text, dim=self.glob.dim_full)[0]  # rebuild 768-d
            vc = self._coarse(v)
            mid = self._restore_from_archive(
                arow, text, float(arow["w"]), v, vc, arow["timezone"], now_unix
            )
            row = self.store.one("SELECT * FROM memories WHERE id=?", (mid,))
            if row is None:
                continue
            cos_full = max(float(cosine(q, v)) for q in q_mat)  # full-dim cos, consistent with live cands
            candidates[mid] = {
                "id": mid,
                "cos": cos_full,
                "r": self.retrievability(row, now_unix),
                "row": row,
                "cluster_id": row["cluster_id"],
                "v": self._coarse(unpack_vec(row["v_full"])),
            }
            restored += 1

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split a turn into paragraphs (blank-line separated blocks).

        Falls back to individual non-empty lines when there are no blank lines,
        and to the whole text when there are no newlines at all. Always returns
        at least one chunk."""
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) <= 1:
            blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return blocks or [text.strip() or text]

    def _recall_gate(self, cos: float, r: float) -> bool:
        """Functional forgetting: a memory is unreachable unless weighted(cos, r) clears the threshold."""
        g = self.memory.gate_w_cos * cos + self.memory.gate_w_r * r
        if self.memory.recall_noise_sigma > 0:
            g += random.gauss(0.0, self.memory.recall_noise_sigma)
        return g >= self.memory.gate_theta

    def _apply_spreading(self, candidates: Dict[str, Dict]) -> None:
        """Spreading activation within the candidate set: boost a memory whose cluster
        siblings are themselves query-relevant (ACT-R associative term, zero extra I/O)."""
        groups: Dict[str, List[Dict]] = {}
        for c in candidates.values():
            if c["cluster_id"] is not None:
                groups.setdefault(c["cluster_id"], []).append(c)
        for c in candidates.values():
            sibs = [s for s in groups.get(c["cluster_id"], []) if s["id"] != c["id"]]
            c["spread"] = (
                self.memory.spread_gamma * max(s["r"] * s["cos"] for s in sibs) if sibs else 0.0
            )

    def _retrieval_score(self, row, cos: float, r: float) -> float:
        """Combine cosine, retrievability, importance, frequency and confidence into a single ranking score."""
        return (
            self.memory.alpha * cos
            + self.memory.beta * r
            + self.memory.delta * float(row["w"])
            + self.memory.eta * math.log1p(float(row["freq"]))
            + self.memory.zeta * float(row["confidence"] or 0.0)
        )

    def _pack_retrieved(self, ranked: List[Dict], now_unix: float):
        """Format top-ranked memories into a text pack for the LLM, respecting budget_chars."""
        used_chars = 0
        lines: List[str] = []
        recalled: List[RecalledItem] = []
        packed_ids: List[str] = []
        for item in ranked:
            row = item["row"]
            updated = int(row["updated_at_unix"])
            local = _fmt_local(updated, row["timezone"])
            line = f"[updated={local} (unix={updated})] {row['text']}\n"
            if used_chars + len(line) > self.glob.budget_chars:
                continue
            lines.append(line)
            used_chars += len(line)
            packed_ids.append(row["id"])
            recalled.append(
                RecalledItem(
                    mem_id=row["id"],
                    text=row["text"],
                    score=item["score"],
                    extra={
                        "cos": round(item["cos"], 3),
                        "w": row["w"],
                        "confidence": round(float(row["confidence"] or 0.0), 3),
                        "r": round(item["r"], 3),
                        "S_days": round(self._stability(row) / 86400.0, 2),
                        "freq": int(float(row["freq"])),
                        "spread": round(item.get("spread", 0.0), 3),
                        "cluster_id": row["cluster_id"],
                        "updated_at_unix": updated,
                        "accessed_at_unix": int(row["accessed_at_unix"]),
                        "timezone": row["timezone"],
                    },
                )
            )
            if used_chars >= self.glob.budget_chars:
                break
        return RetrieveResult(pack_text="".join(lines), recalled=recalled), packed_ids

    def _reinforce_ids(self, ids: List[str], now_unix: float) -> None:
        """Apply successful-recall reinforcement to every packed memory ID."""
        for mid in ids:
            r = self.store.one("SELECT * FROM memories WHERE id=?", (mid,))
            if r:
                self._reinforce(r, now_unix)

    def _apply_interference(self, candidates: Dict[str, Dict], packed: set, now_unix: float) -> None:
        """Retrieval-induced forgetting: recalled items mildly weaken un-recalled competitors
        in the same cluster (their stability shrinks, so future retrievability drops)."""
        if self.memory.interference_decay <= 0:
            return
        packed_clusters = {candidates[i]["cluster_id"] for i in packed if i in candidates}
        packed_clusters.discard(None)
        for c in candidates.values():
            if c["id"] in packed or c["cluster_id"] not in packed_clusters:
                continue
            new_S = self._stability(c["row"]) * (1.0 - self.memory.interference_decay)
            self.store.exec("UPDATE memories SET stability=? WHERE id=?", (new_S, c["id"]))

    def _mmr(self, items: List[Dict]) -> List[Dict]:
        """Maximal Marginal Relevance re-ranking: trade off relevance against diversity (coarse-vector cosine)."""
        if not items:
            return []
        lam = self.memory.lambda_div
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        selected: List[Dict] = []
        pool = items[:]
        while pool:
            best, best_val = None, -1e9
            for it in pool:
                if not selected:
                    val = it["score"]
                else:
                    max_sim = max(cosine(it["v"], s["v"]) for s in selected)
                    val = lam * it["score"] - (1 - lam) * max_sim
                if val > best_val:
                    best, best_val = it, val
            selected.append(best)
            pool.remove(best)
        return selected

    # -- dreaming (LLM-driven consolidation) --------------------------- #
    def dream(self, now_unix: Optional[float] = None, max_clusters: Optional[int] = None, force: bool = False) -> List[Dict]:
        now_unix = self.now_unix() if now_unix is None else now_unix
        limit = self.memory.dream_max_clusters if max_clusters is None else max_clusters
        ranked = self._dream_candidates(now_unix, force=force)
        results = [self._dream_cluster(cid, prio, now_unix) for cid, prio in ranked[:limit]]
        self._enforce_capacity(now_unix)
        return results

    def _dream_candidates(self, now_unix: float, force: bool = False) -> List[tuple]:
        out: List[tuple] = []
        for c in self.store.query("SELECT id, last_dreaming_unix FROM clusters"):
            members = self.store.query("SELECT updated_at_unix, v_full FROM memories WHERE cluster_id=?", (c["id"],))
            size = len(members)
            if size < self.memory.dream_min_size:
                continue
            since_dream = now_unix - float(c["last_dreaming_unix"])
            if not force and since_dream < self.memory.dream_min_interval_seconds:
                continue
            ups = [float(m["updated_at_unix"]) for m in members]
            spread = max(ups) - min(ups)
            coarse = [self._coarse(unpack_vec(m["v_full"])) for m in members]
            cen = _normalize(np.mean(np.stack(coarse), axis=0))
            dispersion = 1.0 - float(np.mean([np.dot(vc, cen) for vc in coarse]))
            priority = (
                self.memory.dream_w_size * math.log(1 + size)
                + self.memory.dream_w_spread * (spread / max(1e-6, self.memory.dream_spread_norm))
                + self.memory.dream_w_age * (since_dream / max(1e-6, self.memory.dream_age_norm))
                + self.memory.dream_w_disp * dispersion
            )
            if priority < self.memory.dream_priority_floor:
                continue
            out.append((c["id"], priority))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _dream_cluster(self, cid: str, priority: float, now_unix: float) -> Dict:
        members = self.store.query("SELECT * FROM memories WHERE cluster_id=?", (cid,))
        members = sorted(members, key=lambda r: float(r["updated_at_unix"]))[: self.memory.dream_max_members]
        payload = [
            {
                "id": m["id"],
                "text": m["text"],
                "w": round(float(m["w"]), 2),
                "provenance": m["provenance"],
                "updated_at_unix": int(m["updated_at_unix"]),
                "updated_at_local": _fmt_local(m["updated_at_unix"], m["timezone"]),
                "timezone": m["timezone"],
                "r": round(self.retrievability(m, now_unix), 2),
            }
            for m in members
        ]
        before = [
            {k: p[k] for k in ("id", "text", "w", "provenance", "updated_at_unix", "timezone")}
            for p in payload
        ]
        decision = self.llm.dream_cluster(payload) or {}
        action = decision.get("action", "none")
        new_mems = decision.get("memories") or []

        if action == "none" or not new_mems:
            self.store.exec("UPDATE clusters SET last_dreaming_unix=? WHERE id=?", (now_unix, cid))
            return {"cluster_id": cid, "action": "none", "priority": round(priority, 3),
                    "before": before, "after": [], "deleted_ids": []}

        # Dreaming no longer raises stability; cap at the best source.
        member_S = [self._stability(m) for m in members]
        seed_S = max(member_S) if member_S else self.memory.stab_base_seconds
        seed_conf = max((float(m["confidence"] or 0.0) for m in members), default=self.memory.confidence_inferred)
        source_ids = [m["id"] for m in members]
        summary_of = _shorten(" / ".join(m["text"] for m in members), self.memory.mem_max_chars)
        dream_id = uuid7()

        deleted = [m["id"] for m in members]
        for m in members:
            self.store.exec("DELETE FROM memories WHERE id=?", (m["id"],))

        affected = {cid}
        after = []
        for nm in new_mems:
            row = self._insert_dreamed(nm, seed_S, seed_conf, source_ids, summary_of, action, dream_id, now_unix)
            if row is None:
                continue
            affected.add(row["cluster_id"])
            after.append({k: row[k] for k in ("id", "text", "w", "provenance", "timezone")})

        for acid in affected:
            if self.store.count("memories", "cluster_id=?", (acid,)) == 0:
                self.store.exec("DELETE FROM clusters WHERE id=?", (acid,))
            else:
                self.store.exec("UPDATE clusters SET last_dreaming_unix=? WHERE id=?", (now_unix, acid))

        return {"cluster_id": cid, "action": action if action in ("merge", "split") else "merge",
                "priority": round(priority, 3), "before": before, "after": after, "deleted_ids": deleted}

    def _insert_dreamed(self, nm, seed_S, seed_conf, source_ids, summary_of, action, dream_id, now_unix):
        """Insert one memory produced by a dreaming (consolidation) pass, preserving lineage metadata."""
        text = str(nm.get("text", "")).strip()
        if not text:
            return None
        if len(text) > self.memory.mem_max_chars:
            text = _shorten(text, self.memory.mem_max_chars)
        w = _clamp01(float(nm.get("w", 0.6)))
        provenance = nm.get("provenance", "inferred")
        confidence = max(seed_conf, self._confidence(provenance)) if provenance == "user" else seed_conf
        timezone = nm.get("timezone") or self.glob.default_timezone
        v = self.provider.encode_document(text, dim=self.glob.dim_full)[0]
        lineage = {
            "source_ids": json.dumps(source_ids, ensure_ascii=False),
            "summary_of": summary_of,
            "dream_action": action,
            "created_by_dream_id": dream_id,
        }
        _, row = self._insert_memory(text, v, w, provenance, confidence, seed_S, timezone, now_unix, lineage=lineage)
        return row

    # -- maintenance / introspection ----------------------------------- #
    def maintain(self, turn: int) -> None:
        """Periodic housekeeping: purge empty clusters, enforce capacity limits, checkpoint WAL."""
        now_unix = self.now_unix()
        for c in self.store.query("SELECT id FROM clusters"):
            if self.store.count("memories", "cluster_id=?", (c["id"],)) == 0:
                self.store.exec("DELETE FROM clusters WHERE id=?", (c["id"],))
        self._enforce_capacity(now_unix)
        # Truncate WAL to prevent unbounded -wal file growth during long operation.
        try:
            self.store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass

    def stats(self) -> Dict[str, int]:
        return {
            "memories": self.store.count("memories"),
            "clusters": self.store.count("clusters"),
            "archive": self.store.count("archive"),
        }

    def total_records(self) -> int:
        return self.store.count("memories")

    def snapshot(self) -> Dict[str, List[Dict]]:
        return {
            "memories": self.store.rows_as_dicts("memories"),
            "clusters": self.store.rows_as_dicts("clusters"),
            "archive": self.store.rows_as_dicts("archive"),
        }

    def vector_mb(self) -> float:
        return self.store.vector_storage_mb([("memories", "v_full"), ("archive", "v_coarse")])

    def db_size_bytes(self) -> int:
        return self.store.db_size_bytes()

    def reset(self) -> None:
        """Wipe all tables (clusters, memories, archive) without deleting the DB file."""
        # Clear clusters first to avoid FK issues if added later, then memories, then archive.
        for table in ("clusters", "memories", "archive"):
            self.store.exec(f"DELETE FROM {table}")


# -- module helpers ---------------------------------------------------- #
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n else vec


def _text_hash(text: str) -> str:
    return hashlib.sha1(str(text).strip().lower().encode("utf-8")).hexdigest()


def _fmt_local(unix, tz) -> str:
    """Format a UNIX time in its stored timezone, e.g. '2026-06-06 14:56 Asia/Tokyo'."""
    try:
        from zoneinfo import ZoneInfo

        dt = datetime.fromtimestamp(float(unix), ZoneInfo(str(tz)))
        return f"{dt:%Y-%m-%d %H:%M} {tz}"
    except Exception:
        dt = datetime.utcfromtimestamp(float(unix))
        return f"{dt:%Y-%m-%d %H:%M} UTC"


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("。", "．", ".", "、", " "):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.5:
            return cut[: idx + 1]
    return cut


def _memory_event(action: str, row, r: float) -> Dict:
    return {
        "action": action,
        "id": row["id"],
        "text": row["text"],
        "w": row["w"],
        "provenance": row["provenance"],
        "confidence": round(float(row["confidence"] or 0.0), 3),
        "r": round(float(r), 3),
        "S_days": round(float(row["stability"] or 0.0) / 86400.0, 2),
        "freq": int(float(row["freq"])),
        "accessed_at_unix": int(row["accessed_at_unix"]),
        "updated_at_unix": int(row["updated_at_unix"]),
        "timezone": row["timezone"],
        "cluster_id": row["cluster_id"],
        "v_full": _vec_label(row["v_full"]),
    }


def _vec_label(blob) -> str:
    return "" if blob is None else f"<{len(blob)}B vec>"


def uuid7() -> str:
    unix_ms = int(time.time() * 1000)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (unix_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0b10 << 62
    value |= rand_b
    return str(uuid.UUID(int=value))
