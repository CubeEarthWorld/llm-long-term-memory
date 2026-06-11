"""Tier movement, capacity enforcement and per-turn maintenance (§5.4, I4)."""
from __future__ import annotations

from core.embedding import truncate_normalize
from core.storage import quantize_int8


class MaintenanceOpsMixin:
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

