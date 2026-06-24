"""WRITE/DELETE pipeline for :class:`LongTermMemory` (§5.1, §5.3)."""
from __future__ import annotations

import numpy as np

from core.embedding import truncate_normalize
from memory.helpers import _shorten, ulid


class WriteOpsMixin:
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
        """Best (id, precise_cos, row) across all tiers; in-band L2/L3 refined at full width (§5.1)."""
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
                vs = self._doc_vec_cached(row)             # text is canonical → re-embed at full width
                cos = float(np.dot(v_full, vs))
            if best is None or cos > best[1]:
                best = (mid, cos, row)
        return best


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

