"""READ pipeline for :class:`LongTermMemory` — retrieve / filter / MMR / pack (§4.2, §5.2)."""
from __future__ import annotations

import math

import numpy as np

from core.base import RetrieveResult
from core.embedding import truncate_normalize
from core.metrics import RecalledItem
from memory.helpers import _split_paragraphs, lam_mmr


class ReadOpsMixin:
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

