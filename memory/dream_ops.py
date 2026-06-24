"""DREAM consolidation for :class:`LongTermMemory` — offline, LLM + embedding (§6)."""
from __future__ import annotations

import numpy as np

from memory.helpers import _cohesion, _fmt_local, _fp, _kmeans, _shorten


class DreamOpsMixin:
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


    def _live_rows_coarse(self, tiers) -> list[tuple]:
        out = []
        for tier in tiers:
            rows, M = self._tier_candidates(tier)
            if not rows:
                continue
            out.extend((row, self._coarse(vec)) for row, vec in zip(rows, M))
        return out

    def _dream_candidates(self, now: float, force: bool = False) -> list[tuple]:
        """Cluster live L1/L2 by coarse vector; rank eligible clusters by priority (§6)."""
        rows = self._live_rows_coarse((1, 2))
        if len(rows) < self.memory.cluster_min:
            return []
        X = np.stack([c for _, c in rows]).astype(np.float32)
        k = max(1, round(len(rows) / 3))
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
            if coh < self.memory.cluster_cohesion_min:
                continue
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

        decision = self.llm.dream_cluster(payload, current_time=self.now_local(now)) or {}
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
            raw_mass = sum_A if is_merge else sum_A / max(1, len(picks))
            # 統合はリハーサル行為: 新規書き込み(mass=1.0)以上に「生きて」いなければ
            # ならない。この床がないと、減衰しきったクラスタは sum_A≈0 → mass 0 となり、
            # 活性が恒久的に 0 のまま同一 dream() 内の _enforce_capacity で真っ先に
            # 追放され、統合結果そのものが失われる。
            per_mass = min(max(raw_mass, 1.0), self.memory.m_max)
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

