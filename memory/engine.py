"""ENGRAM v2 engine (SPEC.md): traces with a forgetting curve, the wake-phase
verbs remember / recall / forget, and the sleep-phase dream.

Every trace is held in RAM (numpy matrix for brute-force cosine); the injected
``Store`` is the durable substrate, the ``EmbeddingProvider`` the index and — for
``dream`` only — the ``LLMClient``. Mirrors the Dart package ``long_term_memory``
operation for operation (see the cross-language conformance test).
"""
from __future__ import annotations

import math
import re
import time

import numpy as np

from config import LongTermMemoryConfig
from memory.model import Memory
from memory.util import clean_text, cues, fmt_local, tz_field, ulid

_SEEDS_PER_BUDGET = 8      # seeds examined per LLM call: dream cost O(budget·capacity·dim)
_CITE = re.compile("《id:([^》]+)》")


class LongTermMemory:
    def __init__(self, store, provider, llm, cfg: LongTermMemoryConfig, timezone: str = "UTC", clock=None):
        self.store = store
        self.provider = provider
        self.llm = llm
        self.cfg = cfg
        self.timezone = timezone             # IANA name stamped onto new traces
        self._clock = clock
        self._traces: dict[str, Memory] = {}
        self._write_times: list[float] = []
        self._last_recall: dict[str, dict] = {}
        self._index: tuple[list[str], np.ndarray] | None = None   # cached (ids, unit-norm matrix)
        self.initialize()

    # ------------------------------------------------------------------ #
    # clock
    # ------------------------------------------------------------------ #
    def now_unix(self) -> float:
        return float(self._clock()) if self._clock else time.time()

    def set_clock(self, fn) -> None:
        self._clock = fn

    def now_local(self, now: float | None = None) -> str:
        now = self.now_unix() if now is None else now
        return fmt_local(now, tz_field(self.timezone, now))

    # ------------------------------------------------------------------ #
    # the forgetting curve (SPEC §3)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _elapsed(m: Memory, now: float) -> float:
        return max(0.0, now - m.last_recall)

    def retrievability(self, m: Memory, now: float) -> float:
        """R = 2^(−max(0, now − last_recall) / stability)."""
        return 2.0 ** (-self._elapsed(m, now) / m.stability)

    def strength(self, m: Memory, now: float) -> float:
        """S·R — proportional to the trace's total remaining retrievability."""
        return m.stability * self.retrievability(m, now)

    def _clamp_stability(self, s: float) -> float:
        return min(max(s, 1.0), self.cfg.max_stability)

    def _activation(self, cos: float) -> float:
        return max(0.0, (cos - self.cfg.cosine_floor) / (1.0 - self.cfg.cosine_floor))

    def _retrieved(self, m: Memory, now: float, a: float) -> Memory:
        s = m.stability * (1 + self.cfg.spacing_gain * a * (1 - self.retrievability(m, now)))
        return m.with_(stability=self._clamp_stability(s), last_recall=int(now))

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        """Load every trace, clamp out-of-range values, re-embed stale vectors."""
        now = int(self.now_unix())
        self._traces.clear()
        self._index = None
        for raw in self.store.load_all():
            m = raw.with_(stability=self._clamp_stability(raw.stability),
                          last_recall=min(raw.last_recall, now), created_at=min(raw.created_at, now))
            self._traces[m.id] = m
            if m != raw:
                self.store.put(m)
        self._reindex()

    def _reindex(self) -> None:
        stale = [m for m in self._traces.values() if m.model_id != self.provider.model_id]
        for i in range(0, len(stale), 64):
            batch = stale[i:i + 64]
            try:
                vecs = self.provider.encode_document([m.text for m in batch])
            except Exception:
                return                                   # embedder offline: stay stale
            with self.store.transaction():
                for m, v in zip(batch, vecs):
                    self._put(m.with_(model_id=self.provider.model_id, vector=v))

    def _search(self) -> tuple[list[Memory], np.ndarray]:
        """Indexed traces and their vector matrix. The matrix is cached and rebuilt only
        when a trace is inserted, removed or re-embedded (not on strength updates)."""
        if self._index is None:
            rows = [m for m in self._traces.values() if m.model_id == self.provider.model_id]
            self._index = ([m.id for m in rows],
                           np.stack([m.vector for m in rows]) if rows else np.zeros((0, 1), dtype=np.float32))
        ids, M = self._index
        return [self._traces[i] for i in ids], M

    def reset(self) -> None:
        self.store.clear()
        self._traces.clear()
        self._write_times.clear()
        self._index = None

    def memories(self) -> list[Memory]:
        return list(self._traces.values())

    def memory(self, mid: str) -> Memory | None:
        return self._traces.get(mid)

    def _put(self, m: Memory) -> None:
        old = self._traces.get(m.id)
        if old is None or old.vector is not m.vector:
            self._index = None
        self._traces[m.id] = m
        self.store.put(m)

    def _remove(self, mid: str) -> None:
        self._traces.pop(mid, None)
        self._index = None
        self.store.remove(mid)

    # ------------------------------------------------------------------ #
    # wake phase (SPEC §4)
    # ------------------------------------------------------------------ #
    def remember(self, text: str, salience: float = 1.0, now: float | None = None) -> dict:
        """Store one proposition. Exact duplicates are rehearsed; everything else is
        inserted (never overwritten). Over capacity the weakest old trace is forgotten."""
        now = self.now_unix() if now is None else now
        text = clean_text(text, self.cfg.text_max)
        if not text:
            return {"action": "rejected", "error": "empty text", "text": ""}
        for m in self._traces.values():
            if m.text == text:
                r = self._retrieved(m, now, 1.0)
                self._put(r)
                return self._event("reinforced", r, now)
        day_ago = now - 86400
        self._write_times = [t for t in self._write_times if t >= day_ago]
        if len(self._write_times) >= self.cfg.writes_per_day:
            return {"action": "rate_limited", "error": "daily write limit", "text": text}
        self._write_times.append(now)
        try:
            v = self.provider.encode_document([text])[0]
        except Exception:
            v = None                                     # embedder offline: index later
        best, nearest = 0.0, None
        if v is not None:
            rows, M = self._search()
            if rows:
                sims = M @ v
                i = int(sims.argmax())
                if float(sims[i]) > best:
                    best, nearest = float(sims[i]), rows[i]
        related = best >= self.cfg.theta_related
        m = Memory(
            id=ulid(int(now * 1000)), text=text, created_at=int(now),
            tz=tz_field(self.timezone, now), last_recall=int(now),
            stability=self._clamp_stability(self.cfg.initial_stability * min(max(float(salience), 0.0), 10.0)),
            consolidated=v is not None and not related and not self._stale(),
            model_id=self.provider.model_id if v is not None else "",
            vector=v if v is not None else np.zeros(0, dtype=np.float32),
        )
        with self.store.transaction():                     # one operation = one commit
            self._put(m)
            if related and nearest.consolidated:
                self._put(nearest.with_(consolidated=False))   # reconsolidation: old trace labile too
            evicted = self._enforce_capacity(now)
        return self._event("inserted", m, now, cos=round(best, 3),
                           evicted=None if evicted is None else evicted.text)

    def _stale(self) -> bool:
        return any(m.model_id != self.provider.model_id for m in self._traces.values())

    def recall(self, query: str, now: float | None = None) -> dict:
        """Inject ≤inject_n relevant traces (half-activation strengthening; ``cite`` completes it)."""
        now = self.now_unix() if now is None else now
        self._last_recall.clear()
        parts = cues(query, self.cfg.max_cues)
        rows, M = self._search()
        if not parts or not rows:
            return {"pack_text": "", "recalled": []}
        try:
            qs = self.provider.encode_query(parts)
        except Exception:
            return {"pack_text": "", "recalled": []}
        cos = (M @ qs.T).max(axis=1)
        R = self._retrievabilities(rows, now)
        act = np.maximum(0.0, (cos - self.cfg.cosine_floor) / (1.0 - self.cfg.cosine_floor))
        score = act * (self.cfg.alpha + (1 - self.cfg.alpha) * R)
        floor = max(self.cfg.min_score, self.cfg.relative_score * float(score.max()))
        pool = sorted((i for i in range(len(rows)) if score[i] >= floor), key=lambda i: -score[i])
        lines, recalled = [], []
        with self.store.transaction():                     # one operation = one commit
            self._pack(self._mmr(pool, score, M), rows, score, cos, R, now, lines, recalled)
        return {"pack_text": "".join(lines), "recalled": recalled}

    def _pack(self, picks, rows, score, cos, R, now, lines, recalled) -> None:
        used = 0
        for i in picks:
            m = rows[i]
            line = f"[{m.created_at} {m.tz}] {m.text}　《id:{m.id}》\n"
            if lines and used + len(line) > self.cfg.budget_chars:
                continue
            lines.append(line)
            used += len(line)
            item = {"id": m.id, "text": m.text, "score": round(float(score[i]), 3), "cos": round(float(cos[i]), 3),
                    "R": round(float(R[i]), 3), "stability": round(m.stability), "created_at": m.created_at, "tz": m.tz}
            recalled.append(item)
            self._last_recall[m.id] = {"cos": float(cos[i]), "R": float(R[i]), "score": float(score[i])}
            self._put(self._retrieved(m, now, 0.5 * self._activation(float(cos[i]))))

    def _mmr(self, pool: list[int], score: np.ndarray, M: np.ndarray) -> list[int]:
        """next = argmax[score − λ·max cos(m, selected)]."""
        if not pool:
            return []
        P = M[pool]
        s = score[pool].astype(np.float64)
        max_sim = np.zeros(len(pool))
        alive = np.ones(len(pool), dtype=bool)
        out: list[int] = []
        while alive.any() and len(out) < self.cfg.inject_n:
            vals = np.where(alive, s - self.cfg.mmr_lambda * max_sim, -np.inf)
            j = int(vals.argmax())
            out.append(pool[j])
            alive[j] = False
            max_sim = np.maximum(max_sim, (P @ P[j]).astype(np.float64))
        return out

    def cite(self, reply_text: str) -> list[str]:
        """Complete the strengthening of last-recall traces cited as 《id:…》 in the reply."""
        ids = []
        for mid in _CITE.findall(reply_text or ""):
            c = self._last_recall.pop(mid, None)
            m = self._traces.get(mid) if c else None
            if m is None:
                continue
            g = self.cfg.spacing_gain * (1 - c["R"])
            partial, full = 1 + g * 0.5 * self._activation(c["cos"]), 1 + g
            self._put(m.with_(stability=self._clamp_stability(m.stability * full / partial)))
            ids.append(mid)
        return ids

    def forget(self, mid: str) -> dict:
        """id-only physical delete."""
        m = self._traces.get(mid)
        if m is None:
            return {"action": "not_found", "id": mid}
        self._remove(mid)
        return {"action": "deleted", "id": mid, "text": m.text}

    @staticmethod
    def _column(rows: list[Memory], field: str) -> np.ndarray:
        return np.fromiter((getattr(m, field) for m in rows), dtype=np.float64, count=len(rows))

    def _retrievabilities(self, rows: list[Memory], now: float) -> np.ndarray:
        S = self._column(rows, "stability")
        return 2.0 ** (-np.maximum(0.0, now - self._column(rows, "last_recall")) / S)

    def _enforce_capacity(self, now: float) -> Memory | None:
        """Forget the lowest-ranked trace while over capacity. Traces inside the grace
        period are protected unless they exceed a tenth of the capacity or nothing
        older exists (ties: old before young, then insertion order)."""
        victim = None
        while len(self._traces) > self.cfg.capacity:
            rows = list(self._traces.values())
            S = self._column(rows, "stability")
            rank = np.log2(S) - np.maximum(0.0, now - self._column(rows, "last_recall")) / S
            age = now - self._column(rows, "created_at")
            young = (age >= 0) & (age < self.cfg.grace_period)
            old_i = int(np.where(young, np.inf, rank).argmin()) if not young.all() else -1
            pick = old_i
            if old_i < 0 or int(young.sum()) > self.cfg.capacity // 10:
                young_i = int(np.where(young, rank, np.inf).argmin())
                if old_i < 0 or rank[young_i] < rank[old_i]:
                    pick = young_i
            victim = rows[pick]
            self._remove(victim.id)
        return victim

    # ------------------------------------------------------------------ #
    # sleep phase (SPEC §5)
    # ------------------------------------------------------------------ #
    def _seeds(self) -> list[Memory]:
        """Labile traces, most stable first (what carries the most evidence integrates first)."""
        return sorted((m for m in self._search()[0] if not m.consolidated), key=lambda m: -m.stability)

    def _cluster(self, seed: Memory) -> list[Memory]:
        rows, M = self._search()
        sims = M @ seed.vector if rows else []
        near = sorted(((float(sims[i]), i) for i in range(len(rows))
                       if sims[i] >= self.cfg.theta_related and rows[i].id != seed.id), key=lambda t: -t[0])
        return [seed] + [rows[i] for _, i in near[: self.cfg.dream_max_members - 1]]

    def clusters(self) -> list[list[Memory]]:
        """The clusters the next dream would hand to the LLM (no LLM call)."""
        out, taken = [], set()
        for seed in self._seeds():
            if seed.id in taken:
                continue
            cluster = self._cluster(seed)
            if len(cluster) < 2:
                continue
            out.append(cluster)
            taken.update(m.id for m in cluster)
        return out

    def dream(self, budget: int | None = None, now: float | None = None) -> list[dict]:
        """Offline consolidation — the only place traces are rewritten."""
        now = self.now_unix() if now is None else now
        self.store.backup()
        self._reindex()
        left = self.cfg.dream_budget if budget is None else int(budget)
        reports, failed = [], set()
        for s in self._seeds()[: _SEEDS_PER_BUDGET * max(left, 0)]:
            seed = self._traces.get(s.id)
            if seed is None or seed.consolidated or s.id in failed:
                continue
            if left <= 0:
                break
            cluster = self._cluster(seed)
            if len(cluster) < 2:
                self._put(seed.with_(consolidated=True))
                continue
            left -= 1
            report = self._adjudicate(cluster, now)
            if report["action"] == "error":
                failed.update(m.id for m in cluster)
            reports.append(report)
        return reports

    def _adjudicate(self, cluster: list[Memory], now: float) -> dict:
        before = [{"id": m.id, "text": m.text, "R": round(self.retrievability(m, now), 2)} for m in cluster]
        members = [{"id": m.id, "text": m.text, "local_time": fmt_local(m.created_at, m.tz),
                    "timezone": m.tz, "R": round(self.retrievability(m, now), 2)} for m in cluster]
        try:
            decision = self.llm.dream_cluster(members, current_time=self.now_local(now)) or {}
        except Exception as e:  # noqa: BLE001
            return {"action": "error", "before": before, "after": [], "error": f"{type(e).__name__}: {e}"}
        texts: list[str] = []
        for t in decision.get("memories") or []:
            c = clean_text(t, self.cfg.text_max)
            if c and c not in texts:
                texts.append(c)
        gists = [] if len(texts) > len(cluster) else self._gists(texts, cluster, now)
        if not gists:
            with self.store.transaction():
                for m in cluster:
                    self._put(m.with_(consolidated=True))
            return {"action": "keep", "before": before, "after": []}
        with self.store.transaction():
            for m in cluster:
                self.store.remove(m.id)
            for g in gists:
                self.store.put(g)
        for m in cluster:
            self._traces.pop(m.id, None)
        for g in gists:
            self._traces[g.id] = g
        self._index = None
        self._enforce_capacity(now)
        return {"action": "replace", "before": before,
                "after": [{"id": g.id, "text": g.text, "stability": round(g.stability)} for g in gists]}

    def _gists(self, texts: list[str], cluster: list[Memory], now: float) -> list[Memory]:
        """Replacement traces: rejected wholesale if any text is unrelated to every member
        (confabulation guard). Stability = strongest member + live evidence of the others;
        R conserves the cluster's total strength (encoded into last_recall)."""
        if not texts:
            return []
        vecs = self.provider.encode_document(texts)
        M = np.stack([m.vector for m in cluster])
        if float((M @ vecs.T).max(axis=0).min()) < self.cfg.gist_min_cosine:
            return []
        strongest = max(cluster, key=lambda m: m.stability)
        sum_sr = sum(self.strength(m, now) for m in cluster)
        stability = self._clamp_stability(strongest.stability + sum_sr - self.strength(strongest, now))
        r = min(1.0, sum_sr / stability)
        elapsed = min(64.0, -math.log2(r)) if r > 0 else 64.0
        last_recall = int(round(now - stability * elapsed))
        tz = tz_field(self.timezone, now)
        return [Memory(id=ulid(int(now * 1000)), text=t, created_at=int(now), tz=tz, last_recall=last_recall,
                       stability=stability, consolidated=True, model_id=self.provider.model_id, vector=v)
                for t, v in zip(texts, vecs)]

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def _event(self, action: str, m: Memory, now: float, **extra) -> dict:
        return {"action": action, "id": m.id, "text": m.text, "stability": round(m.stability),
                "R": round(self.retrievability(m, now), 3), "consolidated": m.consolidated, **extra}

    def stats(self) -> dict[str, int]:
        rows = list(self._traces.values())
        return {"records": len(rows), "labile": sum(1 for m in rows if not m.consolidated),
                "unindexed": sum(1 for m in rows if m.model_id != self.provider.model_id)}

    def total_records(self) -> int:
        return len(self._traces)

    def snapshot(self) -> list[dict]:
        now = self.now_unix()
        return [{"id": m.id, "text": m.text, "local": fmt_local(m.created_at, m.tz), "tz": m.tz,
                 "stability_days": round(m.stability / 86400, 2), "R": round(self.retrievability(m, now), 3),
                 "strength_days": round(self.strength(m, now) / 86400, 2), "consolidated": m.consolidated,
                 "last_recall": m.last_recall, "model_id": m.model_id}
                for m in sorted(self._traces.values(), key=lambda m: -self.strength(m, now))]

    def vector_mb(self) -> float:
        return sum(m.vector.nbytes for m in self._traces.values()) / (1024 * 1024)

    def db_size_bytes(self) -> int:
        return self.store.db_size_bytes()
