"""Per-turn runner: recall → converse (+ save_memory / delete_memory tools) → cite.

Generation is confined to the converse step; everything else is distance and
arithmetic inside the engine (SPEC §1)."""
from __future__ import annotations

import time
from collections import Counter

from .metrics import MetricsRecorder, RecalledItem, TurnMetrics

# Actions that mean a memory was created or strengthened this turn.
_SAVE_ACTIONS = {"inserted", "reinforced"}


class TurnRunner:
    def __init__(self, provider, llm, recorder: MetricsRecorder, system):
        self.provider = provider
        self.llm = llm
        self.recorder = recorder
        self.system = system

    def run_turn(self, turn: int, utterance: str) -> TurnMetrics:
        system = self.system
        metrics = TurnMetrics(turn=turn, system_id="llm_long_term_memory", utterance=utterance)

        # 1) RECALL — inject ≤ budget_chars of relevant traces.
        result, logic_ms, embed_ms = self._timed(lambda: system.recall(utterance))
        metrics.retrieve_ms = logic_ms
        metrics.embed_ms += embed_ms
        metrics.pack_text = result["pack_text"]
        metrics.pack_chars = len(result["pack_text"])
        metrics.pack_n = len(result["recalled"])
        metrics.recalled = [RecalledItem(r["id"], r["text"], r["score"],
                                         {k: v for k, v in r.items() if k not in ("id", "text", "score")})
                            for r in result["recalled"]]

        # 2) CONVERSE — generation + tools; then 3) CITE — used memories are strengthened.
        (response, events, write_ms), logic_ms, embed_ms = self._timed(
            lambda: self._converse_and_write(result["pack_text"], utterance))
        metrics.llm_ms += max(0.0, logic_ms - write_ms)
        metrics.write_ms += write_ms
        metrics.embed_ms += embed_ms
        metrics.response = response.text
        metrics.prompt = response.prompt
        metrics.written_rows = events
        metrics.written_ids = [e.get("id") for e in events if e.get("id")]
        metrics.write_note = _summarize_events(events)
        metrics.cited = system.cite(response.text)

        metrics.total_records = system.total_records()
        metrics.counts = system.stats()
        metrics.db_size_bytes = system.db_size_bytes()
        metrics.vector_mb = system.vector_mb()
        self.recorder.add(metrics)
        return metrics

    def _converse_and_write(self, pack_text: str, utterance: str):
        """Run the tool-calling exchange; fall back to extraction if nothing was saved."""
        system = self.system
        events: list[dict] = []
        write_ms = 0.0
        max_writes = self.system.glob.max_writes_per_turn

        def _saved() -> int:
            return sum(1 for e in events if e.get("action") in _SAVE_ACTIONS)

        def _save(text: str = "", salience: float = 1.0, **_ignore):
            nonlocal write_ms
            if _saved() >= max_writes:
                events.append({"action": "rate_limited", "error": "per-turn save limit", "text": text})
                return {"ok": False, "action": "rate_limited"}
            t0 = time.perf_counter()
            try:
                ev = system.remember(text, salience=float(salience))
            except (TypeError, ValueError):
                ev = system.remember(text)
            write_ms += (time.perf_counter() - t0) * 1000.0
            events.append(ev)
            return {"ok": ev.get("action") in _SAVE_ACTIONS, "action": ev.get("action"), "id": ev.get("id")}

        def _delete(id: str = "", **_ignore):  # noqa: A002 — tool param name is 'id'
            nonlocal write_ms
            t0 = time.perf_counter()
            ev = system.forget(id)
            write_ms += (time.perf_counter() - t0) * 1000.0
            events.append(ev)
            return {"ok": ev.get("action") == "deleted", "action": ev.get("action")}

        current_time = system.now_local()
        conv = self.llm.converse(pack_text, utterance, {"save_memory": _save, "delete_memory": _delete},
                                 current_time=current_time)
        if not _saved() and self.system.glob.tool_fallback:
            for text in self.llm.extract_save_candidates(utterance, conv.text, current_time=current_time,
                                                         known=pack_text)[:max_writes]:
                t0 = time.perf_counter()
                events.append(system.remember(text))
                write_ms += (time.perf_counter() - t0) * 1000.0
        return conv, events, write_ms

    def _timed(self, fn):
        self.provider.pop_ms()
        t0 = time.perf_counter()
        result = fn()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        embed_ms = self.provider.pop_ms()
        return result, max(0.0, wall_ms - embed_ms), embed_ms


def _summarize_events(events: list[dict]) -> str:
    """Human-readable one-line summary of this turn's memory mutations."""
    if not events:
        return "保存なし"
    label = {"inserted": "新規", "reinforced": "強化", "deleted": "削除", "rejected": "却下",
             "rate_limited": "レート上限", "not_found": "対象なし"}
    counts = Counter(e.get("action", "?") for e in events)
    return " / ".join(f"{label.get(k, k)} {v}" for k, v in counts.items())
