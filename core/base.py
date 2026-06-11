"""Shared memory-system protocol and per-turn runner (ENGRAM v1.1).

A turn is ``retrieve → converse(+tools) → maintain`` (§5.2 read, §5.1 write via
the ``save_memory`` tool, §5.4 machine movement during maintain). Generation is
confined to the converse step; everything else is distance and arithmetic.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field

from .embedding import EmbeddingProvider
from .llm_client import LLMClient
from .metrics import MetricsRecorder, RecalledItem, TurnMetrics

# Actions that mean a memory was created or strengthened this turn.
_SAVE_ACTIONS = {"inserted", "updated", "conflict", "reinforced"}


@dataclass
class RetrieveResult:
    pack_text: str
    recalled: list[RecalledItem] = field(default_factory=list)


class MemorySystem(ABC):
    """Interface implemented by the ENGRAM storage/retrieval layer."""

    system_id: str = "memory"
    system_name: str = "Memory System"

    @abstractmethod
    def retrieve(self, query: str, turn: int) -> RetrieveResult: ...
    @abstractmethod
    def save_memory(self, text: str, now: float | None = None) -> dict: ...
    @abstractmethod
    def delete_memory(self, mid: str, hard: bool = False) -> dict: ...
    @abstractmethod
    def maintain(self, turn: int) -> None: ...
    @abstractmethod
    def dream(self, *args, **kwargs) -> list[dict]: ...
    @abstractmethod
    def stats(self) -> dict[str, int]: ...
    @abstractmethod
    def total_records(self) -> int: ...
    @abstractmethod
    def snapshot(self) -> dict[str, list[dict]]: ...
    @abstractmethod
    def vector_mb(self) -> float: ...
    @abstractmethod
    def db_size_bytes(self) -> int: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def now_unix(self) -> float: ...
    @abstractmethod
    def set_clock(self, fn) -> None: ...


class TurnRunner:
    """Runs retrieve → converse(+save/delete tools) → maintain for one user turn."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        llm: LLMClient,
        recorder: MetricsRecorder,
        memory_system: MemorySystem,
    ):
        self.provider = provider
        self.llm = llm
        self.recorder = recorder
        self.memory_system = memory_system

    def run_turn(self, turn: int, utterance: str) -> TurnMetrics:
        system = self.memory_system
        metrics = TurnMetrics(turn=turn, system_id=system.system_id, utterance=utterance)

        # 1) READ (§5.2) — inject ≤1024 chars of recalled memory.
        result, logic_ms, embed_ms = self._timed_with_embedding(lambda: system.retrieve(utterance, turn))
        metrics.retrieve_ms = logic_ms
        metrics.embed_ms += embed_ms
        metrics.pack_text = result.pack_text
        metrics.pack_chars = len(result.pack_text)
        metrics.pack_n = len(result.recalled)
        metrics.recalled = result.recalled

        # 2) CONVERSE (§5.1 write via tools) — generation + save_memory/delete_memory.
        (response, events, write_ms), logic_ms, embed_ms = self._timed_with_embedding(
            lambda: self._converse_and_write(system, result.pack_text, utterance)
        )
        metrics.llm_ms += max(0.0, logic_ms - write_ms)
        metrics.write_ms += write_ms
        metrics.embed_ms += embed_ms
        metrics.response = response.text
        metrics.prompt = response.prompt
        metrics.written_rows = events
        metrics.written_ids = [e.get("id") for e in events if e.get("id")]
        metrics.write_note = _summarize_events(events)

        # 3) MAINTAIN (§5.4) — tombstone sweep + demotion/eviction, no LLM.
        _, logic_ms, embed_ms = self._timed_with_embedding(lambda: system.maintain(turn))
        metrics.maintain_ms = logic_ms
        metrics.embed_ms += embed_ms

        metrics.total_records = system.total_records()
        metrics.counts = system.stats()
        metrics.db_size_bytes = system.db_size_bytes()
        metrics.vector_mb = system.vector_mb()

        self.recorder.add(metrics)
        return metrics

    def _converse_and_write(self, system: MemorySystem, pack_text: str, utterance: str):
        """Run the tool-calling exchange; fall back to extraction if nothing was saved."""
        events: list[dict] = []
        write_ms = 0.0

        def _save(text: str = "", **_ignore):
            nonlocal write_ms
            t0 = time.perf_counter()
            ev = system.save_memory(text)
            write_ms += (time.perf_counter() - t0) * 1000.0
            events.append(ev)
            return {"ok": ev.get("action") in _SAVE_ACTIONS, "action": ev.get("action"),
                    "id": ev.get("id"), "tier": ev.get("tier")}

        def _delete(id: str = "", **_ignore):  # noqa: A002 — tool param name is 'id'
            nonlocal write_ms
            t0 = time.perf_counter()
            ev = system.delete_memory(id)
            write_ms += (time.perf_counter() - t0) * 1000.0
            events.append(ev)
            return {"ok": ev.get("action") in ("tombstoned", "deleted"), "action": ev.get("action")}

        tools = {"save_memory": _save, "delete_memory": _delete}
        conv = self.llm.converse(pack_text, utterance, tools)

        saved = any(e.get("action") in _SAVE_ACTIONS for e in events)
        if not saved and getattr(system.memory, "tool_fallback", True):
            cands = self.llm.extract_save_candidates(utterance, conv.text)
            for text in cands[: getattr(system.memory, "max_writes_per_turn", 8)]:
                t0 = time.perf_counter()
                events.append(system.save_memory(text))
                write_ms += (time.perf_counter() - t0) * 1000.0
        return conv, events, write_ms

    def _timed_with_embedding(self, fn):
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
    counts = Counter(e.get("action", "?") for e in events)
    label = {"inserted": "新規", "updated": "更新(墓標化)", "conflict": "競合", "reinforced": "強化",
             "tombstoned": "削除(墓標)", "deleted": "削除", "rejected": "却下",
             "rate_limited": "レート上限", "not_found": "対象なし"}
    return " / ".join(f"{label.get(k, k)} {v}" for k, v in counts.items())
