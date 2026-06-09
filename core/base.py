"""Shared memory-system protocol and per-turn runner."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from .embedding import EmbeddingProvider
from .llm_client import LLMClient
from .metrics import MetricsRecorder, RecalledItem, TurnMetrics


@dataclass
class RetrieveResult:
    pack_text: str
    recalled: List[RecalledItem] = field(default_factory=list)


@dataclass
class WriteResult:
    written_ids: List = field(default_factory=list)
    written_rows: List[Dict] = field(default_factory=list)
    note: str = ""
    extract_ms: float = 0.0


class MemorySystem(ABC):
    """Interface implemented by the LLM Long-Term Memory storage/retrieval layer."""

    system_id: str = "memory"
    system_name: str = "Memory System"

    @abstractmethod
    def retrieve(self, query: str, turn: int) -> RetrieveResult: ...

    @abstractmethod
    def write(self, turn: int, user_text: str, assistant_text: str) -> WriteResult: ...

    @abstractmethod
    def maintain(self, turn: int) -> None: ...

    @abstractmethod
    def stats(self) -> Dict[str, int]: ...

    @abstractmethod
    def total_records(self) -> int: ...

    @abstractmethod
    def snapshot(self) -> Dict[str, List[Dict]]: ...

    @abstractmethod
    def vector_mb(self) -> float: ...

    @abstractmethod
    def db_size_bytes(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class TurnRunner:
    """Runs retrieve -> respond -> write -> maintain for one user turn."""

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

        result, logic_ms, embed_ms = self._timed_with_embedding(lambda: system.retrieve(utterance, turn))
        metrics.retrieve_ms = logic_ms
        metrics.embed_ms += embed_ms
        metrics.pack_text = result.pack_text
        metrics.pack_chars = len(result.pack_text)
        metrics.pack_n = len(result.recalled)
        metrics.recalled = result.recalled

        response = self.llm.respond(result.pack_text, utterance)
        metrics.llm_ms += response.latency_ms
        metrics.response = response.text
        metrics.prompt = response.prompt

        write_result, logic_ms, embed_ms = self._timed_with_embedding(lambda: system.write(turn, utterance, response.text))
        metrics.write_ms = logic_ms
        metrics.embed_ms += embed_ms
        metrics.llm_ms += write_result.extract_ms
        metrics.written_ids = write_result.written_ids
        metrics.written_rows = write_result.written_rows
        metrics.write_note = write_result.note

        _, logic_ms, embed_ms = self._timed_with_embedding(lambda: system.maintain(turn))
        metrics.maintain_ms = logic_ms
        metrics.embed_ms += embed_ms

        metrics.total_records = system.total_records()
        metrics.counts = system.stats()
        metrics.db_size_bytes = system.db_size_bytes()
        metrics.vector_mb = system.vector_mb()

        self.recorder.add(metrics)
        return metrics

    def _timed_with_embedding(self, fn):
        self.provider.pop_ms()
        t0 = time.perf_counter()
        result = fn()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        embed_ms = self.provider.pop_ms()
        return result, max(0.0, wall_ms - embed_ms), embed_ms
