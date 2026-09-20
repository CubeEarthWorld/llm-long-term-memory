"""Per-turn metrics for the LLM Long-Term Memory prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import SYSTEM_ID, SYSTEM_TITLE


@dataclass
class TurnMetrics:
    turn: int
    utterance: str

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    llm_ms: float = 0.0
    write_ms: float = 0.0

    total_records: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    db_size_bytes: int = 0
    vector_mb: float = 0.0

    pack_chars: int = 0
    pack_n: int = 0
    recalled: list[dict[str, Any]] = field(default_factory=list)   # engine recall items
    pack_text: str = ""
    prompt: str = ""
    response: str = ""

    cited: list[str] = field(default_factory=list)
    written_rows: list[dict[str, Any]] = field(default_factory=list)
    write_note: str = ""

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.retrieve_ms + self.llm_ms + self.write_ms

    def to_detail_dict(self) -> dict[str, Any]:
        """Return a rich detail dict suitable for API responses and JSON export."""
        return {
            "title": SYSTEM_TITLE,
            "response": self.response,
            "write_note": self.write_note,
            "records": self.total_records,
            "pack_chars": self.pack_chars,
            "pack_n": self.pack_n,
            "pack_text": self.pack_text,
            "prompt": self.prompt,
            "written": self.written_rows,
            "cited": self.cited,
            "times": {
                "total": round(self.total_ms, 1),
                "llm": round(self.llm_ms, 1),
                "retrieve": round(self.retrieve_ms, 1),
                "write": round(self.write_ms, 1),
                "embed": round(self.embed_ms, 1),
            },
            "recalled": self.recalled,
        }

    def row(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "system": SYSTEM_ID,
            "utterance": self.utterance,
            "embed_ms": round(self.embed_ms, 1),
            "retrieve_ms": round(self.retrieve_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "write_ms": round(self.write_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "records": self.total_records,
            **{k: self.counts.get(k, 0) for k in ("labile", "unindexed")},
            "pack_chars": self.pack_chars,
            "pack_n": self.pack_n,
            "db_kb": round(self.db_size_bytes / 1024, 1),
            "vector_mb": round(self.vector_mb, 3),
        }


class MetricsRecorder:
    def __init__(self, max_history: int = 1000):
        self.max_history: int = max_history
        self.history: list[TurnMetrics] = []

    def add(self, metrics: TurnMetrics) -> None:
        self.history.append(metrics)
        # Sliding window to prevent unbounded RAM growth during very long sessions.
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def reset(self) -> None:
        self.history.clear()

    def rows(self) -> list[dict[str, Any]]:
        return [m.row() for m in self.history]

    def for_turn(self, turn: int) -> TurnMetrics | None:
        return next((m for m in self.history if m.turn == turn), None)


def invariants(rows: list[dict[str, Any]], mem) -> dict[str, bool]:
    """The two per-turn invariants over ``rows``, keyed by their display label
    (``mem`` is a LongTermMemoryConfig)."""
    return {
        f"全 pack <= {mem.budget_chars}字 (注入予算)": all(r["pack_chars"] <= mem.budget_chars for r in rows),
        f"全 records <= {mem.capacity}件 (capacity)": all(r["records"] <= mem.capacity for r in rows),
    }


def final_stats(memory) -> dict[str, Any]:
    """Final engine figures (counts + sizes) as reported by the API, the CLI and the dump."""
    return {**memory.stats(), "vector_mb": round(memory.vector_mb(), 3),
            "db_kb": round(memory.store.db_size_bytes() / 1024, 1)}
