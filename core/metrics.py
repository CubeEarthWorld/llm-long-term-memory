"""Per-turn metrics for the LLM Long-Term Memory prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RecalledItem:
    mem_id: Any
    text: str
    score: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnMetrics:
    turn: int
    system_id: str
    utterance: str

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    llm_ms: float = 0.0
    write_ms: float = 0.0
    maintain_ms: float = 0.0

    total_records: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    db_size_bytes: int = 0
    vector_mb: float = 0.0

    pack_chars: int = 0
    pack_n: int = 0
    recalled: List[RecalledItem] = field(default_factory=list)
    pack_text: str = ""
    prompt: str = ""
    response: str = ""

    written_ids: List[Any] = field(default_factory=list)
    written_rows: List[Dict[str, Any]] = field(default_factory=list)
    write_note: str = ""

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.retrieve_ms + self.llm_ms + self.write_ms + self.maintain_ms

    def to_detail_dict(self, title: str = "") -> Dict[str, Any]:
        """Return a rich detail dict suitable for API responses and JSON export."""
        return {
            "title": title,
            "response": self.response,
            "write_note": self.write_note,
            "records": self.total_records,
            "pack_chars": self.pack_chars,
            "pack_n": self.pack_n,
            "pack_text": self.pack_text,
            "prompt": self.prompt,
            "written": self.written_rows,
            "times": {
                "total": round(self.total_ms, 1),
                "llm": round(self.llm_ms, 1),
                "retrieve": round(self.retrieve_ms, 1),
                "write": round(self.write_ms, 1),
                "maintain": round(self.maintain_ms, 1),
                "embed": round(self.embed_ms, 1),
            },
            "recalled": [
                {"id": item.mem_id, "text": item.text, "score": round(item.score, 3), **item.extra}
                for item in self.recalled
            ],
        }

    def row(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "system": self.system_id,
            "utterance": self.utterance,
            "embed_ms": round(self.embed_ms, 1),
            "retrieve_ms": round(self.retrieve_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "write_ms": round(self.write_ms, 1),
            "maintain_ms": round(self.maintain_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "records": self.total_records,
            "pack_chars": self.pack_chars,
            "pack_n": self.pack_n,
            "db_kb": round(self.db_size_bytes / 1024, 1),
            "vector_mb": round(self.vector_mb, 3),
        }


class MetricsRecorder:
    def __init__(self, max_history: int = 1000):
        self.max_history: int = max_history
        self.history: List[TurnMetrics] = []

    def add(self, metrics: TurnMetrics) -> None:
        self.history.append(metrics)
        # Sliding window to prevent unbounded RAM growth during very long sessions.
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def reset(self) -> None:
        self.history.clear()

    def dataframe(self, system_id: Optional[str] = None) -> pd.DataFrame:
        rows = [m.row() for m in self.history if system_id is None or m.system_id == system_id]
        return pd.DataFrame(rows)

    def for_turn(self, turn: int, system_id: str) -> Optional[TurnMetrics]:
        return next((m for m in self.history if m.turn == turn and m.system_id == system_id), None)

    def latest_turn(self) -> int:
        return max((m.turn for m in self.history), default=0)
