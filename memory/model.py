"""The memory trace — the canonical record (SPEC §2)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Memory:
    """One trace. Text is canonical; ``vector`` is a derived index under ``model_id``."""

    id: str
    text: str
    created_at: int          # 64-bit Unix seconds when the proposition was stated
    tz: str                  # 'IANA;+HH:MM'
    last_recall: int         # decay baseline
    stability: float         # half-life of retrievability, seconds
    consolidated: bool       # integrated by the dream phase
    model_id: str            # embedding model of ``vector`` ('' = not indexed yet)
    vector: np.ndarray       # unit-norm float32

    def with_(self, **changes) -> "Memory":
        return replace(self, **changes)

    def to_json(self) -> dict:
        return {
            "id": self.id, "text": self.text, "created_at": self.created_at, "tz": self.tz,
            "last_recall": self.last_recall, "stability": self.stability,
            "consolidated": self.consolidated, "model_id": self.model_id,
            "vector": self.vector.astype(np.float32).tolist(),
        }
