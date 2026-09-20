"""The memory trace — the canonical record (SPEC §2)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, eq=False)   # a generated __eq__ would raise on the ndarray field
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
    cue: str = ""            # the question this fact answers; what the dream searches the past with

    def with_(self, **changes) -> "Memory":
        return replace(self, **changes)
