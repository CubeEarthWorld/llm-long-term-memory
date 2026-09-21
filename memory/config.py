"""The ENGRAM v2.1 engine parameters (SPEC §6) — mirrors the Dart ``EngramConfig``
one to one. Lives beside the engine so importing ``memory`` never pulls in the
app's root ``config`` module (which reads ``secrets/.env`` at import time)."""
from __future__ import annotations

from dataclasses import dataclass

_DAY = 24 * 60 * 60.0     # float: these are float knobs, and an int default
_YEAR = 365 * _DAY        # would make Config._coerce() truncate 0.5 to 0


@dataclass
class LongTermMemoryConfig:
    """ENGRAM v2.1 parameters (SPEC §6)."""

    capacity: int = 10000
    initial_stability: float = _DAY          # S0
    spacing_gain: float = 3.0                # S ← S·(1 + gain·a·(1−R))
    max_stability: float = 10 * _YEAR        # no immortal memory
    grace_period: float = 3 * _DAY           # consolidation window (hippocampal buffer)
    cosine_floor: float = 0.4                # baseline cosine of unrelated text (EmbeddingGemma ≈ 0.4)
    alpha: float = 0.35                      # retrievability floor in the score
    inject_n: int = 5
    mmr_lambda: float = 0.3
    min_score: float = 0.1
    relative_score: float = 0.6
    budget_chars: int = 1024
    max_cues: int = 8
    theta_related: float = 0.55              # reactivation band: how far a cue reaches into the past
    dream_budget: int = 5                    # LLM calls per dream()
    dream_max_members: int = 8               # blast radius of one verdict
    gist_min_cosine: float = 0.5             # confabulation guard
    text_max: int = 170
    writes_per_day: int = 1000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < self.initial_stability <= self.max_stability):
            raise ValueError("require 0 < initial_stability <= max_stability")
        if not (0 <= self.cosine_floor < 1):
            raise ValueError("cosine_floor must be in [0, 1)")
        if not (0 <= self.alpha <= 1) or not (0 <= self.relative_score <= 1):
            raise ValueError("alpha and relative_score must be in [0, 1]")
        if not (0 < self.theta_related < 1):
            raise ValueError("theta_related must be in (0, 1)")
        if self.dream_max_members < 2 or self.inject_n <= 0 or self.budget_chars <= 0 or self.text_max <= 0:
            raise ValueError("dream_max_members ≥ 2; inject_n, budget_chars, text_max > 0")
