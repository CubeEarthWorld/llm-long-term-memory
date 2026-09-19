"""Configuration: ``GlobalConfig`` (app: LLM, embedding model, timezone, turn
policy) and ``LongTermMemoryConfig`` (the ENGRAM v2 engine parameters, SPEC §6).
The engine parameters mirror the Dart ``EngramConfig`` one to one."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))   # repository root — the one definition
SYSTEM_ID = "llm_long_term_memory"
SYSTEM_TITLE = "LLM Long-Term Memory"

_SECRETS_ENV = os.path.join(ROOT, "secrets", ".env")
if os.path.exists(_SECRETS_ENV):
    load_dotenv(_SECRETS_ENV)
load_dotenv()

_DAY = 24 * 60 * 60
_YEAR = 365 * _DAY


@dataclass
class GlobalConfig:
    # Sliding windows for the UI / RAM only.
    max_turn_log: int = 1000
    max_metrics_history: int = 1000
    # IANA timezone stamped onto memories (offset frozen into 'name;+HH:MM').
    default_timezone: str = os.getenv("MEMORY_TZ", "Asia/Tokyo")
    # EmbeddingGemma GGUF (relative paths resolve inside ./model).
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma-300m-qat-Q4_0.gguf")
    # LLM provider: "deepseek" or "gemini". Keys come from secrets/.env.
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    temperature: float = 0.7
    max_output_tokens: int = 1024
    # Per-turn policy (app side): tool saves applied per turn, and the extraction
    # fallback when a turn saved nothing via tools.
    max_writes_per_turn: int = 8
    tool_fallback: bool = True


@dataclass
class LongTermMemoryConfig:
    """ENGRAM v2 parameters (SPEC §6)."""

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
    theta_related: float = 0.75              # neighbourhood for dreams
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


@dataclass
class Config:
    glob: GlobalConfig = field(default_factory=GlobalConfig)
    memory: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)

    def to_dict(self) -> dict:
        return {"glob": asdict(self.glob), "memory": asdict(self.memory)}

    @staticmethod
    def from_dict(d: dict) -> "Config":
        cfg = Config()
        for section, dc in (("glob", cfg.glob), ("memory", cfg.memory)):
            sub = d.get(section, {})
            for f in fields(dc):
                if f.name in sub:
                    setattr(dc, f.name, _coerce(sub[f.name], getattr(dc, f.name)))
        cfg.memory.validate()
        return cfg


def _coerce(value, default):
    """Cast JSON values to the type of the dataclass default."""
    try:
        if isinstance(default, bool):
            return value in (True, "true", "True", 1, "1")
        if isinstance(default, int):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError):
        return default
    return value
