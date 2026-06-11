"""Configuration for the LLM Long-Term Memory technical prototype.

The internal memory engine implements the **ENGRAM v1.1** specification
(see ``ENGRAM_spec_v1_1.md``): a 3-tier store (L1 episodic / L2 semantic /
L3 schema), activation ``A = mass·2^(−Δt/τ)``, distance-only identity
judgement, and LLM use confined to write / read / dream.

The product name, file layout, and DB filename are kept; only the algorithm,
schema, parameters, and UI semantics are ENGRAM. ``GlobalConfig`` holds
embedding / LLM / injection settings; ``LongTermMemoryConfig`` holds the
ENGRAM constants of spec §8.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

from dotenv import load_dotenv

# Load secrets from git-ignored directory first, then project root as fallback.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SECRETS_ENV = os.path.join(_BASE_DIR, "secrets", ".env")
if os.path.exists(_SECRETS_ENV):
    load_dotenv(_SECRETS_ENV)
load_dotenv()

# Duration constants (seconds) for the ENGRAM half-lives.
_DAY = 24 * 60 * 60
_YEAR = 365 * _DAY


@dataclass
class GlobalConfig:
    # Max chars of recalled memory injected into the LLM (spec §8: 注入予算 1024).
    budget_chars: int = 1024
    # Max in-memory turn log entries (sliding window) — UI/RAM bound only.
    max_turn_log: int = 1000
    # Max metrics history entries retained in RAM.
    max_metrics_history: int = 1000

    # Default IANA timezone stamped onto memories at write time.
    # Falls back to MEMORY_TZ env, else Asia/Tokyo. ENGRAM stores 'name;+offset'.
    default_timezone: str = os.getenv("MEMORY_TZ", "Asia/Tokyo")

    # Embedding model. EmbeddingGemma native dimension is 768 (L1). MRL truncation
    # to 256 (L2) / 128 (L3) is applied per tier; see LongTermMemoryConfig.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
    dim_full: int = 768

    # LLM provider: "deepseek" or "gemini". Keys come from .env.
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class LongTermMemoryConfig:
    """ENGRAM v1.1 parameters (spec §8). Constants may be tuned only with a spec record."""

    # -- Tier capacities (墓標込み, I4) and MRL vector format (§8 dim/dtype) -- #
    cap1: int = 1000          # L1 episodic
    cap2: int = 3000          # L2 semantic
    cap3: int = 6000          # L3 schema
    dim1: int = 768           # L1 768d f32
    dim2: int = 256           # L2 256d int8
    dim3: int = 128           # L3 128d int8

    # -- Activation half-lives (seconds): L1=7d, L2=90d, L3=3y (§4.1, §8) -- #
    tau1: float = 7 * _DAY
    tau2: float = 90 * _DAY
    tau3: float = 3 * _YEAR

    # -- Activation / reinforcement (§4.1) -- #
    m_max: float = 64.0               # mass / activation cap (I1)
    refractory_seconds: float = 3600  # min interval between mass bonuses (不応期)

    # -- Retrieval score (§4.2) -- #
    alpha: float = 0.35               # activation floor in score
    inject_n: int = 5                 # memories injected per READ
    mmr_lambda: float = 0.3           # MMR diversity penalty
    score_thresholds: tuple[float, ...] = (0.1, 0.2)  # MMR前のスコア閾値（段階的緩和）

    # -- Identity thresholds (document–document, §4.3) -- #
    theta_same: float = 0.97          # ≥ → supersede (再固定化)
    theta_conflict: float = 0.85      # 0.85–0.97 → conflict queue
    precise_margin: float = 0.03      # WRITE: re-embed at 768d within this band (§5.1)

    # -- Tier movement hysteresis (§4.4) -- #
    theta_up: float = 16.0            # A ≥ → promote (L2/L3 → L1, in dream)
    theta_down: float = 4.0           # A < → demotion candidate

    # -- Atomicity of text (§3.1, §8) -- #
    text_max: int = 170               # max chars of a memory proposition
    text_hard_max: int = 1024         # hard byte bound (§8.1, I14)
    gen_max: int = 7                  # consolidation generation cap (I7)

    # -- Dream (§6, §8) -- #
    dream_budget: int = 5             # adjudications per dream() call (0..4096)
    dream_budget_hard: int = 4096     # "∞" bound (§8.1)
    cluster_min: int = 3              # min members for an eligible cluster
    cluster_cohesion_min: float = 0.6  # minimum mean pairwise cosine for cluster eligibility (§6)
    dream_max_members: int = 64       # members handed to the LLM per adjudication
    snapshot_gens: int = 8            # DB snapshot ring before dream (§8)

    # -- Ring buffers (§8, I11) -- #
    conflict_cap: int = 256
    dream_log_cap: int = 512

    # -- Housekeeping (§6) -- #
    tombstone_sweep_pct: float = 0.10     # sweep when tombstones exceed 10% of a tier
    tombstone_sweep_age: float = 7 * _DAY  # ...or older than 7 days

    # -- Soft-side write rate limit (§5.1; final bound 16384/day is §8.1) -- #
    write_rate_per_day: int = 2000
    max_writes_per_turn: int = 8          # cap save_memory tool calls applied per turn

    # -- Robustness net (§ plan): if a turn saved nothing via tools, run one
    #    extraction fallback that proposes propositions through the same save path. -- #
    tool_fallback: bool = True

    # -- Safety hard bounds (§8.1) used by validation (I14) -- #
    hard_memory_rows: int = 16384
    hard_vec_rows: int = 32768
    decay_exp_cap: float = 65536.0        # max(0,Δt)/τ above this → A=0 (was 64→~192yr; now ~196kyr)


@dataclass
class Config:
    glob: GlobalConfig = field(default_factory=GlobalConfig)
    memory: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)

    def to_dict(self) -> dict:
        return {
            "glob": asdict(self.glob),
            "memory": asdict(self.memory),
        }

    @staticmethod
    def from_dict(d: dict) -> "Config":
        cfg = Config()
        for section, dc in (("glob", cfg.glob), ("memory", cfg.memory)):
            sub = d.get(section, {})
            for f in fields(dc):
                if f.name in sub:
                    setattr(dc, f.name, _coerce(sub[f.name], getattr(dc, f.name)))
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
        if isinstance(default, tuple):
            # JSON arrays arrive as list; also accept comma-separated string.
            if isinstance(value, (list, tuple)):
                return tuple(_coerce(v, default[0]) if default else v for v in value)
            if isinstance(value, str):
                parts = [p.strip() for p in value.split(",") if p.strip()]
                return tuple(_coerce(p, default[0]) if default else p for p in parts)
            return default
    except (TypeError, ValueError):
        return default
    return value


def default_config() -> Config:
    return Config()
