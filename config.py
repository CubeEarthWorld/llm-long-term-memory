"""Configuration for the LLM Long-Term Memory technical prototype.

The project is intentionally kept as a small proof of concept: one memory system,
one SQLite store, and a no-build web UI. Names and boundaries are still explicit
enough to support future packaging or MCP server work without adding that layer
now.
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


@dataclass
class GlobalConfig:
    # Max chars of recalled memory handed to the LLM.
    budget_chars: int = 1024
    # Active (hot) memory records cap.
    total_cap: int = 1000
    # Archived (cold, savings) records cap. Bounds total storage together with total_cap.
    archive_cap: int = 5000
    # Max in-memory turn log entries (sliding window). Prevents unbounded RAM growth during very long sessions.
    max_turn_log: int = 1000
    # Max metrics history entries retained in RAM.
    max_metrics_history: int = 1000

    # Minimum cosine similarity required for recall.
    sim_floor: float = 0.3

    # Default IANA timezone stamped onto memories at write time (overridable per write).
    # Falls back to MEMORY_TZ env, else Asia/Tokyo. Recall / dreaming pass it to the LLM.
    default_timezone: str = os.getenv("MEMORY_TZ", "Asia/Tokyo")

    # EmbeddingGemma native dimension and MRL truncation for coarse clustering/diversity.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
    dim_full: int = 768
    dim_coarse: int = 256

    # LLM provider: "deepseek" or "gemini". Keys come from .env.
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class LongTermMemoryConfig:
    mem_max_chars: int = 300
    tau_dup: float = 0.92
    # Cosine threshold for assigning a memory to an existing cluster.
    tau_link: float = 0.75
    # -- Stability / forgetting (FSRS-style: power-law retrievability) -- #
    # Base stability (seconds). Importance w extends it; confidence scales the labile start.
    stab_base_seconds: float = 14 * 24 * 60 * 60
    kappa: float = 3.0                 # importance -> stability multiplier
    forget_beta: float = 0.5           # power-law exponent: r = (1 + dt/S)^(-beta)
    stab_growth_c: float = 1.0         # stability gain on recall: S *= 1 + c*(1-r) (spacing/testing effect). max 2x when r=0.
    labile_frac: float = 0.25          # low-confidence (inferred) memories start more fragile
    freq_seed: float = 1.0             # initial access count on insert
    reinforce_inc: float = 1.0         # added to freq (access count) on each access
    # Upper bounds to prevent unbounded numeric growth over very long use.
    # max_freq is kept low enough that eta*log1p(freq) does not dominate the retrieval
    # score over cosine similarity (alpha*cos). At 10k accesses, log1p ≈ 9.21, eta*9.21 ≈ 0.18
    # which is well below alpha*cos (max 0.55).
    max_freq: float = 10_000.0        # cap cumulative access count
    max_stability_seconds: float = 10 * 365 * 24 * 60 * 60  # cap stability (~10 years)
    min_residency_seconds: float = 24 * 60 * 60

    # Confidence (reliability) axis, derived from provenance. Separate from importance (w) and stability (S).
    confidence_user: float = 0.9
    confidence_inferred: float = 0.6

    # Retrieval score: alpha*cos + beta*r + delta*w + eta*log1p(freq) + zeta*confidence (+ spreading).
    alpha: float = 0.55
    beta: float = 0.2
    delta: float = 0.2
    eta: float = 0.02              # reduced from 0.05 so freq does not dominate long-term ranking
    zeta: float = 0.05
    lambda_div: float = 0.5
    k_retrieve: int = 80
    spread_gamma: float = 0.15         # spreading activation weight (within candidate set)

    # -- Forgetting model: recall gate + two-stage archive + interference -- #
    # Recall gate (functional forgetting): retrievable iff gate_w_cos*cos + gate_w_r*r (+noise) >= gate_theta.
    gate_w_cos: float = 0.6
    gate_w_r: float = 0.4
    gate_theta: float = 0.2
    recall_noise_sigma: float = 0.0    # >0 makes recall probabilistic (ACT-R logistic noise)
    # active -> archive (savings) -> bounded permanent deletion.
    r_archive_floor: float = 0.1       # below this (after grace) a memory is archived (inaccessible)
    r_hard_floor: float = 0.05         # even protected memories can be archived below this
    archive_grace_seconds: float = 7 * 24 * 60 * 60
    # Max memories to archive in a single maintenance pass (prevents mass-forgetting after long downtime).
    max_decay_per_turn: int = 5
    # Max memories to evict (archive) in a single capacity-enforcement pass.
    # Prevents a single turn from stalling when the DB is far over total_cap after a very long downtime.
    max_evict_per_turn: int = 10
    tau_savings: float = 0.85          # coarse-cosine threshold to recognize a reappearance
    savings_gain: float = 1.6          # stability head-start when restoring from archive (relearning saving)
    # 思い出し (recall): retrieve also searches the archive; an archived memory whose coarse-cosine
    # to the cue is >= tau_recall is restored mid-retrieve. Looser than tau_savings (near-duplicate)
    # so a semantically related — not just identical — archived memory can be recovered by a cue.
    tau_recall: float = 0.6
    max_recall_per_turn: int = 3       # cap archived memories restored (思い出し) per retrieve
    protect_confidence: float = 0.85   # high-confidence & high-w memories resist eviction down to r_hard_floor
    protect_w: float = 0.9
    # Maximum age (seconds) for protection. Even protected memories lose protection after this.
    protect_max_seconds: float = 10 * 365 * 24 * 60 * 60  # ~10 years
    interference_decay: float = 0.05   # retrieval-induced forgetting: competitor stability *= (1 - this)
    consolidation_gain: float = 1.5    # dream: merged memories become more durable (gist -> semantic)

    # -- Dreaming (memory consolidation) ------------------------------- #
    dream_min_size: int = 2                       # skip clusters smaller than this
    dream_min_interval_seconds: float = 24 * 60 * 60  # cooldown since last_dreaming_unix
    dream_max_members: int = 20                   # cap members handed to the LLM
    dream_max_clusters: int = 1                   # clusters processed per dream() call
    dream_priority_floor: float = 0.0             # minimum priority to bother dreaming
    # Priority = w_size*log(1+size) + w_spread*(spread/spread_norm)
    #          + w_age*(since_dream/age_norm) + w_disp*dispersion
    dream_w_size: float = 1.0
    dream_w_spread: float = 1.0
    dream_w_age: float = 1.0
    dream_w_disp: float = 0.5
    dream_spread_norm: float = 14 * 24 * 60 * 60
    dream_age_norm: float = 14 * 24 * 60 * 60


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
    except (TypeError, ValueError):
        return default
    return value


def default_config() -> Config:
    return Config()
