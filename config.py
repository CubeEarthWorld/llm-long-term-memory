"""App configuration: ``GlobalConfig`` (LLM, embedding model, timezone, turn
policy) plus the engine's own ``LongTermMemoryConfig`` (``memory/config.py``),
which the two are bundled into as ``Config``."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

from dotenv import load_dotenv

from memory.config import LongTermMemoryConfig

ROOT = os.path.dirname(os.path.abspath(__file__))   # repository root — the one definition
SYSTEM_ID = "llm_long_term_memory"
SYSTEM_TITLE = "LLM Long-Term Memory"

_SECRETS_ENV = os.path.join(ROOT, "secrets", ".env")
if os.path.exists(_SECRETS_ENV):
    load_dotenv(_SECRETS_ENV)
load_dotenv()



@dataclass
class GlobalConfig:
    # Sliding windows for the UI / RAM only.
    max_turn_log: int = 1000
    max_metrics_history: int = 1000
    # IANA timezone stamped onto memories (offset frozen into 'name;+HH:MM').
    default_timezone: str = os.getenv("MEMORY_TZ", "Asia/Tokyo")
    # Embedding GGUF (relative paths resolve inside ./model). Any model works; the
    # retrieval prompts are the model's own (EmbeddingGemma here, "query: "/"passage: "
    # for E5, "" for a symmetric model), and the three cosine thresholds in
    # LongTermMemoryConfig must be recalibrated when the model changes.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma-300m-qat-Q4_0.gguf")
    embed_query_prefix: str = os.getenv("EMBED_QUERY_PREFIX", "task: search result | query: ")
    embed_document_prefix: str = os.getenv("EMBED_DOCUMENT_PREFIX", "title: none | text: ")
    # LLM provider: "deepseek" or "gemini". Keys come from secrets/.env.
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    temperature: float = 0.7
    # Reasoning models spend thousands of tokens before the answer; too low a cap returns
    # an empty body (finish_reason=length). Measured on deepseek-flash dreaming: 500-1100
    # reasoning tokens typically, but hard clusters exceeded 4096 in 14% of calls.
    max_output_tokens: int = 8192
    # Per-turn policy (app side): tool saves applied per turn, and the extraction
    # fallback when a turn saved nothing via tools.
    max_writes_per_turn: int = 8
    tool_fallback: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        # A window of 0 would make ``rows[-n:]`` return every row instead of none,
        # i.e. grow the log without bound and empty the persisted turn_log.
        self.max_turn_log = max(1, int(self.max_turn_log))
        self.max_metrics_history = max(1, int(self.max_metrics_history))


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
        cfg.glob.validate()
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
