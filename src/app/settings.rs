//! App configuration: [`Settings`] (LLM, embedding model, timezone, turn policy) plus
//! the engine's [`Config`], bundled as [`AppConfig`]; paths and the `.env` lookup.

use crate::config::params;
use crate::Config;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// The app root holding `model/`, `data/` and `secrets/`: `LTM_ROOT`, else the working
/// directory or the nearest ancestor of the executable that has one of them (so
/// `target/release/ltm` finds the checkout it was built in), else the working directory.
pub fn root() -> PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        if let Some(r) = std::env::var_os("LTM_ROOT") {
            return PathBuf::from(r);
        }
        let cwd = std::env::current_dir().unwrap_or_default();
        let is_root = |d: &std::path::Path| ["model", "data", "secrets"].iter().any(|s| d.join(s).is_dir());
        let exe = std::env::current_exe().ok();
        std::iter::once(cwd.as_path())
            .chain(exe.iter().flat_map(|e| e.ancestors().skip(1)))
            .find(|d| is_root(d))
            .map_or(cwd.clone(), std::path::Path::to_path_buf)
    })
    .clone()
}

/// Runtime output: the SQLite DB, the seed CSV, CLI results.
pub fn data_dir() -> PathBuf {
    std::env::var_os("MEMORY_DATA_DIR").map_or_else(|| root().join("data"), PathBuf::from)
}

/// The SQLite file (`MEMORY_DB_NAME` is a file name inside the data dir).
pub fn db_path() -> PathBuf {
    let name = env("MEMORY_DB_NAME", "llm_long_term_memory.db");
    data_dir().join(PathBuf::from(name).file_name().map_or_else(|| "llm_long_term_memory.db".into(), PathBuf::from))
}

/// The user-editable seed scenario.
pub fn seed_csv_path() -> PathBuf {
    data_dir().join("seed.csv")
}

/// `KEY=VALUE` lines of `secrets/.env`, then `./.env` (the first definition wins).
fn dotenv() -> &'static HashMap<String, String> {
    static VARS: OnceLock<HashMap<String, String>> = OnceLock::new();
    VARS.get_or_init(|| {
        let mut vars = HashMap::new();
        for file in [root().join("secrets").join(".env"), root().join(".env")] {
            for line in std::fs::read_to_string(file).unwrap_or_default().lines() {
                let line = line.trim();
                let line = line.strip_prefix("export ").unwrap_or(line);
                if let Some((k, v)) = line.split_once('=').filter(|_| !line.starts_with('#')) {
                    let v = v.trim();
                    let v = v.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(v);
                    vars.entry(k.trim().to_owned()).or_insert_with(|| v.to_owned());
                }
            }
        }
        vars
    })
}

/// A process environment variable, else a `.env` entry, else `default`.
pub fn env(key: &str, default: &str) -> String {
    std::env::var(key).ok().or_else(|| dotenv().get(key).cloned()).unwrap_or_else(|| default.to_owned())
}

params! {
    /// App-level settings (environment defaults, editable in the UI).
    pub struct Settings {
        /// Turn log kept in RAM and in the DB.
        max_turn_log: usize = 1000,
        /// Metrics rows kept in RAM.
        max_metrics_history: usize = 1000,
        /// IANA timezone stamped onto memories.
        default_timezone: String = env("MEMORY_TZ", "Asia/Tokyo"),
        /// GGUF file (relative paths resolve inside `./model`), or the model name at `embedding_url`.
        embedding_model: String = env("EMBEDDING_MODEL", "embeddinggemma-300m-qat-Q4_0.gguf"),
        /// An OpenAI-compatible embeddings endpoint; empty = run the GGUF in-process (llama.cpp).
        embedding_url: String = env("EMBEDDING_URL", ""),
        /// The model's query prompt (EmbeddingGemma's by default; "query: " for E5, "" for symmetric models).
        embed_query_prefix: String = env("EMBED_QUERY_PREFIX", "task: search result | query: "),
        /// The model's document prompt.
        embed_document_prefix: String = env("EMBED_DOCUMENT_PREFIX", "title: none | text: "),
        /// "claude-code" (Claude via the Claude Code CLI, answers cached per prompt),
        /// "deepseek" (any OpenAI-compatible endpoint) or "gemini".
        llm_provider: String = env("LLM_PROVIDER", "claude-code"),
        /// Claude model used through Claude Code.
        claude_model: String = env("CLAUDE_MODEL", "claude-haiku-5-5"),
        /// The Claude Code executable.
        claude_cli: String = env("CLAUDE_CLI", "claude"),
        /// Prompt-keyed answer cache: a replayed scenario asks the model only once.
        llm_cache: String = env("LLM_CACHE", "data/llm_cache.json"),
        deepseek_model: String = env("DEEPSEEK_MODEL", "deepseek-flash"),
        deepseek_base_url: String = env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        gemini_model: String = env("GEMINI_MODEL", "gemini-3.5-flash"),
        temperature: f64 = 0.7,
        /// Reasoning models spend thousands of tokens before answering; too low a cap
        /// returns an empty body (finish_reason=length).
        max_output_tokens: usize = 8192,
        /// Tool saves applied per turn.
        max_writes_per_turn: usize = 8,
        /// Extract facts with one extra call when a turn saved nothing through tools.
        tool_fallback: bool = true,
    }
}

/// Settings + engine parameters, as the UI edits them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AppConfig {
    pub glob: Settings,
    pub memory: Config,
}

impl AppConfig {
    /// `{"glob": {...}, "memory": {...}}`.
    pub fn to_json(&self) -> Value {
        json!({"glob": self.glob.to_json(), "memory": self.memory.to_json()})
    }

    /// Defaults overwritten by the sections of `v`; rejects invalid engine parameters.
    /// Which executable runs (`claude_cli`) and where the answer cache is written
    /// (`llm_cache`) come from the environment only, never from a request body.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let mut cfg = Self::default();
        let (cli, cache) = (cfg.glob.claude_cli.clone(), cfg.glob.llm_cache.clone());
        cfg.glob.merge_json(&v["glob"]);
        (cfg.glob.claude_cli, cfg.glob.llm_cache) = (cli, cache);
        cfg.memory.merge_json(&v["memory"]);
        // A window of 0 would keep nothing, not everything.
        cfg.glob.max_turn_log = cfg.glob.max_turn_log.max(1);
        cfg.glob.max_metrics_history = cfg.glob.max_metrics_history.max(1);
        cfg.memory.validate()?;
        Ok(cfg)
    }
}
