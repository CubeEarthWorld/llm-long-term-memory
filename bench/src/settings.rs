//! Tool configuration: [`Settings`] (embedding model, Claude Code, timezone, turn policy)
//! plus the engine's [`Config`], bundled as [`AppConfig`]; paths and the `.env` lookup.

use engram::Config;
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

/// Tool settings, from the environment (or `.env`).
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Turn log kept in RAM and in the DB.
    pub max_turn_log: usize,
    /// Metrics rows kept in RAM.
    pub max_metrics_history: usize,
    /// IANA timezone stamped onto memories.
    pub default_timezone: String,
    /// GGUF file (relative paths resolve inside `./model`).
    pub embedding_model: String,
    /// The model's query prompt (EmbeddingGemma's by default; "query: " for E5, "" for symmetric models).
    pub embed_query_prefix: String,
    /// The model's document prompt.
    pub embed_document_prefix: String,
    /// Claude model used through Claude Code.
    pub claude_model: String,
    /// The Claude Code executable.
    pub claude_cli: String,
    /// Prompt-keyed answer cache: a replayed scenario asks the model only once.
    pub llm_cache: String,
    /// Tool saves applied per turn.
    pub max_writes_per_turn: usize,
    /// Extract facts with one extra call when a turn saved nothing through tools.
    pub tool_fallback: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_turn_log: 1000,
            max_metrics_history: 1000,
            default_timezone: env("MEMORY_TZ", "Asia/Tokyo"),
            embedding_model: env("EMBEDDING_MODEL", "embeddinggemma-300m-qat-Q4_0.gguf"),
            embed_query_prefix: env("EMBED_QUERY_PREFIX", "task: search result | query: "),
            embed_document_prefix: env("EMBED_DOCUMENT_PREFIX", "title: none | text: "),
            claude_model: env("CLAUDE_MODEL", "claude-haiku-5-5"),
            claude_cli: env("CLAUDE_CLI", "claude"),
            llm_cache: env("LLM_CACHE", "data/llm_cache.json"),
            max_writes_per_turn: 8,
            tool_fallback: true,
        }
    }
}

/// Settings + engine parameters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AppConfig {
    pub glob: Settings,
    pub memory: Config,
}
