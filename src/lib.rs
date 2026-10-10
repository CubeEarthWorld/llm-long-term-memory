//! ENGRAM v2.1 — a trace-based long-term memory for LLM applications (see `SPEC.md`).
//!
//! The engine ([`Engram`]) holds every trace in RAM and talks to three injected
//! seams: a durable [`Store`], an [`Embedder`] (text → unit vectors) and — for
//! [`Engram::dream`] only — an [`Adjudicator`] backed by any LLM. Everything else
//! is arithmetic over the forgetting curve and embedding geometry.
//!
//! ```
//! use engram::{Config, Engram, MemoryStore, Remembered};
//! # use engram::Embedder;
//! # struct Hash;
//! # impl Embedder for Hash {
//! #     fn model_id(&self) -> &str { "hash" }
//! #     fn dimension(&self) -> usize { 2 }
//! #     fn embed_documents(&mut self, t: &[&str]) -> engram::Result<Vec<Vec<f32>>> {
//! #         Ok(t.iter().map(|s| vec![s.len() as f32, 1.0]).collect())
//! #     }
//! #     fn embed_queries(&mut self, t: &[&str]) -> engram::Result<Vec<Vec<f32>>> { self.embed_documents(t) }
//! # }
//! let mut memory = Engram::open(Box::new(MemoryStore::default()), Box::new(Hash), Config::default(), "UTC")?;
//! let Remembered::Inserted { trace, .. } = memory.remember("the user lives in Kyoto", 1.0, "")? else { panic!() };
//! assert_eq!(memory.memory(&trace.id).unwrap().text, "the user lives in Kyoto");
//! # Ok::<(), engram::Error>(())
//! ```

mod config;
mod dream;
mod embed;
mod engine;
mod id;
mod json;
mod store;
mod text;
mod time;
mod trace;
mod vector;

#[cfg(feature = "app")]
pub mod app;
#[cfg(feature = "sqlite")]
mod sqlite;

pub use config::{Config, Param};
pub use dream::{Adjudicator, DreamDecision, DreamMember, DreamRequest, DREAM_SYSTEM_PROMPT};
pub use embed::Embedder;
#[cfg(feature = "http")]
pub use embed::http::HttpEmbedder;
#[cfg(feature = "llama")]
pub use embed::llama::LlamaEmbedder;
pub use engine::{DreamAction, DreamReport, Engram, Recall, Recalled, Remembered, Stats};
pub use id::{deterministic_ids, ulid};
pub use json::{json_texts, relaxed_json};
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStore;
pub use store::{MemoryStore, Store, Write};
pub use text::{clean_text, cues, shorten};
pub use time::{fmt_local, parse_offset, system_clock, tz_field, Clock, ManualClock};
pub use trace::Trace;

/// Worker threads this crate may use: 80% of the logical CPUs (at least one), so the
/// machine stays responsive while a model or a benchmark runs.
pub fn cpu_budget() -> usize {
    std::thread::available_parallelism().map_or(1, |n| (n.get() * 8 / 10).max(1))
}

/// Errors from a store, an embedder or an adjudicator: any error type, boxed.
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// `Result` with the crate's boxed [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
