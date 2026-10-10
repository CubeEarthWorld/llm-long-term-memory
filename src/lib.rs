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

pub use config::Config;
pub use dream::{Adjudicator, DreamDecision, DreamMember, DreamRequest, DREAM_SYSTEM_PROMPT};
pub use embed::Embedder;
pub use engine::{DreamAction, DreamReport, Engram, Recall, Recalled, Remembered, Stats};
pub use id::{ulid_ids, IdGen};
pub use json::relaxed_json;
pub use store::{MemoryStore, Store, Write};
pub use text::cues;
pub use time::{fmt_local, system_clock, Clock, ManualClock};
pub use trace::Trace;

/// Errors from a store, an embedder or an adjudicator: any error type, boxed.
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// `Result` with the crate's boxed [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
