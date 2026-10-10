//! The local app around the engine: settings, the LLM client, the per-turn runner,
//! the session (turn log, seed replay, dream), the web server, the CLI and benchmarks.

pub mod bench;
pub mod claude;
pub mod cli;
pub mod llm;
pub mod seed;
pub mod server;
pub mod session;
pub mod settings;
pub mod turn;

use crate::{Embedder, HttpEmbedder, LlamaEmbedder, Result};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const SYSTEM_ID: &str = "llm_long_term_memory";
pub const TITLE: &str = "LLM Long-Term Memory";

/// `x` rounded to `digits` decimals (display only). The binary value is rounded
/// exactly (ties to even on the true decimal expansion), so 0.81875 — stored just below
/// — reads 0.8187, not the 0.8188 that scaling by 10^digits would give.
pub fn round_to(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().unwrap_or(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rounding_is_exact() {
        assert_eq!(super::round_to(0.81875, 4), 0.8187);
        assert_eq!(super::round_to(2.5, 0), 2.0);
        assert_eq!(super::round_to(1.23456, 3), 1.235);
    }
}

/// Time spent inside an embedder, shared between the engine's embedder and the app.
#[derive(Clone, Debug, Default)]
pub struct EmbedTimer(Arc<AtomicU64>);

impl EmbedTimer {
    /// Milliseconds accumulated so far.
    pub fn peek_ms(&self) -> f64 {
        self.0.load(Ordering::Relaxed) as f64 / 1e6
    }
    /// Milliseconds accumulated, then reset.
    pub fn take_ms(&self) -> f64 {
        self.0.swap(0, Ordering::Relaxed) as f64 / 1e6
    }
    fn add(&self, since: Instant) {
        self.0.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// An embedder that adds its call time to an [`EmbedTimer`].
pub struct Timed<E> {
    pub inner: E,
    pub timer: EmbedTimer,
}

impl<E: Embedder> Embedder for Timed<E> {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let t0 = Instant::now();
        let out = self.inner.embed_documents(texts);
        self.timer.add(t0);
        out
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let t0 = Instant::now();
        let out = self.inner.embed_queries(texts);
        self.timer.add(t0);
        out
    }
}

/// Runs `f` and splits its wall time into (result, non-embedding ms, embedding ms).
pub fn timed<T>(timer: &EmbedTimer, f: impl FnOnce() -> T) -> (T, f64, f64) {
    timer.take_ms();
    let t0 = Instant::now();
    let out = f();
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let embed = timer.take_ms();
    (out, (wall - embed).max(0.0), embed)
}

/// The embedder used while the configured one cannot be loaded: every call fails, which
/// the engine treats as offline (writes stay unindexed, recall falls back to recency).
pub struct Offline;

impl Embedder for Offline {
    fn model_id(&self) -> &str {
        ""
    }
    fn dimension(&self) -> usize {
        0
    }
    fn embed_documents(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err("no embedding model".into())
    }
    fn embed_queries(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err("no embedding model".into())
    }
}

/// The configured embedder and its status line. A loaded model is cached per process
/// (keyed by source and prompts) and shared by handle, so rebuilding a session with the
/// same embedding settings does not reload it.
pub fn embedder(s: &settings::Settings) -> Result<(Box<dyn Embedder>, String)> {
    #[derive(Clone)]
    enum Loaded {
        Local(LlamaEmbedder),
        Http(HttpEmbedder),
    }
    static CACHE: Mutex<Option<(Vec<String>, Loaded)>> = Mutex::new(None);
    let key = vec![s.embedding_url.clone(), s.embedding_model.clone(), s.embed_query_prefix.clone(), s.embed_document_prefix.clone()];
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let loaded = match cache.as_ref() {
        Some((k, l)) if *k == key => l.clone(),
        _ => {
            let l = if s.embedding_url.is_empty() {
                let path = Path::new(&s.embedding_model);
                let path = if path.is_absolute() { path.to_path_buf() } else { settings::root().join("model").join(path) };
                Loaded::Local(LlamaEmbedder::load(&path, &s.embed_query_prefix, &s.embed_document_prefix)?)
            } else {
                let api_key = settings::env("EMBEDDING_API_KEY", "");
                Loaded::Http(HttpEmbedder::connect(&s.embedding_url, &s.embedding_model, &api_key, &s.embed_query_prefix, &s.embed_document_prefix)?)
            };
            *cache = Some((key, l.clone()));
            l
        }
    };
    Ok(match loaded {
        Loaded::Local(e) => {
            let status = format!("OK — {} (dim={}, llama.cpp)", e.model_id(), e.dimension());
            (Box::new(e), status)
        }
        Loaded::Http(e) => {
            let status = format!("OK — {} (dim={}, {})", e.model_id(), e.dimension(), s.embedding_url);
            (Box::new(e), status)
        }
    })
}
