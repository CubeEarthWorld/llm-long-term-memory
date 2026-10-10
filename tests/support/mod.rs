//! Deterministic doubles shared by the integration tests: a token-overlap embedder
//! bit-for-bit identical to the Dart package's (FNV-1a token seeds → splitmix64
//! streams), failing embedders, canned adjudicators and an engine factory.
#![allow(dead_code)]

pub mod workload;

use engram::{Config, DreamDecision, DreamRequest, Embedder, Engram, ManualClock, MemoryStore, Result, Store, Trace, Write};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const DAY: f64 = 86_400.0;
pub const START: f64 = 1_700_000_000.0;

/// The suite's base configuration: the fake embedder is centred at zero.
pub fn cfg() -> Config {
    Config { cosine_floor: 0.0, ..Config::default() }
}

/// Lowercased `[a-z0-9]+` words, then every kana/CJK/half-width-katakana character.
pub fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    let mut word = String::new();
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            word.push(c);
        } else if !word.is_empty() {
            out.push(std::mem::take(&mut word));
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out.extend(lower.chars().filter(|&c| ('\u{3040}'..='\u{9FFF}').contains(&c) || ('\u{FF66}'..='\u{FF9D}').contains(&c)).map(String::from));
    out
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3))
}

fn splitmix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// `cos ≈ shared / √(n1·n2)` for texts with overlapping token sets.
pub struct FakeEmbedder {
    pub dimension: usize,
    pub model_id: String,
    cache: HashMap<String, Vec<f64>>,
}

impl FakeEmbedder {
    pub fn new() -> Self {
        Self::with(64, "fake/token-overlap")
    }

    pub fn with(dimension: usize, model_id: &str) -> Self {
        Self { dimension, model_id: model_id.into(), cache: HashMap::new() }
    }

    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        let toks = tokens(text);
        let dim = self.dimension;
        let mut v = vec![if toks.is_empty() { 1.0 } else { 0.0 }; dim];
        for t in &toks {
            let tv = self.cache.entry(t.clone()).or_insert_with(|| {
                let mut state = fnv1a(t);
                (0..dim)
                    .map(|_| {
                        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                        (splitmix(state) >> 11) as f64 / 9_007_199_254_740_992.0 * 2.0 - 1.0
                    })
                    .collect()
            });
            for (a, b) in v.iter_mut().zip(tv.iter()) {
                *a += b;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter().map(|&x| (if norm == 0.0 { x } else { x / norm }) as f32).collect()
    }
}

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed(t)).collect())
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_documents(texts)
    }
}

/// An unavailable model.
pub struct BrokenEmbedder;

impl Embedder for BrokenEmbedder {
    fn model_id(&self) -> &str {
        "fake/broken"
    }
    fn dimension(&self) -> usize {
        64
    }
    fn embed_documents(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err("embedder offline".into())
    }
    fn embed_queries(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err("embedder offline".into())
    }
}

/// Records every query batch; can fail them, or mis-size one cue.
pub struct CountingEmbedder {
    pub inner: FakeEmbedder,
    pub offline_queries: bool,
    pub bad_cue: Option<String>,
    pub batches: Arc<Mutex<Vec<Vec<String>>>>,
}

impl CountingEmbedder {
    pub fn new(offline_queries: bool, bad_cue: Option<&str>) -> (Self, Arc<Mutex<Vec<Vec<String>>>>) {
        let batches = Arc::new(Mutex::new(Vec::new()));
        (Self { inner: FakeEmbedder::new(), offline_queries, bad_cue: bad_cue.map(Into::into), batches: batches.clone() }, batches)
    }
}

impl Embedder for CountingEmbedder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn dimension(&self) -> usize {
        self.inner.dimension
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(texts)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.batches.lock().unwrap().push(texts.iter().map(|t| t.to_string()).collect());
        if self.offline_queries {
            return Err("embedder offline".into());
        }
        let half = self.inner.dimension / 2;
        Ok(texts.iter().map(|t| if Some(*t) == self.bad_cue.as_deref() { vec![0.0; half] } else { self.inner.embed(t) }).collect())
    }
}

/// Merges the cluster into one gist (texts joined), absorbing every candidate.
pub fn merge_to_gist(r: &DreamRequest) -> Result<DreamDecision> {
    Ok(DreamDecision {
        memories: vec![r.members.iter().map(|m| m.text.as_str()).collect::<Vec<_>>().join(" / ")],
        absorbed_ids: r.members[1..].iter().map(|m| m.id.clone()).collect(),
    })
}

pub fn always_keep(_: &DreamRequest) -> Result<DreamDecision> {
    Ok(DreamDecision::keep())
}

pub fn always_fail(_: &DreamRequest) -> Result<DreamDecision> {
    Err("llm down".into())
}

/// An engine over `store` (default: in RAM) with a virtual clock at [`START`].
/// A [`MemoryStore`] that outlives an engine: clones share the rows, so a test can
/// reopen "the same database" or write to it behind the engine's back.
#[derive(Clone, Default)]
pub struct SharedStore(Arc<Mutex<MemoryStore>>);

impl Store for SharedStore {
    fn load_all(&self) -> Result<Vec<Trace>> {
        self.0.lock().unwrap().load_all()
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.0.lock().unwrap().write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        self.0.lock().unwrap().clear()
    }
}

pub fn build_with(cfg: Config, embedder: Box<dyn Embedder>, store: Box<dyn Store>) -> (Engram, ManualClock) {
    let clock = ManualClock::new(START);
    let engine = Engram::with_clock(store, embedder, cfg, "Asia/Tokyo", clock.clock()).expect("engine");
    (engine, clock)
}

pub fn build(cfg: Config) -> (Engram, ManualClock) {
    build_with(cfg, Box::new(FakeEmbedder::new()), Box::new(MemoryStore::default()))
}

/// Inserts `text` and returns the new trace's id (panics on any other outcome).
pub fn insert(m: &mut Engram, text: &str, salience: f64, cue: &str) -> String {
    match m.remember(text, salience, cue).expect("remember") {
        engram::Remembered::Inserted { trace, .. } => trace.id,
        other => panic!("expected an insert, got {other:?}"),
    }
}

/// A unique temporary directory for one test.
pub fn temp_dir(name: &str) -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "engram-test-{}-{}-{}",
        std::process::id(),
        name,
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Structural JSON equality where numbers (or numeric strings, as the Dart traces
/// write them) are compared as exact f64s. Returns the first differing path.
pub fn same_json(a: &serde_json::Value, b: &serde_json::Value, path: &str) -> std::result::Result<(), String> {
    use serde_json::Value;
    let num = |v: &Value| match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok().filter(|_| s.contains(|c: char| c.is_ascii_digit())),
        _ => None,
    };
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Err(format!("{path}: length {} vs {}", x.len(), y.len()));
            }
            x.iter().zip(y).enumerate().try_for_each(|(i, (p, q))| same_json(p, q, &format!("{path}[{i}]")))
        }
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                same_json(x.get(k).unwrap_or(&Value::Null), y.get(k).unwrap_or(&Value::Null), &format!("{path}.{k}"))?;
            }
            Ok(())
        }
        _ => match (num(a), num(b)) {
            (Some(p), Some(q)) if p == q => Ok(()),
            _ if a == b => Ok(()),
            _ => Err(format!("{path}: {a} vs {b}")),
        },
    }
}
