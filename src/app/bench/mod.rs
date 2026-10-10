//! Benchmarks on the real corpus (`bench/corpus`, `bench/updates`) and synthetic loads.
//!
//! * `forget`     — decay + capacity eviction against FIFO / LRU / random / unbounded arms
//! * `sweep`      — the 155-run grid of `forget` (sharded), and `--report`
//! * `supersede`  — overwrite forgetting through the dream (`--analyze` / `--run`, real LLM)
//! * `generality` — embedding models and dimensions, with and without calibration
//! * `cache`      — embed the corpus once with the configured model (`bench/embeddings*.bin`)
//! * `perf`       — latency and memory of every operation on synthetic 768-d vectors
//!
//! Measurements run through [`crate::Engram::peek`], so probing never strengthens what it measures.

mod forget;
mod generality;
mod perf;
mod supersede;
mod sweep;

use super::settings::{root, AppConfig};
use crate::{cues, Embedder, Result};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DAY: f64 = 86_400.0;

/// `ltm bench <name> …`. Returns the process exit code.
pub fn main(args: &[String]) -> i32 {
    let rest = args.get(1..).unwrap_or_default();
    let result = match args.first().map(String::as_str) {
        Some("forget") => forget::main(rest),
        Some("sweep") => sweep::main(rest),
        Some("supersede") => supersede::main(rest),
        Some("generality") => generality::main(rest),
        Some("cache") => cache_main(rest),
        Some("perf") => perf::main(rest),
        _ => Err("usage: ltm bench <forget|sweep|supersede|generality|cache|perf> [--help]".into()),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

pub fn bench_dir() -> PathBuf {
    root().join("bench")
}

/// `--flag value` pairs and bare `--switch`es.
pub struct Flags(HashMap<String, String>);

impl Flags {
    pub fn parse(args: &[String]) -> Self {
        let mut m = HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let k = args[i].trim_start_matches("--").to_owned();
            if args.get(i + 1).is_some_and(|v| !v.starts_with("--")) {
                m.insert(k, args[i + 1].clone());
                i += 2;
            } else {
                m.insert(k, String::new());
                i += 1;
            }
        }
        Self(m)
    }
    pub fn has(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
    pub fn str(&self, k: &str, default: &str) -> String {
        self.0.get(k).filter(|v| !v.is_empty()).cloned().unwrap_or_else(|| default.to_owned())
    }
    pub fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        self.0.get(k).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
    /// A whitespace- or comma-separated list.
    pub fn list<T: std::str::FromStr>(&self, k: &str) -> Vec<T> {
        self.0.get(k).map_or_else(Vec::new, |v| v.split([',', ' ']).filter_map(|x| x.parse().ok()).collect())
    }
}

// ------------------------------------------------------------------ corpus

#[derive(Clone, Debug)]
pub struct Gold {
    pub fact: String,
    pub probes: Vec<String>,
    pub distractor: String,
}

pub struct Corpus {
    pub gold: Vec<Gold>,
    pub noise: Vec<String>,
}

fn json_files(dir: &Path) -> Vec<Value> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect())
        .unwrap_or_default();
    files.sort();
    files.iter().filter_map(|f| serde_json::from_str(&std::fs::read_to_string(f).ok()?).ok()).collect()
}

fn s(v: &Value) -> String {
    v.as_str().map_or_else(String::new, |x| x.trim().to_owned())
}

/// The shards merged, duplicate texts dropped (first wins), and any distractor that is
/// itself a gold fact cleared.
pub fn load_corpus(dir: &Path) -> Corpus {
    let (mut gold, mut noise, mut seen) = (Vec::new(), Vec::new(), HashSet::new());
    for shard in json_files(dir) {
        for g in shard["gold"].as_array().into_iter().flatten() {
            let fact = s(&g["fact"]);
            let probes: Vec<String> = g["probes"].as_array().into_iter().flatten().map(s).take(2).collect();
            if !fact.is_empty() && !probes.is_empty() && seen.insert(fact.clone()) {
                gold.push(Gold { fact, probes, distractor: s(&g["distractor"]) });
            }
        }
        for n in shard["noise"].as_array().into_iter().flatten().map(s) {
            if !n.is_empty() && seen.insert(n.clone()) {
                noise.push(n);
            }
        }
    }
    let facts: HashSet<String> = gold.iter().map(|g| g.fact.clone()).collect();
    for g in &mut gold {
        if facts.contains(&g.distractor) {
            g.distractor.clear();
        }
    }
    Corpus { gold, noise }
}

/// An old → new fact update with the probe that should now find the new one, and the
/// cue each side's `save_memory` would have written.
#[derive(Clone, Debug)]
pub struct Update {
    pub old: String,
    pub new: String,
    pub probe: String,
    pub cue_old: String,
    pub cue_new: String,
}

pub fn load_updates(dir: &Path) -> Vec<Update> {
    json_files(dir)
        .iter()
        .flat_map(|f| f["pairs"].as_array().cloned().unwrap_or_default())
        .map(|p| Update { old: s(&p["old"]), new: s(&p["new"]), probe: s(&p["probe"]), cue_old: s(&p["cue_old"]), cue_new: s(&p["cue_new"]) })
        .filter(|u| !u.old.is_empty() && !u.new.is_empty() && !u.probe.is_empty() && u.old != u.new)
        .collect()
}

/// Every document text and every query cue the forgetting benchmark embeds.
fn corpus_texts(c: &Corpus, max_cues: usize) -> (Vec<String>, Vec<String>) {
    let mut docs: Vec<String> = c.gold.iter().map(|g| g.fact.clone()).collect();
    docs.extend(c.gold.iter().filter(|g| !g.distractor.is_empty()).map(|g| g.distractor.clone()));
    docs.extend(c.noise.iter().cloned());
    let queries: Vec<String> = c.gold.iter().flat_map(|g| g.probes.iter().flat_map(|p| cues(p, max_cues))).collect();
    let dedup = |mut v: Vec<String>| {
        v.sort();
        v.dedup();
        v
    };
    (dedup(docs), dedup(queries))
}

// ------------------------------------------------------------- embeddings

/// Precomputed vectors served without a model (`ltm bench cache` writes them). Clones
/// share the tables.
#[derive(Clone)]
pub struct CachedEmbedder {
    model_id: String,
    dimension: usize,
    docs: Arc<HashMap<String, Vec<f32>>>,
    queries: Arc<HashMap<String, Vec<f32>>>,
}

const CACHE_MAGIC: &[u8] = b"ENGC1\n";

impl CachedEmbedder {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e} — build it with `ltm bench cache`", path.display()))?;
        let mut at = CACHE_MAGIC.len();
        if !bytes.starts_with(CACHE_MAGIC) {
            return Err(format!("{} is not an embedding cache", path.display()).into());
        }
        let u32_at = |at: &mut usize| -> Result<usize> {
            let b = bytes.get(*at..*at + 4).ok_or("truncated cache")?;
            *at += 4;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        };
        let text = |at: &mut usize| -> Result<String> {
            let n = u32_at(at)?;
            let b = bytes.get(*at..*at + n).ok_or("truncated cache")?;
            *at += n;
            Ok(String::from_utf8_lossy(b).into_owned())
        };
        let model_id = text(&mut at)?;
        let (dimension, n_docs, n_queries) = (u32_at(&mut at)?, u32_at(&mut at)?, u32_at(&mut at)?);
        let table = |n: usize, at: &mut usize| -> Result<HashMap<String, Vec<f32>>> {
            (0..n)
                .map(|_| {
                    let k = text(at)?;
                    let b = bytes.get(*at..*at + 4 * dimension).ok_or("truncated cache")?;
                    *at += 4 * dimension;
                    Ok((k, b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()))
                })
                .collect()
        };
        let docs = table(n_docs, &mut at)?;
        let queries = table(n_queries, &mut at)?;
        Ok(Self { model_id, dimension, docs: Arc::new(docs), queries: Arc::new(queries) })
    }

    pub fn save(path: &Path, model_id: &str, dimension: usize, docs: &[(String, Vec<f32>)], queries: &[(String, Vec<f32>)]) -> Result<()> {
        let mut out = CACHE_MAGIC.to_vec();
        let put_text = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u32).to_le_bytes());
            out.extend(s.as_bytes());
        };
        put_text(&mut out, model_id);
        for n in [dimension, docs.len(), queries.len()] {
            out.extend((n as u32).to_le_bytes());
        }
        for (k, v) in docs.iter().chain(queries) {
            put_text(&mut out, k);
            out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        std::fs::write(path, out)?;
        Ok(())
    }

    fn get(table: &HashMap<String, Vec<f32>>, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| table.get(*t).cloned().ok_or_else(|| format!("embedding cache miss: {t:?} — rebuild with `ltm bench cache`").into())).collect()
    }
}

impl Embedder for CachedEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Self::get(&self.docs, texts)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Self::get(&self.queries, texts)
    }
}

/// Matryoshka truncation: the first `dimension` components, renormalised. The model id
/// carries the dimension, so stored vectors of another length are re-embedded.
pub struct Truncated<E> {
    pub base: E,
    pub dimension: usize,
    model_id: String,
}

impl<E: Embedder> Truncated<E> {
    pub fn new(base: E, dimension: usize) -> Self {
        let model_id = format!("{}#d{dimension}", base.model_id());
        Self { base, dimension, model_id }
    }
    fn cut(&self, vs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        vs.into_iter().map(|v| crate::vector::normalized(&v[..self.dimension.min(v.len())])).collect()
    }
}

impl<E: Embedder> Embedder for Truncated<E> {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let v = self.base.embed_documents(texts)?;
        Ok(self.cut(v))
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let v = self.base.embed_queries(texts)?;
        Ok(self.cut(v))
    }
}

/// The default cache path for a model file (`embeddings.bin` for the default model).
pub fn cache_path(model: &str) -> PathBuf {
    let default = AppConfig::default().glob.embedding_model;
    let stem = Path::new(model).file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    bench_dir().join(if model == default { "embeddings.bin".to_owned() } else { format!("embeddings_{stem}.bin") })
}

/// Embeds the corpus once with `glob`'s model into `path`.
pub fn build_cache(corpus: &Corpus, cfg: &AppConfig, path: &Path) -> Result<()> {
    let (mut e, _) = super::embedder(&cfg.glob)?;
    let (docs, queries) = corpus_texts(corpus, cfg.memory.max_cues);
    println!("embedding {} docs + {} queries with {} …", docs.len(), queries.len(), e.model_id());
    let t0 = std::time::Instant::now();
    let embed = |e: &mut Box<dyn Embedder>, texts: &[String], query: bool| -> Result<Vec<(String, Vec<f32>)>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(64) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            let vs = if query { e.embed_queries(&refs)? } else { e.embed_documents(&refs)? };
            out.extend(chunk.iter().cloned().zip(vs.iter().map(|v| crate::vector::normalized(v))));
        }
        Ok(out)
    };
    let d = embed(&mut e, &docs, false)?;
    let q = embed(&mut e, &queries, true)?;
    CachedEmbedder::save(path, e.model_id(), e.dimension(), &d, &q)?;
    println!("wrote {} ({:.1} MB, {:.0}s)", path.display(), std::fs::metadata(path)?.len() as f64 / 1e6, t0.elapsed().as_secs_f64());
    Ok(())
}

fn cache_main(args: &[String]) -> Result<()> {
    let f = Flags::parse(args);
    let mut cfg = AppConfig::default();
    if f.has("model") {
        cfg.glob.embedding_model = f.str("model", "");
    }
    if f.has("query-prefix") {
        cfg.glob.embed_query_prefix = f.str("query-prefix", "");
    }
    if f.has("document-prefix") {
        cfg.glob.embed_document_prefix = f.str("document-prefix", "");
    }
    let path = f.0.get("out").filter(|o| !o.is_empty()).map_or_else(|| cache_path(&cfg.glob.embedding_model), PathBuf::from);
    build_cache(&load_corpus(&bench_dir().join("corpus")), &cfg, &path)
}

// ------------------------------------------------------------------ helpers

/// A Mersenne Twister (MT19937 seeded with `init_by_array`, rejection-sampled
/// `randrange`, Fisher–Yates `shuffle`): the benchmark timelines are fixed per seed.
pub struct PyRandom {
    mt: [u32; 624],
    idx: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> Self {
        let mut key: Vec<u32> = Vec::new();
        let mut s = seed;
        while s > 0 {
            key.push(s as u32);
            s >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = Self { mt: [0; 624], idx: 624 };
        r.mt[0] = 19_650_218;
        for i in 1..624 {
            r.mt[i] = 1_812_433_253u32.wrapping_mul(r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1_664_525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
        }
        r.mt[0] = 0x8000_0000;
        r
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7FFF_FFFF);
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 == 1 { 0x9908_B0DF } else { 0 };
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }

    /// A uniform integer in `[0, n)` (`n < 2^32`).
    pub fn randrange(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = (self.next_u32() >> (32 - k)) as usize;
            if r < n {
                return r;
            }
        }
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.randrange(i + 1);
            v.swap(i, j);
        }
    }
}

/// Percentile by linear interpolation between closest ranks, `q` in [0, 100].
pub fn percentile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return f64::NAN;
    }
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

pub fn r4(x: f64) -> f64 {
    super::round_to(x, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_reference_sequence() {
        let mut r = PyRandom::new(1);
        let mut x: Vec<u32> = (0..10).collect();
        r.shuffle(&mut x);
        assert_eq!(x, [6, 8, 9, 7, 5, 3, 0, 4, 1, 2]);
        assert_eq!((0..5).map(|_| r.randrange(1000)).collect::<Vec<_>>(), [807, 214, 96, 499, 29]);
        assert_eq!(r.next_u32(), 3_836_767_462);
        let mut r = PyRandom::new(7);
        assert_eq!((0..3).map(|_| r.randrange(1 << 30)).collect::<Vec<_>>(), [695_425_564, 323_946_139, 847_876_999]);
        let mut r = PyRandom::new((1 << 40) + 5);
        assert_eq!((0..5).map(|_| r.randrange(97)).collect::<Vec<_>>(), [64, 66, 34, 84, 3]);
    }

    #[test]
    fn percentile_is_linear() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.5);
        assert_eq!(percentile(&[5.0], 90.0), 5.0);
    }
}
