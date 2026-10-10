//! Latency and memory of every engine operation on synthetic unit vectors (every 10th
//! fact a near-duplicate of the previous one, so dreams have labile pairs), over a
//! file-backed SQLite store as in the app. `--vectors` replays a recorded dataset
//! (`u32 dim, u32 count, then u32 len + UTF-8 text + dim f32` per entry).
//!
//! `ltm bench perf [--n 10000] [--dim 768] [--vectors FILE]`

use super::{Flags, PyRandom};
use crate::{Config, DreamDecision, Embedder, Engram, ManualClock, Result, SqliteStore};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Serves precomputed vectors by text.
#[derive(Clone)]
struct Table(Arc<HashMap<String, Vec<f32>>>, usize);

impl Embedder for Table {
    fn model_id(&self) -> &str {
        "synthetic"
    }
    fn dimension(&self) -> usize {
        self.1
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.0.get(*t).cloned().ok_or_else(|| format!("no vector for {t:?}").into())).collect()
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_documents(texts)
    }
}

/// Standard normal deviates (Box–Muller over MT19937).
fn gaussian(rng: &mut PyRandom) -> f64 {
    let u = |rng: &mut PyRandom| (rng.randrange(1 << 30) as f64 + 0.5) / f64::from(1u32 << 30);
    let (a, b) = (u(rng), u(rng));
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

fn synthetic(n: usize, dim: usize) -> (Vec<String>, Vec<String>, HashMap<String, Vec<f32>>) {
    let mut rng = PyRandom::new(42);
    let unit = |v: Vec<f64>| crate::vector::normalized(&v.iter().map(|&x| x as f32).collect::<Vec<_>>());
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let v = if i % 10 == 9 {
            vecs[i - 1].iter().map(|&x| f64::from(x) + 0.012 * gaussian(&mut rng)).collect()
        } else {
            (0..dim).map(|_| gaussian(&mut rng)).collect()
        };
        vecs.push(unit(v));
    }
    let texts: Vec<String> = (0..n).map(|i| format!("fact number {i} about topic {}", i % 97)).collect();
    let queries: Vec<String> = (0..50).map(|j| format!("query {j}")).collect();
    let mut table: HashMap<String, Vec<f32>> = texts.iter().cloned().zip(vecs.iter().cloned()).collect();
    for (j, q) in queries.iter().enumerate() {
        let base = &vecs[(j * 37) % (n - 300)];
        table.insert(q.clone(), unit(base.iter().map(|&x| f64::from(x) + 0.03 * gaussian(&mut rng)).collect()));
    }
    (texts, queries, table)
}

/// Texts, queries, their vectors and the dimension.
type Dataset = (Vec<String>, Vec<String>, HashMap<String, Vec<f32>>, usize);

fn recorded(path: &str) -> Result<Dataset> {
    let b = std::fs::read(path)?;
    let u32_at = |at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
    let (dim, count) = (u32_at(0), u32_at(4));
    let (mut at, mut keys, mut table) = (8, Vec::new(), HashMap::new());
    for _ in 0..count {
        let len = u32_at(at);
        let key = String::from_utf8_lossy(&b[at + 4..at + 4 + len]).into_owned();
        at += 4 + len;
        let v = b[at..at + 4 * dim].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        at += 4 * dim;
        keys.push(key.clone());
        table.insert(key, v);
    }
    let (texts, queries): (Vec<String>, Vec<String>) = keys.into_iter().partition(|k| !k.starts_with("query "));
    Ok((texts, queries, table, dim))
}

/// Resident and peak resident memory of this process, MiB.
fn rss_mb() -> (f64, f64) {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Counters {
            cb: u32,
            page_faults: u32,
            peak_working_set: usize,
            working_set: usize,
            rest: [usize; 6],
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
        }
        let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Counters::default() };
        // SAFETY: the pseudo-handle is always valid and `c` is a correctly sized PROCESS_MEMORY_COUNTERS.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0;
        let mb = |x: usize| x as f64 / 1_048_576.0;
        if ok { (mb(c.working_set), mb(c.peak_working_set)) } else { (0.0, 0.0) }
    }
    #[cfg(not(windows))]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let kb = |key: &str| status.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0;
        (kb("VmRSS:"), kb("VmHWM:"))
    }
}

pub fn main(args: &[String]) -> Result<()> {
    let f = Flags::parse(args);
    let base_rss = rss_mb().0;
    let (texts, queries, table, dim) = if f.has("vectors") {
        recorded(&f.str("vectors", ""))?
    } else {
        let (n, dim) = (f.num("n", 10_000), f.num("dim", 768));
        let (t, q, table) = synthetic(n + 300, dim);
        (t, q, table, dim)
    };
    let n = texts.len() - 300;
    let embedder = Table(Arc::new(table), dim);
    let dir = std::env::temp_dir().join(format!("engram-perf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = dir.join("p.db");
    let cfg = Config { capacity: n, writes_per_day: usize::MAX, ..Config::default() };
    let clock = ManualClock::new(1_700_000_000.0);
    let mut m = Engram::with_clock(Box::new(SqliteStore::open(&db)?), Box::new(embedder.clone()), cfg.clone(), "Asia/Tokyo", clock.clock())?;
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    for text in &texts[..n] {
        clock.advance(60.0);
        m.remember(text, 1.0, "")?;
    }
    let fill_ms = ms(t);
    let t = Instant::now();
    for text in &texts[n..n + 200] {
        clock.advance(60.0);
        m.remember(text, 1.0, "")?;
    }
    let remember_ms = ms(t) / 200.0;
    let (t, mut hits) = (Instant::now(), 0);
    for q in &queries {
        clock.advance(60.0);
        hits += m.recall(q)?.recalled.len();
    }
    let recall_ms = ms(t) / queries.len() as f64;
    let t = Instant::now();
    let clusters = m.clusters(None).len();
    let clusters_ms = ms(t);
    let t = Instant::now();
    let reports = m.dream(&mut |_: &crate::DreamRequest| Ok(DreamDecision::keep()), Some(5))?.len();
    let dream_ms = ms(t);
    let stats = m.stats();
    let (rss, peak) = rss_mb();
    drop(m);
    let t = Instant::now();
    let loaded = Engram::with_clock(Box::new(SqliteStore::open(&db)?), Box::new(embedder), cfg, "Asia/Tokyo", clock.clock())?.memories().len();
    let load_ms = ms(t);
    let _ = std::fs::remove_dir_all(&dir);
    println!("{}", serde_json::to_string_pretty(&json!({
        "n": n, "dim": dim, "base_rss_mb": base_rss,
        "fill_ms_per_op": fill_ms / n as f64, "remember_at_capacity_ms": remember_ms, "recall_ms": recall_ms, "recall_hits": hits,
        "clusters_ms": clusters_ms, "clusters_n": clusters, "dream5_ms": dream_ms, "dream_reports": reports,
        "stats": {"records": stats.records, "labile": stats.labile, "unindexed": stats.unindexed},
        "rss_mb": rss, "peak_rss_mb": peak, "load_ms": load_ms, "loaded": loaded,
    }))?);
    Ok(())
}
