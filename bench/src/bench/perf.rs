//! Latency, memory and disk of every engine operation on synthetic unit vectors (every
//! 10th fact a near-duplicate of the previous one, so dreams have labile pairs).
//! Vectors are a function of the fact number, so no table of them is held; above
//! 20,000 facts the store is filled in bulk and the operations are timed at capacity.
//! The store is the bundled [`Store`] in a file.
//!
//! `ltm bench perf [--n 10000] [--dim 768]`

use super::Flags;
use engram::{Config, DreamDecision, Embedder, Engram, ManualClock, Result, Storage, Store, Trace, Write};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODEL: &str = "synthetic";
const T0: f64 = 1_700_000_000.0;
const STEP: f64 = 60.0;

/// SplitMix64: a reproducible stream per seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A standard normal deviate (Box–Muller).
    fn gaussian(&mut self) -> f64 {
        let u = |r: &mut Self| ((r.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let (a, b) = (u(self), u(self));
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

fn unit(v: Vec<f64>) -> Vec<f32> {
    crate::normalized(&v.iter().map(|&x| x as f32).collect::<Vec<_>>())
}

/// Fact `i`: random, or (every 10th) a near-duplicate of fact `i − 1`.
fn fact(i: u64, dim: usize) -> Vec<f32> {
    let random = |i: u64| -> Vec<f64> {
        let mut r = Rng(i);
        (0..dim).map(|_| r.gaussian()).collect()
    };
    if i % 10 == 9 {
        let mut r = Rng(i ^ 0xD1CE);
        unit(random(i - 1).iter().map(|&x| x / (dim as f64).sqrt() + 0.012 * r.gaussian() / (dim as f64).sqrt() * 40.0).collect())
    } else {
        unit(random(i))
    }
}

/// Query `j`: a noisy copy of a stored fact.
fn query(j: u64, n: u64, dim: usize) -> Vec<f32> {
    let base = fact((j * 37) % n.saturating_sub(300).max(1), dim);
    let mut r = Rng(j ^ 0xBEEF);
    unit(base.iter().map(|&x| f64::from(x) + 0.03 * r.gaussian() / (dim as f64).sqrt() * 10.0).collect())
}

/// Embeds `"fact {i}"` and `"query {j}"` by number.
struct Synthetic {
    n: u64,
    dim: usize,
}

impl Synthetic {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| {
                let (kind, num) = t.split_once(' ').ok_or("unknown text")?;
                let i: u64 = num.parse()?;
                Ok(if kind == "query" { query(i, self.n, self.dim) } else { fact(i, self.dim) })
            })
            .collect()
    }
}

impl Embedder for Synthetic {
    fn model_id(&self) -> &str {
        MODEL
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }
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

/// Writes facts `0..n` straight into the store, as `remember` would have left them.
fn bulk_fill(store: &mut dyn Storage, n: u64, dim: usize) -> Result<()> {
    let mut ids = engram::ulid_ids();
    for chunk in (0..n).collect::<Vec<_>>().chunks(10_000) {
        let traces: Vec<Trace> = chunk
            .iter()
            .map(|&i| {
                let t = T0 + i as f64 * STEP;
                Trace {
                    id: ids((t * 1000.0) as i64, ""),
                    text: format!("fact {i}"),
                    created_at: t as i64,
                    tz: "Asia/Tokyo;+09:00".into(),
                    last_recall: t as i64,
                    stability: 86_400.0,
                    consolidated: i % 10 != 9,
                    cue: String::new(),
                    model_id: MODEL.into(),
                    vector: fact(i, dim),
                }
            })
            .collect();
        store.write(&traces.iter().map(Write::Insert).collect::<Vec<_>>())?;
    }
    Ok(())
}

/// A scratch directory removed on every exit path, errors included.
struct ScratchDir(PathBuf);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn disk_mb(dir: &Path) -> f64 {
    std::fs::read_dir(dir).map_or(0.0, |d| d.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum::<u64>() as f64 / 1_048_576.0)
}

pub fn main(args: &[String]) -> Result<()> {
    let f = Flags::parse(args);
    let (n, dim): (u64, usize) = (f.num("n", 10_000)?, f.num("dim", 768)?);
    if n == 0 || dim == 0 {
        return Err("--n and --dim must be positive".into());
    }
    let base_rss = rss_mb().0;
    let dir = std::env::temp_dir().join(format!("engram-perf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let scratch = ScratchDir(dir.clone()); // declared before the stores, so dropped after them
    let db = dir.join("p.redb");
    let cfg = Config { capacity: n as usize, writes_per_day: usize::MAX, ..Config::default() };
    let clock = ManualClock::new(T0);
    let embedder = || Box::new(Synthetic { n, dim });
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    let mut store = Store::open(&db)?;
    let t = Instant::now();
    let bulk = n > 20_000;
    if bulk {
        bulk_fill(&mut store, n, dim)?;
    }
    let mut m = Engram::with_clock(Box::new(store), embedder(), cfg.clone(), "Asia/Tokyo", clock.clock())?;
    if !bulk {
        for i in 0..n {
            clock.set(T0 + i as f64 * STEP);
            m.remember(&format!("fact {i}"), 1.0, "")?;
        }
    }
    let fill_ms = ms(t);
    clock.set(T0 + n as f64 * STEP);
    let t = Instant::now();
    for i in n..n + 200 {
        clock.advance(STEP);
        m.remember(&format!("fact {i}"), 1.0, "")?;
    }
    let remember_ms = ms(t) / 200.0;
    let (t, mut hits) = (Instant::now(), 0);
    for j in 0..50 {
        clock.advance(STEP);
        hits += m.recall(&format!("query {j}"))?.recalled.len();
    }
    let recall_ms = ms(t) / 50.0;
    let t = Instant::now();
    let clusters = m.clusters(None)?.len();
    let clusters_ms = ms(t);
    let t = Instant::now();
    let reports = m.dream(&mut |_: &engram::DreamRequest| Ok(DreamDecision::keep()), Some(5))?.len();
    let dream_ms = ms(t);
    let stats = m.stats()?;
    let (rss, peak) = rss_mb();
    drop(m);
    let disk = disk_mb(&dir);
    let t = Instant::now();
    let reopened = Engram::with_clock(Box::new(Store::open(&db)?), embedder(), cfg, "Asia/Tokyo", clock.clock())?;
    let open_ms = ms(t);
    let first = Instant::now();
    let _ = reopened.memory("nothing")?;
    let first_ms = ms(first);
    drop(reopened);
    drop(scratch);
    println!("{}", serde_json::to_string_pretty(&json!({
        "n": n, "dim": dim, "fill": if bulk { "bulk" } else { "remember" },
        "fill_ms_per_op": fill_ms / n as f64, "remember_at_capacity_ms": remember_ms, "recall_ms": recall_ms, "recall_hits": hits,
        "clusters_ms": clusters_ms, "clusters_n": clusters, "dream5_ms": dream_ms, "dream_reports": reports,
        "records": stats.records, "labile": stats.labile,
        "base_rss_mb": base_rss, "rss_mb": rss, "peak_rss_mb": peak, "disk_mb": disk,
        "reopen_ms": open_ms, "first_lookup_ms": first_ms,
    }))?);
    Ok(())
}
