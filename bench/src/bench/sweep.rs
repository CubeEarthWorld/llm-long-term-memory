//! The forgetting benchmark's grid: A — arms × timeline length, B — capacity
//! sensitivity, C — salience × dream. One JSON file per run in the results directory
//! (existing files are skipped, so an interrupted sweep resumes); `--report` prints the
//! medians per configuration.
//!
//! `ltm bench sweep [--jobs N] [--results DIR]`   then   `ltm bench sweep --report`

use super::forget::{run, Spec};
use super::{bench_dir, load_corpus, CachedEmbedder, Flags};
use engram::{Config, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

fn spec(arm: &str, span: f64, capacity: usize, salience: f64, dream: bool, seed: u64) -> Spec {
    Spec { arm: arm.into(), seed, span_days: span, capacity, rehearsed_frac: 0.5, salience, dream }
}

/// The 155 distinct runs.
pub fn grid() -> Vec<Spec> {
    let mut runs = Vec::new();
    for seed in 1..=5 {
        for arm in ["engram", "lru", "fifo", "random", "none"] {
            for span in [30.0, 365.0, 3650.0, 36_500.0] {
                runs.push(spec(arm, span, 1000, 1.0, true, seed));
            }
        }
        for arm in ["engram", "lru"] {
            for cap in [500, 1000, 2000, 4000] {
                runs.push(spec(arm, 365.0, cap, 1.0, true, seed));
            }
        }
        for sal in [1.0, 3.0, 10.0] {
            for dream in [true, false] {
                runs.push(spec("engram", 3650.0, 1000, sal, dream, seed));
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    runs.retain(|r| seen.insert(name(r)));
    runs
}

pub fn name(r: &Spec) -> String {
    format!("{}_span{}_cap{}_sal{}_{}_s{}", r.arm, r.span_days, r.capacity, r.salience, if r.dream { "keep" } else { "off" }, r.seed)
}

pub fn main(args: &[String]) -> Result<()> {
    let f = Flags::parse(args);
    let dir = f.0.get("results").filter(|d| !d.is_empty()).map_or_else(|| bench_dir().join("results"), PathBuf::from);
    if f.has("report") {
        return report(&dir);
    }
    std::fs::create_dir_all(&dir)?;
    let corpus = load_corpus(&bench_dir().join("corpus"));
    let cache = CachedEmbedder::load(&f.0.get("cache").filter(|c| !c.is_empty()).map_or_else(|| bench_dir().join("embeddings.bin"), PathBuf::from))?;
    let todo: Vec<Spec> = grid().into_iter().filter(|r| !dir.join(format!("{}.json", name(r))).exists()).collect();
    let jobs = f.num("jobs", crate::cpu_budget()).clamp(1, crate::cpu_budget());
    let (next, done, failure) = (AtomicUsize::new(0), AtomicUsize::new(0), Mutex::new(None));
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                while let Some(r) = todo.get(next.fetch_add(1, Ordering::SeqCst)) {
                    match run(&corpus, r, Config::default(), Box::new(cache.clone())).and_then(|res| {
                        Ok(std::fs::write(dir.join(format!("{}.json", name(r))), serde_json::to_string(&res)?)?)
                    }) {
                        Ok(()) => println!("[{}/{}] {}", done.fetch_add(1, Ordering::SeqCst) + 1, todo.len(), name(r)),
                        Err(e) => *failure.lock().unwrap() = Some(format!("{}: {e}", name(r))),
                    }
                }
            });
        }
    });
    failure.into_inner().unwrap().map_or(Ok(()), |e| Err(e.into()))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn report(dir: &std::path::Path) -> Result<()> {
    let mut rows: Vec<Value> = Vec::new();
    for e in std::fs::read_dir(dir)? {
        rows.push(serde_json::from_str(&std::fs::read_to_string(e?.path())?)?);
    }
    println!("{} runs\n", rows.len());
    let key = |r: &Value| (r["arm"].as_str().unwrap_or("").to_owned(), r["span_days"].as_f64().unwrap_or(0.0).to_bits(), r["capacity"].as_u64().unwrap_or(0), r["salience"].as_f64().unwrap_or(0.0).to_bits(), r["dream"].as_str().unwrap_or("").to_owned());
    let mut groups: BTreeMap<(String, u64, u64, u64, String), Vec<&Value>> = BTreeMap::new();
    for r in &rows {
        groups.entry(key(r)).or_default().push(r);
    }
    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0).then(f64::from_bits(a.1).total_cmp(&f64::from_bits(b.1))).then(a.2.cmp(&b.2)).then(f64::from_bits(a.3).total_cmp(&f64::from_bits(b.3))).then(a.4.cmp(&b.4)));
    println!("{:<7} {:>7} {:>5} {:>4} {:>5} {:>2} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>4}",
             "arm", "span_d", "cap", "sal", "dream", "n", "reh_surv", "reh_top1", "reh_hit5", "dor_surv", "noise_surv", "dist_ahead", "evict_prec", "bad");
    for k in keys {
        let g = &groups[&k];
        let med = |path: &[&str]| median(g.iter().map(|r| path.iter().fold(*r, |v, p| &v[*p]).as_f64().unwrap_or(0.0)).collect());
        let bad: u64 = g.iter().map(|r| r["invariants"].as_object().map_or(0, |o| o.values().filter_map(Value::as_i64).sum::<i64>().max(0) as u64) + r["forget_api"]["leaked"].as_u64().unwrap_or(0)).sum();
        println!("{:<7} {:>7} {:>5} {:>4} {:>5} {:>2} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>4}",
                 k.0, f64::from_bits(k.1), k.2, f64::from_bits(k.3), k.4, g.len(),
                 med(&["final", "rehearsed", "survival"]), med(&["final", "rehearsed", "top1"]), med(&["final", "rehearsed", "hit@5"]),
                 med(&["final", "dormant", "survival"]), med(&["final", "noise_survival"]), med(&["final", "rehearsed", "distractor_ahead"]),
                 med(&["evict_precision"]), bad);
    }
    Ok(())
}
