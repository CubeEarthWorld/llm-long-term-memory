//! Forgetting benchmark (SPEC §3/§4): do used facts survive while unused chatter is
//! forgotten? A virtual timeline writes chatter past capacity while half the gold facts
//! are rehearsed at expanding intervals; at checkpoints every gold fact is probed with a
//! paraphrased question. Baseline arms replace strength-ordered eviction with FIFO, LRU,
//! random or none on the same timeline.
//!
//! `ltm bench forget --arm engram --span-days 365 --capacity 500 --seed 1`

use super::{bench_dir, load_corpus, r4, CachedEmbedder, Corpus, Flags, PyRandom, DAY};
use engram::{Config, DreamDecision, Embedder, Engram, ManualClock, Remembered, Result, Store};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// Rehearsal times as fractions of the span (expanding intervals).
const REHEARSALS: [f64; 6] = [0.002, 0.008, 0.03, 0.1, 0.3, 0.7];
const CHECKPOINTS: [f64; 5] = [0.01, 0.1, 0.3, 0.6, 1.0];
pub const ARMS: [&str; 5] = ["engram", "fifo", "lru", "random", "none"];

/// One run of the timeline.
#[derive(Clone, Debug)]
pub struct Spec {
    pub arm: String,
    pub seed: u64,
    pub span_days: f64,
    pub capacity: usize,
    pub rehearsed_frac: f64,
    pub salience: f64,
    pub dream: bool,
}

impl Spec {
    /// Rejects a timeline that cannot be run (an unknown arm, a non-positive span or
    /// capacity, a rehearsed fraction outside [0, 1], a non-finite salience).
    pub fn validate(&self) -> Result<()> {
        let bad = if !ARMS.contains(&self.arm.as_str()) {
            format!("unknown arm {}", self.arm)
        } else if !(self.span_days.is_finite() && self.span_days > 0.0) {
            format!("--span-days must be positive, got {}", self.span_days)
        } else if self.capacity == 0 {
            "--capacity must be positive".into()
        } else if !(0.0..=1.0).contains(&self.rehearsed_frac) {
            format!("--rehearsed-frac must be in [0, 1], got {}", self.rehearsed_frac)
        } else if !self.salience.is_finite() {
            format!("--salience must be finite, got {}", self.salience)
        } else {
            return Ok(());
        };
        Err(bad.into())
    }
}

enum Event {
    Noise(usize),
    Rehearse(usize),
    Dream,
    Measure,
}

struct Run<'a> {
    m: Engram,
    clock: ManualClock,
    spec: &'a Spec,
    rng: PyRandom,
    kind: HashMap<&'a str, &'static str>,
    evicted: HashMap<&'static str, usize>,
}

impl Run<'_> {
    /// Writes one text and applies the arm's eviction, counting what each eviction took.
    fn write(&mut self, text: &str, salience: f64) -> Result<()> {
        let r = self.m.remember(text, salience, "")?;
        let mut victims = Vec::new();
        match self.spec.arm.as_str() {
            "engram" => {
                if let Remembered::Inserted { evicted: Some(v), .. } = r {
                    victims.push(v.text);
                }
            }
            "none" => {}
            arm => {
                loop {
                    let rows = crate::all_traces(&self.m)?;
                    if rows.len() <= self.spec.capacity {
                        break;
                    }
                    let first_min = |key: &dyn Fn(&engram::Trace) -> i64| (0..rows.len()).min_by_key(|&i| (key(&rows[i]), i)).expect("rows");
                    let i = match arm {
                        "fifo" => first_min(&|t| t.created_at),
                        "lru" => first_min(&|t| t.last_recall),
                        _ => self.rng.randrange(rows.len()),
                    };
                    let id = rows[i].id.clone();
                    victims.push(self.m.forget(&id)?.ok_or("listed trace vanished")?.text);
                }
            }
        }
        for v in victims {
            *self.evicted.entry(self.kind.get(v.as_str()).copied().unwrap_or("noise")).or_default() += 1;
        }
        Ok(())
    }
}

/// Probes every gold fact without side effects and summarises survival and ranking.
fn measure(m: &mut Engram, corpus: &Corpus, rehearsed: &HashSet<&str>, t0: f64) -> Result<Value> {
    let alive: HashSet<String> = crate::all_traces(m)?.into_iter().map(|t| t.text).collect();
    let mut out = json!({"records": alive.len(), "day": ((m.now() - t0) / DAY).round() as i64});
    for (group, used) in [("rehearsed", true), ("dormant", false)] {
        let items: Vec<_> = corpus.gold.iter().filter(|g| rehearsed.contains(g.fact.as_str()) == used).collect();
        let (mut top1, mut hit5, mut mrr, mut probes, mut beaten) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for g in &items {
            for p in &g.probes {
                let texts: Vec<String> = m.peek(p)?.recalled.into_iter().map(|c| c.trace.text).collect();
                probes += 1.0;
                let has_distractor = !g.distractor.is_empty();
                if let Some(i) = texts.iter().position(|t| *t == g.fact) {
                    let rank = (i + 1) as f64;
                    top1 += f64::from(u8::from(i == 0));
                    hit5 += 1.0;
                    mrr += 1.0 / rank;
                    if has_distractor && texts[..i].contains(&g.distractor) {
                        beaten += 1.0;
                    }
                } else if has_distractor && texts.contains(&g.distractor) {
                    beaten += 1.0;
                }
            }
        }
        let (n, p) = (items.len().max(1) as f64, f64::max(probes, 1.0));
        out[group] = json!({
            "n": items.len(),
            "survival": r4(items.iter().filter(|g| alive.contains(&g.fact)).count() as f64 / n),
            "top1": r4(top1 / p), "hit@5": r4(hit5 / p), "mrr": r4(mrr / p), "distractor_ahead": r4(beaten / p),
        });
    }
    let distractors: Vec<&str> = corpus.gold.iter().map(|g| g.distractor.as_str()).filter(|d| !d.is_empty()).collect();
    out["noise_survival"] = json!(r4(corpus.noise.iter().filter(|n| alive.contains(*n)).count() as f64 / corpus.noise.len().max(1) as f64));
    out["distractor_survival"] = json!(r4(distractors.iter().filter(|d| alive.contains(**d)).count() as f64 / distractors.len().max(1) as f64));
    Ok(out)
}

/// Violations of SPEC §7 at `now` (all zero when healthy; `store_mismatch` is -1 when
/// the store could not count).
pub fn invariants(m: &Engram, capacity: usize) -> Result<Value> {
    let now = m.now();
    let all = crate::all_traces(m)?;
    let rs: Vec<f64> = all.iter().map(|t| m.retrievability(t, now)).collect();
    let max_s = m.config().max_stability;
    Ok(json!({
        "nan": rs.iter().filter(|r| r.is_nan()).count(),
        "r_out_of_range": rs.iter().filter(|r| !(0.0..=1.0).contains(*r)).count(),
        "stability_out_of_range": all.iter().filter(|t| !(1.0..=max_s).contains(&t.stability)).count(),
        "over_capacity": all.len().saturating_sub(capacity),
        "store_mismatch": m.stats().map_or(-1, |s| s.records as i64 - all.len() as i64),
    }))
}

/// How many violations a run's `invariants` and `forget_api` record: every count by
/// magnitude, so a negative mismatch or the -1 sentinel adds instead of cancelling.
pub fn violations(run: &Value) -> u64 {
    let sum = |v: &Value| v.as_object().map_or(0, |o| o.values().filter_map(Value::as_i64).map(i64::unsigned_abs).sum::<u64>());
    sum(&run["invariants"]) + run["forget_api"]["leaked"].as_u64().unwrap_or(0) + run["forget_api"]["store_mismatch"].as_i64().map_or(0, i64::unsigned_abs)
}

/// Explicit forgetting (SPEC §4 forget): a deleted fact is never recalled again.
fn forget_api_check(m: &mut Engram, corpus: &Corpus) -> Result<Value> {
    let alive: HashMap<String, String> = crate::all_traces(m)?.into_iter().map(|t| (t.text, t.id)).collect();
    let targets: Vec<_> = corpus.gold.iter().filter(|g| alive.contains_key(&g.fact)).take(10).collect();
    for g in &targets {
        m.forget(&alive[&g.fact])?;
    }
    let mut leaked = 0;
    for (g, p) in targets.iter().flat_map(|g| g.probes.iter().map(move |p| (g, p))) {
        leaked += usize::from(m.peek(p)?.recalled.iter().any(|c| c.trace.text == g.fact));
    }
    Ok(json!({"deleted": targets.len(), "leaked": leaked, "store_mismatch": m.stats()?.records as i64 - crate::all_traces(m)?.len() as i64}))
}

/// Runs one timeline and returns its result record.
pub fn run(corpus: &Corpus, spec: &Spec, mut cfg: Config, embedder: Box<dyn Embedder>) -> Result<Value> {
    spec.validate()?;
    let engine_cap = if spec.arm == "engram" { spec.capacity } else { 1_000_000_000 };
    let arm_cap = if spec.arm == "none" { engine_cap } else { spec.capacity };
    cfg.capacity = engine_cap;
    let (model_id, dimension) = (embedder.model_id().to_owned(), embedder.dimension());
    let clock = ManualClock::new(1_700_000_000.0);
    let m = Engram::with_clock(Box::new(Store::in_memory()?), embedder, cfg, "Asia/Tokyo", clock.clock())?;
    let mut kind: HashMap<&str, &'static str> = corpus.gold.iter().map(|g| (g.fact.as_str(), "gold")).collect();
    kind.extend(corpus.gold.iter().filter(|g| !g.distractor.is_empty()).map(|g| (g.distractor.as_str(), "distractor")));
    let evicted = ["gold", "distractor", "noise"].into_iter().map(|k| (k, 0)).collect();
    let mut run = Run { m, clock, spec, rng: PyRandom::new(spec.seed), kind, evicted };

    let mut gold: Vec<&super::Gold> = corpus.gold.iter().collect();
    run.rng.shuffle(&mut gold);
    let rehearsed: HashSet<&str> = gold[..(gold.len() as f64 * spec.rehearsed_frac) as usize].iter().map(|g| g.fact.as_str()).collect();
    let t0 = run.clock.get();
    let mut written_at = Vec::with_capacity(gold.len());
    for g in &gold {
        run.clock.advance(600.0);
        run.write(&g.fact, spec.salience)?;
        written_at.push(run.clock.get());
        if !g.distractor.is_empty() {
            run.clock.advance(600.0);
            run.write(&g.distractor, spec.salience)?;
        }
    }
    let mut noise: Vec<&String> = corpus.noise.iter().collect();
    run.rng.shuffle(&mut noise);
    let span = spec.span_days * DAY;
    let base = run.clock.get();
    let step = span / noise.len().max(1) as f64;

    let mut events: Vec<(f64, u8, Event)> = (0..noise.len()).map(|i| (base + (i + 1) as f64 * step, 0, Event::Noise(i))).collect();
    for (gi, g) in gold.iter().enumerate().filter(|(_, g)| rehearsed.contains(g.fact.as_str())) {
        let _ = g;
        events.extend(REHEARSALS.iter().map(|f| (written_at[gi] + f * span, 1, Event::Rehearse(gi))));
    }
    if spec.dream {
        events.extend((1..=40).map(|k| (base + f64::from(k) * span / 40.0, 2, Event::Dream)));
    }
    events.extend(CHECKPOINTS.iter().map(|f| (base + f * span, 3, Event::Measure)));
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))); // stable

    let mut keep = |_: &engram::DreamRequest| Ok(DreamDecision::keep()); // measure forgetting itself
    let (mut series, mut rehearsals, mut hits) = (Vec::new(), 0, 0);
    for (when, _, event) in &events {
        run.clock.set(*when);
        match event {
            Event::Noise(i) => run.write(noise[*i], 1.0)?, // chatter is never salient
            Event::Rehearse(gi) => {
                rehearsals += 1;
                let g = gold[*gi];
                let r = run.m.recall(&g.probes[0])?;
                if let Some(c) = r.recalled.iter().find(|c| c.trace.text == g.fact) {
                    hits += 1;
                    run.m.cite(&format!("《id:{}》", c.trace.id))?; // actually used = full strengthening
                }
            }
            Event::Dream => {
                run.m.dream(&mut keep, Some(2))?;
            }
            Event::Measure => series.push(measure(&mut run.m, corpus, &rehearsed, t0)?),
        }
    }
    let total: usize = run.evicted.values().sum();
    let ev = &run.evicted;
    let mut result = json!({
        "arm": spec.arm, "seed": spec.seed, "span_days": spec.span_days, "capacity": spec.capacity,
        "model_id": model_id, "dimension": dimension, "salience": spec.salience, "rehearsed_frac": spec.rehearsed_frac,
        "dream": if spec.dream { "keep" } else { "off" },
        "corpus": {"gold": gold.len(), "noise": noise.len(), "distractor": gold.iter().filter(|g| !g.distractor.is_empty()).count()},
        "written": gold.len() + noise.len() + gold.iter().filter(|g| !g.distractor.is_empty()).count(),
        "rehearsals": rehearsals, "rehearsal_hits": hits,
        "evicted": {"gold": ev["gold"], "distractor": ev["distractor"], "noise": ev["noise"]},
        "final": series.last().cloned().unwrap_or_else(|| json!({})), "series": series,
        "invariants": invariants(&run.m, arm_cap)?,
        "evict_precision": if total > 0 { json!(r4(ev["noise"] as f64 / total as f64)) } else { Value::Null },
    });
    result["forget_api"] = forget_api_check(&mut run.m, corpus)?;
    Ok(result)
}

pub fn main(args: &[String]) -> Result<()> {
    let f = Flags::parse(args);
    if f.has("help") {
        println!("ltm bench forget [--arm engram|fifo|lru|random|none] [--seed 1] [--span-days 365] [--capacity 500] [--rehearsed-frac 0.5] [--salience 1] [--dream keep|off] [--cache PATH] [--out PATH]");
        return Ok(());
    }
    let spec = Spec {
        arm: f.str("arm", "engram"),
        seed: f.num("seed", 1)?,
        span_days: f.num("span-days", 365.0)?,
        capacity: f.num("capacity", 500)?,
        rehearsed_frac: f.num("rehearsed-frac", 0.5)?,
        salience: f.num("salience", 1.0)?,
        dream: f.str("dream", "keep") != "off",
    };
    spec.validate()?;
    let corpus = load_corpus(&bench_dir().join("corpus"))?;
    let cache = CachedEmbedder::load(&f.0.get("cache").filter(|c| !c.is_empty()).map_or_else(|| bench_dir().join("embeddings.bin"), Into::into))?;
    let res = run(&corpus, &spec, Config::default(), Box::new(cache))?;
    let text = serde_json::to_string_pretty(&res)?;
    if f.has("out") {
        crate::write_atomic(std::path::Path::new(&f.str("out", "")), &text)?;
    }
    println!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrunnable_specs_are_rejected() {
        let ok = Spec { arm: "engram".into(), seed: 1, span_days: 365.0, capacity: 500, rehearsed_frac: 0.5, salience: 1.0, dream: true };
        assert!(ok.validate().is_ok());
        for bad in [
            Spec { rehearsed_frac: 1.5, ..ok.clone() },
            Spec { rehearsed_frac: f64::NAN, ..ok.clone() },
            Spec { span_days: 0.0, ..ok.clone() },
            Spec { capacity: 0, ..ok.clone() },
            Spec { arm: "lifo".into(), ..ok.clone() },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn violations_never_cancel() {
        let r = json!({"invariants": {"nan": 0, "over_capacity": 1, "store_mismatch": -1}, "forget_api": {"leaked": 2, "store_mismatch": -3}});
        assert_eq!(violations(&r), 7);
        assert_eq!(violations(&json!({"invariants": {"nan": 0}, "forget_api": {"leaked": 0, "store_mismatch": 0}})), 0);
    }
}
