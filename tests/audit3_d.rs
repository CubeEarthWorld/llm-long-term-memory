//! Round-3 audit, lane D: long horizons, scale and performance regressions.
//! Run: CARGO_TARGET_DIR=C:/t/r3d cargo test --release --test audit3_d -j 3
//! Heavy measurements (100k traces) are `#[ignore]`: add `-- --ignored --nocapture --test-threads=1`.
//!
//! Part 1: SPEC section 8's "3000-year simulation" at reduced scale, with a shadow copy of the
//! store and a brute-force oracle for eviction (the weakest of the traces past the grace period).
//! Part 2: edge clocks (48-bit ULID horizon, i64 / f64 extremes).
//! Part 3: cost bounds and measurements (rows read per operation, latency, memory).

mod support;

use engram::{
    Config, DreamDecision, DreamRequest, Embedder, Engram, Hit, ManualClock, Remembered, Result, Storage, Store, Trace, View, Write,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use support::{cfg, merge_to_gist, FakeEmbedder, DAY};

const YEAR: f64 = 365.0 * DAY;
const DEAD: f64 = -64.0;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 20
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn unit(&mut self) -> f64 {
        (self.next() % (1 << 40)) as f64 / (1u64 << 40) as f64
    }
}

// ------------------------------------------------------------------ the spy store

#[derive(Default)]
struct Counters {
    find_calls: AtomicU64,
    find_rows: AtomicU64,
    band_calls: AtomicU64,
    band_rows: AtomicU64,
    count_calls: AtomicU64,
    near_calls: AtomicU64,
    write_calls: AtomicU64,
}

impl Counters {
    fn snap(&self) -> [u64; 7] {
        [
            self.find_calls.load(SeqCst),
            self.find_rows.load(SeqCst),
            self.band_calls.load(SeqCst),
            self.band_rows.load(SeqCst),
            self.count_calls.load(SeqCst),
            self.near_calls.load(SeqCst),
            self.write_calls.load(SeqCst),
        ]
    }
}

fn delta(a: [u64; 7], b: [u64; 7]) -> [u64; 7] {
    let mut d = [0; 7];
    for i in 0..7 {
        d[i] = b[i] - a[i];
    }
    d
}

/// What the oracle needs to know about a row.
type Shadow = BTreeMap<String, (i64, i64, f64)>; // id -> (created_at, last_recall, stability)

struct Probe {
    shadow: Mutex<Shadow>,
    violations: Mutex<Vec<String>>,
    expect_evict: AtomicBool,
    checks: AtomicU64,
    victims: AtomicU64,
    inexact: AtomicU64,
    worst_gap: Mutex<f64>,
    worst_desc: Mutex<String>,
    clock: ManualClock,
    cap: usize,
    grace: f64,
    track: bool,
}

fn now_of(clock: &ManualClock) -> f64 {
    let t = clock.get();
    if t.is_nan() { 0.0 } else { t.clamp(f64::MIN, f64::MAX) }
}

fn rank_of(row: (i64, i64, f64), now: f64) -> f64 {
    let (_, last, s) = row;
    s.log2() - (now - last as f64).max(0.0) / s
}

impl Probe {
    /// The brute-force eviction oracle, run on the pre-state of the write that inserts `new`.
    fn check(&self, writes: &[Write<'_>]) {
        let Some(new) = writes.iter().find_map(|w| if let Write::Insert(t) = w { Some(*t) } else { None }) else { return };
        let shadow = self.shadow.lock().unwrap();
        if shadow.contains_key(&new.id) {
            return;
        }
        let now = now_of(&self.clock);
        let removed: Vec<&str> = writes.iter().filter_map(|w| if let Write::Remove(id) = w { Some(*id) } else { None }).collect();
        let excess = (shadow.len() + 1).saturating_sub(self.cap);
        self.checks.fetch_add(1, SeqCst);
        let fail = |m: String| self.violations.lock().unwrap().push(m);
        if removed.len() != excess {
            fail(format!("eviction removed {} traces, capacity needs {excess} (store {}, cap {})", removed.len(), shadow.len(), self.cap));
            return;
        }
        if excess == 0 {
            return;
        }
        let newrow = (new.created_at, new.last_recall, new.stability);
        let cutoff = (now - self.grace) as i64;
        let young = shadow.values().filter(|r| r.0 > cutoff).count() + usize::from(newrow.0 > cutoff);
        let mut pool: Vec<f64> = Vec::new();
        if young <= self.cap / 10 {
            pool = shadow.values().filter(|r| r.0 <= cutoff).map(|&r| rank_of(r, now)).collect();
        }
        if pool.len() < excess {
            pool = shadow.values().map(|&r| rank_of(r, now)).chain([rank_of(newrow, now)]).collect();
        }
        pool.sort_by(f64::total_cmp);
        pool.truncate(excess);
        let mut got: Vec<f64> = removed
            .iter()
            .map(|id| if *id == new.id { rank_of(newrow, now) } else { shadow.get(*id).map_or(f64::NAN, |&r| rank_of(r, now)) })
            .collect();
        got.sort_by(f64::total_cmp);
        self.victims.fetch_add(excess as u64, SeqCst);
        for (g, e) in got.iter().zip(&pool) {
            let same = (g - e).abs() <= 1e-9 * e.abs().max(1.0) || (*g <= DEAD && *e <= DEAD);
            if !same {
                self.inexact.fetch_add(1, SeqCst);
                let gap = g - e; // > 0: a stronger trace than the weakest was evicted
                let mut w = self.worst_gap.lock().unwrap();
                if gap.abs() > w.abs() {
                    *w = gap;
                    let band = |s: f64| (2.0 * s.max(1.0).log2()).floor() as i64;
                    let weakest = shadow.iter().filter(|(_, r)| r.0 <= cutoff || young > self.cap / 10).map(|(id, &r)| (rank_of(r, now), id, r)).min_by(|a, b| a.0.total_cmp(&b.0));
                    let victim = removed.iter().filter_map(|id| shadow.get(*id).map(|&r| (rank_of(r, now), *id, r))).max_by(|a, b| a.0.total_cmp(&b.0));
                    let fmt = |x: Option<(f64, &str, (i64, i64, f64))>| x.map_or("-".into(), |(rk, id, r)| {
                        let bn = shadow.values().filter(|q| band(q.2) == band(r.2)).count();
                        format!("[{id} rank {rk:.3} S {:.3e} band {} (band holds {bn}) dt {:.3e} age {:.3e}]", r.2, band(r.2), now - r.1 as f64, now - r.0 as f64)
                    });
                    let weakest = weakest.map(|(a, b, c)| (a, b.as_str(), c));
                    let victim = victim.map(|(a, b, c)| (a, b, c));
                    *self.worst_desc.lock().unwrap() = format!("victim {} vs weakest {} (excess {excess}, young {young})", fmt(victim), fmt(weakest));
                }
                if gap < 0.0 {
                    fail(format!("evicted a trace ranked {g:.6} below the oracle's {e:.6} (protected young row taken?) now={now}"));
                }
            }
        }
    }

    fn apply(&self, writes: &[Write<'_>]) {
        let mut s = self.shadow.lock().unwrap();
        for w in writes {
            match *w {
                Write::Insert(t) => {
                    s.insert(t.id.clone(), (t.created_at, t.last_recall, t.stability));
                }
                Write::State { id, last_recall, stability, .. } => {
                    if let Some(r) = s.get_mut(id) {
                        r.1 = last_recall;
                        r.2 = stability;
                    }
                }
                Write::Remove(id) => {
                    s.remove(id);
                }
                Write::Index { .. } => {}
            }
        }
    }
}

struct Spy {
    inner: Store,
    c: Arc<Counters>,
    probe: Option<Arc<Probe>>,
}

impl Storage for Spy {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        let out = self.inner.find(view, limit)?;
        self.c.find_calls.fetch_add(1, SeqCst);
        self.c.find_rows.fetch_add(out.len() as u64, SeqCst);
        if matches!(view, View::Band { .. }) {
            self.c.band_calls.fetch_add(1, SeqCst);
            self.c.band_rows.fetch_add(out.len() as u64, SeqCst);
        }
        Ok(out)
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        self.c.count_calls.fetch_add(1, SeqCst);
        self.inner.count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        self.c.near_calls.fetch_add(1, SeqCst);
        self.inner.near(q, model_id, min_cos, k, stated_by)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.c.write_calls.fetch_add(1, SeqCst);
        if let Some(p) = &self.probe {
            if p.expect_evict.load(SeqCst) {
                p.check(writes);
            }
        }
        self.inner.write(writes)?;
        if let Some(p) = &self.probe {
            if p.track {
                p.apply(writes);
            }
        }
        Ok(())
    }
    fn clear(&mut self) -> Result<()> {
        if let Some(p) = &self.probe {
            p.shadow.lock().unwrap().clear();
        }
        self.inner.clear()
    }
}

fn probe(clock: &ManualClock, cap: usize, grace: f64) -> Arc<Probe> {
    Arc::new(Probe {
        shadow: Mutex::new(BTreeMap::new()),
        violations: Mutex::new(Vec::new()),
        expect_evict: AtomicBool::new(false),
        checks: AtomicU64::new(0),
        victims: AtomicU64::new(0),
        inexact: AtomicU64::new(0),
        worst_gap: Mutex::new(0.0),
        worst_desc: Mutex::new(String::new()),
        clock: clock.clone(),
        cap,
        grace,
        track: true,
    })
}

fn row(id: &str, text: &str, created_at: i64, last_recall: i64, stability: f64, consolidated: bool) -> Trace {
    Trace {
        id: id.into(),
        text: text.into(),
        created_at,
        tz: "UTC;+00:00".into(),
        last_recall,
        stability,
        consolidated,
        cue: String::new(),
        model_id: String::new(),
        vector: Vec::new(),
    }
}

// ------------------------------------------------------------------ section 7 checks

fn check_invariants(m: &Engram, p: &Probe, cfg: &Config, keys: &[String], when: &str) {
    let rows = support::rows(m);
    assert!(rows.len() <= cfg.capacity, "{when}: {} rows > capacity {}", rows.len(), cfg.capacity);
    assert_eq!(m.stats().unwrap().records, rows.len(), "{when}: stats().records");
    let now = now_of(&p.clock);
    let mut ids = std::collections::BTreeSet::new();
    for t in &rows {
        assert!(ids.insert(t.id.clone()), "{when}: duplicate id {}", t.id);
        assert!(t.stability.is_finite() && (1.0..=cfg.max_stability).contains(&t.stability), "{when}: S {}", t.stability);
        let n = t.text.chars().count();
        assert!((1..=cfg.text_max).contains(&n) && !t.text.contains(['《', '》']), "{when}: text {:?}", t.text);
        assert!(!t.id.is_empty() && !t.id.contains(['《', '》', '\n']), "{when}: id");
        if !t.vector.is_empty() {
            let n2: f64 = t.vector.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            assert!((n2 - 1.0).abs() < 1e-3, "{when}: non unit vector {n2}");
            assert!(!t.model_id.is_empty(), "{when}: vector without model");
        }
        for at in [now, now + 10.0 * YEAR, now - 40.0 * DAY] {
            let r = m.retrievability(t, at);
            assert!((0.0..=1.0).contains(&r), "{when}: R {r} for {t:?}");
        }
    }
    let shadow = p.shadow.lock().unwrap();
    assert_eq!(shadow.len(), rows.len(), "{when}: shadow size");
    for t in &rows {
        assert_eq!(shadow.get(&t.id), Some(&(t.created_at, t.last_recall, t.stability)), "{when}: shadow of {}", t.id);
    }
    for k in keys {
        assert!(rows.iter().any(|t| &t.text == k), "{when}: key fact lost: {k}");
    }
}

// ------------------------------------------------------------------ the simulation

struct Scenario {
    name: &'static str,
    cap: usize,
    steps: u64,
    seed: u64,
    writes_per_day: usize,
    /// Fill the store with this many facts at S_max (last recall spread over ten years) first.
    saturate: usize,
    /// Mean seconds between operations.
    mean_gap: f64,
    bursts: bool,
}

struct Outcome {
    virtual_years: f64,
    remembers: u64,
    full_remembers: u64,
    dreams: u64,
    replaces: u64,
    max_band_rows: u64,
    mean_band_rows: f64,
    checks: u64,
    inexact: u64,
    worst_gap: f64,
    worst_desc: String,
    secs: f64,
}

fn embedder() -> Box<dyn Embedder> {
    Box::new(FakeEmbedder::with(32, "fake/token-overlap"))
}

fn note(rng: &mut Rng) -> String {
    format!("note n{} topic{} w{} x{}", rng.below(2000), rng.below(97), rng.below(2000), rng.below(300))
}

fn run(sc: &Scenario) -> Outcome {
    let t0 = Instant::now();
    let config = Config { capacity: sc.cap, writes_per_day: sc.writes_per_day, theta_related: 0.75, ..cfg() };
    let clock = ManualClock::new(support::START);
    let c = Arc::new(Counters::default());
    let p = probe(&clock, sc.cap, config.grace_period);
    let mut spy = Spy { inner: Store::in_memory().unwrap(), c: c.clone(), probe: Some(p.clone()) };
    let mut rng = Rng(sc.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let start = support::START;
    for i in 0..sc.saturate {
        // facts that were last recalled 0..10 years ago, all at the stability ceiling
        let ago = rng.unit() * 10.0 * YEAR;
        let t = row(&format!("00{i:024}"), &format!("old fact f{i} about thing{}", i % 23), (start - ago - YEAR) as i64, (start - ago) as i64, config.max_stability, true);
        spy.write(&[Write::Insert(&t)]).unwrap();
    }
    let keys: Vec<String> = ["user likes matcha ice cream", "user was born in kyoto japan", "user has a cat named tama", "user works as a nurse in osaka", "user plays the cello"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if sc.saturate > 0 {
        // in a saturated store the key facts are themselves ten-year facts (a new memory that is
        // not recalled within its grace period is the weakest row there is)
        for (i, k) in keys.iter().enumerate() {
            let t = row(&format!("01{i:024}"), k, start as i64 - 400 * 86_400, start as i64 - 86_400, config.max_stability, true);
            spy.write(&[Write::Insert(&t)]).unwrap();
        }
    }
    let mut m = Engram::with_clock(Box::new(spy), embedder(), config.clone(), "UTC", clock.clock()).unwrap();
    if sc.saturate > 0 {
        while m.reindex(100_000).unwrap() > 0 {}
    } else {
        for k in &keys {
            m.remember(k, 2.0, "").unwrap();
        }
    }
    let mut adj_rng = Rng(sc.seed ^ 0xABCD);
    let mut adj = move |r: &DreamRequest| -> Result<DreamDecision> {
        match adj_rng.below(10) {
            0 => Err("llm down".into()),
            1..=3 => Ok(DreamDecision::keep()),
            4 => Ok(DreamDecision {
                memories: vec![format!("{} part a", r.members[0].text), format!("{} part b", r.members[0].text)],
                absorbed_ids: r.members[1..].iter().map(|x| x.id.clone()).collect(),
            }),
            _ => merge_to_gist(r),
        }
    };
    let (mut remembers, mut full_remembers, mut dreams, mut replaces) = (0u64, 0u64, 0u64, 0u64);
    let (mut max_band, mut sum_band) = (0u64, 0u64);
    let mut recent: Vec<String> = Vec::new();
    let mut burst_left = 0u32;
    let mut excursions = 0;
    for step in 0..sc.steps {
        let warm = step < 600;
        // ---- the clock
        let r = rng.below(100_000);
        if !warm && r < 250 {
            clock.advance(10.0 * YEAR); // a decade of silence
        } else if !warm && r < 330 {
            clock.advance(-(1.0 + rng.unit() * 30.0) * DAY); // an NTP step back
        } else if !warm && r < 336 && excursions < 4 {
            excursions += 1;
            let t = clock.get();
            clock.set(t + (1.0 + rng.unit() * 30.0) * YEAR); // a forward fault, undone
            let k = &keys[rng.below(keys.len() as u64) as usize];
            m.recall(k).unwrap();
            clock.set(t);
        } else if !warm && r < 700 && sc.bursts && burst_left == 0 {
            burst_left = 5 + rng.below(150) as u32;
        } else if burst_left > 0 {
            burst_left -= 1;
            clock.advance(rng.unit() * 3.0);
        } else {
            clock.advance(rng.unit() * 2.0 * sc.mean_gap);
        }
        // ---- the operation
        let op = if burst_left > 0 { 0 } else { rng.below(100) };
        // key facts: recalled when their retrievability has dropped (the spacing effect rewards it), and
        // periodically whatever it is: each must come back, and none may be forgotten.
        for (ki, k) in keys.iter().enumerate() {
            let stored = m.store().find(View::Text(k), 1).unwrap().pop();
            let due = stored.as_ref().is_none_or(|t| m.retrievability(t, now_of(&clock)) < 0.7) || (step + ki as u64) % 11 == 0;
            if !due {
                continue;
            }
            let rec = m.recall(k).unwrap();
            assert!(
                rec.recalled.iter().any(|c| &c.trace.text == k),
                "{}: step {step}: key fact {k:?} not recalled (clock {}, stored: {:?})",
                sc.name,
                clock.get(),
                stored.map(|t| (t.created_at, t.last_recall, t.stability, t.consolidated))
            );
            if rng.below(2) == 0 {
                m.cite(&rec.pack_text).unwrap();
            }
        }
        match op {
            0..=59 => {
                let text = if !recent.is_empty() && rng.below(4) == 0 {
                    let mut t = recent[rng.below(recent.len() as u64) as usize].clone(); // a near duplicate: the dream has work
                    t = t.replace(&format!(" x{}", t.rsplit('x').next().unwrap_or("")), &format!(" x{}", rng.below(300)));
                    t
                } else {
                    note(&mut rng)
                };
                let cue = if rng.below(5) == 0 && !recent.is_empty() { recent[rng.below(recent.len() as u64) as usize].clone() } else { String::new() };
                let sal = if rng.below(8) == 0 { rng.unit() * 10.0 } else { 1.0 };
                let before = c.snap();
                p.expect_evict.store(true, SeqCst);
                let out = m.remember(&text, sal, &cue).unwrap();
                p.expect_evict.store(false, SeqCst);
                if let Remembered::Inserted { .. } = out {
                    remembers += 1;
                    if p.shadow.lock().unwrap().len() >= sc.cap {
                        full_remembers += 1;
                        let d = delta(before, c.snap());
                        max_band = max_band.max(d[3]);
                        sum_band += d[3];
                    }
                    recent.push(text);
                    if recent.len() > 64 {
                        recent.remove(0);
                    }
                }
            }
            60..=74 => {
                let q = if rng.below(2) == 0 && !recent.is_empty() { recent[rng.below(recent.len() as u64) as usize].clone() } else { note(&mut rng) };
                let rec = m.recall(&q).unwrap();
                if rng.below(2) == 0 {
                    m.cite(&rec.pack_text).unwrap();
                }
            }
            75..=84 => {
                dreams += 1;
                let reports = m.dream(&mut adj, Some(2)).unwrap();
                replaces += reports.iter().filter(|r| r.action == engram::DreamAction::Replace).count() as u64;
            }
            85..=86 => {
                let n = p.shadow.lock().unwrap().len() as u64;
                if n > 0 {
                    let id = p.shadow.lock().unwrap().keys().nth(rng.below(n) as usize).cloned().unwrap();
                    let is_key = m.memory(&id).unwrap().is_some_and(|t| keys.contains(&t.text));
                    if !is_key {
                        m.forget(&id).unwrap();
                    }
                }
            }
            87..=89 => {
                let _ = m.peek(&note(&mut rng)).unwrap();
            }
            _ => {
                m.reindex(8).unwrap();
            }
        }
        if step % 997 == 0 {
            check_invariants(&m, &p, &config, &keys, &format!("{} step {step}", sc.name));
        }
        {
            let v = p.violations.lock().unwrap();
            assert!(v.is_empty(), "{}: step {step}: {:?}", sc.name, &v[..v.len().min(3)]);
        }
    }
    check_invariants(&m, &p, &config, &keys, &format!("{} end", sc.name));
    for k in &keys {
        let top = &m.recall(k).unwrap().recalled;
        assert!(top.first().is_some_and(|c| &c.trace.text == k), "{}: key fact {k:?} is not the top recall at the end", sc.name);
    }
    let virtual_years = (clock.get() - start) / YEAR;
    Outcome {
        virtual_years,
        remembers,
        full_remembers,
        dreams,
        replaces,
        max_band_rows: max_band,
        mean_band_rows: sum_band as f64 / full_remembers.max(1) as f64,
        checks: p.checks.load(SeqCst),
        inexact: p.inexact.load(SeqCst),
        worst_gap: *p.worst_gap.lock().unwrap(),
        worst_desc: p.worst_desc.lock().unwrap().clone(),
        secs: t0.elapsed().as_secs_f64(),
    }
}

fn report(sc: &Scenario, o: &Outcome) {
    println!(
        "[sim {:<14}] cap {:>3} steps {:>7} | virtual {:>7.0} y | remember {:>6} (full {:>6}) dream {:>5} replace {:>4} | band rows/remember mean {:>6.1} max {:>5} | eviction checks {:>6}, not-weakest {:>4} (worst gap {:+.3} log2) | {:.1}s",
        sc.name, sc.cap, sc.steps, o.virtual_years, o.remembers, o.full_remembers, o.dreams, o.replaces, o.mean_band_rows, o.max_band_rows, o.checks, o.inexact, o.worst_gap, o.secs
    );
}

fn assert_outcome(sc: &Scenario, o: &Outcome) {
    report(sc, o);
    if o.inexact > 0 {
        println!("    worst deviation: {}", o.worst_desc);
    }
    assert!(o.virtual_years >= 1000.0, "{}: only {:.0} virtual years", sc.name, o.virtual_years);
    assert!(o.full_remembers > 100, "{}: the store was never full", sc.name);
    // SPEC section 8: no operation reads more than ~60 bands x 200 rows to evict.
    assert!(o.max_band_rows <= 60 * 200, "{}: {} band rows read for one remember", sc.name, o.max_band_rows);
    // The eviction is weakest-first among alive traces. A deviation is a stronger trace evicted
    // than the oracle's weakest (the BAND_CAP bound); count it, tolerate a vanishing share, never a big gap.
    assert!(o.worst_gap.abs() <= 8.0 && o.inexact * 50 <= o.checks, "{}: {} of {} evictions were not the weakest (worst gap {})", sc.name, o.inexact, o.checks, o.worst_gap);
}

#[test]
fn sim_cap50() {
    let sc = Scenario { name: "cap50", cap: 50, steps: 90_000, seed: 11, writes_per_day: 400, saturate: 0, mean_gap: 3.0 * 3600.0, bursts: true };
    assert_outcome(&sc, &run(&sc));
}

#[test]
fn sim_cap120() {
    let sc = Scenario { name: "cap120", cap: 120, steps: 90_000, seed: 12, writes_per_day: 1000, saturate: 0, mean_gap: 2.0 * 3600.0, bursts: true };
    assert_outcome(&sc, &run(&sc));
}

#[test]
fn sim_cap500() {
    let sc = Scenario { name: "cap500", cap: 500, steps: 90_000, seed: 13, writes_per_day: 100_000, saturate: 0, mean_gap: 1800.0, bursts: true };
    assert_outcome(&sc, &run(&sc));
}

#[test]
fn sim_saturated_cap200() {
    // 200 facts at the stability ceiling fill the store; notes arrive about twice a day.
    let sc = Scenario { name: "saturated200", cap: 200, steps: 60_000, seed: 14, writes_per_day: 1000, saturate: 195, mean_gap: 12.0 * 3600.0, bursts: false };
    assert_outcome(&sc, &run(&sc));
}

#[test]
fn sim_saturated_cap500() {
    let sc = Scenario { name: "saturated500", cap: 500, steps: 60_000, seed: 15, writes_per_day: 1000, saturate: 495, mean_gap: 6.0 * 3600.0, bursts: true };
    assert_outcome(&sc, &run(&sc));
}

/// In a store saturated with ten-year facts a new memory lives through its grace period, and then the
/// fact recalled longest ago yields its place (SPEC section 8).
#[test]
fn saturated_store_new_memory_lives_grace_then_oldest_fact_yields() {
    let config = Config { capacity: 100, writes_per_day: 10_000, ..cfg() };
    let clock = ManualClock::new(support::START);
    let c = Arc::new(Counters::default());
    let p = probe(&clock, 100, config.grace_period);
    let mut spy = Spy { inner: Store::in_memory().unwrap(), c, probe: Some(p.clone()) };
    for i in 0..100 {
        let ago = (i as f64 + 1.0) * 30.0 * DAY; // fact i was last recalled i+1 months ago: fact 99 is the oldest
        let t = row(&format!("00{i:024}"), &format!("fact number {i}"), (support::START - ago - YEAR) as i64, (support::START - ago) as i64, config.max_stability, true);
        spy.write(&[Write::Insert(&t)]).unwrap();
    }
    let mut m = Engram::with_clock(Box::new(spy), embedder(), config, "UTC", clock.clock()).unwrap();
    let mut mine = Vec::new();
    for i in 0..8 {
        clock.advance(6.0 * 3600.0);
        p.expect_evict.store(true, SeqCst);
        let Remembered::Inserted { trace, evicted, .. } = m.remember(&format!("brand new thing {i}"), 1.0, "").unwrap() else { panic!("insert") };
        p.expect_evict.store(false, SeqCst);
        mine.push(trace.id.clone());
        assert_eq!(evicted.unwrap().text, format!("fact number {}", 99 - i), "the oldest fact goes first");
    }
    assert!(p.violations.lock().unwrap().is_empty());
    for id in &mine {
        assert!(m.memory(id).unwrap().is_some(), "a new memory died inside its grace period");
    }
    // after the grace period (and a flood of new ones) the notes themselves are the weakest and go
    clock.advance(10.0 * DAY);
    for i in 0..30 {
        clock.advance(3600.0);
        m.remember(&format!("later note {i}"), 1.0, "").unwrap();
    }
    assert!(support::rows(&m).len() <= 100);
    assert!(mine.iter().filter(|id| m.memory(id).unwrap().is_some()).count() < mine.len(), "stale notes never yield to newer memories");
}

/// LOW (documented approximation, SPEC section 3: "an upper-limit band is approximate", yet the same
/// paragraph says the result equals a full comparison for usable traces): a band of 400 rows whose
/// order by `last_recall` disagrees with the order by rank (rows with a lower S were recalled later,
/// and dt/S is ~45): the weakest rows lie behind BAND_CAP = 128 rows, so a stronger row is evicted.
#[test]
#[ignore = "accepted: a band beyond BAND_CAP rows is an approximation (SPEC §3)"]
fn bandcap_repro_a_dense_aged_band_evicts_a_stronger_trace_than_the_weakest() {
    let n = 400usize;
    let config = Config { capacity: n, writes_per_day: 100_000, ..cfg() };
    let clock = ManualClock::new(support::START);
    let c = Arc::new(Counters::default());
    let p = probe(&clock, n, config.grace_period);
    let mut spy = Spy { inner: Store::in_memory().unwrap(), c, probe: Some(p.clone()) };
    let now = support::START as i64;
    for i in 0..n {
        let (s, last) = if i < 300 { (86_400.0, now - 3_600_000 + i as i64 * 1000) } else { (66_000.0, now - 3_000_000 + i as i64 * 10) };
        let t = row(&format!("00{i:024}"), &format!("aged note {i}"), last - 86_400 * 30, last, s, true);
        spy.write(&[Write::Insert(&t)]).unwrap();
    }
    let mut m = Engram::with_clock(Box::new(spy), embedder(), config, "UTC", clock.clock()).unwrap();
    p.expect_evict.store(true, SeqCst);
    let Remembered::Inserted { evicted, .. } = m.remember("a brand new note", 1.0, "").unwrap() else { panic!("insert") };
    println!("evicted {:?}; oracle: {}", evicted.map(|t| t.text), p.worst_desc.lock().unwrap());
    assert!(p.violations.lock().unwrap().is_empty());
    assert_eq!(p.inexact.load(SeqCst), 0, "evicted a trace {:+.2} log2 strength above the weakest: {}", *p.worst_gap.lock().unwrap(), p.worst_desc.lock().unwrap());
}

// ------------------------------------------------------------------ clock and id edges

#[test]
fn ulid_48_bit_horizon_ids_stay_valid_unique_and_monotonic() {
    let horizon = ((1u64 << 48) - 1) as f64 / 1000.0; // year 10889
    let config = Config { capacity: 40, writes_per_day: 100_000, ..cfg() };
    let clock = ManualClock::new(horizon - 10.0);
    let mut m = Engram::with_clock(Box::new(Store::in_memory().unwrap()), embedder(), config, "UTC", clock.clock()).unwrap();
    let mut ids: Vec<String> = Vec::new();
    for i in 0..60 {
        clock.advance(1.0); // crosses the horizon after ten steps
        let Remembered::Inserted { trace, .. } = m.remember(&format!("horizon note {i} z{}", i * 7), 1.0, "").unwrap() else { panic!("insert") };
        assert_eq!(trace.id.len(), 26, "{}", trace.id);
        ids.push(trace.id);
    }
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "ids must keep sorting in creation order across the 48-bit horizon: {:?}", &ids[8..14]);
    let mut uniq = ids.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), ids.len());
    // beyond the horizon everything still works (the documented rewind only affects id order)
    clock.advance(1.0e9);
    for i in 0..5 {
        m.remember(&format!("beyond {i} q{i}"), 1.0, "").unwrap();
    }
    m.recall("horizon note 3 z21").unwrap();
    m.dream(&mut merge_to_gist, Some(3)).unwrap();
    assert!(support::rows(&m).len() <= 40);
}

#[test]
fn clock_extremes_never_panic_and_keep_the_invariants() {
    let edges: [(&str, f64); 14] = [
        ("zero", 0.0),
        ("minus one", -1.0),
        ("nan", f64::NAN),
        ("inf", f64::INFINITY),
        ("neg inf", f64::NEG_INFINITY),
        ("f64::MAX", f64::MAX),
        ("f64::MIN", f64::MIN),
        ("2^63", 9.223_372_036_854_775_8e18),
        ("-2^63", -9.223_372_036_854_775_8e18),
        ("2^53", 9.007_199_254_740_992e15),
        ("year 10000", 253_402_300_800.0),
        ("jiff max", 253_402_207_200.0),
        ("jiff min", -377_705_023_201.0),
        ("1e300", 1e300),
    ];
    let mut failures = Vec::new();
    for (name, t) in edges {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let config = Config { capacity: 6, writes_per_day: 100_000, theta_related: 0.6, ..cfg() };
            let clock = ManualClock::new(t);
            let mut m = Engram::with_clock(Box::new(Store::in_memory().unwrap()), embedder(), config.clone(), "Asia/Tokyo", clock.clock()).unwrap();
            for i in 0..14 {
                if i % 3 == 0 {
                    clock.set(if t.is_finite() { t + i as f64 * 0.5 } else { t });
                }
                let cue = if i % 4 == 3 { "note about thing" } else { "" };
                m.remember(&format!("edge note {i} about thing{}", i % 3), 1.0 + (i % 4) as f64, cue).unwrap();
                let rec = m.recall("edge note about thing").unwrap();
                m.cite(&rec.pack_text).unwrap();
                if i % 5 == 0 {
                    m.dream(&mut merge_to_gist, Some(2)).unwrap();
                }
            }
            let _ = m.now_local();
            let rows = support::rows(&m);
            assert!(rows.len() <= config.capacity, "capacity {}", rows.len());
            for r in &rows {
                assert!(r.stability.is_finite() && (1.0..=config.max_stability).contains(&r.stability));
                for at in [t, 0.0, 1e18] {
                    let x = m.retrievability(r, at);
                    assert!((0.0..=1.0).contains(&x), "R {x}");
                }
                let _ = engram::fmt_local(r.created_at, &r.tz);
                let _ = engram::fmt_local(r.last_recall, &r.tz);
            }
            assert_eq!(m.stats().unwrap().records, rows.len());
        }));
        if r.is_err() {
            failures.push(name);
        }
    }
    assert!(failures.is_empty(), "clock edges that panic or break an invariant: {failures:?}");
}

#[test]
fn fmt_local_i64_edges() {
    for t in [i64::MIN, i64::MIN + 1, -1, 0, i64::MAX - 1, i64::MAX, 253_402_300_800, 1 << 48] {
        for tz in ["UTC;+00:00", "X;-23:59", "Y;+23:59", "garbage", ""] {
            let s = engram::fmt_local(t, tz);
            assert!(s.contains(' '), "{t} {tz}: {s}");
        }
    }
}

// ------------------------------------------------------------------ cost and scale

const DIM: usize = 768;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic unit vector (uniform on the sphere, up to the generator) per seed.
fn unit_vec(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x1234_5678;
    let mut v: Vec<f32> = (0..DIM)
        .map(|_| {
            // sum of four uniforms: close enough to Gaussian for a direction
            let mut a = 0.0f32;
            for _ in 0..4 {
                s = splitmix(s);
                a += (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            a
        })
        .collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

/// `base` blended with a random direction: cosine to `base` near `cos`.
fn near_vec(base: &[f32], cos: f32, seed: u64) -> Vec<f32> {
    let o = unit_vec(seed);
    let sin = (1.0 - cos * cos).sqrt();
    let mut v: Vec<f32> = base.iter().zip(&o).map(|(b, o)| cos * b + sin * o).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3))
}

/// Text -> a random unit vector (documents and queries alike): "hot:N" texts are near the hot centre.
struct Hash768 {
    model: String,
    hot: Vec<f32>,
}

impl Hash768 {
    fn new(model: &str) -> Self {
        Self { model: model.into(), hot: unit_vec(0xBEEF) }
    }
}

impl Embedder for Hash768 {
    fn model_id(&self) -> &str {
        &self.model
    }
    fn dimension(&self) -> usize {
        DIM
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| if t.starts_with("hot") { near_vec(&self.hot, 0.97, fnv(t)) } else { unit_vec(fnv(t)) })
            .collect())
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_documents(texts)
    }
}

const NOW: i64 = 1_800_000_000;

/// How the rows of a perf store are laid out.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Stability log-uniform over bands `lo..=hi`, last recalled within two half-lives.
    Bands { lo: u32, hi: u32 },
    /// Same bands, every row recalled within the last minute.
    JustRecalled { lo: u32, hi: u32 },
    /// Same bands, every row at the same rank (log2 strength): the adversary of the window bound.
    Tied { lo: u32, hi: u32 },
    /// Every row created within the grace period (a flood).
    Flood,
}

fn stability_in(lo: u32, hi: u32, rng: &mut Rng, smax: f64) -> f64 {
    let a = f64::from(lo) / 2.0;
    let b = (f64::from(hi) + 1.0) / 2.0;
    (a + rng.unit() * (b - a)).exp2().min(smax)
}

fn make_row(i: u64, layout: Layout, rng: &mut Rng, smax: f64, with_vector: bool, model: &str) -> Trace {
    let (s, last, created, labile);
    match layout {
        Layout::Bands { lo, hi } => {
            s = stability_in(lo, hi, rng, smax);
            last = NOW - (s * rng.unit() * 2.0) as i64;
            created = last - (rng.unit() * 2.0 * YEAR) as i64;
            labile = false;
        }
        Layout::JustRecalled { lo, hi } => {
            s = stability_in(lo, hi, rng, smax);
            last = NOW - rng.below(60) as i64;
            created = last - (rng.unit() * 2.0 * YEAR) as i64;
            labile = false;
        }
        Layout::Tied { lo, hi } => {
            s = stability_in(lo, hi, rng, smax);
            let rank = -2.0 + (rng.unit() - 0.5) * 0.2;
            last = NOW - (s * (s.log2() - rank)) as i64;
            created = last - (rng.unit() * YEAR) as i64;
            labile = false;
        }
        Layout::Flood => {
            s = 86_400.0;
            last = NOW - rng.below(2 * 86_400) as i64;
            created = last;
            labile = false;
        }
    }
    let mut t = row(&format!("00{i:024}"), &format!("r{i}"), created, last, s, !labile);
    if with_vector {
        t.model_id = model.into();
        t.vector = unit_vec(i ^ 0xA5A5_0000_0000);
    }
    t
}

fn load(store: &mut Store, n: u64, layout: Layout, smax: f64, model: &str) {
    let mut rng = Rng(7);
    let mut i = 0;
    while i < n {
        let end = (i + 1000).min(n);
        let rows: Vec<Trace> = (i..end).map(|j| make_row(j, layout, &mut rng, smax, true, model)).collect();
        store.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
        i = end;
    }
}

fn engine_over(store: Store, cap: usize, c: Arc<Counters>, clock: &ManualClock, model: &str) -> Engram {
    let config = Config { capacity: cap, writes_per_day: 10_000_000, ..Config::default() };
    Engram::with_clock(Box::new(Spy { inner: store, c, probe: None }), Box::new(Hash768::new(model)), config, "UTC", clock.clock()).unwrap()
}

struct Measure {
    ms_mean: f64,
    ms_max: f64,
    rows: f64,
    band_rows: f64,
    finds: f64,
    counts: f64,
    nears: f64,
}

/// Runs `op` `n` times, reporting latency and counters per operation.
fn measure(c: &Counters, n: usize, mut op: impl FnMut(usize)) -> Measure {
    let before = c.snap();
    let (mut total, mut max) = (0.0f64, 0.0f64);
    for i in 0..n {
        let t = Instant::now();
        op(i);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        total += ms;
        max = max.max(ms);
    }
    let d = delta(before, c.snap());
    let per = |x: u64| x as f64 / n as f64;
    Measure { ms_mean: total / n as f64, ms_max: max, rows: per(d[1]), band_rows: per(d[3]), finds: per(d[0]), counts: per(d[4]), nears: per(d[5]) }
}

fn line(name: &str, n: u64, m: &Measure) {
    println!(
        "| {name:<46} | {n:>7} | {:>8.2} | {:>8.2} | {:>8.0} | {:>8.0} | {:>6.0} | {:>5.1} | {:>5.1} |",
        m.ms_mean, m.ms_max, m.rows, m.band_rows, m.finds, m.counts, m.nears
    );
}

fn header() {
    println!("| operation | N | ms mean | ms max | rows/op | band rows/op | find calls | count calls | near calls |");
    println!("|---|---|---|---|---|---|---|---|---|");
}

fn rss_mb() -> f64 {
    let pid = std::process::id();
    let out = std::process::Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"]).output();
    out.ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.rsplit("\",\"").next().map(str::to_owned))
        .and_then(|s| s.chars().filter(char::is_ascii_digit).collect::<String>().parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}


/// Eviction cost at one size: store full, idle for `idle` seconds.
fn evict_case(n: u64, layout: Layout, idle: f64, name: &str, ops: usize) -> Measure {
    let mut store = Store::in_memory().unwrap();
    let config = Config::default();
    load(&mut store, n, layout, config.max_stability, "hash768");
    let clock = ManualClock::new(NOW as f64 + idle);
    let c = Arc::new(Counters::default());
    let mut m = engine_over(store, n as usize, c.clone(), &clock, "hash768");
    let r = measure(&c, ops, |i| {
        clock.advance(1.0);
        let out = m.remember(&format!("fresh {name} {i}"), 1.0, "").unwrap();
        assert!(matches!(out, Remembered::Inserted { .. }));
    });
    line(name, n, &r);
    r
}

/// The ceiling this audit holds the cost of one remember to (SPEC section 8: an index read per band).
const BAND_ROW_BOUND: f64 = 60.0 * 200.0;

#[test]
fn cost_bounds_at_10k() {
    header();
    let n = 10_000;
    for (name, layout, idle) in [
        ("evict full idle 1y, 1 band", Layout::Bands { lo: 56, hi: 56 }, YEAR),
        ("evict full idle 1y, 10 bands", Layout::Bands { lo: 47, hi: 56 }, YEAR),
        ("evict full idle 1y, 41 bands", Layout::Bands { lo: 16, hi: 56 }, YEAR),
        ("evict full idle 1y, 57 bands", Layout::Bands { lo: 0, hi: 56 }, YEAR),
        ("evict full just recalled, 57 bands", Layout::JustRecalled { lo: 0, hi: 56 }, 0.0),
        ("evict full tied ranks, 57 bands", Layout::Tied { lo: 0, hi: 56 }, 0.0),
        ("evict full flood (all young)", Layout::Flood, 0.0),
    ] {
        let r = evict_case(n, layout, idle, name, 10);
        assert!(r.band_rows <= BAND_ROW_BOUND, "{name}: {} band rows read per remember (bound {BAND_ROW_BOUND})", r.band_rows);
        assert!(r.ms_max < 400.0, "{name}: one remember took {:.1} ms", r.ms_max);
    }
}

/// A store whose bands are tied (log2 strength equal everywhere) makes every band's window grow to
/// BAND_CAP: the rows read per remember are bounded by bands x (excess + 8 + 4x + cap).
#[test]
fn eviction_still_picks_the_weakest_in_a_tied_store() {
    let n = 3_000u64;
    let mut store = Store::in_memory().unwrap();
    let config = Config::default();
    load(&mut store, n, Layout::Tied { lo: 0, hi: 56 }, config.max_stability, "hash768");
    let clock = ManualClock::new(NOW as f64);
    let c = Arc::new(Counters::default());
    let p = probe(&clock, n as usize, config.grace_period);
    {
        // learn the shadow from the store
        let all = store.find(View::All, usize::MAX).unwrap();
        let mut sh = p.shadow.lock().unwrap();
        for t in all {
            sh.insert(t.id, (t.created_at, t.last_recall, t.stability));
        }
    }
    let cfgm = Config { capacity: n as usize, writes_per_day: 10_000_000, ..Config::default() };
    let mut m = Engram::with_clock(Box::new(Spy { inner: store, c: c.clone(), probe: Some(p.clone()) }), Box::new(Hash768::new("hash768")), cfgm, "UTC", clock.clock()).unwrap();
    for i in 0..40 {
        clock.advance(1.0);
        p.expect_evict.store(true, SeqCst);
        m.remember(&format!("fresh tied {i}"), 1.0, "").unwrap();
        p.expect_evict.store(false, SeqCst);
    }
    println!("tied store: {} evictions checked, {} not the weakest, worst gap {:+.4} log2", p.victims.load(SeqCst), p.inexact.load(SeqCst), *p.worst_gap.lock().unwrap());
    assert!(p.violations.lock().unwrap().is_empty(), "{:?}", p.violations.lock().unwrap());
    assert_eq!(p.inexact.load(SeqCst), 0, "{} evictions were not the weakest trace (worst gap {:+.4} log2 strength)", p.inexact.load(SeqCst), *p.worst_gap.lock().unwrap());
}

/// near() widening: junk rows (unusable) nearer than the answer must not make one recall cost more than ~100 ms
/// at 10k, nor read an unbounded number of vectors.
#[test]
fn recall_with_unusable_neighbours_stays_bounded_at_10k() {
    let n = 10_000u64;
    let mut store = Store::in_memory().unwrap();
    load(&mut store, n, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
    let mut emb = Hash768::new("hash768");
    let q = emb.embed_queries(&["what is the answer"]).unwrap().remove(0);
    for u in 0..2000u64 {
        // an unusable row (empty text) whose vector is closer to the query than anything else
        let mut t = row(&format!("zz{u:024}"), "", NOW - 100, NOW - 100, 86_400.0, true);
        t.model_id = "hash768".into();
        t.vector = near_vec(&q, 0.9, u + 5);
        store.write(&[Write::Insert(&t)]).unwrap();
    }
    let good = {
        let mut t = row("zzgood", "the real answer", NOW - 100, NOW - 100, 86_400.0, true);
        t.model_id = "hash768".into();
        t.vector = near_vec(&q, 0.8, 99);
        store.write(&[Write::Insert(&t)]).unwrap();
        t
    };
    let clock = ManualClock::new(NOW as f64);
    let c = Arc::new(Counters::default());
    let mut m = engine_over(store, 20_000, c.clone(), &clock, "hash768");
    let t = Instant::now();
    let rec = m.peek("what is the answer").unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let d = c.snap();
    println!("recall behind 2000 unusable rows at 10k: {ms:.1} ms, {} near calls, {} rows", d[5], d[1]);
    assert!(rec.recalled.iter().any(|r| r.trace.id == good.id), "the real answer is hidden behind the junk");
    assert!(ms < 1500.0 && d[5] <= 12, "{ms:.1} ms, {} near calls", d[5]);
}

/// MEDIUM (regression of this session's widening `Engram::find`): rows the engine cannot use (empty
/// text) that are listed as stale are re-read by every cued `remember` (`find(Stale, 1)` widens until
/// it finds a usable row or the view ends) and by every `reindex`/`dream` (`find(Stale, limit)`): the
/// cost is O(junk) per operation instead of O(1). SPEC section 8: no operation scans the store.
#[test]
fn stale_unusable_rows_are_not_rescanned_by_every_cued_remember_and_dream() {
    let mut store = Store::in_memory().unwrap();
    for u in 0..5_000u64 {
        let t = row(&format!("zz{u:024}"), "", NOW - 100, NOW - 100, 86_400.0, true);
        store.write(&[Write::Insert(&t)]).unwrap();
    }
    let clock = ManualClock::new(NOW as f64);
    let c = Arc::new(Counters::default());
    let mut m = engine_over(store, 100_000, c.clone(), &clock, "hash768");
    let r = measure(&c, 3, |i| {
        clock.advance(1.0);
        m.remember(&format!("with cue {i}"), 1.0, &format!("cue {i}")).unwrap();
    });
    println!("cued remember with 5000 stale unusable rows: {:.1} ms, {:.0} rows read, {:.0} finds", r.ms_mean, r.rows, r.finds);
    let d = measure(&c, 3, |_| {
        m.dream(&mut support::always_keep, None).unwrap();
    });
    println!("dream with 5000 stale unusable rows: {:.1} ms, {:.0} rows read, {:.0} finds", d.ms_mean, d.rows, d.finds);
    // Bounded (JUNK_READ + doubling), not proportional to the junk: 5000 junk rows cost about 3000 rows read.
    assert!(r.rows <= 4096.0, "a cued remember read {} rows to learn that no stale row exists", r.rows);
    assert!(d.rows <= 16_384.0, "a dream read {} rows", d.rows); // bounded by 8 x limit + JUNK_READ per read, not by the junk
}

// ----------------------------------------------------------- the heavy, ignored measurements

#[test]
#[ignore = "100k traces: run with --ignored --nocapture --test-threads=1"]
fn perf_table_eviction() {
    header();
    for n in [1_000u64, 10_000, 100_000] {
        for (name, layout, idle) in [
            ("evict full idle 1y, 1 band", Layout::Bands { lo: 56, hi: 56 }, YEAR),
            ("evict full idle 1y, 10 bands", Layout::Bands { lo: 47, hi: 56 }, YEAR),
            ("evict full idle 1y, 41 bands", Layout::Bands { lo: 16, hi: 56 }, YEAR),
            ("evict full idle 1y, 57 bands", Layout::Bands { lo: 0, hi: 56 }, YEAR),
            ("evict full just recalled, 57 bands", Layout::JustRecalled { lo: 0, hi: 56 }, 0.0),
            ("evict full tied ranks, 57 bands", Layout::Tied { lo: 0, hi: 56 }, 0.0),
            ("evict full flood (all young)", Layout::Flood, 0.0),
        ] {
            evict_case(n, layout, idle, name, if n >= 100_000 { 5 } else { 10 });
        }
    }
}

#[test]
#[ignore = "100k traces: run with --ignored --nocapture --test-threads=1"]
fn perf_table_basic_ops() {
    header();
    for n in [1_000u64, 10_000, 100_000] {
        let mut store = Store::in_memory().unwrap();
        let t = Instant::now();
        load(&mut store, n, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
        println!("(load {n}: {:.1} s, rss {:.0} MB)", t.elapsed().as_secs_f64(), rss_mb());
        // a labile queue of 40 seeds (own vectors, no cues): the dream does one near() search for each
        for i in 0..40u64 {
            let mut t = make_row(10_000_000 + i, Layout::Flood, &mut Rng(i + 1), 3.15e8, true, "hash768");
            t.consolidated = false;
            t.created_at = NOW - 10;
            store.write(&[Write::Insert(&t)]).unwrap();
        }
        let clock = ManualClock::new(NOW as f64);
        let c = Arc::new(Counters::default());
        let mut m = engine_over(store, n as usize + 100_000, c.clone(), &clock, "hash768");
        let r = measure(&c, 20, |i| {
            clock.advance(1.0);
            m.remember(&format!("plain {i}"), 1.0, "").unwrap();
        });
        line("remember (not full)", n, &r);
        let r = measure(&c, 20, |i| {
            clock.advance(1.0);
            m.remember(&format!("plain cue {i}"), 1.0, &format!("what is plain {i}")).unwrap();
        });
        line("remember with cue (not full)", n, &r);
        let r = measure(&c, 20, |i| {
            clock.advance(1.0);
            m.recall(&format!("question {i}")).unwrap();
        });
        line("recall, 1 cue", n, &r);
        let r = measure(&c, 10, |i| {
            clock.advance(1.0);
            m.recall(&format!("q{i} one\nq{i} two\nq{i} three\nq{i} four\nq{i} five\nq{i} six\nq{i} seven\nq{i} eight")).unwrap();
        });
        line("recall, 8 cues", n, &r);
        let r = measure(&c, 20, |i| {
            m.peek(&format!("question {i}")).unwrap();
        });
        line("peek, 1 cue", n, &r);
        let r = measure(&c, 50, |_| {
            m.stats().unwrap();
        });
        line("stats()", n, &r);
        let labile = m.stats().unwrap().labile;
        let r = measure(&c, 1, |_| {
            m.dream(&mut support::always_keep, None).unwrap();
        });
        line(&format!("dream (budget 5, {labile} labile seeds, keep)"), n, &r);
        let r = measure(&c, 3, |_| {
            m.dream(&mut support::always_keep, None).unwrap();
        });
        line("dream (budget 5, queue drained)", n, &r);
    }
}

#[test]
#[ignore = "100k traces: run with --ignored --nocapture --test-threads=1"]
fn perf_table_widening_reindex_seeds_count() {
    header();
    for n in [10_000u64, 100_000] {
        // ---- recall widening: a hot cluster of related traces
        for hot in [0u64, 100, 1000, 5000, 20_000] {
            let mut store = Store::in_memory().unwrap();
            load(&mut store, n, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
            let mut emb = Hash768::new("hash768");
            for h in 0..hot {
                let text = format!("hot {h}");
                let mut t = row(&format!("zz{h:024}"), &text, NOW - 1000, NOW - 1000, 20.0 * DAY, true);
                t.model_id = "hash768".into();
                t.vector = emb.embed_documents(&[&text]).unwrap().remove(0);
                store.write(&[Write::Insert(&t)]).unwrap();
            }
            let clock = ManualClock::new(NOW as f64);
            let c = Arc::new(Counters::default());
            let mut m = engine_over(store, 200_000, c.clone(), &clock, "hash768");
            let r = measure(&c, 3, |_| {
                m.peek("hot query").unwrap();
            });
            line(&format!("peek, {hot} related traces (hot cluster)"), n, &r);
        }
        // ---- unusable neighbours
        for junk in [0u64, 100, 1000, 10_000] {
            let mut store = Store::in_memory().unwrap();
            load(&mut store, n, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
            let mut emb = Hash768::new("hash768");
            let q = emb.embed_queries(&["junk query"]).unwrap().remove(0);
            for u in 0..junk {
                let mut t = row(&format!("zy{u:024}"), "", NOW - 100, NOW - 100, 86_400.0, true);
                t.model_id = "hash768".into();
                t.vector = near_vec(&q, 0.9, u + 5);
                store.write(&[Write::Insert(&t)]).unwrap();
            }
            let mut good = row("zzgood", "the real answer", NOW - 100, NOW - 100, 86_400.0, true);
            good.model_id = "hash768".into();
            good.vector = near_vec(&q, 0.8, 99);
            store.write(&[Write::Insert(&good)]).unwrap();
            let clock = ManualClock::new(NOW as f64);
            let c = Arc::new(Counters::default());
            let mut m = engine_over(store, 200_000, c.clone(), &clock, "hash768");
            let mut found = true;
            let r = measure(&c, 3, |_| {
                found = m.peek("junk query").unwrap().recalled.iter().any(|x| x.trace.id == "zzgood");
            });
            line(&format!("peek behind {junk} unusable rows (answer found: {found})"), n, &r);
            assert!(found || junk > 8192, "the answer is hidden behind only {junk} unusable rows");
        }
        // ---- reindex: a model switch makes every row stale
        {
            let mut store = Store::in_memory().unwrap();
            load(&mut store, n, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "old-model");
            let clock = ManualClock::new(NOW as f64);
            let c = Arc::new(Counters::default());
            let mut m = engine_over(store, 200_000, c.clone(), &clock, "new-model");
            let r = measure(&c, 3, |_| {
                assert_eq!(m.reindex(320).unwrap(), 320);
            });
            line("reindex(320), all rows stale", n, &r);
            let r = measure(&c, 3, |i| {
                m.remember(&format!("while stale {i}"), 1.0, "").unwrap();
            });
            line("remember while the store is stale", n, &r);
            let r = measure(&c, 3, |_| {
                m.peek("anything").unwrap();
            });
            line("peek while the store is stale (new model)", n, &r);
        }
        // ---- seeds paging with unusable labile rows
        for junk in [0u64, 10_000] {
            let mut store = Store::in_memory().unwrap();
            load(&mut store, n, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
            for u in 0..junk {
                let mut t = row(&format!("000{u:023}"), "", NOW - 100, NOW - 100, 86_400.0, false);
                t.model_id = "hash768".into();
                t.vector = unit_vec(u + 1_000_000);
                store.write(&[Write::Insert(&t)]).unwrap();
            }
            let clock = ManualClock::new(NOW as f64);
            let c = Arc::new(Counters::default());
            let mut m = engine_over(store, 200_000, c.clone(), &clock, "hash768");
            for i in 0..8 {
                clock.advance(1.0);
                m.remember(&format!("usable seed {i}"), 1.0, &format!("cue {i}")).unwrap();
            }
            let r = measure(&c, 3, |_| {
                m.set_dream_cursor("");
                m.dream(&mut support::always_keep, Some(1)).unwrap();
            });
            line(&format!("dream(budget 1) behind {junk} unusable labile rows"), n, &r);
            let r = measure(&c, 3, |_| {
                m.stats().unwrap();
            });
            line(&format!("stats() with {junk} unusable labile rows"), n, &r);
        }
        // ---- Store.count(view)
        {
            let mut store = Store::in_memory().unwrap();
            load(&mut store, n, Layout::Flood, Config::default().max_stability, "hash768");
            let t = Instant::now();
            let young = store.count(View::Young { created_after: NOW - 3 * 86_400 }).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!("| Store.count(Young) over {young} young rows | {n} | {ms:.2} | | | | | | |");
            let t = Instant::now();
            let all = store.count(View::All).unwrap();
            println!("| Store.count(All) = {all} | {n} | {:.3} | | | | | | |", t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let b = store.count(View::Band { band: 32, created_by: i64::MAX }).unwrap();
            println!("| Store.count(Band 32) = {b} | {n} | {:.2} | | | | | | |", t.elapsed().as_secs_f64() * 1e3);
        }
    }
}

#[test]
#[ignore = "100k traces: run with --ignored --nocapture --test-threads=1"]
fn perf_table_structural() {
    header();
    for n in [10_000u64, 100_000] {
        // ---- stale unusable rows (empty text, another model): reindex / remember-with-cue / dream re-read them
        for junk in [0u64, 1_000, 10_000, 100_000] {
            let mut store = Store::in_memory().unwrap();
            load(&mut store, n.min(20_000), Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
            for u in 0..junk {
                let t = row(&format!("zz{u:024}"), "", NOW - 100, NOW - 100, 86_400.0, true);
                store.write(&[Write::Insert(&t)]).unwrap();
            }
            for u in 0..5u64 {
                let t = row(&format!("zy{u:024}"), &format!("usable stale {u}"), NOW - 100, NOW - 100, 86_400.0, true);
                store.write(&[Write::Insert(&t)]).unwrap();
            }
            let clock = ManualClock::new(NOW as f64);
            let c = Arc::new(Counters::default());
            let mut m = engine_over(store, 500_000, c.clone(), &clock, "hash768");
            let r = measure(&c, 1, |_| {
                assert_eq!(m.reindex(5).unwrap(), 5);
            });
            line(&format!("reindex(5) behind {junk} stale unusable rows"), n.min(20_000), &r);
            let r = measure(&c, 3, |i| {
                m.remember(&format!("with cue {i}"), 1.0, &format!("cue {i}")).unwrap();
            });
            line(&format!("remember+cue with {junk} stale unusable rows"), n.min(20_000), &r);
            let r = measure(&c, 2, |_| {
                m.dream(&mut support::always_keep, None).unwrap();
            });
            line(&format!("dream (empty queue) with {junk} stale unusable rows"), n.min(20_000), &r);
        }
        // ---- identical vectors: an indivisible list is read whole
        for dup in [1_000u64, 20_000, 100_000] {
            if dup > n {
                continue;
            }
            let mut store = Store::in_memory().unwrap();
            let v = unit_vec(5);
            for chunk in (0..dup).collect::<Vec<_>>().chunks(1000) {
                let rows: Vec<Trace> = chunk
                    .iter()
                    .map(|&i| {
                        let mut t = row(&format!("0{i:025}"), &format!("same vector {i}"), NOW - 1000, NOW - 1000, 86_400.0, true);
                        t.model_id = "hash768".into();
                        t.vector = v.clone();
                        t
                    })
                    .collect();
                store.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
            }
            let clock = ManualClock::new(NOW as f64);
            let c = Arc::new(Counters::default());
            let mut m = engine_over(store, 500_000, c.clone(), &clock, "hash768");
            let r = measure(&c, 3, |i| {
                m.peek(&format!("query {i}")).unwrap();
            });
            line(&format!("peek over {dup} identical vectors"), dup, &r);
            let r = measure(&c, 3, |i| {
                m.remember(&format!("identical neighbour {i}"), 1.0, "").unwrap();
            });
            line(&format!("remember over {dup} identical vectors"), dup, &r);
        }
    }
}

#[test]
#[ignore = "100k traces on disk: run with --ignored --nocapture --test-threads=1"]
fn perf_purge_and_memory() {
    for n in [10_000u64, 100_000] {
        let path = support::temp_dir("r3d-purge").join("s.redb");
        let rss0 = rss_mb();
        {
            let mut store = Store::open(&path).unwrap();
            let t = Instant::now();
            load(&mut store, n, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
            println!("load {n} into a file store: {:.1} s, rss {:.0} -> {:.0} MB, file {:.0} MB", t.elapsed().as_secs_f64(), rss0, rss_mb(), std::fs::metadata(&path).unwrap().len() as f64 / 1048576.0);
        }
        // damage one row
        {
            let db = redb::Database::open(&path).unwrap();
            let wx = db.begin_write().unwrap();
            {
                let mut rows = wx.open_table(redb::TableDefinition::<&str, &[u8]>::new("rows")).unwrap();
                rows.insert(format!("00{:024}", n / 2).as_str(), b"{not json".as_slice()).unwrap();
            }
            wx.commit().unwrap();
        }
        let rss1 = rss_mb();
        let mut store = Store::open(&path).unwrap();
        let t = Instant::now();
        store.write(&[Write::Remove(&format!("00{:024}", n / 2))]).unwrap();
        println!("| purge: remove one damaged row | {n} | {:.1} ms |", t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(store.count(View::All).unwrap(), n as usize - 1);
        // the same through an insert over the damaged id
        let clock = ManualClock::new(NOW as f64);
        let c = Arc::new(Counters::default());
        let mut m = engine_over(store, n as usize + 10, c.clone(), &clock, "hash768");
        // cold, then warm queries against the file (1 GiB page cache)
        let t = Instant::now();
        m.peek("cold query").unwrap();
        let cold = t.elapsed().as_secs_f64() * 1e3;
        let r = measure(&c, 10, |i| {
            m.peek(&format!("warm query {i}")).unwrap();
        });
        println!("file store {n}: cold peek {cold:.1} ms, warm peek {:.1} ms; rss after reopen {:.0} -> {:.0} MB (cache 1 GiB cap)", r.ms_mean, rss1, rss_mb());
    }
}

/// The engine's own memory at 100k: the Engram struct, the Store's centroid cache (private) estimated from the list count.
#[test]
#[ignore = "100k traces: run with --ignored --nocapture --test-threads=1"]
fn perf_memory_100k() {
    let rss0 = rss_mb();
    let mut store = Store::in_memory().unwrap();
    load(&mut store, 100_000, Layout::Bands { lo: 20, hi: 56 }, Config::default().max_stability, "hash768");
    let rss1 = rss_mb();
    let clock = ManualClock::new(NOW as f64);
    let c = Arc::new(Counters::default());
    let mut m = engine_over(store, 100_000, c.clone(), &clock, "hash768");
    m.peek("warm").unwrap();
    let rss2 = rss_mb();
    for i in 0..30 {
        clock.advance(1.0);
        m.remember(&format!("mem {i}"), 1.0, "").unwrap();
        m.recall(&format!("mem q {i}")).unwrap();
    }
    let rss3 = rss_mb();
    println!(
        "memory at 100k x 768-d (in-memory redb): baseline {rss0:.0} MB; after load {rss1:.0} MB (raw vectors {:.0} MB); after first query {rss2:.0} MB; after 30 remember+recall {rss3:.0} MB",
        100_000.0 * 768.0 * 4.0 / 1048576.0
    );
    println!("Engram struct: {} bytes inline; its own heap is the parameters, <= writes_per_day f64 write times and <= inject_n exposures", std::mem::size_of::<Engram>());
}

// -------------------------------------------------------------- one-off structural probes

/// The store keeps its `Young` view scan O(young); the engine calls it on every full remember.
#[test]
fn young_count_cost_is_flat_in_the_store_size_when_few_are_young() {
    let mut times = Vec::new();
    for n in [2_000u64, 20_000] {
        let mut store = Store::in_memory().unwrap();
        let mut rng = Rng(3);
        let rows: Vec<Trace> = (0..n).map(|i| { let mut t = make_row(i, Layout::Bands { lo: 20, hi: 56 }, &mut rng, 3.15e8, false, ""); t.created_at = NOW - 10 * 86_400 - i as i64; t }).collect();
        for chunk in rows.chunks(1000) {
            store.write(&chunk.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
        }
        let t = Instant::now();
        for _ in 0..200 {
            assert_eq!(store.count(View::Young { created_after: NOW - 3 * 86_400 }).unwrap(), 0);
        }
        times.push(t.elapsed().as_secs_f64() * 1e6 / 200.0);
    }
    println!("count(Young) with nothing young: {:.1} us at 2k, {:.1} us at 20k", times[0], times[1]);
    assert!(times[1] < times[0] * 8.0 + 200.0, "count(Young) grows with the store: {times:?}");
}

/// A dream's seed paging must not re-read the same unusable labile rows without bound.
#[test]
fn seed_paging_cost_is_bounded_by_the_queue_not_repeated_per_dream_at_10k_junk() {
    let mut store = Store::in_memory().unwrap();
    for u in 0..10_000u64 {
        let mut t = row(&format!("000{u:023}"), "", NOW - 100, NOW - 100, 86_400.0, false);
        t.model_id = "hash768".into();
        t.vector = unit_vec(u + 1_000_000);
        store.write(&[Write::Insert(&t)]).unwrap();
    }
    let clock = ManualClock::new(NOW as f64);
    let c = Arc::new(Counters::default());
    let mut m = engine_over(store, 100_000, c.clone(), &clock, "hash768");
    for i in 0..8 {
        clock.advance(1.0);
        m.remember(&format!("usable seed {i}"), 1.0, &format!("cue {i}")).unwrap();
    }
    let r = measure(&c, 3, |_| {
        m.set_dream_cursor("");
        m.dream(&mut support::always_keep, Some(1)).unwrap();
    });
    println!("dream behind 10k unusable labile rows: {:.1} ms, {:.0} rows read, {:.0} finds", r.ms_mean, r.rows, r.finds);
    assert!(r.ms_max < 1000.0 && r.rows < 12_000.0, "{:.1} ms, {} rows", r.ms_max, r.rows);
}

#[allow(dead_code)]
fn unused(_: HashMap<u8, u8>) {}
