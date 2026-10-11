//! Round-2 audit A: dream cursor, eviction windows, commit/cite/forget, unit-norm rule.
//! A FAILING test here is a confirmed defect; see audit/round2_a.md. Public API only.

mod support;

use engram::{Config, DreamAction, Hit, Remembered, Storage, Trace, View, Write};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use support::*;

const MODEL: &str = "fake/token-overlap";

fn row(id: &str, text: &str, created: i64, last_recall: i64, stability: f64, consolidated: bool, indexed: bool) -> Trace {
    Trace {
        id: id.into(),
        text: text.into(),
        created_at: created,
        tz: "UTC;+00:00".into(),
        last_recall,
        stability,
        consolidated,
        cue: String::new(),
        model_id: if indexed { MODEL.into() } else { String::new() },
        vector: if indexed { FakeEmbedder::new().embed(text) } else { Vec::new() },
    }
}

/// A store that counts the rows its `find` hands out and the calls made.
#[derive(Clone, Default)]
struct Counting {
    inner: SharedStore,
    rows: Arc<AtomicUsize>,
    finds: Arc<AtomicUsize>,
}

impl Storage for Counting {
    fn find(&self, view: View<'_>, limit: usize) -> engram::Result<Vec<Trace>> {
        let r = self.inner.find(view, limit)?;
        self.rows.fetch_add(r.len(), Ordering::SeqCst);
        self.finds.fetch_add(1, Ordering::SeqCst);
        Ok(r)
    }
    fn count(&self, view: View<'_>) -> engram::Result<usize> {
        self.inner.count(view)
    }
    fn near(&self, q: &[f32], m: &str, c: f64, k: usize, s: Option<i64>) -> engram::Result<Vec<Hit>> {
        self.inner.near(q, m, c, k, s)
    }
    fn write(&mut self, w: &[Write<'_>]) -> engram::Result<()> {
        self.inner.write(w)
    }
    fn clear(&mut self) -> engram::Result<()> {
        self.inner.clear()
    }
}

fn fill(store: &mut impl Storage, n: usize, mk: impl Fn(usize) -> Trace) {
    let mut i = 0;
    while i < n {
        let hi = (i + 5000).min(n);
        let ts: Vec<Trace> = (i..hi).map(&mk).collect();
        let ws: Vec<Write<'_>> = ts.iter().map(Write::Insert).collect();
        store.write(&ws).unwrap();
        i = hi;
    }
}

fn rid(i: usize) -> String {
    format!("01{:024}", i)
}

/// Rows read by one remember into an idle full store of `n` rows whose stabilities are `stab(i)`
/// and last-recall ages are `age(i)` seconds.
fn read_cost(n: usize, stab: impl Fn(usize) -> f64, age: impl Fn(usize) -> i64) -> (usize, usize, f64) {
    let mut store = Counting::default();
    fill(&mut store, n, |i| {
        let a = age(i);
        row(&rid(i), &format!("fact {i}"), START as i64 - a - 10 * 86400, START as i64 - a, stab(i), true, false)
    });
    let c = Config { capacity: n, ..cfg() };
    let (mut m, _clock) = build_with(c, Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    store.rows.store(0, Ordering::SeqCst);
    store.finds.store(0, Ordering::SeqCst);
    let t0 = std::time::Instant::now();
    let r = m.remember("a brand new fact about nothing", 1.0, "").unwrap();
    let dt = t0.elapsed().as_secs_f64();
    assert!(matches!(r, Remembered::Inserted { .. }));
    (store.rows.load(Ordering::SeqCst), store.finds.load(Ordering::SeqCst), dt)
}

#[test]
#[ignore = "measurement"]
fn measure_victims_cost() {
    let day = 86400.0;
    for &n in &[1_000usize, 10_000, 100_000] {
        let one = read_cost(n, |_| day, |i| (i as i64 % 3600) + 40 * 86400);
        println!("N={n} one band (S=1d, burst, idle 40d): rows={} finds={} t={:.3}s", one.0, one.1, one.2);
        let one2 = read_cost(n, |_| day, |i| (i as i64) * 30);
        println!("N={n} one band (S=1d, spread 30s apart): rows={} finds={} t={:.3}s", one2.0, one2.1, one2.2);
        let spread = read_cost(n, |i| day * 1.5f64.powi((i % 20) as i32), |i| 5 * 86400 + (i as i64 * 7) % 9999);
        println!("N={n} 20 stabilities: rows={} finds={} t={:.3}s", spread.0, spread.1, spread.2);
        let wide = read_cost(n, |i| 2f64.powf(0.5 * ((i % 56) as f64) + 0.25), |i| 3 * 86400 + (i as i64 * 7) % 9999);
        println!("N={n} 56 bands (half-octave): rows={} finds={} t={:.3}s", wide.0, wide.1, wide.2);
        // Every band equally weak (rank = -10 + jitter): no band's bound can be proven above the weakest.
        let eq = read_cost(n, |i| 2f64.powf(0.5 * ((i % 41) as f64) + 0.25), |i| {
            let s = 2f64.powf(0.5 * ((i % 41) as f64) + 0.25);
            (s * (s.log2() + 10.0) + s * 0.01 * ((i / 41) % 50) as f64) as i64
        });
        println!("N={n} 41 bands equal rank: rows={} finds={} t={:.3}s", eq.0, eq.1, eq.2);
    }
}

// ------------------------------------------------------------------ helpers

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn rank_of(t: &Trace, now: f64, smax: f64) -> f64 {
    let s = if t.stability.is_nan() { 1.0 } else { t.stability }.max(1.0).min(smax);
    let dt = (now - (t.last_recall.min(now as i64)) as f64).max(0.0);
    s.log2() - dt / s
}

/// Runs one remember into a store of `n` random rows (capacity `cap`) and returns
/// (evicted ids, expected victim ranks by brute force over the eligible rows, actual ranks).
fn evict_round(seed: u64, n: usize, cap: usize, wild: bool) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng(seed | 1);
    let mut store = SharedStore::default();
    let c = Config { capacity: cap, ..cfg() };
    let smax = c.max_stability;
    let now = START;
    let mut ts = Vec::new();
    for i in 0..n {
        let s = if wild {
            // below 1, in band, above S_max
            (rng.unit() * 32.0 - 1.0).exp2()
        } else {
            (rng.unit() * 2.0 + 16.0).exp2()
        };
        let age = if wild && rng.unit() < 0.1 { -(rng.unit() * 1e5) } else { rng.unit() * 3e7 * rng.unit() };
        let lr = now as i64 - age as i64;
        let young = rng.unit() < 0.03;
        let created = if young { now as i64 - (rng.unit() * 2.0 * 86400.0) as i64 } else { lr.min(now as i64) - 5 * 86400 };
        ts.push(row(&rid(i), &format!("w{i} alpha{}", i % 7), created, lr, s, true, false));
    }
    let ws: Vec<Write<'_>> = ts.iter().map(Write::Insert).collect();
    store.write(&ws).unwrap();
    let (mut m, _clock) = build_with(c.clone(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let before = rows(&m);
    let r = m.remember("fresh unrelated sentence", 1.0, "").unwrap();
    assert!(matches!(r, Remembered::Inserted { .. }));
    let after: HashSet<String> = rows(&m).into_iter().map(|t| t.id).collect();
    let gone: Vec<&Trace> = before.iter().filter(|t| !after.contains(&t.id)).collect();
    let excess = (n + 1).saturating_sub(cap);
    // brute force: protected = young unless flood
    let cutoff = (now - c.grace_period) as i64;
    let young = before.iter().filter(|t| t.created_at > cutoff).count() + 1;
    let mut pool: Vec<f64> = before.iter().filter(|t| young > cap / 10 || t.created_at <= cutoff).map(|t| rank_of(t, now, smax)).collect();
    if pool.len() < excess || young > cap / 10 {
        pool = before.iter().map(|t| rank_of(t, now, smax)).collect();
    }
    pool.sort_by(f64::total_cmp);
    let mut expect: Vec<f64> = pool.into_iter().take(excess).map(|r| r.max(-64.0)).collect();
    let mut got: Vec<f64> = gone.iter().map(|t| rank_of(t, now, smax).max(-64.0)).collect();
    expect.sort_by(f64::total_cmp);
    got.sort_by(f64::total_cmp);
    (expect, got)
}

#[test]
fn eviction_matches_brute_force_within_the_band_cap() {
    let mut bad = Vec::new();
    for seed in 1..=25u64 {
        for &(n, cap, wild) in &[(299usize, 299usize, false), (299, 299, true), (330, 300, true), (330, 300, false)] {
            let (e, g) = evict_round(seed * 7919, n, cap, wild);
            // the young-protected pool is only claimed exact when enough old rows exist
            if e.len() != g.len() || e.iter().zip(&g).any(|(a, b)| (a - b).abs() > 1e-6) {
                bad.push((seed, n, cap, wild, e, g));
            }
        }
    }
    assert!(bad.is_empty(), "{} mismatches, first: {:?}", bad.len(), bad.first());
}

/// Documented approximation: a band is read to at most BAND_CAP = 512 rows (SPEC 3).
/// One dense band whose first 512 rows are stronger (S = 1.41 lo) than the ones behind them.
#[test]
#[ignore = "accepted: SPEC 3 caps one band at 512 rows, so a weaker row deeper than that is not found"]
fn eviction_beyond_the_band_cap_is_approximate() {
    let mut store = SharedStore::default();
    let c = Config { capacity: 2000, ..cfg() };
    let now = START;
    let d = 20.0 * 65536.0;
    let ts: Vec<Trace> = (0..2000)
        .map(|i| {
            let s = if i < 512 { 92_000.0 } else { 65_600.0 };
            let lr = now as i64 - d as i64 + i as i64 * 100;
            row(&rid(i), &format!("w{i}"), lr - 5 * 86400, lr, s, true, false)
        })
        .collect();
    let ws: Vec<Write<'_>> = ts.iter().map(Write::Insert).collect();
    store.write(&ws).unwrap();
    let (mut m, _c) = build_with(c.clone(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let weakest = ts.iter().min_by(|a, b| rank_of(a, now, c.max_stability).total_cmp(&rank_of(b, now, c.max_stability))).unwrap().id.clone();
    m.remember("fresh unrelated sentence", 1.0, "").unwrap();
    assert!(mem(&m, &weakest).is_none(), "the weakest row {weakest} survived");
}

// ------------------------------------------------------------------ the dream cursor

fn labile(i: usize) -> Trace {
    row(&rid(i), &format!("alpha beta gamma delta n{i}"), START as i64 - 100, START as i64 - 100, 86400.0, false, true)
}

fn seed_store(n: usize) -> SharedStore {
    let mut store = SharedStore::default();
    let mut anchor = row(&rid(0), "alpha beta gamma delta anchor", START as i64 - 1000, START as i64 - 1000, 86400.0, true, true);
    anchor.id = rid(0);
    let mut ts = vec![anchor];
    ts.extend((1..=n).map(labile));
    let ws: Vec<Write<'_>> = ts.iter().map(Write::Insert).collect();
    store.write(&ws).unwrap();
    store
}

fn examined(reports: &[engram::DreamReport]) -> Vec<String> {
    reports.iter().filter(|r| !r.before.is_empty()).map(|r| r.before[0].id.clone()).collect()
}

/// Model: the cycle after `cursor` over the live labile ids, `b` of them.
fn rotation(live: &[String], cursor: &str, b: usize) -> Vec<String> {
    let mut v: Vec<String> = live.iter().filter(|x| x.as_str() > cursor).cloned().collect();
    v.extend(live.iter().filter(|x| x.as_str() <= cursor).cloned());
    v.truncate(b);
    v
}

/// Under random arrivals, forgets (also of the cursor seed) and budgets, with an LLM that
/// never decides, every dream examines exactly the next seeds of the cycle.
#[test]
#[ignore = "sound: the cursor cycle matches the model (kept as a regression guard)"]
fn cursor_cycle_matches_the_model() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let store = seed_store(25);
    let (mut m, _c) = build_with(Config { dream_budget: 3, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let mut live: Vec<String> = (1..=25).map(rid).collect();
    let mut next = 26;
    let mut cursor = String::new();
    for round in 0..200 {
        match rng.next() % 4 {
            0 => {
                store.clone().write(&[Write::Insert(&labile(next))]).unwrap();
                live.push(rid(next));
                next += 1;
            }
            1 if live.len() > 3 => {
                let k = if rng.next().is_multiple_of(2) { live.iter().position(|x| *x == cursor).unwrap_or(0) } else { (rng.next() as usize) % live.len() };
                let id = live.remove(k);
                m.forget(&id).unwrap();
            }
            _ => {}
        }
        let b = 1 + (rng.next() as usize) % 4;
        let want = rotation(&live, &cursor, b.min(live.len()));
        let got = examined(&m.dream(&mut always_fail, Some(b)).unwrap());
        assert_eq!(got, want, "round {round}, cursor {cursor}");
        if let Some(l) = got.last() {
            cursor = l.clone();
        }
    }
}

/// A host that reopens the engine per dream (a CLI run, an app start) loses the in-memory
/// cursor: with `n > budget` undecidable seeds only the first `budget` are ever examined.
#[test]
fn reopened_engine_still_reaches_every_seed() {
    let store = seed_store(20);
    let mut seen: HashSet<String> = HashSet::new();
    let mut cursor = String::new(); // the host keeps it between runs (Engram::dream_cursor)
    for _ in 0..25 {
        let (mut m, _c) = build_with(Config { dream_budget: 1, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(store.clone()));
        m.set_dream_cursor(&cursor);
        seen.extend(examined(&m.dream(&mut always_fail, Some(1)).unwrap()));
        cursor = m.dream_cursor().to_owned();
    }
    assert!(seen.len() >= 20, "only {} of 20 undecidable seeds were ever examined in 25 dreams", seen.len());
}

// ------------------------------------------------------------------ the unit-norm rule

fn scaled(id: &str, text: &str, k: f32) -> Trace {
    let mut t = row(id, text, START as i64 - 10 * 86400, START as i64 - 10 * 86400, 86400.0, true, true);
    for x in &mut t.vector {
        *x *= k;
    }
    t
}

/// A row whose index vector is off unit length (a quantising store, an older writer) is
/// "unindexed" for the engine (SPEC 2): the engine must heal it like any unindexed row,
/// by re-embedding it. `Stale` lists only rows of other models, so the row is never repaired
/// and stays invisible to recall forever.
#[test]
fn off_norm_row_is_reindexed_not_lost() {
    let mut store = SharedStore::default();
    store.write(&[Write::Insert(&scaled(&rid(1), "the cat sat on the mat", 1.04))]).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    // A rescaled vector is not searched with; the scrub in reindex() embeds the row again.
    let n = m.reindex(10).unwrap();
    m.dream(&mut always_keep, Some(1)).unwrap();
    let found = m.recall("the cat sat on the mat").unwrap().recalled.len();
    assert!(n > 0 || found > 0, "reindex repaired {n} rows; recall finds {found}: the row is lost to search for good");
}

/// A store that keeps vectors as int8 (scale = max |x| / 127) and hands them back dequantised:
/// "precision and quantisation are the store's freedom" (SPEC 2). Norm drift of such vectors
/// exceeds 1e-3, and the engine then drops them from search.
struct Int8(SharedStore);

fn q8(t: &mut Trace) {
    let m = t.vector.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if m > 0.0 {
        for x in &mut t.vector {
            *x = (*x / m * 127.0).round() / 127.0 * m;
        }
    }
}

impl Storage for Int8 {
    fn find(&self, v: View<'_>, l: usize) -> engram::Result<Vec<Trace>> {
        let mut r = self.0.find(v, l)?;
        r.iter_mut().for_each(q8);
        Ok(r)
    }
    fn count(&self, v: View<'_>) -> engram::Result<usize> {
        self.0.count(v)
    }
    fn near(&self, q: &[f32], m: &str, c: f64, k: usize, s: Option<i64>) -> engram::Result<Vec<Hit>> {
        let mut h = self.0.near(q, m, c, k, s)?;
        h.iter_mut().for_each(|x| q8(&mut x.trace));
        Ok(h)
    }
    fn write(&mut self, w: &[Write<'_>]) -> engram::Result<()> {
        self.0.write(w)
    }
    fn clear(&mut self) -> engram::Result<()> {
        self.0.clear()
    }
}

#[test]
fn int8_quantised_store_keeps_its_rows_searchable() {
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(Int8(SharedStore::default())));
    let texts: Vec<String> = (0..60).map(|i| format!("distinct fact number{i} about topic{} and thing{}", i * 3, i * 7)).collect();
    for t in &texts {
        insert(&mut m, t, 1.0, "");
    }
    let lost: Vec<&String> = texts.iter().filter(|t| m.peek(t).unwrap().recalled.iter().all(|r| &r.trace.text != *t)).collect();
    assert!(lost.is_empty(), "{} of {} rows are unfindable through an int8-quantising store, e.g. {:?}", lost.len(), texts.len(), lost.first());
}

// ------------------------------------------------------------------ commit / cite / forget

use std::sync::atomic::AtomicBool;

#[derive(Clone, Default)]
struct Flaky {
    inner: SharedStore,
    fail: Arc<AtomicBool>,
}

impl Storage for Flaky {
    fn find(&self, v: View<'_>, l: usize) -> engram::Result<Vec<Trace>> {
        self.inner.find(v, l)
    }
    fn count(&self, v: View<'_>) -> engram::Result<usize> {
        self.inner.count(v)
    }
    fn near(&self, q: &[f32], m: &str, c: f64, k: usize, s: Option<i64>) -> engram::Result<Vec<Hit>> {
        self.inner.near(q, m, c, k, s)
    }
    fn write(&mut self, w: &[Write<'_>]) -> engram::Result<()> {
        if self.fail.load(Ordering::SeqCst) {
            return Err("disk full".into());
        }
        self.inner.write(w)
    }
    fn clear(&mut self) -> engram::Result<()> {
        self.inner.clear()
    }
}

#[test]
#[ignore = "sound: cite is retryable after a store error and counts once"]
fn cite_retry_boosts_once() {
    let store = Flaky::default();
    let (mut m, clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let id = insert(&mut m, "the cat sat on the mat", 1.0, "");
    clock.advance(10.0 * 86400.0);
    let r = m.recall("the cat sat on the mat").unwrap();
    assert_eq!(r.recalled.len(), 1);
    let s0 = mem(&m, &id).unwrap().stability;
    let reply = format!("yes 《id:{id}》 and again 《id:{id}》");
    store.fail.store(true, Ordering::SeqCst);
    assert!(m.cite(&reply).is_err());
    store.fail.store(false, Ordering::SeqCst);
    assert_eq!(m.cite(&reply).unwrap(), vec![id.clone()]);
    let s1 = mem(&m, &id).unwrap().stability;
    assert!(s1 > s0);
    assert!(m.cite(&reply).unwrap().is_empty(), "a spent citation must not boost twice");
    assert_eq!(mem(&m, &id).unwrap().stability, s1);
}

#[test]
#[ignore = "sound: a removed trace takes its exposure with it"]
fn removed_trace_leaves_no_exposure_for_a_reused_id() {
    let mut store = SharedStore::default();
    let (mut m, clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let id = insert(&mut m, "the cat sat on the mat", 1.0, "");
    clock.advance(10.0 * 86400.0);
    assert_eq!(m.recall("the cat sat on the mat").unwrap().recalled.len(), 1);
    m.forget(&id).unwrap();
    let mut t = row(&id, "an unrelated dog barked", START as i64, clock.get() as i64 - 5 * 86400, 86400.0, true, true);
    t.created_at = START as i64;
    store.write(&[Write::Insert(&t)]).unwrap();
    assert!(m.cite(&format!("《id:{id}》")).unwrap().is_empty());
    assert_eq!(mem(&m, &id).unwrap().stability, 86400.0);
}

#[test]
#[ignore = "sound: rows the engine cannot use can still be forgotten"]
fn forget_removes_unusable_rows() {
    let mut store = SharedStore::default();
    let mut empty = row("01EMPTY", "x", START as i64 - 5, START as i64 - 5, 86400.0, true, false);
    empty.text = String::new();
    let bad_id = row("01BAD\u{300a}ID", "text", START as i64 - 5, START as i64 - 5, 86400.0, true, false);
    let nan = row("01NAN", "nan stability", START as i64 - 5, START as i64 - 5, 5.0, true, false);
    store.write(&[Write::Insert(&empty), Write::Insert(&bad_id), Write::Insert(&nan)]).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    for id in ["01EMPTY", "01BAD\u{300a}ID", "01NAN"] {
        assert!(m.forget(id).unwrap().is_some(), "{id}");
        assert!(m.store().find(View::Id(id), 1).unwrap().is_empty(), "{id} still stored");
    }
}

/// SPEC 3: "one band is read up to 512 rows". The window grows 9 -> 36 -> 144 -> 512 and each
/// step re-reads the band from its head, so one dense band costs 701 rows, not 512.
#[test]
fn one_band_read_stays_within_the_stated_cap() {
    let (rows, _finds, _t) = read_cost(5000, |_| 86400.0, |i| 40 * 86400 + i as i64 % 3600);
    assert!(rows <= 512, "one remember read {rows} rows from a single band (SPEC 3: at most 512)");
}

#[test]
#[ignore = "sound: the embedder's own f32 normalisation at 3072 dims passes the unit-norm rule"]
fn engine_normalised_3072_dim_vectors_are_never_discarded() {
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::with(3072, "fake/3072")), Box::new(SharedStore::default()));
    let texts: Vec<String> = (0..80).map(|i| format!("fact {i} about subject{} with object{}", i * 5, i * 11)).collect();
    for t in &texts {
        insert(&mut m, t, 1.0, "");
    }
    let lost = texts.iter().filter(|t| m.peek(t).unwrap().recalled.iter().all(|r| &r.trace.text != *t)).count();
    assert_eq!(lost, 0);
}

#[test]
#[ignore = "measurement"]
fn measure_seed_paging_over_unusable_rows() {
    for &u in &[100usize, 1000, 5000] {
        let mut store = Counting::default();
        let mut ts: Vec<Trace> = (1..=7).map(labile).collect();
        for i in 0..u {
            let mut t = labile(100 + i);
            t.text = String::new();
            ts.push(t);
        }
        ts.extend((100 + u..110 + u).map(labile));
        let ws: Vec<Write<'_>> = ts.iter().map(Write::Insert).collect();
        store.write(&ws).unwrap();
        let (m, _c) = build_with(Config { dream_budget: 1, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(store.clone()));
        store.finds.store(0, Ordering::SeqCst);
        store.rows.store(0, Ordering::SeqCst);
        let t0 = std::time::Instant::now();
        let mut m = m;
        let c = m.clusters(Some(1)).unwrap();
        println!("U={u}: clusters={} finds={} rows={} t={:?}", c.len(), store.finds.load(Ordering::SeqCst), store.rows.load(Ordering::SeqCst), t0.elapsed());
    }
}

// ------------------------------------------- top-k taken before unusable rows are dropped

fn junk(i: usize, like: &str) -> Trace {
    let mut t = row(&rid(500 + i), "x", START as i64 - 5000, START as i64 - 5000, 86400.0, true, true);
    t.text = String::new(); // unusable: SPEC 2 "text empty rows are not used"
    t.vector = FakeEmbedder::new().embed(like);
    t
}

/// `remember` asks the store for the single nearest trace and only then validates it: an
/// unusable row (or one with an off-norm vector) that happens to be nearest hides the usable
/// related trace right behind it, so the write is born consolidated and never dreamed.
#[test]
fn unusable_nearest_row_does_not_hide_the_related_trace() {
    let mut store = SharedStore::default();
    let old = row(&rid(1), "alpha beta gamma delta old", START as i64 - 9000, START as i64 - 9000, 86400.0, true, true);
    store.write(&[Write::Insert(&old), Write::Insert(&junk(1, "alpha beta gamma delta new"))]).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    match m.remember("alpha beta gamma delta new", 1.0, "").unwrap() {
        Remembered::Inserted { trace, cosine, .. } => {
            assert!(cosine > 0.5 && !trace.consolidated, "cosine {cosine}, consolidated {}: the usable related trace was missed", trace.consolidated)
        }
        other => panic!("{other:?}"),
    }
}

/// The dream's candidate search has the same shape: `dream_max_members` hits are taken, then
/// validated. With that many unusable rows nearer than the real candidate, the seed finds
/// "nothing to reactivate" and is settled as consolidated without ever being compared.
#[test]
fn unusable_rows_do_not_hide_dream_candidates() {
    let mut store = SharedStore::default();
    let old = row(&rid(1), "alpha beta gamma delta old", START as i64 - 9000, START as i64 - 9000, 86400.0, true, true);
    let seed = labile(2);
    let mut control = SharedStore::default();
    control.write(&[Write::Insert(&old), Write::Insert(&seed)]).unwrap();
    let (mut c, _k) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(control));
    assert!(c.dream(&mut merge_to_gist, Some(1)).unwrap().iter().any(|r| r.action == DreamAction::Replace), "control");
    let mut ts = vec![old, seed];
    ts.extend((0..8).map(|i| junk(i, "alpha beta gamma delta n2")));
    let ws: Vec<Write<'_>> = ts.iter().map(Write::Insert).collect();
    store.write(&ws).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let reports = m.dream(&mut merge_to_gist, Some(1)).unwrap();
    assert!(reports.iter().any(|r| r.action == DreamAction::Replace), "the seed was settled without meeting its older candidate: {reports:?}");
}
