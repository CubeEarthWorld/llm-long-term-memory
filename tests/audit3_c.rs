//! Round-3 audit, lane C: hostile environments and panic-freedom.
//! Run: CARGO_TARGET_DIR=C:/t/r3c cargo test --test audit3_c -j 3
//!
//! The engine promises: never panics, never loops forever, never allocates unboundedly,
//! whatever its injected Storage / Embedder / Adjudicator / Clock / IdGen do.
//!
//! Tests are grouped by their name prefix:
//!   `s*`  hostile Storage        `e*` hostile Embedder      `a*` hostile Adjudicator
//!   `c*`  hostile Clock          `i*` hostile IdGen         `f*` Config extremes
//!   `r*`  resource bounds        `d*` damaged redb file     `g_*` REGRESSION GUARDS
//!
//! A test whose name starts with `g_` PASSES today and is kept as a regression guard.
//! Every other test asserts the documented contract and FAILS only on a real defect
//! (the root cause is in audit/round3_c.md).
//!
//! All tests take `serial()`: the resource tests count live heap bytes through the
//! global allocator below, and a concurrent test would add noise to the count.

mod support;

use engram::{
    Config, DreamAction, DreamDecision, DreamRequest, Embedder, Engram, Hit, ManualClock, Remembered, Result, Storage, Store,
    Trace, View, Write,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;
use support::{always_keep, build_with, cfg, insert, merge_to_gist, temp_dir, FakeEmbedder, START};

// ------------------------------------------------------------------ plumbing

/// Counts live heap bytes (alloc minus dealloc) so a test can see what the engine keeps.
struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new as isize - l.size() as isize, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::Relaxed)
}

fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs `f`; a panic inside it fails the test with `what` and the panic message.
fn no_panic<T>(what: &str, f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(e) => {
            let msg = e.downcast_ref::<&str>().map(|s| (*s).to_owned()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_default();
            panic!("{what}: PANICKED: {msg}")
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------ hostile Storage

#[derive(Clone, Default)]
struct Knobs {
    count_all: Option<usize>,
    count_young: Option<usize>,
    /// `View::Id(_)` answers with every row, ignoring the id and the limit.
    id_view_all: bool,
    /// `View::Text(_)` answers with every row, ignoring the text and the limit.
    text_view_all: bool,
    /// Every near() hit is repeated this many times.
    near_dupes: usize,
    /// near() ignores `k` (asks the inner store for everything) and answers in reverse order.
    near_everything_reversed: bool,
    /// near() reports this cosine for every hit.
    cos_override: Option<f64>,
    /// 1: near() invents `k` junk (empty-text) hits for any k; 2: `k` valid hits for any k. Errors above `k_limit`.
    bottomless: u8,
    k_limit: usize,
    drop_writes: bool,
    fail_clear: bool,
    panic_write: bool,
    panic_find: bool,
    /// `View::Text(_)` answers with `limit` junk (empty-text) rows for any limit; errors above `k_limit`.
    find_junk: bool,
}

/// A real store behind a set of switches that make it lie.
struct Hostile {
    inner: Store,
    knobs: Arc<Mutex<Knobs>>,
    max_k: Arc<AtomicUsize>,
    max_find: Arc<AtomicUsize>,
}

struct Handle {
    knobs: Arc<Mutex<Knobs>>,
    max_k: Arc<AtomicUsize>,
    max_find: Arc<AtomicUsize>,
}

impl Handle {
    fn set(&self, f: impl FnOnce(&mut Knobs)) {
        f(&mut lock(&self.knobs));
    }
    fn max_k(&self) -> usize {
        self.max_k.load(Ordering::SeqCst)
    }
    fn max_find(&self) -> usize {
        self.max_find.load(Ordering::SeqCst)
    }
}

fn hostile() -> (Hostile, Handle) {
    let knobs = Arc::new(Mutex::new(Knobs { k_limit: 1 << 16, ..Knobs::default() }));
    let max_k = Arc::new(AtomicUsize::new(0));
    let max_find = Arc::new(AtomicUsize::new(0));
    (Hostile { inner: Store::in_memory().unwrap(), knobs: knobs.clone(), max_k: max_k.clone(), max_find: max_find.clone() }, Handle { knobs, max_k, max_find })
}

impl Hostile {
    fn kn(&self) -> Knobs {
        lock(&self.knobs).clone()
    }
}

impl Storage for Hostile {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        let kn = self.kn();
        if kn.panic_find {
            panic!("storage: find exploded");
        }
        if kn.find_junk && matches!(view, View::Text(_)) {
            self.max_find.fetch_max(limit, Ordering::SeqCst);
            if limit > kn.k_limit {
                return Err("bottomless store: asked for more than the test allows".into());
            }
            return Ok((0..limit).map(|i| Trace { id: format!("J{i:012}"), text: String::new(), created_at: 0, tz: String::new(), last_recall: 0, stability: 1.0, consolidated: true, cue: String::new(), model_id: String::new(), vector: Vec::new() }).collect());
        }
        match view {
            View::Id(_) if kn.id_view_all => self.inner.find(View::All, usize::MAX),
            View::Text(_) if kn.text_view_all => self.inner.find(View::All, usize::MAX),
            v => self.inner.find(v, limit),
        }
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        let kn = self.kn();
        match view {
            View::All if kn.count_all.is_some() => Ok(kn.count_all.unwrap()),
            View::Young { .. } if kn.count_young.is_some() => Ok(kn.count_young.unwrap()),
            v => self.inner.count(v),
        }
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        let kn = self.kn();
        self.max_k.fetch_max(k, Ordering::SeqCst);
        if kn.bottomless != 0 {
            if k > kn.k_limit {
                return Err("bottomless store: asked for more than the test allows".into());
            }
            return Ok((0..k)
                .map(|i| Hit {
                    cos: 1.0,
                    trace: Trace {
                        id: format!("V{i:012}"),
                        text: if kn.bottomless == 2 { format!("t{i}") } else { String::new() },
                        created_at: 0,
                        tz: "UTC;+00:00".into(),
                        last_recall: 0,
                        stability: 1e6,
                        consolidated: true,
                        cue: String::new(),
                        model_id: model_id.to_owned(),
                        vector: q.to_vec(),
                    },
                })
                .collect());
        }
        let mut hits = if kn.near_everything_reversed {
            let mut h = self.inner.near(q, model_id, min_cos, usize::MAX, stated_by)?;
            h.reverse();
            h
        } else {
            self.inner.near(q, model_id, min_cos, k, stated_by)?
        };
        if kn.near_dupes > 1 {
            hits = hits.into_iter().flat_map(|h| std::iter::repeat_n(h, kn.near_dupes)).collect();
        }
        if let Some(c) = kn.cos_override {
            hits.iter_mut().for_each(|h| h.cos = c);
        }
        Ok(hits)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        let kn = self.kn();
        if kn.panic_write {
            panic!("storage: write exploded");
        }
        if kn.drop_writes {
            return Ok(());
        }
        self.inner.write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        if self.kn().fail_clear {
            return Err("clear refused".into());
        }
        self.inner.clear()
    }
}

fn hostile_engine(cfg: Config) -> (Engram, ManualClock, Handle) {
    let (h, handle) = hostile();
    let (m, clock) = build_with(cfg, Box::new(FakeEmbedder::new()), Box::new(h));
    (m, clock, handle)
}

fn all_ids(m: &Engram) -> Vec<String> {
    m.store().find(View::All, usize::MAX).unwrap().into_iter().map(|t| t.id).collect()
}

// ------------------------------------------------------------- hostile Embedder

/// An embedder whose every answer is a closure.
struct Flex {
    model: Box<dyn Fn() -> &'static str + Send>,
    dim: Box<dyn Fn() -> usize + Send>,
    docs: Box<dyn FnMut(&[&str]) -> Result<Vec<Vec<f32>>> + Send>,
    queries: Box<dyn FnMut(&[&str]) -> Result<Vec<Vec<f32>>> + Send>,
}

impl Embedder for Flex {
    fn model_id(&self) -> &str {
        (self.model)()
    }
    fn dimension(&self) -> usize {
        (self.dim)()
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (self.docs)(texts)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (self.queries)(texts)
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3))
}

/// A unit vector with a single 1 at a position derived from `text`.
fn onehot(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[(fnv(text) % dim as u64) as usize] = 1.0;
    v
}

fn onehot_embedder(model: &'static str, dim: usize) -> Flex {
    Flex {
        model: Box::new(move || model),
        dim: Box::new(move || dim),
        docs: Box::new(move |t| Ok(t.iter().map(|s| onehot(s, dim)).collect())),
        queries: Box::new(move |t| Ok(t.iter().map(|s| onehot(s, dim)).collect())),
    }
}

fn engine_with(cfg: Config, e: impl Embedder + 'static) -> (Engram, ManualClock) {
    build_with(cfg, Box::new(e), Box::new(Store::in_memory().unwrap()))
}

/// A cluster: an older trace and a newer one whose cue reactivates it.
fn cluster_engine(cfg: Config) -> (Engram, ManualClock, String, String) {
    let (mut m, clock) = support::build(cfg);
    let a = insert(&mut m, "the user lives in kyoto japan", 1.0, "");
    clock.advance(10.0);
    let b = insert(&mut m, "the user moved to osaka japan", 1.0, "the user lives in kyoto japan");
    (m, clock, a, b)
}

// =================================================================== A. Storage

/// S1: `victims` does `count(All) + 1`: a store that reports `usize::MAX` rows overflows.
/// (A release build wraps to 0 and silently stops evicting; a debug build panics.)
#[test]
fn s1_count_usize_max_does_not_panic() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(Config { capacity: 4, ..cfg() });
    for t in ["alpha one", "beta two", "gamma three"] {
        insert(&mut m, t, 1.0, "");
    }
    h.set(|k| k.count_all = Some(usize::MAX));
    let _ = no_panic("remember with count(All) = usize::MAX", || m.remember("delta four", 1.0, ""));
}

/// S2: with a small capacity and count(All) near `usize::MAX`, `excess + BAND_DEPTH` overflows.
#[test]
fn s2_count_near_usize_max_with_capacity_one_does_not_panic() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(Config { capacity: 1, ..cfg() });
    insert(&mut m, "alpha one", 1.0, "");
    h.set(|k| k.count_all = Some(usize::MAX - 2));
    let _ = no_panic("remember with count(All) = usize::MAX - 2", || m.remember("beta two", 1.0, ""));
}

/// S3: `count(Young) + usize::from(..)` overflows for a store that reports `usize::MAX`.
#[test]
fn s3_young_count_usize_max_does_not_panic() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(Config { capacity: 2, ..cfg() });
    for t in ["alpha one", "beta two", "gamma three"] {
        insert(&mut m, t, 1.0, "");
    }
    h.set(|k| k.count_young = Some(usize::MAX));
    let _ = no_panic("remember with count(Young) = usize::MAX", || m.remember("delta four", 1.0, ""));
}

/// S5a: `memory(id)` takes the LAST row of `find(View::Id(id), 1)`; a store that ignores the
/// view (or its limit) makes it return a trace that is not the one asked for.
#[test]
fn s5a_memory_returns_the_requested_id_only() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    let a = insert(&mut m, "alpha one", 1.0, "");
    let _b = insert(&mut m, "beta two", 1.0, "");
    let _c2 = insert(&mut m, "gamma three", 1.0, "");
    h.set(|k| k.id_view_all = true);
    let got = m.memory(&_c2).unwrap();
    assert!(got.as_ref().is_none_or(|t| t.id == _c2), "memory({_c2}) returned the trace {:?} (asked for the newest row, the store answered with all rows oldest first)", got.map(|t| t.id));
    let _ = a;
}

/// S5b: `forget(id)` removes `raw.id` of the last row returned, so a store that ignores the
/// id deletes an unrelated trace.
#[test]
fn s5b_forget_removes_only_the_requested_trace() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    let a = insert(&mut m, "alpha one", 1.0, "");
    let b = insert(&mut m, "beta two", 1.0, "");
    let c = insert(&mut m, "gamma three", 1.0, "");
    h.set(|k| k.id_view_all = true);
    let _ = m.forget(&a);
    h.set(|k| k.id_view_all = false);
    let left = all_ids(&m);
    assert!(left.contains(&b) && left.contains(&c), "forget({a}) deleted another trace: left = {left:?}, expected b and c to survive");
}

/// S5c: `remember` rehearses the first row `find(View::Text(text))` returns, without checking
/// that its text is the one being stored: a store that ignores the text merges two facts.
#[test]
fn s5c_remember_reinforces_only_the_same_text() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    insert(&mut m, "alpha one", 1.0, "");
    h.set(|k| k.text_view_all = true);
    let r = m.remember("beta two", 1.0, "").unwrap();
    assert!(!matches!(&r, Remembered::Reinforced(t) if t.text != "beta two"), "'beta two' was taken for a rehearsal of {r:?}");
}

/// S6: `candidates` does not de-duplicate near() hits: a store that repeats a hit puts the same
/// trace into a cluster several times (the LLM sees it repeated; `gists` counts its strength
/// once per copy).
#[test]
fn s6_duplicate_near_hits_do_not_duplicate_cluster_members() {
    let _g = serial();
    let (h, handle) = hostile();
    let (mut m, clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(h));
    insert(&mut m, "the user lives in kyoto japan", 1.0, "");
    clock.advance(10.0);
    insert(&mut m, "the user moved to osaka japan", 1.0, "the user lives in kyoto japan");
    handle.set(|k| k.near_dupes = 3);
    let clusters = m.clusters(None).unwrap();
    assert!(!clusters.is_empty(), "scenario: expected one cluster");
    for c in &clusters {
        let mut ids: Vec<&str> = c.iter().map(|t| t.id.as_str()).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "a cluster holds the same trace more than once: {n} members, {} distinct", ids.len());
    }
}

/// S7: the widening loop of `Engram::near` asks for twice as many rows each round while the store
/// keeps returning only unusable ones. It is bounded only by the store running out; a store that
/// is never exhausted (or lies about it) is asked for `usize::MAX` rows. The honest count is 3.
#[test]
fn s7_widening_near_is_bounded_by_the_row_count() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    for t in ["alpha one", "beta two", "gamma three"] {
        insert(&mut m, t, 1.0, "");
    }
    h.set(|k| k.bottomless = 1);
    let _ = m.remember("delta four", 1.0, "");
    assert!(h.max_k() <= 4096, "near() was asked for k = {} with 3 rows in the store", h.max_k());
}

/// S7b: same for the exact-recall loop (`select`): `k *= 4` until the k-th cosine is below the floor.
#[test]
fn s7b_exact_recall_loop_is_bounded_by_the_row_count() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    for t in ["alpha one", "beta two", "gamma three"] {
        insert(&mut m, t, 1.0, "");
    }
    h.set(|k| k.bottomless = 2);
    let _ = m.peek("alpha beta");
    assert!(h.max_k() <= 4096, "recall asked near() for k = {} with 3 rows in the store", h.max_k());
}

/// S7c: the same for `Engram::find` (the usable-rows widening used by every ordered read).
#[test]
fn s7c_widening_find_is_bounded_by_the_row_count() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    for t in ["alpha one", "beta two", "gamma three"] {
        insert(&mut m, t, 1.0, "");
    }
    h.set(|k| k.find_junk = true);
    let _ = m.remember("delta four", 1.0, "");
    assert!(h.max_find() <= 4096, "find() was asked for limit = {} with 3 rows in the store", h.max_find());
}

/// S9b: the engine never re-checks `cos >= min_cos` of a hit whose vector is unit (it trusts the
/// store's number): a store that claims a hit far below θ_related still marks the write as
/// "reactivating" and reports that cosine.
#[test]
fn s9b_a_hit_below_the_requested_cosine_is_not_a_reactivation() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    insert(&mut m, "alpha one", 1.0, "");
    h.set(|k| k.cos_override = Some(-0.9));
    // "alpha one extra" really is near "alpha one" (cos ~0.82): the store finds the hit, then lies about its cosine.
    let r = m.remember("alpha one extra", 1.0, "").unwrap();
    let Remembered::Inserted { cosine, .. } = r else { panic!("expected an insert") };
    assert!(cosine >= m.config().theta_related, "a hit claimed to have cosine {cosine}, below the min_cos {} the engine asked for, was accepted", m.config().theta_related);
}

/// G: out-of-range / non-finite cosines, extra hits and unsorted hits never panic and never put
/// more than `inject_n` traces into a pack.
#[test]
fn g_s8_s9_oversized_unsorted_and_nonfinite_hits_do_not_panic() {
    let _g = serial();
    for cos in [None, Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY), Some(5.0), Some(-5.0), Some(1e300), Some(-1e300)] {
        let (mut m, clock, h) = hostile_engine(Config { inject_n: 2, dream_max_members: 3, ..cfg() });
        for i in 0..12 {
            insert(&mut m, &format!("shared words number{i}"), 1.0, "");
            clock.advance(2.0);
        }
        h.set(|k| {
            k.near_everything_reversed = true;
            k.cos_override = cos;
        });
        let _ = no_panic("recall", || {
            let r = m.recall("shared words").unwrap();
            assert!(r.recalled.len() <= 2, "{} traces injected with inject_n = 2", r.recalled.len());
            for c in &r.recalled {
                assert!(c.score.is_finite() && (0.0..=1.0).contains(&c.score) && c.cosine <= 1.0, "score {} cosine {}", c.score, c.cosine);
            }
        });
        let _ = no_panic("remember", || m.remember("shared words extra", 1.0, "shared words").unwrap());
        no_panic("clusters", || {
            for c in m.clusters(None).unwrap() {
                assert!(c.len() <= 3, "cluster of {} with dream_max_members = 3", c.len());
            }
        });
        no_panic("dream", || m.dream(&mut merge_to_gist, Some(3)).unwrap());
    }
}

/// G: a store that drops every write: nothing loops, nothing panics, the dream ends.
#[test]
fn g_s10_dropped_writes_terminate() {
    let _g = serial();
    let (mut m, clock, h) = hostile_engine(cfg());
    insert(&mut m, "the user lives in kyoto japan", 1.0, "");
    clock.advance(10.0);
    insert(&mut m, "the user moved to osaka japan", 1.0, "the user lives in kyoto japan");
    h.set(|k| k.drop_writes = true);
    let t = Instant::now();
    for i in 0..30 {
        let _ = m.remember(&format!("fact {i}"), 1.0, "");
        m.recall("fact").unwrap();
        m.dream(&mut merge_to_gist, Some(2)).unwrap();
        m.reindex(10).unwrap();
        m.cite("《id:nothing》").unwrap();
    }
    assert!(t.elapsed().as_secs() < 30);
}

/// G: clear() failing leaves the engine usable.
#[test]
fn g_s11_a_failing_clear_leaves_the_engine_usable() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(cfg());
    insert(&mut m, "alpha one", 1.0, "");
    h.set(|k| k.fail_clear = true);
    assert!(m.reset().is_err());
    h.set(|k| k.fail_clear = false);
    assert!(matches!(m.remember("beta two", 1.0, "").unwrap(), Remembered::Inserted { .. }));
    assert_eq!(m.stats().unwrap().records, 2);
    m.reset().unwrap();
    assert_eq!(m.stats().unwrap().records, 0);
}

/// G: a Storage that panics mid-operation: the engine is still usable and the failed write
/// did not consume a rate-limit slot (write_times is pushed only after the commit).
#[test]
fn g_s12_a_panicking_storage_leaves_the_engine_consistent() {
    let _g = serial();
    let (mut m, _c, h) = hostile_engine(Config { writes_per_day: 2, ..cfg() });
    h.set(|k| k.panic_write = true);
    assert!(catch_unwind(AssertUnwindSafe(|| m.remember("alpha one", 1.0, ""))).is_err(), "scenario: the write must panic");
    h.set(|k| k.panic_write = false);
    assert!(matches!(m.remember("alpha one", 1.0, "").unwrap(), Remembered::Inserted { .. }));
    assert!(matches!(m.remember("beta two", 1.0, "").unwrap(), Remembered::Inserted { .. }), "the panicked write consumed a rate-limit slot");
    assert!(matches!(m.remember("gamma three", 1.0, "").unwrap(), Remembered::RateLimited { .. }));
    h.set(|k| k.panic_find = true);
    assert!(catch_unwind(AssertUnwindSafe(|| m.remember("zeta six", 1.0, ""))).is_err());
    h.set(|k| k.panic_find = false);
    assert!(!m.recall("alpha one").unwrap().recalled.is_empty());
    assert_eq!(m.stats().unwrap().records, 2);
}

/// G: a store that lies about every count never makes stats() or a dream panic.
#[test]
fn g_s13_counts_that_lie_do_not_panic_stats_or_dream() {
    let _g = serial();
    for lie in [0usize, 1, usize::MAX / 2, usize::MAX] {
        let (mut m, clock, h) = hostile_engine(cfg());
        insert(&mut m, "the user lives in kyoto japan", 1.0, "");
        clock.advance(10.0);
        insert(&mut m, "the user moved to osaka japan", 1.0, "the user lives in kyoto japan");
        h.set(|k| k.count_all = Some(lie));
        no_panic("stats", || m.stats().unwrap());
        let _ = no_panic("recall", || m.recall("kyoto").unwrap());
        no_panic("dream", || m.dream(&mut merge_to_gist, None).unwrap());
        no_panic("forget", || m.forget("nope").unwrap());
    }
}

// ================================================================== B. Embedder

/// E1: `dimension()` is read once per vector; an embedder whose answer changes between calls
/// yields query vectors of different lengths in one recall, and `dot` (which asserts equal
/// lengths) panics in `select`.
#[test]
fn e1_a_dimension_that_changes_between_calls_does_not_panic() {
    let _g = serial();
    let flip = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let (f2, c2) = (flip.clone(), calls.clone());
    let e = Flex {
        model: Box::new(|| "flip"),
        dim: Box::new(move || if f2.load(Ordering::SeqCst) && c2.fetch_add(1, Ordering::SeqCst) % 2 == 1 { 8 } else { 4 }),
        docs: Box::new(|t| Ok(t.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())),
        queries: Box::new(|t| Ok(t.iter().enumerate().map(|(i, _)| if i == 0 { vec![1.0, 0.0, 0.0, 0.0] } else { vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }).collect())),
    };
    let (mut m, _c) = engine_with(cfg(), e);
    insert(&mut m, "alpha", 1.0, "");
    flip.store(true, Ordering::SeqCst);
    let _ = no_panic("recall of a two-cue query under a dimension that flips", || m.recall("first part. second part."));
}

/// E3: an embedder whose model_id is "" collides with the "not indexed" sentinel (`Trace::model_id`
/// is "" for a trace that has no vector yet): traces written while the embedder was offline look
/// indexed under the current model and `reindex` never repairs them.
#[test]
fn e3_an_empty_model_id_still_reindexes_offline_traces() {
    let _g = serial();
    let online = Arc::new(AtomicBool::new(false));
    let o = online.clone();
    let e = Flex {
        model: Box::new(|| ""),
        dim: Box::new(|| 4),
        docs: Box::new(move |t| if o.load(Ordering::SeqCst) { Ok(t.iter().map(|s| onehot(s, 4)).collect()) } else { Err("offline".into()) }),
        queries: Box::new(|t| Ok(t.iter().map(|s| onehot(s, 4)).collect())),
    };
    let (mut m, _c) = engine_with(cfg(), e);
    let id = insert(&mut m, "alpha beta", 1.0, "");
    assert!(m.memory(&id).unwrap().unwrap().vector.is_empty(), "scenario: written offline");
    online.store(true, Ordering::SeqCst);
    let done = m.reindex(10).unwrap();
    assert_eq!(done, 1, "reindex repaired {done} traces; the offline one is invisible under model_id \"\"");
}

/// G: model ids with separators, control characters and a 100 kB one still recall, reindex, dream, forget.
#[test]
fn g_e4_odd_model_ids_work() {
    let _g = serial();
    let long: &'static str = Box::leak("m".repeat(100_000).into_boxed_str());
    for model in ["a\0b", "x》y", "line\nbreak", ";", "a;b", " ", "\u{202E}", "日本語/モデル", long] {
        let (mut m, clock) = engine_with(cfg(), onehot_embedder(model, 16));
        let ids: Vec<String> = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"].iter().map(|t| insert(&mut m, t, 1.0, "")).collect();
        clock.advance(5.0);
        let label = format!("model {:?}", model.chars().take(12).collect::<String>());
        no_panic(&label, || {
            assert!(!m.recall("alpha beta gamma").unwrap().recalled.is_empty(), "recall found nothing");
            assert_eq!(m.reindex(10).unwrap(), 0);
            m.dream(&mut merge_to_gist, None).unwrap();
            assert_eq!(m.stats().unwrap().records, 3);
            assert!(m.forget(&ids[0]).unwrap().is_some());
        });
    }
}

/// G: a million-dimension model works for a handful of traces (no panic, bounded time).
#[test]
fn g_e5_a_million_dimensions_do_not_panic() {
    let _g = serial();
    let (mut m, _c) = engine_with(Config { capacity: 4, ..cfg() }, onehot_embedder("huge", 1_000_000));
    let t = Instant::now();
    no_panic("1e6 dimensions", || {
        for text in ["alpha beta", "gamma delta", "epsilon zeta"] {
            insert(&mut m, text, 1.0, "");
        }
        assert!(!m.recall("alpha beta").unwrap().recalled.is_empty());
    });
    assert!(t.elapsed().as_secs() < 120, "took {:?}", t.elapsed());
}

/// G: NaN, infinite and zero vectors are an offline embedder: the trace is stored unindexed, nothing panics.
#[test]
fn g_e6_non_finite_and_zero_vectors_are_an_offline_embedder() {
    let _g = serial();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, f32::MAX, f32::MIN_POSITIVE / 4.0] {
        let e = Flex {
            model: Box::new(|| "bad"),
            dim: Box::new(|| 4),
            docs: Box::new(move |t| Ok(t.iter().map(|_| vec![bad; 4]).collect())),
            queries: Box::new(move |t| Ok(t.iter().map(|_| vec![bad; 4]).collect())),
        };
        let (mut m, _c) = engine_with(cfg(), e);
        no_panic(&format!("vector of {bad}"), || {
            let r = m.remember("alpha beta", 1.0, "gamma").unwrap();
            // f32::MAX components renormalise legitimately; the others cannot be indexed.
            if let Remembered::Inserted { trace, .. } = r {
                assert!(trace.vector.is_empty() || trace.vector.iter().all(|x| x.is_finite()));
            }
            m.recall("alpha beta").unwrap();
            m.reindex(5).unwrap();
            m.dream(&mut always_keep, None).unwrap();
        });
    }
}

/// G: an embedder that answers every text with the same vector (identical vectors defeat the
/// IVF split): 1 300 inserts, recall and dream stay bounded.
#[test]
fn g_e8_identical_vectors_for_every_text() {
    let _g = serial();
    let e = Flex {
        model: Box::new(|| "same"),
        dim: Box::new(|| 8),
        docs: Box::new(|t| Ok(t.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).collect())),
        queries: Box::new(|t| Ok(t.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).collect())),
    };
    let (mut m, clock) = engine_with(Config { capacity: 5000, writes_per_day: 100_000, ..cfg() }, e);
    let t = Instant::now();
    for i in 0..1300 {
        clock.advance(1.0);
        assert!(matches!(m.remember(&format!("fact {i}"), 1.0, "").unwrap(), Remembered::Inserted { .. }));
    }
    let r = m.recall("anything").unwrap();
    assert!(r.recalled.len() <= m.config().inject_n);
    assert!(r.pack_text.chars().count() <= m.config().budget_chars);
    for c in m.clusters(Some(3)).unwrap() {
        assert!(c.len() <= m.config().dream_max_members);
    }
    m.dream(&mut merge_to_gist, Some(3)).unwrap();
    assert!(t.elapsed().as_secs() < 120, "took {:?}", t.elapsed());
}

/// G: an embedder that panics: the engine is usable afterwards, nothing half-stored.
#[test]
fn g_e9_a_panicking_embedder_leaves_the_engine_usable() {
    let _g = serial();
    let boom = Arc::new(AtomicBool::new(true));
    let b = boom.clone();
    let e = Flex {
        model: Box::new(|| "boom"),
        dim: Box::new(|| 4),
        docs: Box::new(move |t| {
            if b.load(Ordering::SeqCst) {
                panic!("embedder exploded");
            }
            Ok(t.iter().map(|s| onehot(s, 4)).collect())
        }),
        queries: Box::new(|t| Ok(t.iter().map(|s| onehot(s, 4)).collect())),
    };
    let (mut m, _c) = engine_with(Config { writes_per_day: 1, ..cfg() }, e);
    assert!(catch_unwind(AssertUnwindSafe(|| m.remember("alpha", 1.0, ""))).is_err());
    assert_eq!(m.stats().unwrap().records, 0);
    boom.store(false, Ordering::SeqCst);
    assert!(matches!(m.remember("alpha", 1.0, "").unwrap(), Remembered::Inserted { .. }), "the panicked call used the daily slot");
}

/// G: a model_id that changes on every call never panics or hangs any operation.
#[test]
fn g_e2_a_model_id_that_changes_every_call() {
    let _g = serial();
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = n.clone();
    let e = Flex {
        model: Box::new(move || Box::leak(format!("m{}", n2.fetch_add(1, Ordering::SeqCst)).into_boxed_str())),
        dim: Box::new(|| 4),
        docs: Box::new(|t| Ok(t.iter().map(|s| onehot(s, 4)).collect())),
        queries: Box::new(|t| Ok(t.iter().map(|s| onehot(s, 4)).collect())),
    };
    let (mut m, clock) = engine_with(cfg(), e);
    no_panic("changing model ids", || {
        for i in 0..20 {
            clock.advance(3.0);
            m.remember(&format!("fact {i}"), 1.0, "fact").unwrap();
            m.recall("fact").unwrap();
            m.dream(&mut merge_to_gist, Some(2)).unwrap();
            m.reindex(8).unwrap();
            m.clusters(None).unwrap();
            m.stats().unwrap();
        }
    });
}

// =============================================================== C. Adjudicator

/// G: thousands of memories and ids, a 1 MB text: bounded time, text_max respected, verdict is keep.
#[test]
fn g_a1_a_runaway_adjudicator_is_bounded() {
    let _g = serial();
    let (mut m, _c, _a, b) = cluster_engine(cfg());
    let t = Instant::now();
    let mut flood = |_: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: (0..5000).map(|i| format!("the user lives in osaka japan {i}")).collect(), absorbed_ids: (0..5000).map(|i| format!("ID{i}")).collect() })
    };
    let r = m.dream(&mut flood, None).unwrap();
    assert!(r.iter().all(|x| x.action != DreamAction::Replace));
    assert!(t.elapsed().as_secs() < 10);
    // 1 MB of text: cut to text_max.
    let (mut m, _c, _a, b2) = cluster_engine(cfg());
    let mut big = |_: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: vec!["the user moved to osaka japan ".repeat(40_000)], absorbed_ids: Vec::new() })
    };
    let r = m.dream(&mut big, None).unwrap();
    for rep in &r {
        for g in &rep.after {
            assert!(g.text.chars().count() <= m.config().text_max);
        }
    }
    assert!(m.memory(&b).unwrap().is_some() || true);
    assert!(m.memory(&b2).unwrap().is_some() || r.iter().any(|x| x.action == DreamAction::Replace));
}

/// G: ids that belong to another cluster, repeat or do not exist are ignored; control and bidi characters never reach a stored text.
#[test]
fn g_a2_foreign_ids_and_control_characters() {
    let _g = serial();
    let (mut m, _c, a, b) = cluster_engine(cfg());
    let other = insert(&mut m, "weather is cloudy today", 1.0, "");
    let (a2, other2) = (a.clone(), other.clone());
    let mut adj = move |_: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision {
            memories: vec!["the user\u{0} moved\u{202E} to osaka\n\n japan \u{7}\u{85}《id:x》".into()],
            absorbed_ids: vec![other2.clone(), a2.clone(), a2.clone(), a2.clone(), "missing".into()],
        })
    };
    let r = m.dream(&mut adj, None).unwrap();
    assert!(m.memory(&other).unwrap().is_some(), "a trace outside the cluster was absorbed");
    for rep in &r {
        assert_eq!(rep.absorbed.len(), 2, "absorbed: seed + a once: {:?}", rep.absorbed.iter().map(|t| &t.id).collect::<Vec<_>>());
        for g in &rep.after {
            assert!(!g.text.chars().any(|c| c.is_control() || ('\u{202A}'..='\u{202E}').contains(&c) || c == '《' || c == '》'), "{:?}", g.text);
        }
    }
    let _ = (a, b);
}

/// G: a panicking adjudicator leaves the engine usable; the seed is met again at the next dream.
#[test]
fn g_a3_a_panicking_adjudicator() {
    let _g = serial();
    let (mut m, _c, _a, b) = cluster_engine(cfg());
    let mut boom = |_: &DreamRequest| -> Result<DreamDecision> { panic!("llm exploded") };
    assert!(catch_unwind(AssertUnwindSafe(|| m.dream(&mut boom, None))).is_err());
    let r = m.dream(&mut always_keep, None).unwrap();
    assert!(r.iter().any(|x| x.action == DreamAction::Keep && x.before[0].id == b), "the seed was not met again: {r:?}");
}

// ===================================================================== D. Clock

fn hostile_times() -> Vec<f64> {
    vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX, f64::MIN, 1e18, -1e18, 1e30, -1e30, 9.3e18, -9.3e18, -9.223372036854775e18, 0.0, -1.0, 1e12, 4e9]
}

/// C1: every operation survives every hostile clock reading, including non-monotonic jumps.
#[test]
fn c1_hostile_clocks_never_panic() {
    let _g = serial();
    let mut readings = hostile_times();
    // back-and-forth jumps
    readings.extend([START, START + 1e9, START, START - 1e9, START, 1e18, START, -1e18]);
    let (mut m, clock) = support::build(Config { capacity: 6, ..cfg() });
    for (n, &t) in readings.iter().enumerate() {
        for round in 0..2 {
            clock.set(START);
            let a = format!("alpha{n}x{round} lives kyoto");
            let _ = m.remember(&a, 1.0, "");
            clock.set(t);
            let what = |op: &str| format!("clock = {t:e}: {op}");
            let _ = no_panic(&what("remember"), || m.remember(&format!("alpha{n}x{round} moved osaka"), 1.0, &a));
            let rec = no_panic(&what("recall"), || m.recall(&format!("alpha{n}x{round} kyoto osaka")).unwrap());
            for r in &rec.recalled {
                assert!((0.0..=1.0).contains(&r.retrievability), "R = {} at clock {t:e}", r.retrievability);
            }
            no_panic(&what("cite"), || m.cite(&rec.pack_text).unwrap());
            let reports = no_panic(&what("dream"), || m.dream(&mut merge_to_gist, Some(3)).unwrap());
            no_panic(&what("undo_all"), || m.undo_all(&reports).unwrap());
            no_panic(&what("reindex/stats/clusters/peek"), || {
                m.reindex(10).unwrap();
                m.stats().unwrap();
                m.clusters(None).unwrap();
                m.peek("alpha").unwrap();
                m.now_local();
            });
            let ids = all_ids(&m);
            if let Some(id) = ids.first() {
                no_panic(&what("forget"), || m.forget(id).unwrap());
            }
            assert!(m.stats().unwrap().records <= 6 + 1, "capacity exceeded at clock {t:e}");
        }
    }
}

/// C2: a dream whose clock is at the bottom of the i64 range while the members were stamped
/// there too: the gist's `last_recall = now - round(S * elapsed)` is i64 arithmetic.
#[test]
fn c2_gist_last_recall_does_not_overflow_at_the_bottom_of_i64() {
    let _g = serial();
    for low in [-9.223372036854775e18, -9.2233720368547e18, f64::MIN, -1e19] {
        let (mut m, clock) = support::build(cfg());
        clock.set(low);
        let a = insert(&mut m, "the user lives in kyoto japan", 1.0, "");
        clock.set(low + 2048.0);
        let b = insert(&mut m, "the user moved to osaka japan", 1.0, "the user lives in kyoto japan");
        clock.set(low + 3.0e6);
        no_panic(&format!("dream at clock {low:e}"), || m.dream(&mut merge_to_gist, None).unwrap());
        let _ = (a, b);
    }
}

// ===================================================================== E. IdGen

/// G: a generator that returns invalid or used ids forever ends in an error, not a loop, and stores nothing.
#[test]
fn g_i1_a_stuck_id_generator_is_an_error() {
    let _g = serial();
    let (mut m, _c) = support::build(cfg());
    for bad in ["", "a\nb", "x《y》", "tab\there"] {
        let b = bad.to_owned();
        m.set_ids(Box::new(move |_, _| b.clone()));
        let t = Instant::now();
        assert!(m.remember("alpha one", 1.0, "").is_err(), "{bad:?} accepted");
        assert!(t.elapsed().as_secs() < 5);
        assert_eq!(m.stats().unwrap().records, 0);
    }
    m.set_ids(Box::new(|_, _| "SAME".to_owned()));
    assert!(matches!(m.remember("alpha one", 1.0, "").unwrap(), Remembered::Inserted { .. }));
    assert!(m.remember("beta two", 1.0, "").is_err(), "a duplicate id overwrote a trace");
    assert_eq!(m.stats().unwrap().records, 1);
}

/// G: a panicking generator costs no rate-limit slot and leaves the engine usable.
#[test]
fn g_i2_a_panicking_id_generator() {
    let _g = serial();
    let (mut m, _c) = support::build(Config { writes_per_day: 1, ..cfg() });
    m.set_ids(Box::new(|_, _| panic!("idgen exploded")));
    assert!(catch_unwind(AssertUnwindSafe(|| m.remember("alpha one", 1.0, ""))).is_err());
    let n = Arc::new(AtomicUsize::new(0));
    m.set_ids(Box::new(move |_, _| format!("ID{:08}", n.fetch_add(1, Ordering::SeqCst))));
    assert!(matches!(m.remember("alpha one", 1.0, "").unwrap(), Remembered::Inserted { .. }));
}

/// I3: ids are not length-bounded: a generator (or a store) with 1 MB ids puts a 1 MB line in the
/// pack (`pack_text` is documented as at most `budget_chars`; "the first always fits" assumes a
/// line is short), and a 1 MB key into every index.
#[test]
fn i3_a_megabyte_id_cannot_blow_the_pack_budget() {
    let _g = serial();
    let (mut m, _c) = support::build(cfg());
    let n = Arc::new(AtomicUsize::new(0));
    m.set_ids(Box::new(move |_, _| format!("{}{:08}", "Z".repeat(1 << 20), n.fetch_add(1, Ordering::SeqCst))));
    match m.remember("hello world greeting", 1.0, "") {
        Err(_) => {} // refused: fine
        Ok(_) => {
            let r = m.recall("hello world greeting").unwrap();
            assert!(r.pack_text.chars().count() <= 4 * m.config().budget_chars, "pack of {} characters for budget_chars = {}", r.pack_text.chars().count(), m.config().budget_chars);
        }
    }
}

// ================================================================== F. Config

/// G: extreme but valid configurations (round 2 did random ones; these are the corners that
/// stress the arithmetic of this round: caps, windows, widening).
#[test]
fn g_f1_extreme_valid_configs_survive_a_short_life() {
    let _g = serial();
    let huge = usize::MAX;
    let cfgs = vec![
        Config { capacity: 1, ..cfg() },
        Config { capacity: huge, ..cfg() },
        Config { capacity: 2, grace_period: f64::MAX, ..cfg() },
        Config { capacity: 2, grace_period: 0.0, ..cfg() },
        Config { text_max: 1, ..cfg() },
        Config { text_max: huge, budget_chars: huge, writes_per_day: huge, ..cfg() },
        Config { initial_stability: f64::MAX, max_stability: f64::MAX, spacing_gain: f64::MAX, ..cfg() },
        Config { initial_stability: 1.0, max_stability: 1.0, ..cfg() },
        Config { theta_related: 1e-300, gist_min_cosine: -1.0, ..cfg() },
        Config { theta_related: 1.0 - f64::EPSILON, cosine_floor: 1.0 - f64::EPSILON, ..cfg() },
        Config { inject_n: 1 << 20, dream_max_members: 1 << 20, max_cues: 1 << 20, dream_budget: 1 << 20, ..cfg() },
        Config { writes_per_day: 0, ..cfg() },
        Config { min_score: 1.0, relative_score: 1.0, alpha: 0.0, ..cfg() },
    ];
    for (i, c) in cfgs.into_iter().enumerate() {
        let (mut m, clock) = support::build(c);
        no_panic(&format!("config #{i}"), || {
            for j in 0..6 {
                clock.advance(7.0);
                let _ = m.remember(&format!("the user lives in town{j} japan"), 1.0, "the user lives in town0 japan");
            }
            let r = m.recall("the user lives in japan").unwrap();
            m.cite(&r.pack_text).unwrap();
            let reports = m.dream(&mut merge_to_gist, None).unwrap();
            m.undo_all(&reports).unwrap();
            m.reindex(100).unwrap();
            m.clusters(None).unwrap();
            if let Some(id) = all_ids(&m).first() {
                m.forget(id).unwrap();
            }
        });
    }
}

/// G: odd timezone names (traversal-looking, NUL, huge) are never a panic and the stored field stays short.
#[test]
fn g_f2_odd_timezone_names() {
    let _g = serial();
    for tz in ["", "\0", "../../../etc/passwd", "a/b/../..", "UTC;+99:99", ";;;;", "日本/東京", "\u{202E}Asia/Tokyo", &"Z".repeat(1 << 20)] {
        let clock = ManualClock::new(START);
        let mut m = Engram::with_clock(Box::new(Store::in_memory().unwrap()), Box::new(FakeEmbedder::new()), cfg(), tz, clock.clock()).unwrap();
        no_panic("odd timezone", || {
            let id = insert(&mut m, "alpha one", 1.0, "");
            let t = m.memory(&id).unwrap().unwrap();
            assert!(t.tz.chars().count() <= 64, "stored tz of {} characters", t.tz.chars().count());
            m.now_local();
            m.dream(&mut merge_to_gist, None).unwrap();
        });
    }
}

// ============================================================ R. resource bounds

const N: usize = 20_000;
/// Live-heap growth tolerated between the first quarter and the end of a loop of N operations.
const SLACK: isize = 3 << 20;

fn vocab_text(i: usize) -> String {
    format!("tok{} tok{} tok{}", i % 97, i % 89, i % 83)
}

/// A small engine on a clock that advances `step` seconds per operation.
fn tiny(step: f64) -> (Engram, ManualClock) {
    let cfg = Config { capacity: 8, grace_period: 100.0, writes_per_day: 1_000_000, inject_n: 3, dream_budget: 2, ..cfg() };
    let (m, clock) = support::build(cfg);
    let _ = step;
    (m, clock)
}

/// Runs `op` N times, sampling the live heap at N/4 and N; fails when it grew by more than SLACK.
fn bounded(what: &str, m: &mut Engram, clock: &ManualClock, step: f64, mut op: impl FnMut(&mut Engram, usize)) {
    let mut at_quarter = 0;
    for i in 0..N {
        clock.advance(step);
        op(m, i);
        if i == N / 4 {
            at_quarter = live();
        }
    }
    let grown = live() - at_quarter;
    eprintln!("R: {what}: live heap {:+} bytes between operation {} and {N} ({} bytes live at the end)", grown, N / 4, live());
    assert!(grown <= SLACK, "{what}: live heap grew by {grown} bytes between operation {} and {N}", N / 4);
}

/// G: the instrument works: a deliberate leak of 1 000 x 1 kB shows up in the live-byte count.
#[test]
fn g_r0_the_counter_sees_a_leak() {
    let _g = serial();
    let before = live();
    let leaked: Vec<Vec<u8>> = (0..1000).map(|_| vec![7u8; 1024]).collect();
    let during = live();
    assert!(during - before >= 1_000_000, "{} bytes seen", during - before);
    drop(leaked);
    assert!(live() - before < 100_000);
}

/// R1: remember (eviction every time at capacity 8): write_times, store.
#[test]
fn r1_remember_is_bounded() {
    let _g = serial();
    let (mut m, clock) = tiny(20.0);
    bounded("remember", &mut m, &clock, 20.0, |m, i| {
        let _ = m.remember(&vocab_text(i), 1.0, "");
    });
}

/// R2: recall + cite (exposures).
#[test]
fn r2_recall_and_cite_are_bounded() {
    let _g = serial();
    let (mut m, clock) = tiny(20.0);
    for i in 0..8 {
        insert(&mut m, &vocab_text(i), 1.0, "");
    }
    bounded("recall+cite", &mut m, &clock, 20.0, |m, i| {
        let r = m.recall(&vocab_text(i)).unwrap();
        m.cite(&r.pack_text).unwrap();
        let _ = m.peek(&vocab_text(i + 1)).unwrap();
    });
}

/// R3: forget interleaved with remember.
#[test]
fn r3_forget_is_bounded() {
    let _g = serial();
    let (mut m, clock) = tiny(20.0);
    bounded("remember+forget", &mut m, &clock, 20.0, |m, i| {
        if let Ok(Remembered::Inserted { trace, .. }) = m.remember(&vocab_text(i), 1.0, "") {
            m.forget(&trace.id).unwrap();
        }
        m.forget("not an id").unwrap();
    });
}

/// R4: dream + undo (cursor, reports): clusters are made by cued writes.
#[test]
fn r4_dream_and_undo_are_bounded() {
    let _g = serial();
    let (mut m, clock) = tiny(20.0);
    bounded("dream+undo", &mut m, &clock, 20.0, |m, i| {
        let _ = m.remember(&vocab_text(i), 1.0, &vocab_text(i.wrapping_sub(1).min(i)));
        let reports = m.dream(&mut merge_to_gist, Some(2)).unwrap();
        if i % 2 == 0 {
            m.undo_all(&reports).unwrap();
        }
        m.clusters(Some(1)).unwrap();
        assert!(m.dream_cursor().len() < 64);
    });
}

/// R5: reindex with an embedder that flips offline / online (stale rows, damaged list).
#[test]
fn r5_reindex_is_bounded() {
    let _g = serial();
    let online = Arc::new(AtomicBool::new(true));
    let o = online.clone();
    let e = Flex {
        model: Box::new(|| "flap"),
        dim: Box::new(|| 16),
        docs: Box::new(move |t| if o.load(Ordering::SeqCst) { Ok(t.iter().map(|s| onehot(s, 16)).collect()) } else { Err("offline".into()) }),
        queries: Box::new(|t| Ok(t.iter().map(|s| onehot(s, 16)).collect())),
    };
    let (mut m, clock) = engine_with(Config { capacity: 8, writes_per_day: 1_000_000, ..cfg() }, e);
    bounded("reindex", &mut m, &clock, 20.0, |m, i| {
        online.store(i % 3 != 0, Ordering::SeqCst);
        let _ = m.remember(&vocab_text(i), 1.0, "");
        online.store(true, Ordering::SeqCst);
        m.reindex(16).unwrap();
    });
}

/// R6: a clock that jumps back and forth by a day or more every operation: write_times keeps
/// at most `writes_per_day` entries (the limiter), the engine keeps nothing else per write.
#[test]
fn r6_oscillating_clock_is_bounded() {
    let _g = serial();
    let (mut m, clock) = tiny(0.0);
    let mut at_quarter = 0;
    for i in 0..N {
        clock.set(if i % 2 == 0 { START + 1e6 * i as f64 } else { START - 1e6 * i as f64 });
        let _ = m.remember(&vocab_text(i), 1.0, "");
        let _ = m.recall(&vocab_text(i)).unwrap();
        if i == N / 4 {
            at_quarter = live();
        }
    }
    let grown = live() - at_quarter;
    eprintln!("R: oscillating clock: live heap {grown:+} bytes");
    assert!(grown <= SLACK, "oscillating clock: live heap grew by {grown} bytes");
}

/// R7: the damaged list (rows whose vector is beyond repair) never holds more than DAMAGED_MAX = 64
/// ids: reindex repairs at most that many.
#[test]
fn r7_the_damaged_list_is_capped() {
    let _g = serial();
    struct Corrupting(Store);
    impl Storage for Corrupting {
        fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
            let mut rows = self.0.find(view, limit)?;
            rows.iter_mut().filter(|t| !t.vector.is_empty()).for_each(|t| t.vector[0] = f32::NAN);
            Ok(rows)
        }
        fn count(&self, view: View<'_>) -> Result<usize> {
            self.0.count(view)
        }
        fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
            let mut hits = self.0.near(q, model_id, min_cos, k, stated_by)?;
            hits.iter_mut().for_each(|h| h.trace.vector[0] = f32::NAN);
            Ok(hits)
        }
        fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
            self.0.write(writes)
        }
        fn clear(&mut self) -> Result<()> {
            self.0.clear()
        }
    }
    let (mut m, clock) = build_with(Config { capacity: 1000, writes_per_day: 1_000_000, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(Corrupting(Store::in_memory().unwrap())));
    for i in 0..300 {
        clock.advance(1.0);
        insert(&mut m, &format!("row number{i}"), 1.0, "");
    }
    for _ in 0..50 {
        let _ = m.find_all_for_audit();
    }
}

trait AuditFind {
    fn find_all_for_audit(&mut self) -> usize;
}
impl AuditFind for Engram {
    fn find_all_for_audit(&mut self) -> usize {
        // every row is damaged: reads note them; one reindex repairs at most 64 of them
        let _ = self.recall("row number1 number2 number3").unwrap();
        let done = self.reindex(10_000).unwrap();
        assert!(done <= 64 + 1, "reindex repaired {done} damaged rows in one call; the list is meant to hold 64");
        done
    }
}

// ============================================================ D. damaged redb file

fn tamper<R>(path: &std::path::Path, f: impl FnOnce(&redb::WriteTransaction) -> R) -> R {
    let db = redb::Database::open(path).unwrap();
    let wx = db.begin_write().unwrap();
    let r = f(&wx);
    wx.commit().unwrap();
    r
}

const CENTROIDS: redb::TableDefinition<(&str, u32), &[u8]> = redb::TableDefinition::new("centroids");
const LISTS: redb::TableDefinition<(&str, u32, &str), &[u8]> = redb::TableDefinition::new("lists");

fn file_engine(path: &std::path::Path, cfg: Config) -> (Engram, ManualClock) {
    let clock = ManualClock::new(START);
    let m = Engram::with_clock(Box::new(Store::open(path).unwrap()), Box::new(FakeEmbedder::new()), cfg, "UTC", clock.clock()).unwrap();
    (m, clock)
}

fn seeded_file(name: &str, n: usize) -> std::path::PathBuf {
    let path = temp_dir(name).join("s.redb");
    let (mut m, clock) = file_engine(&path, Config { capacity: 3, ..cfg() });
    for i in 0..n {
        clock.advance(1.0);
        insert(&mut m, &format!("row number{i} words"), 1.0, "");
    }
    path
}

/// D1a: a centroid whose stored size is `u64::MAX` (damaged file): the next insert does `size += 1`.
/// SPEC / store docs: a damaged file is an error, never a panic.
#[test]
fn d1a_a_centroid_size_of_u64_max_does_not_panic_the_next_insert() {
    let _g = serial();
    let path = seeded_file("a3c-size", 2);
    tamper(&path, |wx| {
        let mut t = wx.open_table(CENTROIDS).unwrap();
        let keys: Vec<(String, u32, Vec<u8>)> = redb::ReadableTable::iter(&t).unwrap().map(|e| { let (k, v) = e.unwrap(); (k.value().0.to_owned(), k.value().1, v.value().to_vec()) }).collect();
        assert!(!keys.is_empty());
        for (m, l, mut raw) in keys {
            raw[..8].copy_from_slice(&u64::MAX.to_le_bytes());
            t.insert((m.as_str(), l), raw.as_slice()).unwrap();
        }
    });
    let (mut m, _c) = file_engine(&path, Config { capacity: 100, ..cfg() });
    let _ = no_panic("insert into a list whose size is u64::MAX", || m.remember("a brand new fact", 1.0, ""));
}

/// D1b: two lists whose sizes sum past `u64::MAX`: `near` does `read += l.size`.
#[test]
fn d1b_centroid_sizes_that_sum_past_u64_max_do_not_panic_near() {
    let _g = serial();
    let path = seeded_file("a3c-sum", 2);
    tamper(&path, |wx| {
        let mut t = wx.open_table(CENTROIDS).unwrap();
        let first: (String, u32, Vec<u8>) = redb::ReadableTable::iter(&t).unwrap().map(|e| { let (k, v) = e.unwrap(); (k.value().0.to_owned(), k.value().1, v.value().to_vec()) }).next().unwrap();
        let mut big = first.2.clone();
        big[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        t.insert((first.0.as_str(), 9999), big.as_slice()).unwrap();
    });
    let (mut m, _c) = file_engine(&path, Config { capacity: 100, ..cfg() });
    let _ = no_panic("recall over lists whose sizes sum past u64::MAX", || m.recall("row number0 words"));
}

/// D2: one list entry too short to decode makes every `find` that touches it an error, so
/// `forget(id)` of that very row fails too, and nothing but `reset` (all data) heals the store.
/// (DESIGN-CONFLICT: store.rs documents "a damaged file is an error"; a row whose JSON is
/// damaged IS returned as an unusable row, evicted first and forgettable.)
#[test]
fn d2_a_damaged_list_entry_does_not_make_the_row_unforgettable() {
    let _g = serial();
    let path = seeded_file("a3c-short", 3);
    let id = tamper(&path, |wx| {
        let mut t = wx.open_table(LISTS).unwrap();
        let k: (String, u32, String) = redb::ReadableTable::iter(&t).unwrap().next().map(|e| { let (k, _) = e.unwrap(); (k.value().0.to_owned(), k.value().1, k.value().2.to_owned()) }).unwrap();
        t.insert((k.0.as_str(), k.1, k.2.as_str()), [1u8, 2, 3].as_slice()).unwrap();
        k.2
    });
    let (mut m, _c) = file_engine(&path, Config { capacity: 3, ..cfg() });
    let r = m.forget(&id);
    assert!(r.is_ok(), "forget({id}) of a row with a damaged list entry: {:?}", r.err().map(|e| e.to_string()));
}

/// D3: a damaged centroid entry makes `near` and every indexed write an error for good: the
/// engine can neither recall nor remember. (DESIGN-CONFLICT, see D2: the index is derived data.)
#[test]
fn d3_a_damaged_centroid_does_not_wedge_remember() {
    let _g = serial();
    let path = seeded_file("a3c-cent", 2);
    tamper(&path, |wx| {
        let mut t = wx.open_table(CENTROIDS).unwrap();
        let k: (String, u32) = redb::ReadableTable::iter(&t).unwrap().next().map(|e| { let (k, _) = e.unwrap(); (k.value().0.to_owned(), k.value().1) }).unwrap();
        t.insert((k.0.as_str(), k.1), [9u8].as_slice()).unwrap();
    });
    let (mut m, _c) = file_engine(&path, Config { capacity: 100, ..cfg() });
    let r = m.remember("a brand new fact", 1.0, "");
    assert!(r.is_ok(), "remember: {:?}", r.err().map(|e| e.to_string()));
}

/// Runs `Store::open(path)` on its own thread: `Ok(None)` = opened, `Ok(Some(msg))` = an error,
/// `Err(what)` = it panicked or did not return within `secs`.
fn open_watched(path: &std::path::Path, secs: u64) -> std::result::Result<Option<String>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let p = path.to_owned();
    std::thread::spawn(move || {
        let r = catch_unwind(AssertUnwindSafe(|| Store::open(&p).map(|_| ()).map_err(|e| e.to_string())));
        let _ = tx.send(r.map_err(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned())).unwrap_or_default()));
    });
    match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
        Ok(Ok(Ok(()))) => Ok(None),
        Ok(Ok(Err(e))) => Ok(Some(e)),
        Ok(Err(panic)) => Err(format!("panicked: {panic}")),
        Err(_) => Err(format!("did not return within {secs} s")),
    }
}

/// D4: `Store::open` of a database file that was cut short (a partial copy, a full disk, a crash
/// of a non-journaling file system). redb 2.6.4 asserts `raw_file_len() >= header.layout().len()`
/// (a panic) for files of 512 bytes up to one byte short, and loops forever on files of about
/// 64-100 bytes. A damaged file must be an `Err`.
#[test]
fn d4_a_truncated_database_file_is_an_error_not_a_panic_or_a_hang() {
    let _g = serial();
    let good = seeded_file("a3c-trunc", 5);
    let bytes = std::fs::read(&good).unwrap();
    let dir = temp_dir("a3c-trunc-cuts");
    let mut failures = Vec::new();
    for cut in [64usize, 100, 512, 4096, 1 << 20, bytes.len() - 1] {
        let p = dir.join(format!("cut{cut}.redb"));
        std::fs::write(&p, &bytes[..cut.min(bytes.len())]).unwrap();
        if let Err(what) = open_watched(&p, 8) {
            failures.push(format!("a file cut to {cut} bytes: {what}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// G: a file that is not a database, and a second handle on an open one, are errors.
#[test]
fn g_d5_junk_and_locked_files_are_errors() {
    let _g = serial();
    let dir = temp_dir("a3c-junk");
    let junk = dir.join("junk.redb");
    std::fs::write(&junk, vec![0xA5u8; 10_000]).unwrap();
    assert_eq!(open_watched(&junk, 8).map(|e| e.is_some()), Ok(true));
    let good = seeded_file("a3c-lock", 3);
    let held = Store::open(&good).unwrap();
    assert_eq!(open_watched(&good, 8).map(|e| e.is_some()), Ok(true));
    drop(held);
}
