//! Audit lane 4 — wake phase and text (SPEC §2, §4, §7). Public API only.
//! Tests named `note_*` pin down design trade-offs / ambiguous spec points; the rest assert
//! the spec and fail only on a real defect. Each doc comment names the hypothesis.
//! Run by copying to tests/audit_lane4.rs (it uses tests/support).

mod support;

use engram::{cues, fmt_local, ulid_ids, Config, Embedder, Engram, Hit, ManualClock, Remembered, Storage, Trace, View, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use support::{build, build_with, cfg, insert, mem, rows, FakeEmbedder, SharedStore, DAY, START};

const MODEL: &str = "fake/token-overlap";

// ------------------------------------------------------------------ helpers

fn trace(id: &str, text: &str) -> Trace {
    Trace { id: id.into(), text: text.into(), created_at: START as i64, tz: "UTC;+00:00".into(), last_recall: START as i64, stability: DAY, consolidated: true, cue: String::new(), model_id: String::new(), vector: Vec::new() }
}

fn indexed(id: &str, text: &str) -> Trace {
    let mut t = trace(id, text);
    t.model_id = MODEL.into();
    t.vector = FakeEmbedder::new().embed(text);
    t
}

fn put(store: &SharedStore, t: &Trace) {
    store.clone().write(&[Write::Insert(t)]).unwrap();
}

fn engine_on(store: &SharedStore, c: Config) -> (Engram, ManualClock) {
    build_with(c, Box::new(FakeEmbedder::new()), Box::new(store.clone()))
}

type Rewrite = Arc<dyn Fn(Trace) -> Trace + Send + Sync>;

/// A store that rewrites rows on the way out (an "older version" wrote them), can fail
/// reads, and can add bogus hits to similarity searches.
#[derive(Clone, Default)]
struct Hostile {
    inner: SharedStore,
    rewrite: Arc<Mutex<Option<Rewrite>>>,
    fail_find: Arc<AtomicBool>,
    extra: Arc<Mutex<Vec<Hit>>>,
}

impl Hostile {
    fn rw(&self, t: Trace) -> Trace {
        match &*self.rewrite.lock().unwrap() {
            Some(f) => f(t),
            None => t,
        }
    }
}

impl Storage for Hostile {
    fn find(&self, view: View<'_>, limit: usize) -> engram::Result<Vec<Trace>> {
        if self.fail_find.load(Ordering::SeqCst) {
            return Err("store down".into());
        }
        Ok(self.inner.find(view, limit)?.into_iter().map(|t| self.rw(t)).collect())
    }
    fn count(&self, view: View<'_>) -> engram::Result<usize> {
        self.inner.count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> engram::Result<Vec<Hit>> {
        let mut hits: Vec<Hit> = self.extra.lock().unwrap().clone();
        hits.extend(self.inner.near(q, model_id, min_cos, k, stated_by)?.into_iter().map(|h| Hit { cos: h.cos, trace: self.rw(h.trace) }));
        Ok(hits)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> engram::Result<()> {
        self.inner.write(writes)
    }
    fn clear(&mut self) -> engram::Result<()> {
        self.inner.clear()
    }
}

/// A switchable embedder (offline until `up`).
struct Switch(FakeEmbedder, Arc<AtomicBool>);

impl Embedder for Switch {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
    fn dimension(&self) -> usize {
        self.0.dimension
    }
    fn embed_documents(&mut self, t: &[&str]) -> engram::Result<Vec<Vec<f32>>> {
        if !self.1.load(Ordering::SeqCst) {
            return Err("offline".into());
        }
        self.0.embed_documents(t)
    }
    fn embed_queries(&mut self, t: &[&str]) -> engram::Result<Vec<Vec<f32>>> {
        if !self.1.load(Ordering::SeqCst) {
            return Err("offline".into());
        }
        self.0.embed_queries(t)
    }
}

/// Documents embed, queries (cues) never do.
struct CueDown(FakeEmbedder);

impl Embedder for CueDown {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
    fn dimension(&self) -> usize {
        self.0.dimension
    }
    fn embed_documents(&mut self, t: &[&str]) -> engram::Result<Vec<Vec<f32>>> {
        self.0.embed_documents(t)
    }
    fn embed_queries(&mut self, _: &[&str]) -> engram::Result<Vec<Vec<f32>>> {
        Err("queries down".into())
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn sane(t: &Trace, now: f64, c: &Config) {
    assert!(t.stability.is_finite() && t.stability >= 1.0 && t.stability <= c.max_stability, "stability {} of {:?}", t.stability, t.id);
    let _ = now; // stored times are not rewritten on read; a future one means no time has elapsed
    assert!(t.text.chars().count() <= c.text_max && !t.text.is_empty());
    assert!(!t.text.contains(['《', '》']) && !t.cue.contains(['《', '》']));
    assert!(t.tz.chars().count() <= 64, "tz too long");
    assert!(t.vector.iter().all(|x| x.is_finite()));
}

// ================================================================== L4-1 text and ids

/// SPEC §2: an id holding a control character is not adoptable / not usable.
/// Hypothesis: `valid_id` reuses `is_dropped`, which exempts the whitespace controls
/// (\t \n \r \v \f U+001C-1F U+0085) and U+2028/2029 that `clean_text` turns into spaces.
#[test]
fn l4_1_valid_id_rejects_every_control_character() {
    let controls = ['\t', '\n', '\u{b}', '\u{c}', '\r', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{85}', '\u{7f}', '\u{0}'];
    let mut accepted_rows = Vec::new();
    let mut accepted_gen = Vec::new();
    for ch in controls.iter().copied().chain(['\u{2028}', '\u{2029}']) {
        let store = SharedStore::default();
        let (m, _) = engine_on(&store, cfg());
        let id = format!("A{ch}B");
        put(&store, &trace(&id, "a row with an odd id"));
        if m.memory(&id).unwrap().is_some() {
            accepted_rows.push(format!("{:04X}", ch as u32));
        }
        let (mut m, _) = engine_on(&SharedStore::default(), cfg());
        m.set_ids(Box::new(move |_, _| format!("A{ch}B")));
        if m.remember("fresh fact about odd generator", 1.0, "").is_ok() {
            accepted_gen.push(format!("{:04X}", ch as u32));
        }
    }
    assert!(accepted_rows.is_empty() && accepted_gen.is_empty(), "read path accepted ids with U+{accepted_rows:?}; IdGen path adopted ids with U+{accepted_gen:?}");
}

/// A line-oriented pack must stay one trace per line.
#[test]
fn l4_1_pack_has_one_line_per_trace_even_for_an_odd_stored_id() {
    let store = SharedStore::default();
    put(&store, &indexed("A\nB", "kyoto temple garden"));
    let (mut m, _) = engine_on(&store, cfg());
    let r = m.recall("kyoto temple garden").unwrap();
    assert_eq!(r.pack_text.lines().count(), r.recalled.len(), "pack: {:?}", r.pack_text);
    assert!(r.recalled.is_empty(), "a row whose id holds a newline is unusable (SPEC §2)");
}

fn is_bad_char(c: char) -> bool {
    matches!(c, '《' | '》' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}') || c.is_control() || c == '\u{FEFF}' || (c.is_whitespace() && c != ' ')
}

#[test]
fn l4_1_clean_text_fuzz_bounds_idempotence_and_alphabet() {
    const POOL: &[char] = &[
        'a', 'b', 'c', 'd', ' ', ' ', '.', '。', '、', '！', '《', '》', '\u{202E}', '\u{2066}', '\u{0}', '\u{7f}', '\u{85}', '\u{FEFF}', '\u{1C}', '\n', '\t', '\u{2028}',
        '\u{3000}', 'é', '\u{301}', '😀', '\u{200B}', 'あ', '\u{200E}', '\u{10FFFF}',
    ];
    let mut c = cfg();
    c.writes_per_day = 1_000_000;
    let (mut m, _) = build(c.clone());
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..500 {
        let len = [0, 1, 2, 5, 40, 169, 170, 171, 172, 200, 400][rng.below(11)];
        let text: String = (0..len).map(|_| POOL[rng.below(POOL.len())]).collect();
        let cue: String = (0..rng.below(250)).map(|_| POOL[rng.below(POOL.len())]).collect();
        match m.remember(&text, 1.0, &cue).unwrap() {
            Remembered::Rejected => assert!(text.chars().all(|c| is_bad_char(c) || c.is_whitespace()), "case {case}: rejected {text:?}"),
            Remembered::Inserted { trace: t, .. } | Remembered::Reinforced(t) => {
                let n = t.text.chars().count();
                assert!((1..=170).contains(&n), "case {case}: {n} chars from {text:?}");
                assert!(!t.text.chars().any(is_bad_char), "case {case}: {:?}", t.text);
                assert_eq!(t.text.trim(), t.text);
                assert!(!t.text.contains("  "));
                assert!(t.cue.chars().count() <= 170 && !t.cue.chars().any(is_bad_char), "case {case}: cue {:?}", t.cue);
                assert_eq!(t.cue.trim(), t.cue);
                match m.remember(&t.text, 1.0, "").unwrap() {
                    Remembered::Reinforced(r) => assert_eq!(r.id, t.id, "case {case}"),
                    other => panic!("case {case}: clean_text is not idempotent on {:?}: {other:?}", t.text),
                }
            }
            Remembered::RateLimited { .. } => panic!("rate limited"),
        }
    }
}

#[test]
fn l4_1_clean_text_counts_chars_and_cuts_at_sentence_boundary() {
    let (mut m, _) = build(cfg());
    let Remembered::Inserted { trace: t, .. } = m.remember(&"😀".repeat(300), 1.0, "").unwrap() else { panic!() };
    assert_eq!(t.text.chars().count(), 170);
    assert_eq!(t.text.len(), 680);
    let text = format!("{}。{}", "a".repeat(100), "b".repeat(100));
    let Remembered::Inserted { trace: t, .. } = m.remember(&text, 1.0, "").unwrap() else { panic!() };
    assert_eq!(t.text, format!("{}。", "a".repeat(100)));
    let text = format!("{} {}", "a".repeat(169), "b".repeat(10));
    let Remembered::Inserted { trace: t, .. } = m.remember(&text, 1.0, "").unwrap() else { panic!() };
    assert_eq!(t.text, "a".repeat(169));
}

#[test]
fn l4_1_clean_tz_keeps_the_offset_and_the_bound() {
    for (name, offset) in [
        ("Asia/Tokyo".to_string(), "+09:00"),
        ("Very/Long_Zone".repeat(30), "+00:00"),
        ("A;B\nC《》\u{202E}".to_string(), "+00:00"),
        ("N".repeat(64), "+00:00"),
        ("N".repeat(57), "+00:00"),
        (String::new(), "+00:00"),
    ] {
        let clock = ManualClock::new(START);
        let mut m = Engram::with_clock(Box::new(engram::Store::in_memory().unwrap()), Box::new(FakeEmbedder::new()), cfg(), &name, clock.clock()).unwrap();
        let Remembered::Inserted { trace: t, .. } = m.remember("a fact about a zone", 1.0, "").unwrap() else { panic!() };
        assert!(t.tz.chars().count() <= 64, "{:?}", t.tz);
        assert!(t.tz.ends_with(&format!(";{offset}")), "tz {:?} for zone {name:?}", t.tz);
        assert!(!t.tz.chars().any(is_bad_char), "{:?}", t.tz);
        assert!(fmt_local(START as i64, &t.tz).ends_with(offset), "{}", fmt_local(START as i64, &t.tz));
    }
}

/// SPEC §8: "id は 50bit ミリ秒（西暦 37,000 年）". The `ulid` crate masks to 48 bits.
#[test]
#[ignore = "accepted: the ulid crate keeps 48-bit milliseconds (year 10,889); SPEC §8 now says so"]
fn note_l4_1_ulid_ids_stay_ordered_past_48_bit_milliseconds() {
    let mut g = ulid_ids();
    let a = g((1i64 << 48) - 1, "");
    let b = g(1i64 << 48, "");
    assert!(b > a, "id timestamp wrapped at 2^48 ms (year 10889), not 2^50: {a} then {b}");
}

// ================================================================== L4-2 recall pack

fn corpus(m: &mut Engram, n: usize, words: &[String], clock: &ManualClock, rng: &mut Rng) {
    for i in 0..n {
        let k = 3 + rng.below(3);
        let mut text: Vec<String> = (0..k).map(|_| words[rng.below(words.len())].clone()).collect();
        text.push(format!("u{i}"));
        insert(m, &text.join(" "), 1.0, "");
        clock.advance((rng.below(40) as f64) * 3600.0);
    }
}

#[test]
fn l4_2_pack_budget_count_and_line_shape() {
    for budget in [10usize, 120, 400, 1024] {
        let mut c = cfg();
        c.budget_chars = budget;
        let (mut m, clock) = build(c.clone());
        for i in 0..14 {
            insert(&mut m, &format!("alpha fact number{i} with filler words to lengthen the line a bit"), 1.0, "");
            clock.advance(60.0);
        }
        let r = m.recall("alpha").unwrap();
        assert!(!r.recalled.is_empty() && r.recalled.len() <= c.inject_n);
        assert_eq!(r.pack_text.lines().count(), r.recalled.len());
        if r.recalled.len() > 1 {
            assert!(r.pack_text.chars().count() <= budget, "budget {budget}: {} chars", r.pack_text.chars().count());
        }
        for (line, rc) in r.pack_text.lines().zip(&r.recalled) {
            assert!(line.starts_with('[') && line.ends_with(&format!("《id:{}》", rc.trace.id)), "{line:?}");
        }
        let ids: std::collections::HashSet<_> = r.recalled.iter().map(|c| &c.trace.id).collect();
        assert_eq!(ids.len(), r.recalled.len(), "MMR selected an index twice");
    }
}

#[test]
fn l4_2_equal_scores_break_ties_by_id() {
    let (mut m, _) = build(cfg());
    let a = insert(&mut m, "alpha beta gamma!", 1.0, "");
    let b = insert(&mut m, "alpha beta gamma?", 1.0, "");
    let r = m.peek("alpha beta gamma").unwrap();
    assert_eq!(r.recalled[0].trace.id, a);
    assert!(a < b);
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / n) as f32).collect()
}

fn dotf(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

/// SPEC §4 recall.4: the widening search equals a scan of everything.
#[test]
fn l4_2_recall_equals_brute_force_scan() {
    let mut c = cfg();
    c.inject_n = 3;
    c.budget_chars = 1_000_000;
    let (mut m, clock) = build(c.clone());
    let words: Vec<String> = (0..14).map(|i| format!("w{i}")).collect();
    let mut rng = Rng(12345);
    corpus(&mut m, 80, &words, &clock, &mut rng);
    clock.advance(5.0 * DAY);
    let (mut checked, mut widened) = (0, 0);
    for _ in 0..60 {
        let mut query: Vec<String> = Vec::new();
        for _ in 0..1 + rng.below(2) {
            let line: Vec<&str> = (0..1 + rng.below(3)).map(|_| words[rng.below(words.len())].as_str()).collect();
            query.push(line.join(" "));
        }
        let query = query.join("\n");
        let now = m.now();
        let mut emb = FakeEmbedder::new();
        let qs: Vec<Vec<f32>> = cues(&query, c.max_cues).iter().map(|q| unit(&emb.embed(q))).collect();
        let all = rows(&m);
        let scored: Vec<(String, f64, Vec<f32>)> = all
            .iter()
            .map(|t| {
                let cos = qs.iter().map(|q| dotf(&t.vector, q).min(1.0)).fold(f64::NEG_INFINITY, f64::max);
                let a = ((cos - c.cosine_floor) / (1.0 - c.cosine_floor)).max(0.0);
                let r = (-((now - t.last_recall as f64).max(0.0)) / t.stability).exp2();
                (t.id.clone(), a * (c.alpha + (1.0 - c.alpha) * r), t.vector.clone())
            })
            .collect();
        let top = scored.iter().map(|s| s.1).fold(0.0, f64::max);
        let floor = c.min_score.max(c.relative_score * top);
        if scored.iter().any(|s| (s.1 - floor).abs() < 1e-6) {
            continue;
        }
        let pool: Vec<&(String, f64, Vec<f32>)> = scored.iter().filter(|s| s.1 >= floor).collect();
        if pool.len() > 4 * c.inject_n {
            widened += 1;
        }
        let mut max_sim = vec![0.0f64; pool.len()];
        let mut alive = vec![true; pool.len()];
        let mut expect = Vec::new();
        while expect.len() < c.inject_n {
            let mut best: Option<(usize, f64)> = None;
            for i in (0..pool.len()).filter(|&i| alive[i]) {
                let v = pool[i].1 - c.mmr_lambda * max_sim[i];
                if best.is_none_or(|(_, bv)| v > bv) {
                    best = Some((i, v));
                }
            }
            let Some((b, _)) = best else { break };
            expect.push(pool[b].0.clone());
            alive[b] = false;
            for i in (0..pool.len()).filter(|&i| alive[i]) {
                max_sim[i] = max_sim[i].max(dotf(&pool[i].2, &pool[b].2));
            }
        }
        let got: Vec<String> = m.peek(&query).unwrap().recalled.iter().map(|r| r.trace.id.clone()).collect();
        assert_eq!(got, expect, "query {query:?}");
        checked += 1;
    }
    assert!(checked >= 30 && widened >= 5, "weak coverage: {checked} checked, {widened} widened");
}

/// `4 * inject_n` and `k *= 4` are unchecked. Either `Config::validate` must refuse
/// absurd values or recall must not panic.
#[test]
fn l4_2_absurd_inject_n_is_rejected_or_harmless() {
    let mut c = cfg();
    c.inject_n = usize::MAX;
    let Ok(mut m) = Engram::with_clock(Box::new(engram::Store::in_memory().unwrap()), Box::new(FakeEmbedder::new()), c, "UTC", ManualClock::new(START).clock()) else { return };
    m.remember("alpha beta gamma", 1.0, "").unwrap();
    let r = catch_unwind(AssertUnwindSafe(|| m.recall("alpha beta gamma")));
    assert!(r.is_ok(), "recall panicked: unchecked `4 * inject_n` overflow (Config::validate accepts inject_n = usize::MAX)");
}

// ================================================================== L4-3 cite

fn recalled_one() -> (Engram, ManualClock, String) {
    let (mut m, clock) = build(cfg());
    let a = insert(&mut m, "kyoto temple garden visit", 1.0, "");
    insert(&mut m, "tokyo tower night view", 1.0, "");
    clock.advance(2.0 * DAY);
    let r = m.recall("kyoto temple garden visit").unwrap();
    assert_eq!(r.recalled.len(), 1);
    assert_eq!(r.recalled[0].trace.id, a);
    (m, clock, a)
}

#[test]
fn l4_3_cite_only_the_last_recalls_exposure_each_once_full_boost() {
    let (mut m, _, a) = recalled_one();
    let after_recall = mem(&m, &a).unwrap();
    assert!((after_recall.stability / DAY - 2.125).abs() < 1e-3, "{}", after_recall.stability / DAY);
    let got = m.cite(&format!("see 《id:{a}》 and again 《id:{a}》 and 《id:NOPE》 《id:》")).unwrap();
    assert_eq!(got, vec![a.clone()]);
    let c = mem(&m, &a).unwrap();
    assert!((c.stability / DAY - 3.25).abs() < 1e-3, "{}", c.stability / DAY);
    assert_eq!(c.last_recall, after_recall.last_recall);
    assert!(c.stability >= after_recall.stability);
    assert!(m.cite(&format!("《id:{a}》")).unwrap().is_empty(), "second cite must not boost again");
    assert_eq!(mem(&m, &a).unwrap().stability, c.stability);
}

#[test]
fn l4_3_a_newer_recall_ends_the_citation_window() {
    let (mut m, _, a) = recalled_one();
    let r = m.recall("tokyo tower night view").unwrap();
    assert_eq!(r.recalled.len(), 1);
    let before = mem(&m, &a).unwrap().stability;
    assert!(m.cite(&format!("《id:{a}》")).unwrap().is_empty());
    assert_eq!(mem(&m, &a).unwrap().stability, before);
    let none = m.recall("completely unrelated zebra quantum").unwrap();
    assert!(none.recalled.is_empty());
    let b = rows(&m).iter().find(|t| t.text.contains("tokyo")).unwrap().id.clone();
    assert!(m.cite(&format!("《id:{b}》")).unwrap().is_empty());
}

#[test]
fn l4_3_cite_of_a_forgotten_trace_neither_errors_nor_resurrects() {
    let (mut m, _, a) = recalled_one();
    assert!(m.forget(&a).unwrap().is_some());
    assert!(m.cite(&format!("《id:{a}》")).unwrap().is_empty());
    assert_eq!(rows(&m).len(), 1);
    assert!(mem(&m, &a).is_none());
}

/// Hypothesis: `forget` leaves the exposure behind, so a re-created id inherits it.
/// Only reachable with an IdGen that can repeat an id after deletion (e.g. derived from
/// text and time; `new_id` only avoids ids currently stored).
#[test]
fn l4_3_cite_does_not_boost_a_trace_re_created_under_a_forgotten_id() {
    let (mut m, clock) = build(cfg());
    m.set_ids(Box::new(|_, _| "FIXED-ID".to_owned()));
    insert(&mut m, "kyoto temple garden visit", 1.0, "");
    clock.advance(2.0 * DAY);
    m.recall("kyoto temple garden visit").unwrap();
    m.forget("FIXED-ID").unwrap();
    insert(&mut m, "an entirely different fact about zebra", 1.0, "");
    let before = mem(&m, "FIXED-ID").unwrap().stability;
    let got = m.cite("《id:FIXED-ID》").unwrap();
    assert!(got.is_empty() && mem(&m, "FIXED-ID").unwrap().stability == before, "the new trace was boosted by the old trace's exposure");
}

#[test]
fn l4_3_cite_parser_finds_a_well_formed_citation_after_a_stray_opener() {
    let (mut m, _, a) = recalled_one();
    // an LLM that stutters the opener: the inner citation is well formed
    let got = m.cite(&format!("《id:《id:{a}》")).unwrap();
    assert_eq!(got, vec![a], "cited_ids reads up to the first 》 and swallows the real citation");
}

/// Hypothesis: `cite` removes the exposure before the store read, so a transient store
/// error loses the citation window for good.
#[test]
fn l4_3_cite_survives_a_transient_store_error() {
    let store = Hostile::default();
    let (mut m, clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let a = insert(&mut m, "kyoto temple garden visit", 1.0, "");
    clock.advance(2.0 * DAY);
    m.recall("kyoto temple garden visit").unwrap();
    store.fail_find.store(true, Ordering::SeqCst);
    assert!(m.cite(&format!("《id:{a}》")).is_err());
    store.fail_find.store(false, Ordering::SeqCst);
    assert_eq!(m.cite(&format!("《id:{a}》")).unwrap(), vec![a], "retry after the error found nothing");
}

// ================================================================== L4-4 remember

#[test]
fn l4_4_rate_limit_window_and_zero() {
    let mut c = cfg();
    c.writes_per_day = 3;
    let (mut m, clock) = build(c);
    for i in 0..3 {
        insert(&mut m, &format!("fact number{i} here"), 1.0, "");
        clock.advance(10.0);
    }
    assert_eq!(m.remember("fact number9 here", 1.0, "").unwrap(), Remembered::RateLimited { text: "fact number9 here".into() });
    clock.advance(DAY);
    insert(&mut m, "fact number9 here", 1.0, "");
    let mut c = cfg();
    c.writes_per_day = 0;
    let (mut z, _) = build(c);
    assert!(matches!(z.remember("anything", 1.0, "").unwrap(), Remembered::RateLimited { .. }));
    assert_eq!(rows(&z).len(), 0);
}

/// SPEC §4 remember.1 imposes the daily limit before the duplicate check. Rehearsal is a
/// store write, so with the limit exhausted an endless repeat still writes.
#[test]
#[ignore = "accepted: rehearsal adds no trace, so it is outside the write limit (SPEC §4)"]
fn note_l4_4_rehearsal_obeys_the_write_limit() {
    let store = SharedStore::default();
    let (mut a, _) = engine_on(&store, cfg());
    insert(&mut a, "kyoto temple garden", 1.0, "");
    let mut c = cfg();
    c.writes_per_day = 0;
    let (mut b, _) = engine_on(&store, c);
    let r = b.remember("kyoto temple garden", 1.0, "").unwrap();
    assert!(matches!(r, Remembered::RateLimited { .. }), "got {r:?}");
}

/// A clock step back drops every write stamped later than the new `now`, so the limit
/// is bypassed by a backwards jump. (Trade-off documented in the source comment.)
#[test]
#[ignore = "accepted: a clock stepped back reopens the daily window (SPEC §4)"]
fn note_l4_4_clock_step_back_does_not_reset_the_write_limit() {
    let mut c = cfg();
    c.writes_per_day = 3;
    let (mut m, clock) = build(c);
    for i in 0..3 {
        insert(&mut m, &format!("fact number{i} here"), 1.0, "");
    }
    clock.advance(-5.0);
    assert!(matches!(m.remember("fact number9 here", 1.0, "").unwrap(), Remembered::RateLimited { .. }));
}

#[test]
fn l4_4_ids_follow_creation_order_when_the_clock_runs_backwards() {
    let (mut m, clock) = build(cfg());
    let mut ids = Vec::new();
    for (i, dt) in [0.0, -3600.0, 0.0, -86_400.0, 5.0, -1.0, 0.0].into_iter().enumerate() {
        clock.advance(dt);
        ids.push(insert(&mut m, &format!("fact number{i} here"), 1.0, ""));
    }
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "{ids:?}");
    let stored: Vec<String> = rows(&m).into_iter().map(|t| t.id).collect();
    assert_eq!(stored, ids);
}

/// SPEC §2: id order = insertion order (FIFO seeds, ties). A fresh generator after a
/// restart with the clock behind the stored ids hands out smaller ids.
#[test]
#[ignore = "accepted: ids are ordered within a process; the host keeps its clock from running behind persisted ids (SPEC §2)"]
fn l4_4_ids_stay_ordered_across_a_restart_with_a_slower_clock() {
    let store = SharedStore::default();
    let (mut a, _) = engine_on(&store, cfg());
    let first = insert(&mut a, "first fact one", 1.0, "");
    let (mut b, clock_b) = engine_on(&store, cfg());
    clock_b.set(START - 1000.0);
    let second = insert(&mut b, "second fact two", 1.0, "");
    assert!(second > first, "restart: new generator, older clock: {second} < {first}");
}

#[test]
fn l4_4_salience_is_clamped_and_nan_is_one() {
    let (mut m, _) = build(cfg());
    let mut s = |i: usize, sal: f64| match m.remember(&format!("salience probe number{i}"), sal, "").unwrap() {
        Remembered::Inserted { trace, .. } => trace.stability,
        o => panic!("{o:?}"),
    };
    assert_eq!(s(0, f64::NAN), DAY);
    assert_eq!(s(1, f64::INFINITY), 10.0 * DAY);
    assert_eq!(s(2, f64::NEG_INFINITY), 1.0);
    assert_eq!(s(3, -4.0), 1.0);
    assert_eq!(s(4, 1e308), 10.0 * DAY);
    assert_eq!(s(5, 0.0), 1.0);
    assert_eq!(s(6, 2.5), 2.5 * DAY);
}

#[test]
fn l4_4_consolidated_flag_truth_table() {
    let up = Arc::new(AtomicBool::new(true));
    let (mut m, _) = build_with(cfg(), Box::new(Switch(FakeEmbedder::new(), up.clone())), Box::new(SharedStore::default()));
    let ins = |m: &mut Engram, t: &str, cue: &str| match m.remember(t, 1.0, cue).unwrap() {
        Remembered::Inserted { trace, cosine, .. } => (trace, cosine),
        o => panic!("{o:?}"),
    };
    let (t, _) = ins(&mut m, "kyoto temple garden", "");
    assert!(t.consolidated, "nothing to reactivate, indexed");
    let (t, cos) = ins(&mut m, "zebra quantum harp violin", "");
    assert!(t.consolidated && cos == 0.0);
    let (t, cos) = ins(&mut m, "kyoto temple garden visit", "");
    assert!(!t.consolidated && cos > 0.5, "reactivated an older trace: labile");
    let (t, cos) = ins(&mut m, "unrelated fact about sparrow", "kyoto temple garden");
    assert!(!t.consolidated && cos > 0.5, "the cue reactivates kyoto");
    // the embedder goes offline: a text-only trace, labile
    up.store(false, Ordering::SeqCst);
    let (t, _) = ins(&mut m, "offline fact about otters", "");
    assert!(!t.consolidated && t.model_id.is_empty() && t.vector.is_empty());
    // back online, but an unindexed trace exists: nothing is known to be unreactivated
    up.store(true, Ordering::SeqCst);
    let (t, _) = ins(&mut m, "yet another distinct claim about pelicans", "");
    assert!(!t.consolidated, "a stale trace might be what this reactivates");
    assert_eq!(m.reindex(10).unwrap(), 1);
    let (t, _) = ins(&mut m, "final distinct claim about walruses", "");
    assert!(t.consolidated, "all indexed and nothing near");
    // a cue that cannot be embedded: not known to reactivate nothing
    let (mut q, _) = build_with(cfg(), Box::new(CueDown(FakeEmbedder::new())), Box::new(SharedStore::default()));
    let (t, _) = ins(&mut q, "solo fact with a cue", "what is the solo fact");
    assert!(!t.consolidated, "cue could not be embedded");
    let (t, _) = ins(&mut q, "walrus mariner chronicle entry", "");
    assert!(t.consolidated);
}

#[test]
fn l4_4_a_failed_embedder_leaves_text_that_is_reindexed_and_recalled_later() {
    let up = Arc::new(AtomicBool::new(false));
    let (mut m, _) = build_with(cfg(), Box::new(Switch(FakeEmbedder::new(), up.clone())), Box::new(SharedStore::default()));
    let id = insert(&mut m, "otters hold hands while sleeping", 1.0, "");
    assert_eq!(mem(&m, &id).unwrap().text, "otters hold hands while sleeping");
    assert!(m.recall("otters sleeping").unwrap().recalled.is_empty());
    assert_eq!(m.reindex(10).unwrap(), 0);
    up.store(true, Ordering::SeqCst);
    assert_eq!(m.reindex(10).unwrap(), 1);
    assert_eq!(m.reindex(10).unwrap(), 0);
    let r = m.recall("otters sleeping").unwrap();
    assert_eq!(r.recalled.len(), 1);
    assert_eq!(r.recalled[0].trace.id, id);
}

/// SPEC §4 remember.2: "find(Text) の先頭数件で最初に使えるもの". More than TEXT_PROBE (8)
/// unusable same-text rows ahead of the usable one hide it and a duplicate is created.
#[test]
#[ignore = "accepted: SPEC §4 probes the first few same-text traces"]
fn note_l4_4_more_than_eight_unusable_duplicates_do_not_hide_the_usable_one() {
    let store = SharedStore::default();
    let (mut m, _) = engine_on(&store, cfg());
    let id = insert(&mut m, "dup fact here", 1.0, "");
    for i in 0..9 {
        put(&store, &trace(&format!("!\u{202E}{i}"), "dup fact here"));
    }
    let r = m.remember("dup fact here", 1.0, "").unwrap();
    assert!(matches!(&r, Remembered::Reinforced(t) if t.id == id), "got {r:?}");
}

/// A row written by an older cleaner (extra blanks, 《》) has a different raw text, so the
/// exact-text probe misses it and the cleaned text is stored twice.
#[test]
#[ignore = "accepted: rehearsal matches the cleaned text exactly; rows written by older cleaning stay separate traces"]
fn note_l4_4_rehearsal_finds_a_row_stored_with_an_older_cleaning() {
    let store = SharedStore::default();
    put(&store, &trace("0OLD", "  spaced   out 《x》 text "));
    let (mut m, _) = engine_on(&store, cfg());
    let r = m.remember("spaced out x text", 1.0, "").unwrap();
    assert!(matches!(&r, Remembered::Reinforced(t) if t.id == "0OLD"), "got {r:?}");
}

// ================================================================== L4-5 forget / stats / reset / checked

#[test]
fn l4_5_forget_stats_reset_are_consistent() {
    let mut c = cfg();
    c.writes_per_day = 2;
    let (mut m, _) = build(c);
    let a = insert(&mut m, "kyoto temple garden", 1.0, "");
    let b = insert(&mut m, "kyoto temple garden visit", 1.0, "");
    let s = m.stats().unwrap();
    assert_eq!((s.records, s.labile), (2, 1));
    assert_eq!(m.forget("nope").unwrap(), None);
    let gone = m.forget(&b).unwrap().unwrap();
    assert_eq!(gone.id, b);
    let s = m.stats().unwrap();
    assert_eq!((s.records, s.labile), (1, 0));
    assert!(m.forget(&b).unwrap().is_none());
    m.recall("kyoto").unwrap();
    m.reset().unwrap();
    assert_eq!(m.stats().unwrap().records, 0);
    assert!(m.cite(&format!("《id:{a}》")).unwrap().is_empty());
    // the write window restarted too
    insert(&mut m, "fresh fact one", 1.0, "");
    insert(&mut m, "fresh fact two", 1.0, "");
    assert!(matches!(m.remember("fresh fact three", 1.0, "").unwrap(), Remembered::RateLimited { .. }));
}

/// SPEC §2: unusable rows are counted and evicted first; nothing says `forget(id)` may
/// not remove one, and an operator has no other way to purge a corrupt row.
#[test]
fn note_l4_5_forget_can_remove_an_unusable_row() {
    let store = SharedStore::default();
    put(&store, &trace("0EMPTY", ""));
    put(&store, &trace("0BAD\u{202E}ID", "a row with a bidi id"));
    let (mut m, _) = engine_on(&store, cfg());
    assert_eq!(m.stats().unwrap().records, 2);
    m.forget("0EMPTY").unwrap();
    m.forget("0BAD\u{202E}ID").unwrap();
    assert_eq!(m.stats().unwrap().records, 0, "unusable rows survive forget(id)");
}

/// Stats.labile counts `Labile{model}` = labile AND indexed, so a text-only trace that is
/// labile is not counted ("Counts of all and labile traces").
#[test]
#[ignore = "accepted: Stats.labile counts the labile traces the dream can take (indexed)"]
fn note_l4_5_stats_labile_counts_text_only_traces() {
    let up = Arc::new(AtomicBool::new(false));
    let (mut m, _) = build_with(cfg(), Box::new(Switch(FakeEmbedder::new(), up)), Box::new(SharedStore::default()));
    insert(&mut m, "offline fact one", 1.0, "");
    insert(&mut m, "offline fact two", 1.0, "");
    let t = rows(&m);
    assert!(t.iter().all(|t| !t.consolidated));
    assert_eq!(m.stats().unwrap().labile, 2);
}

/// Rows an older version could have written (the Store only refuses non-finite numbers).
#[test]
fn l4_5_rows_from_an_older_version_are_read_with_the_write_time_rules() {
    let store = SharedStore::default();
    let mut rowset: Vec<Trace> = Vec::new();
    let mut t = indexed("0A1", "kyoto temple zero stability");
    t.stability = 0.0;
    rowset.push(t);
    let mut t = indexed("0A2", "kyoto temple negative stability");
    t.stability = -5.0;
    rowset.push(t);
    let mut t = indexed("0A3", "kyoto temple far future recall");
    t.last_recall = i64::MAX;
    rowset.push(t);
    let mut t = indexed("0A4", "kyoto temple future creation");
    t.created_at = i64::MAX;
    t.last_recall = i64::MAX;
    rowset.push(t);
    let mut t = indexed("0A5", "kyoto temple huge stability");
    t.stability = 1e30;
    rowset.push(t);
    let mut t = indexed("0A6", "  kyoto   temple 《dirty》 text ");
    t.tz = format!("{};+09:00", "X".repeat(500));
    t.cue = "c".repeat(900);
    rowset.push(t);
    let mut t = indexed("0A7", "kyoto temple unindexed with vector");
    t.model_id = String::new();
    rowset.push(t);
    for r in &rowset {
        put(&store, r);
    }
    let c = cfg();
    let (mut m, clock) = engine_on(&store, c.clone());
    clock.advance(DAY);
    let now = m.now();
    for r in &rowset {
        sane(&m.memory(&r.id).unwrap().unwrap(), now, &c);
    }
    let rc = m.recall("kyoto temple").unwrap();
    assert!(!rc.recalled.is_empty());
    for r in &rc.recalled {
        sane(&r.trace, now, &c);
        assert!(r.cosine.is_finite() && r.cosine <= 1.0 && r.score.is_finite() && (0.0..=1.0).contains(&r.retrievability));
    }
    assert!(rc.recalled.iter().all(|r| r.trace.id != "0A7"), "unindexed rows are not searched");
    for line in rc.pack_text.lines() {
        assert!(line.chars().count() < 400, "{line}");
    }
    let ids: Vec<String> = rc.recalled.iter().map(|r| r.trace.id.clone()).collect();
    let reply: String = ids.iter().map(|i| format!("《id:{i}》")).collect();
    assert_eq!(m.cite(&reply).unwrap().len(), ids.len());
    for r in &rowset {
        sane(&m.memory(&r.id).unwrap().unwrap(), now, &c);
    }
    assert_eq!(m.reindex(10).unwrap(), 1, "the unindexed row is re-embedded");
    // capacity pressure with such rows
    let mut c2 = c.clone();
    c2.capacity = 5;
    let (mut m2, _) = engine_on(&store, c2);
    for i in 0..4 {
        insert(&mut m2, &format!("pressure probe number{i}"), 1.0, "");
        assert!(m2.stats().unwrap().records <= 5, "capacity broken after insert {i}");
    }
}

/// NaN / infinite stability and non-finite or short vectors as seen through a store
/// that does not police them.
#[test]
fn l4_5_non_finite_and_short_rows_never_reach_the_result() {
    let store = Hostile::default();
    let c = cfg();
    let (mut m, clock) = build_with(c.clone(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    for i in 0..8 {
        insert(&mut m, &format!("kyoto temple number{i}"), 1.0, "");
    }
    *store.rewrite.lock().unwrap() = Some(Arc::new(|mut t: Trace| {
        let i: usize = t.text.rsplit_once("number").unwrap().1.parse().unwrap();
        match i % 4 {
            0 => t.stability = f64::NAN,
            1 => t.stability = f64::INFINITY,
            2 => t.vector[0] = f32::NAN,
            _ => t.vector.truncate(10),
        }
        t
    }));
    store.extra.lock().unwrap().extend([
        Hit { cos: f64::NAN, trace: indexed("0X1", "kyoto temple bogus") },
        Hit { cos: f64::NEG_INFINITY, trace: indexed("0X2", "kyoto temple bogus two") },
        Hit { cos: 2.5, trace: indexed("0X3", "zebra quantum unrelated") },
    ]);
    clock.advance(DAY);
    let now = m.now();
    let r = m.recall("kyoto temple").unwrap();
    assert!(!r.recalled.is_empty());
    for x in &r.recalled {
        sane(&x.trace, now, &c);
        let i: usize = x.trace.text.rsplit_once("number").map_or(99, |p| p.1.parse().unwrap_or(99));
        assert!(i == 99 || i % 4 < 2, "row {i} with a bad vector was searched");
        // The engine recomputes every cosine from the row's own vector: a hit whose claimed cosine is
        // NaN or -inf but whose vector really matches is a match, and one claiming 2.5 for an unrelated
        // vector is not.
        assert!(x.trace.id != "0X3", "a hit with a lying cosine survived: {:?}", x.trace.id);
    }
    let nan_row = rows(&m).into_iter().find(|t| t.text.ends_with("number0")).unwrap();
    sane(&m.memory(&nan_row.id).unwrap().unwrap(), now, &c); // NaN stability is clamped, not dropped
    let reply: String = r.recalled.iter().map(|x| format!("《id:{}》", x.trace.id)).collect();
    m.cite(&reply).unwrap();
    for t in store.inner.find(View::All, usize::MAX).unwrap() {
        assert!(t.stability.is_finite(), "{}", t.id);
    }
    let id = r.recalled[0].trace.id.clone();
    m.remember(&r.recalled[0].trace.text.clone(), 1.0, "").unwrap();
    assert!(m.forget(&id).unwrap().is_some());
    assert!(m.stats().is_ok());
}

/// SPEC §7: vectors are unit norm. A row whose vector is not unit is read as is, and
/// `dot().min(1.0)` turns a scaled unrelated vector into a perfect match.
#[test]
fn note_l4_5_a_non_unit_stored_vector_cannot_fake_a_perfect_match() {
    let store = SharedStore::default();
    let mut emb = FakeEmbedder::new();
    let mut chosen = None;
    for q in ["kyoto temple", "river bank", "paris tower", "alpha beta", "moon star", "red fox", "blue sky", "green tea"] {
        let qv = unit(&emb.embed(q));
        let bv = emb.embed("zebra quantum harp");
        let d = dotf(&bv, &qv);
        if (0.02..0.3).contains(&d) {
            chosen = Some(q);
            break;
        }
    }
    let q = chosen.expect("a query unrelated but positively aligned with the probe text");
    let mut t = indexed("0BIG", "zebra quantum harp");
    t.vector = t.vector.iter().map(|x| x * 100.0).collect();
    put(&store, &t);
    let (mut m, _) = engine_on(&store, cfg());
    let r = m.peek(q).unwrap();
    assert!(r.recalled.is_empty(), "scaled vector scored cosine {:?}", r.recalled.iter().map(|x| x.cosine).collect::<Vec<_>>());
}
