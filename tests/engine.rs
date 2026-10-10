//! remember / recall / cite / forget / capacity (SPEC §4).

mod support;

use support::{mem, rows};
use engram::{Config, DreamAction, DreamDecision, DreamRequest, Embedder, Hit, Remembered, Result, Storage, Store, Trace, View, Write};
use support::{build, build_with, cfg, insert, BrokenEmbedder, FakeEmbedder, SharedStore, DAY};

fn inserted(r: Remembered) -> (Trace, f64, Option<Trace>) {
    match r {
        Remembered::Inserted { trace, cosine, evicted } => (trace, cosine, evicted),
        other => panic!("expected an insert, got {other:?}"),
    }
}

#[test]
fn insert_rehearse_reject() {
    let (mut s, clock) = build(cfg());
    let (a, _, _) = inserted(s.remember("user lives in kyoto", 1.0, "").unwrap());
    assert!(a.consolidated && a.stability == DAY);
    clock.advance(2.0 * DAY);
    let Remembered::Reinforced(b) = s.remember("  user  lives in kyoto ", 1.0, "").unwrap() else { panic!() };
    assert!(b.id == a.id && b.stability > DAY);
    assert_eq!(s.remember("   ", 1.0, "").unwrap(), Remembered::Rejected);
    assert_eq!(rows(&s).len(), 1);
}

/// Returns a half-length vector once, then behaves normally.
struct Glitching(FakeEmbedder, bool);

impl Embedder for Glitching {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
    fn dimension(&self) -> usize {
        self.0.dimension
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if !self.1 {
            self.1 = true;
            return Ok(texts.iter().map(|_| vec![0.0; 32]).collect());
        }
        self.0.embed_documents(texts)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.0.embed_queries(texts)
    }
}

#[test]
fn wrong_dimension_vector_is_treated_as_stale() {
    let (mut s, _) = build_with(cfg(), Box::new(Glitching(FakeEmbedder::new(), false)), Box::new(Store::in_memory().unwrap()));
    s.remember("a trace the provider mis-embedded", 1.0, "").unwrap();
    let stale = |s: &engram::Engram| s.store().count(View::Stale { model_id: "fake/token-overlap" }).unwrap();
    assert_eq!(stale(&s), 1); // never reaches the index
    s.remember("a trace about kyoto embedded normally", 1.0, "").unwrap();
    assert!(!s.recall("kyoto").unwrap().recalled.is_empty());
    assert_eq!(s.reindex(10).unwrap(), 1); // re-embedded from its text
    assert_eq!(stale(&s), 0);
}

#[test]
fn a_row_with_no_text_is_never_served() {
    let mut store = Store::in_memory().unwrap();
    let empty = Trace {
        id: "01EMPTY".into(),
        text: String::new(),
        created_at: 1_700_000_000,
        tz: "UTC;+00:00".into(),
        last_recall: 1_700_000_000,
        stability: DAY,
        consolidated: true,
        cue: String::new(),
        model_id: "fake/token-overlap".into(),
        vector: vec![0.0; 64],
    };
    store.write(&[Write::Insert(&empty)]).unwrap();
    let (s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store));
    assert!(mem(&s, "01EMPTY").is_none());
}

#[test]
fn text_hygiene() {
    let (mut s, _) = build(Config { text_max: 20, ..cfg() });
    let (t, _, _) = inserted(s.remember(&format!("a《id:x》b\n\n c {}", "x".repeat(40)), 1.0, "").unwrap());
    assert!(!t.text.contains('《') && !t.text.contains('\n') && t.text.chars().count() <= 20);
}

#[test]
fn write_that_reactivates_the_past_is_born_labile() {
    let (mut s, _) = build(cfg());
    let (old, _, _) = inserted(s.remember("cat dog bird fish", 1.0, "").unwrap());
    assert!(old.consolidated); // nothing to reactivate
    let (r, cos, _) = inserted(s.remember("cat dog bird fish!", 1.0, "").unwrap());
    assert!(cos > 0.9 && !r.consolidated);
    assert!(mem(&s, &old.id).unwrap().consolidated); // a candidate, not rewritten on the way in
    let ids: Vec<String> = s.clusters(None).unwrap()[0].iter().map(|t| t.id.clone()).collect();
    assert_eq!(ids, [r.id, old.id]);
    assert_eq!(rows(&s).len(), 2);
}

#[test]
fn salience_scales_and_clamps() {
    let (mut s, _) = build(cfg());
    assert_eq!(inserted(s.remember("alpha", 3.0, "").unwrap()).0.stability, 3.0 * DAY);
    assert_eq!(inserted(s.remember("beta", 1e9, "").unwrap()).0.stability, Config::default().max_stability / 365.0);
    let zero = inserted(s.remember("gamma", 0.0, "").unwrap()).0;
    assert!(zero.stability == 1.0 && s.retrievability(&zero, zero.last_recall as f64 + 100.0) < 1e-20);
    assert_eq!(inserted(s.remember("delta", f64::NAN, "").unwrap()).0.stability, DAY);
}

#[test]
fn daily_write_limit() {
    let (mut s, clock) = build(Config { writes_per_day: 2, ..cfg() });
    insert(&mut s, "a1 b1", 1.0, "");
    insert(&mut s, "a2 b2", 1.0, "");
    assert_eq!(s.remember("a3 b3", 1.0, "").unwrap(), Remembered::RateLimited { text: "a3 b3".into() });
    clock.advance(1.01 * DAY);
    insert(&mut s, "a3 b3", 1.0, "");
}

#[test]
fn embedder_offline_keeps_text() {
    let db = SharedStore::default();
    let (mut broken, _) = build_with(cfg(), Box::new(BrokenEmbedder), Box::new(db.clone()));
    insert(&mut broken, "kept while offline", 1.0, "");
    assert!(broken.recall("kept while offline").unwrap().recalled.is_empty());
    drop(broken);
    let healed_embedder = FakeEmbedder::with(64, "fake/broken");
    let (mut healed, _) = build_with(cfg(), Box::new(healed_embedder), Box::new(db));
    assert_eq!(healed.reindex(10).unwrap(), 1); // indexed once the model answers
    assert_eq!(healed.recall("kept while offline").unwrap().recalled[0].trace.text, "kept while offline");
}

#[test]
fn eviction_prefers_weak_old_traces() {
    let (mut s, clock) = build(Config { capacity: 3, grace_period: DAY, ..cfg() });
    insert(&mut s, "old weak one", 1.0, "");
    let strong = insert(&mut s, "old strong two", 10.0, "");
    clock.advance(2.0 * DAY);
    insert(&mut s, "fresh three", 1.0, "");
    let (_, _, evicted) = inserted(s.remember("fresh four", 1.0, "").unwrap());
    assert_eq!(evicted.unwrap().text, "old weak one");
    assert!(mem(&s, &strong).is_some());
}

#[test]
fn flood_evicts_its_own_members() {
    let (mut s, _) = build(Config { capacity: 20, grace_period: 1e9, ..cfg() });
    let keep = insert(&mut s, "precious", 10.0, "");
    for i in 0..40 {
        s.remember(&format!("junk {i} x{i}"), 1.0, "").unwrap();
    }
    assert!(rows(&s).len() == 20 && mem(&s, &keep).is_some());
}

#[test]
fn flood_spares_the_old_stable_fact() {
    let (mut s, clock) = build(Config { capacity: 20, grace_period: DAY, ..cfg() });
    let keep = insert(&mut s, "precious fact", 10.0, "");
    clock.advance(2.0 * DAY);
    for i in 0..60 {
        s.remember(&format!("junk {i} x{i}"), 1.0, "").unwrap();
    }
    assert!(rows(&s).len() == 20 && mem(&s, &keep).is_some());
}

#[test]
fn recall_packs_and_strengthens() {
    let (mut s, clock) = build(cfg());
    let cat = inserted(s.remember("user has a cat named tama", 1.0, "").unwrap()).0;
    insert(&mut s, "user works at a bank", 1.0, "");
    clock.advance(DAY);
    let r = s.recall("tell me about the cat tama").unwrap();
    assert_eq!(r.recalled[0].trace.id, cat.id);
    let first = format!("[{} Asia/Tokyo;+09:00] user has a cat named tama　《id:{}》\n", cat.created_at, cat.id);
    assert!(r.pack_text.starts_with(&first));
    let after = mem(&s, &cat.id).unwrap();
    assert!(after.stability > cat.stability && after.last_recall == clock.get() as i64);
}

#[test]
fn recall_nothing_when_nothing_matches() {
    let (mut s, _) = build(Config { min_score: 0.25, ..cfg() });
    insert(&mut s, "alpha beta gamma", 1.0, "");
    assert!(s.recall("zzz yyy").unwrap().recalled.is_empty());
}

#[test]
fn relative_cut_and_multi_cue() {
    let (mut s, _) = build(cfg());
    for t in ["kyoto trip hotel booking", "kyoto weather", "random unrelated note"] {
        insert(&mut s, t, 1.0, "");
    }
    let texts: Vec<String> = s.recall("kyoto trip hotel booking").unwrap().recalled.into_iter().map(|c| c.trace.text).collect();
    assert!(!texts.contains(&"random unrelated note".to_owned()));
    insert(&mut s, "user likes matcha ice cream", 1.0, "");
    insert(&mut s, "user plays tennis on sunday", 1.0, "");
    let texts: Vec<String> =
        s.recall("matcha ice cream is great.\nalso tennis on sunday?").unwrap().recalled.into_iter().map(|c| c.trace.text).collect();
    assert!(texts.contains(&"user likes matcha ice cream".to_owned()) && texts.contains(&"user plays tennis on sunday".to_owned()));
}

#[test]
fn budget_after_first_line() {
    let (mut s, _) = build(Config { budget_chars: 120, ..cfg() });
    for i in 0..4 {
        insert(&mut s, &format!("kyoto note number {i} about kyoto"), 1.0, "");
    }
    assert_eq!(s.recall("kyoto note").unwrap().recalled.len(), 1);
}

#[test]
fn cite_completes_strengthening() {
    let (mut s, clock) = build(cfg());
    let id = insert(&mut s, "user birthday is 1990-05-03", 1.0, "");
    clock.advance(3.0 * DAY);
    s.recall("when is the birthday?").unwrap();
    let half = mem(&s, &id).unwrap().stability;
    assert_eq!(s.cite(&format!("It is on 1990-05-03 《id:{id}》")).unwrap(), std::slice::from_ref(&id));
    assert!(mem(&s, &id).unwrap().stability > half);
    assert!(s.cite(&format!("again 《id:{id}》")).unwrap().is_empty());
}

#[test]
fn forget_is_by_id_only() {
    let (mut s, _) = build(cfg());
    let id = insert(&mut s, "to be forgotten", 1.0, "");
    assert_eq!(s.forget(&id).unwrap().unwrap().text, "to be forgotten");
    assert!(s.forget(&id).unwrap().is_none());
    assert!(rows(&s).is_empty());
}

#[test]
fn rows_read_are_clamped() {
    let db = SharedStore::default();
    let (mut s, clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(db.clone()));
    let id = insert(&mut s, "future", 1.0, "");
    let future = Trace {
        last_recall: clock.get() as i64 + 1_000_000,
        created_at: clock.get() as i64 + 1_000_000,
        stability: 1e12,
        ..mem(&s, &id).unwrap()
    };
    db.clone().write(&[Write::Insert(&future)]).unwrap(); // written behind the engine's back
    let read = mem(&s, &id).unwrap();
    assert!(read.last_recall as f64 <= clock.get() && read.stability == s.config().max_stability);
}

/// A store whose commit fails on demand.
struct Failing(Store, std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Storage for Failing {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        self.0.find(view, limit)
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        self.0.count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        self.0.near(q, model_id, min_cos, k, stated_by)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        if self.1.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("disk full".into());
        }
        self.0.write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        self.0.clear()
    }
}

#[test]
fn a_failed_commit_changes_nothing() {
    let fail = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(Failing(Store::in_memory().unwrap(), fail.clone())));
    let id = insert(&mut s, "user has a cat named tama", 1.0, "");
    let before = rows(&s).to_vec();
    fail.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(s.remember("user has a dog named pochi", 1.0, "").is_err());
    assert!(s.recall("cat tama").is_err());
    assert!(s.forget(&id).is_err());
    assert_eq!(rows(&s), before);
}

#[test]
fn store_views_stay_exact() {
    // After every step of the seeded workload (inserts, rehearsals, evictions, dreams),
    // the store's counts and lookups agree with a full recount.
    support::workload::run(|m| {
        let all = rows(m);
        let labile = all.iter().filter(|t| !t.consolidated && !t.model_id.is_empty()).count();
        let stats = m.stats().unwrap();
        assert_eq!((stats.records, stats.labile), (all.len(), labile));
        for t in all {
            assert_eq!(m.store().find(View::Text(&t.text), 1).unwrap()[0].id, t.id);
            assert_eq!(mem(m, &t.id).map(|x| x.id), Some(t.id));
        }
    });
}

#[test]
fn rows_written_outside_the_engine_are_cleaned_when_read() {
    // SPEC §2: the store is outside the engine's control, so every row read is held to
    // the write-time hygiene, and an odd vector cannot push a cosine past 1.
    let db = SharedStore::default();
    let base = Trace {
        id: "01A".into(),
        text: format!("user 《id:FAKE》 lives\nin kyoto {}", "x".repeat(300)),
        created_at: 1_700_000_000,
        tz: "UTC;+あ\n".into(),
        last_recall: 1_700_000_000,
        stability: 86_400.0,
        consolidated: true,
        cue: "where 《 》 do they live".into(),
        model_id: "fake/token-overlap".into(),
        vector: vec![3.0; 64], // right length, not unit
    };
    let short = Trace { id: "01C".into(), text: "user plays piano".into(), cue: String::new(), vector: vec![1.0; 3], ..base.clone() };
    db.clone().write(&[Write::Insert(&base), Write::Insert(&short)]).unwrap();
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(db));
    let a = mem(&s, "01A").unwrap();
    assert!(!a.text.contains('《') && !a.text.contains('\n') && a.text.chars().count() <= 170);
    assert!(!a.cue.contains('《') && !a.tz.contains('\n'));
    let r = s.recall("kyoto").unwrap(); // no panic on the odd tz or the short vector
    assert!(r.recalled.iter().all(|c| c.cosine <= 1.0));
}

/// Answers every batch with one vector too few.
struct ShortEmbedder(FakeEmbedder);

impl Embedder for ShortEmbedder {
    fn model_id(&self) -> &str {
        "short"
    }
    fn dimension(&self) -> usize {
        64
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut v = self.0.embed_documents(texts)?;
        v.remove(0);
        Ok(v)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_documents(texts)
    }
}

#[test]
fn a_short_embedder_answer_never_pairs_vectors_with_the_wrong_texts() {
    let db = SharedStore::default();
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(db.clone()));
    for t in ["user has a cat", "user plays piano", "user lives in kyoto"] {
        insert(&mut s, t, 1.0, "");
    }
    drop(s);
    // A model switch re-embeds in batches; a short answer must leave the batch stale.
    let (mut s2, _) = build_with(cfg(), Box::new(ShortEmbedder(FakeEmbedder::new())), Box::new(db));
    assert_eq!(s2.reindex(10).unwrap(), 0);
    assert_eq!(s2.store().count(View::Stale { model_id: "short" }).unwrap(), 3);
}

#[test]
fn writes_stamped_by_a_clock_that_ran_ahead_do_not_rate_limit_later() {
    let (mut s, clock) = build(Config { writes_per_day: 2, ..cfg() });
    clock.advance(500.0 * 365.0 * DAY); // a virtual timeline (seed replay) runs far ahead
    insert(&mut s, "a1 b1", 1.0, "");
    insert(&mut s, "a2 b2", 1.0, "");
    clock.set(support::START); // back to the wall clock
    insert(&mut s, "a3 b3", 1.0, "");
}

#[test]
fn parameters_that_break_the_invariants_are_rejected() {
    for bad in [
        Config { max_stability: f64::INFINITY, ..cfg() },
        Config { max_stability: 0.5, initial_stability: 0.25, ..cfg() },
        Config { max_cues: 0, ..cfg() },
        Config { min_score: f64::NAN, ..cfg() },
        Config { gist_min_cosine: -2.0, ..cfg() },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
}

#[test]
fn a_trace_stamped_in_the_future_keeps_its_grace() {
    // A clock that ran ahead and came back leaves a trace "from the future": it has not
    // passed its grace period, so it is protected like any young trace (SPEC §4).
    let (mut s, clock) = build(Config { capacity: 30, grace_period: DAY, ..cfg() });
    for i in 0..29 {
        insert(&mut s, &format!("settled fact {i} q{i}"), 10.0, "");
    }
    clock.advance(10.0 * DAY);
    let ahead = insert(&mut s, "written while the clock ran ahead", 1.0, "");
    clock.set(clock.get() - 5.0 * DAY);
    let (_, _, evicted) = inserted(s.remember("written after the clock came back", 1.0, "").unwrap());
    assert!(evicted.is_some_and(|e| e.id != ahead) && mem(&s, &ahead).is_some());
}

#[test]
fn rows_stored_above_the_stability_cap_can_still_be_evicted() {
    // Written under a higher cap (or by another writer): their band lies above the one
    // the configured cap maps to, and must still be searched for victims.
    let db = SharedStore::default();
    let long_ago = (support::START - 200.0 * 365.0 * DAY) as i64;
    let old: Vec<Trace> = (0..3)
        .map(|i| Trace {
            id: format!("00{i}"),
            text: format!("ancient fact {i}"),
            created_at: long_ago,
            tz: "UTC;+00:00".into(),
            last_recall: long_ago,
            stability: 1e12,
            consolidated: true,
            cue: String::new(),
            model_id: String::new(),
            vector: Vec::new(),
        })
        .collect();
    db.clone().write(&old.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    let (mut s, _) = build_with(Config { capacity: 3, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(db));
    let (t, _, evicted) = inserted(s.remember("a new fact", 1.0, "").unwrap());
    assert!(evicted.is_some_and(|e| e.id != t.id), "the new trace must not evict itself");
    assert!(mem(&s, &t.id).is_some());
}

#[test]
fn a_write_whose_cue_cannot_be_embedded_stays_labile() {
    // Falling back to the trace's own vector cannot show that the cue reactivates nothing.
    let (e, _) = support::CountingEmbedder::new(true, None);
    let (mut s, _) = build_with(cfg(), Box::new(e), Box::new(Store::in_memory().unwrap()));
    let id = insert(&mut s, "the user moved to osaka", 1.0, "where the user lives");
    assert!(!mem(&s, &id).unwrap().consolidated);
    let uncued = insert(&mut s, "zebra unrelated topic", 1.0, "");
    assert!(mem(&s, &uncued).unwrap().consolidated);
}

#[test]
fn recall_widens_past_rows_it_cannot_use() {
    // The store's first k answers are unusable rows (no text): the search must widen,
    // not conclude that nothing else is there.
    let db = SharedStore::default();
    let mut fe = FakeEmbedder::new();
    let junk: Vec<Trace> = (0..4)
        .map(|i| Trace {
            id: format!("0{i}"),
            text: String::new(),
            created_at: support::START as i64,
            tz: "UTC;+00:00".into(),
            last_recall: support::START as i64,
            stability: DAY,
            consolidated: true,
            cue: String::new(),
            model_id: "fake/token-overlap".into(),
            vector: fe.embed("kyoto"),
        })
        .collect();
    db.clone().write(&junk.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    let (mut s, _) = build_with(Config { inject_n: 1, ..cfg() }, Box::new(fe), Box::new(db));
    let id = insert(&mut s, "kyoto temple visit", 1.0, "");
    assert_eq!(s.recall("kyoto").unwrap().recalled.iter().map(|c| c.trace.id.clone()).collect::<Vec<_>>(), vec![id]);
}

#[test]
fn a_store_failure_mid_dream_keeps_the_undo_records_of_applied_verdicts() {
    let fail = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(Failing(Store::in_memory().unwrap(), fail.clone())));
    for t in ["alpha gamma delta one", "alpha gamma delta two", "beta zeta eta one", "beta zeta eta two"] {
        insert(&mut s, t, 1.0, "");
    }
    let mut calls = 0;
    let flip = fail.clone();
    let mut merge = |r: &DreamRequest| -> Result<DreamDecision> {
        calls += 1;
        if calls == 2 {
            flip.store(true, std::sync::atomic::Ordering::SeqCst); // the second verdict's write fails
        }
        support::merge_to_gist(r)
    };
    let reports = s.dream(&mut merge, None).unwrap();
    assert_eq!(reports.iter().map(|r| r.action).collect::<Vec<_>>(), [DreamAction::Replace, DreamAction::Error]);
    fail.store(false, std::sync::atomic::Ordering::SeqCst);
    let absorbed: Vec<String> = reports[0].absorbed.iter().map(|t| t.id.clone()).collect();
    s.undo(&reports[0]).unwrap(); // the applied verdict is still reversible
    assert!(absorbed.iter().all(|id| mem(&s, id).is_some()));
}

#[test]
fn an_unusable_row_counts_against_capacity_and_dies_first() {
    let db = SharedStore::default();
    let blank = Trace {
        id: "000".into(),
        text: "  ".into(),
        created_at: 0,
        tz: "UTC;+00:00".into(),
        last_recall: 0,
        stability: 1e9,
        consolidated: true,
        cue: String::new(),
        model_id: String::new(),
        vector: Vec::new(),
    };
    db.clone().write(&[Write::Insert(&blank)]).unwrap();
    let (mut s, _) = build_with(Config { capacity: 2, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(db));
    let a = insert(&mut s, "first real fact", 1.0, "");
    let (b, _, evicted) = inserted(s.remember("second real fact", 1.0, "").unwrap());
    assert_eq!(evicted.map(|t| t.id), Some("000".into()));
    assert!(mem(&s, &a).is_some() && mem(&s, &b.id).is_some() && s.stats().unwrap().records == 2);
}

/// Answers every search with its rows at cosine NaN.
struct NanCos(Store);

impl Storage for NanCos {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        self.0.find(view, limit)
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        self.0.count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, _: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        Ok(self.0.near(q, model_id, -1.0, k, stated_by)?.into_iter().map(|h| Hit { cos: f64::NAN, ..h }).collect())
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.0.write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        self.0.clear()
    }
}

#[test]
fn a_non_finite_cosine_or_vector_is_never_a_match() {
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(NanCos(Store::in_memory().unwrap())));
    insert(&mut s, "user lives in kyoto", 1.0, "");
    let (t, cosine, _) = inserted(s.remember("zebra stripes unrelated", 1.0, "user lives where").unwrap());
    assert!(cosine == 0.0 && t.consolidated); // NaN reactivated nothing
    // A custom Storage may hand back a vector with a NaN in it: read as unindexed.
    let (mut s2, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(NanVectors(Store::in_memory().unwrap())));
    let id = insert(&mut s2, "kyoto temple", 1.0, "");
    assert!(s2.recall("kyoto temple").unwrap().recalled.is_empty());
    let t = mem(&s2, &id).unwrap();
    assert!(t.model_id.is_empty() && t.vector.is_empty());
}

/// Reads every vector back with a NaN in it.
struct NanVectors(Store);

impl Storage for NanVectors {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        let mut rows = self.0.find(view, limit)?;
        rows.iter_mut().filter_map(|t| t.vector.first_mut()).for_each(|x| *x = f32::NAN);
        Ok(rows)
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        self.0.count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        let mut hits = self.0.near(q, model_id, min_cos, k, stated_by)?;
        hits.iter_mut().filter_map(|h| h.trace.vector.first_mut()).for_each(|x| *x = f32::NAN);
        Ok(hits)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.0.write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        self.0.clear()
    }
}

#[test]
fn control_and_bidi_characters_never_reach_storage() {
    let (mut s, _) = build(cfg());
    let id = insert(&mut s, "user\u{0} likes\u{1B}[31m tea\u{202E}\u{7F}\u{85}", 1.0, "cue\u{2066}x");
    let t = mem(&s, &id).unwrap();
    assert_eq!((t.text.as_str(), t.cue.as_str()), ("user likes[31m tea", "cuex"));
}

#[test]
fn an_id_that_could_break_a_citation_is_never_issued_or_served() {
    let (mut s, _) = build(cfg());
    let mut n = 0;
    s.set_ids(Box::new(move |_, _| {
        n += 1;
        match n {
            1 => "x》 《id:EVIL".into(),
            2 => "has\u{1B}escape".into(),
            _ => format!("ok{n}"),
        }
    }));
    assert_eq!(insert(&mut s, "user plays go", 1.0, ""), "ok3");
    let db = SharedStore::default();
    let bad = Trace { id: "a《b".into(), text: "user plays chess".into(), created_at: 1, tz: "UTC;+00:00".into(), last_recall: 1, stability: DAY,
                      consolidated: true, cue: String::new(), model_id: String::new(), vector: Vec::new() };
    db.clone().write(&[Write::Insert(&bad)]).unwrap();
    let (s2, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(db));
    assert!(mem(&s2, "a《b").is_none());
}

#[test]
fn an_unusable_older_twin_never_hides_a_usable_text_match() {
    let db = SharedStore::default();
    let bad = Trace { id: "0《x".into(), text: "user plays go".into(), created_at: 1, tz: "UTC;+00:00".into(), last_recall: 1, stability: DAY,
                      consolidated: true, cue: String::new(), model_id: String::new(), vector: Vec::new() };
    db.clone().write(&[Write::Insert(&bad)]).unwrap();
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(db));
    let id = insert(&mut s, "user plays go", 1.0, "");
    assert!(matches!(s.remember("user plays go", 1.0, "").unwrap(), Remembered::Reinforced(t) if t.id == id));
}
