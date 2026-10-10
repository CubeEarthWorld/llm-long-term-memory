//! remember / recall / cite / forget / capacity (SPEC §4).
#![cfg(feature = "sqlite")]

mod support;

use engram::{Config, Embedder, MemoryStore, Remembered, Result, SqliteStore, Store, Trace, Write};
use support::{build, build_with, cfg, insert, temp_dir, BrokenEmbedder, FakeEmbedder, DAY};

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
    assert_eq!(s.memories().len(), 1);
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
    let (mut s, _) = build_with(cfg(), Box::new(Glitching(FakeEmbedder::new(), false)), Box::new(MemoryStore::default()));
    s.remember("a trace the provider mis-embedded", 1.0, "").unwrap();
    assert_eq!(s.stats().unindexed, 1); // never reaches the index
    s.remember("a trace about kyoto embedded normally", 1.0, "").unwrap();
    assert!(!s.recall("kyoto").unwrap().recalled.is_empty());
    s.initialize().unwrap(); // reindex re-embeds it
    assert_eq!(s.stats().unindexed, 0);
}

#[test]
fn row_with_no_text_is_skipped_on_load() {
    let mut store = MemoryStore::default();
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
    store.write(&[Write::Put(&empty)]).unwrap();
    let (s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store));
    assert!(s.memories().is_empty());
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
    assert!(s.memory(&old.id).unwrap().consolidated); // a candidate, not rewritten on the way in
    let ids: Vec<String> = s.clusters(None)[0].iter().map(|t| t.id.clone()).collect();
    assert_eq!(ids, [r.id, old.id]);
    assert_eq!(s.memories().len(), 2);
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
    let path = temp_dir("offline").join("shared.db");
    let (mut broken, _) = build_with(cfg(), Box::new(BrokenEmbedder), Box::new(SqliteStore::open(&path).unwrap()));
    insert(&mut broken, "kept while offline", 1.0, "");
    assert!(broken.recall("kept while offline").unwrap().recalled.is_empty());
    drop(broken);
    let healed_embedder = FakeEmbedder::with(64, "fake/broken");
    let (mut healed, _) = build_with(cfg(), Box::new(healed_embedder), Box::new(SqliteStore::open(&path).unwrap()));
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
    assert!(s.memory(&strong).is_some());
}

#[test]
fn flood_evicts_its_own_members() {
    let (mut s, _) = build(Config { capacity: 20, grace_period: 1e9, ..cfg() });
    let keep = insert(&mut s, "precious", 10.0, "");
    for i in 0..40 {
        s.remember(&format!("junk {i} x{i}"), 1.0, "").unwrap();
    }
    assert!(s.memories().len() == 20 && s.memory(&keep).is_some());
}

#[test]
fn flood_spares_the_old_stable_fact() {
    let (mut s, clock) = build(Config { capacity: 20, grace_period: DAY, ..cfg() });
    let keep = insert(&mut s, "precious fact", 10.0, "");
    clock.advance(2.0 * DAY);
    for i in 0..60 {
        s.remember(&format!("junk {i} x{i}"), 1.0, "").unwrap();
    }
    assert!(s.memories().len() == 20 && s.memory(&keep).is_some());
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
    let after = s.memory(&cat.id).unwrap();
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
    let half = s.memory(&id).unwrap().stability;
    assert_eq!(s.cite(&format!("It is on 1990-05-03 《id:{id}》")).unwrap(), std::slice::from_ref(&id));
    assert!(s.memory(&id).unwrap().stability > half);
    assert!(s.cite(&format!("again 《id:{id}》")).unwrap().is_empty());
}

#[test]
fn forget_is_by_id_only() {
    let (mut s, _) = build(cfg());
    let id = insert(&mut s, "to be forgotten", 1.0, "");
    assert_eq!(s.forget(&id).unwrap().unwrap().text, "to be forgotten");
    assert!(s.forget(&id).unwrap().is_none());
    assert!(s.memories().is_empty());
}

#[test]
fn initialize_clamps_and_persists() {
    let path = temp_dir("clamp").join("persist.db");
    let (mut s, clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(SqliteStore::open(&path).unwrap()));
    let id = insert(&mut s, "future", 1.0, "");
    let future = Trace {
        last_recall: clock.get() as i64 + 1_000_000,
        created_at: clock.get() as i64 + 1_000_000,
        stability: 1e12,
        ..s.memory(&id).unwrap().clone()
    };
    drop(s);
    SqliteStore::open(&path).unwrap().write(&[Write::Put(&future)]).unwrap();
    let (s2, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(SqliteStore::open(&path).unwrap()));
    let loaded = &s2.memories()[0];
    assert!(loaded.last_recall as f64 <= clock.get() && loaded.stability == s2.config().max_stability);
    let stored = SqliteStore::open(&path).unwrap().load_all().unwrap();
    assert_eq!(stored[0].stability, s2.config().max_stability); // the clamp is written back
}

/// A store whose commit fails on demand: the engine must not change RAM either.
struct Failing(MemoryStore, std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Store for Failing {
    fn load_all(&self) -> Result<Vec<Trace>> {
        self.0.load_all()
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
fn a_failed_commit_leaves_ram_unchanged() {
    let fail = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(Failing(MemoryStore::default(), fail.clone())));
    let id = insert(&mut s, "user has a cat named tama", 1.0, "");
    let before = s.memories().to_vec();
    fail.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(s.remember("user has a dog named pochi", 1.0, "").is_err());
    assert!(s.recall("cat tama").is_err());
    assert!(s.forget(&id).is_err());
    assert_eq!(s.memories(), before.as_slice());
}

#[test]
fn incremental_counts_and_indexes_stay_exact() {
    // After every step of the seeded workload (inserts, rehearsals, evictions, dreams),
    // the O(1) counters and the id / text indexes agree with a full recount.
    support::workload::run(|m| {
        let all = m.memories();
        let labile = all.iter().filter(|t| !t.consolidated).count();
        assert_eq!((m.stats().records, m.stats().labile, m.stats().unindexed), (all.len(), labile, 0));
        for t in all {
            assert_eq!(m.memory(&t.id).map(|x| &x.text), Some(&t.text));
        }
    });
}

#[test]
fn load_repairs_rows_written_outside_the_engine() {
    // SPEC §2: every row is validated on load. Text, cue and tz are held to the write-time
    // hygiene; a vector that is not finite and unit-norm is re-embedded from the text.
    let path = temp_dir("repair").join("repair.db");
    let mut st = SqliteStore::open(&path).unwrap();
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
    let nan = Trace { id: "01B".into(), text: "user has a cat".into(), cue: String::new(), vector: vec![f32::NAN; 64], ..base.clone() };
    let odd = Trace { id: "01C".into(), text: "user plays piano".into(), cue: String::new(), vector: vec![0.0; 64], ..base.clone() };
    st.write(&[Write::Put(&base), Write::Put(&nan), Write::Put(&odd)]).unwrap();
    st.connection().execute("UPDATE memory SET vector=X'010203' WHERE id='01C'", []).unwrap(); // not whole float32s
    drop(st);
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(SqliteStore::open(&path).unwrap()));
    assert_eq!(s.memories().len(), 3); // nothing lost: the text is canonical
    let a = s.memory("01A").unwrap();
    assert!(!a.text.contains('《') && !a.text.contains('\n') && a.text.chars().count() <= 170);
    assert!(!a.cue.contains('《') && !a.tz.contains('\n'));
    for t in s.memories() {
        let norm: f64 = t.vector.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        assert!((norm - 1.0).abs() < 1e-3, "{} was not re-embedded", t.id);
    }
    assert!(s.recall("kyoto").unwrap().pack_text.matches("《id:").count() >= 1); // no panic on the odd tz
    let stored = SqliteStore::open(&path).unwrap().load_all().unwrap();
    assert!(stored.iter().all(|t| !t.text.contains('《'))); // the repair is written back
}

/// A store whose snapshot fails.
struct NoBackup(MemoryStore);

impl Store for NoBackup {
    fn load_all(&self) -> Result<Vec<Trace>> {
        self.0.load_all()
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.0.write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        self.0.clear()
    }
    fn backup(&mut self) -> Result<Option<String>> {
        Err("disk full".into())
    }
}

#[test]
fn a_failed_snapshot_stops_the_dream() {
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(NoBackup(MemoryStore::default())));
    for t in ["trip kyoto plan schedule note", "trip kyoto hotel schedule note"] {
        insert(&mut s, t, 1.0, "");
    }
    let before = s.memories().to_vec();
    assert!(s.dream(&mut support::merge_to_gist, None).is_err());
    assert_eq!(s.memories(), before.as_slice()); // nothing rewritten without a snapshot
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
    let path = temp_dir("short").join("short.db");
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(SqliteStore::open(&path).unwrap()));
    for t in ["user has a cat", "user plays piano", "user lives in kyoto"] {
        insert(&mut s, t, 1.0, "");
    }
    drop(s);
    // A model switch re-embeds in batches; a short answer must leave the batch stale.
    let (s2, _) = build_with(cfg(), Box::new(ShortEmbedder(FakeEmbedder::new())), Box::new(SqliteStore::open(&path).unwrap()));
    assert_eq!(s2.stats().unindexed, 3);
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
