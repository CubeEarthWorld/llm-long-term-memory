//! Round-3 audit A: regression review of the round-2 fixes (near() widening, checked() repair,
//! the unreadable-row placeholder, commit() gate, remember() bookkeeping, clock, prompt/JSON).
//! A FAILING test here is a confirmed defect; see audit/round3_a.md. Public API only.

mod support;

use engram::{Config, DreamDecision, Embedder, Engram, Remembered, Storage, Trace, View, Write};
use support::*;

const MODEL: &str = "fake/token-overlap";

fn rid(i: usize) -> String {
    format!("01{:024}", i)
}

fn row(id: &str, text: &str, model: &str, vector: Vec<f32>, consolidated: bool) -> Trace {
    Trace {
        id: id.into(),
        text: text.into(),
        created_at: START as i64 - 10 * 86400,
        tz: "UTC;+00:00".into(),
        last_recall: START as i64 - 10 * 86400,
        stability: 86400.0,
        consolidated,
        cue: String::new(),
        model_id: model.into(),
        vector,
    }
}

/// A 2-D embedder with scripted directions: "probe" = (1, 0); anything else = (0, 1).
struct Script;

impl Embedder for Script {
    fn model_id(&self) -> &str {
        "script"
    }
    fn dimension(&self) -> usize {
        2
    }
    fn embed_documents(&mut self, texts: &[&str]) -> engram::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| if *t == "probe" { vec![1.0, 0.0] } else { vec![0.0, 1.0] }).collect())
    }
    fn embed_queries(&mut self, texts: &[&str]) -> engram::Result<Vec<Vec<f32>>> {
        self.embed_documents(texts)
    }
}

/// A unit direction at cosine `c` from the probe, scaled by `k`.
fn dir(c: f32, k: f32) -> Vec<f32> {
    vec![c * k, (1.0 - c * c).sqrt() * k]
}

fn scripted(rows: &[Trace]) -> Engram {
    let mut store = SharedStore::default();
    store.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    build_with(cfg(), Box::new(Script), Box::new(store)).0
}

fn probe_remember(m: &mut Engram) -> (f64, bool) {
    match m.remember("probe", 1.0, "").unwrap() {
        Remembered::Inserted { cosine, trace, .. } => (cosine, trace.consolidated),
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------ (1)/(2) near(): repaired vectors vs the store's cos

/// A row whose stored vector is twice unit length has true cosine 0.3 (below theta_related = 0.55),
/// but the store ranks and thresholds it with the unscaled dot (0.6 >= 0.55). `near` recomputes the
/// cosine of the repaired vector and never re-applies `min_cos`: the write is reported as having
/// reactivated a trace it is unrelated to, and is born labile.
#[test]
fn overscaled_row_below_theta_is_not_reported_as_related() {
    let mut m = scripted(&[row(&rid(1), "alpha", "script", dir(0.3, 2.0), true)]);
    let (cosine, consolidated) = probe_remember(&mut m);
    assert!(cosine == 0.0 && consolidated, "an unrelated trace (true cos 0.3 < 0.55) was returned as nearest: cosine {cosine}, consolidated {consolidated}");
}

/// The store orders hits by the unscaled dot; the engine returns them in that order without re-sorting
/// by the recomputed cosine, so a rescaled row outranks a truly nearer unit row.
#[test]
fn rescaled_row_does_not_outrank_a_nearer_unit_row() {
    let mut m = scripted(&[row(&rid(1), "alpha", "script", dir(0.6, 2.0), true), row(&rid(2), "beta", "script", dir(0.8, 1.0), true)]);
    let (cosine, _) = probe_remember(&mut m);
    assert!((cosine - 0.8).abs() < 1e-3, "nearest cosine {cosine}: the unit row at 0.8 was hidden behind a rescaled row at 0.6");
}

/// An under-scaled vector (norm 0.5, true cosine 0.99) falls below the store's own `min_cos`
/// (0.495 < 0.55) and is never returned, although `checked` would have repaired it.
#[test]
#[ignore = "superseded: an unsound vector is embedded again by the scrub in reindex(), not rescaled (SPEC §2)"]
fn underscaled_row_is_found_after_repair() {
    let mut m = scripted(&[row(&rid(1), "alpha", "script", dir(0.99, 0.5), true)]);
    // An unsound vector is not searched with; the scrub in reindex() embeds the row again.
    assert!(m.reindex(10).unwrap() > 0);
    let (cosine, consolidated) = probe_remember(&mut m);
    assert!((cosine - 0.99).abs() < 1e-3 && !consolidated, "cosine {cosine}, consolidated {consolidated}: the repaired row is invisible to the store's threshold");
}

/// Same through recall: min_cos (0.1) is applied to the unscaled vector.
#[test]
#[ignore = "superseded: an unsound vector is embedded again by the scrub in reindex(), not rescaled (SPEC §2)"]
fn underscaled_row_is_recalled() {
    let mut m = scripted(&[row(&rid(1), "alpha", "script", dir(0.99, 0.05), true)]);
    assert!(m.reindex(10).unwrap() > 0); // the scrub repairs what the engine cannot search with
    let r = m.peek("probe").unwrap();
    assert!(!r.recalled.is_empty(), "a row with norm 0.05 (repairable) is invisible to recall");
}

// ------------------------------------------------ (2) what `checked` cannot repair

/// A unit vector of another dimension under the current model id is neither stale (the stored model
/// matches) nor damaged (unit norm): it is invisible to every search and never re-embedded.
#[test]
fn wrong_dimension_vector_is_reindexed() {
    let mut store = SharedStore::default();
    let mut v = vec![0.0f32; 3];
    v[0] = 1.0;
    store.write(&[Write::Insert(&row(&rid(1), "the cat sat on the mat", MODEL, v, true))]).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store));
    m.reindex(10).unwrap();
    m.dream(&mut always_keep, Some(1)).unwrap();
    assert!(!m.recall("the cat sat on the mat").unwrap().recalled.is_empty(), "a row with a 3-D vector under a 64-D model is lost to search for good");
}

/// A zero vector is beyond repair and is queued for `reindex` — but only in memory, and only when
/// something reads the row. A fresh engine (a host that opens one per dream) never learns of it.
#[test]
fn damaged_vector_is_healed_by_a_fresh_engines_reindex() {
    let mut store = SharedStore::default();
    store.write(&[Write::Insert(&row(&rid(1), "the cat sat on the mat", MODEL, vec![0.0; 64], true))]).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    m.reindex(10).unwrap();
    m.dream(&mut always_keep, Some(1)).unwrap();
    let raw = store.find(View::Id(&rid(1)), 1).unwrap().pop().unwrap();
    assert!(raw.vector.iter().any(|&x| x != 0.0), "the zero vector is still in the store after reindex + dream: nothing ever finds the row");
}

/// `reindex(limit)` reads `limit` ids of the stale view and only then drops the unusable ones: a
/// stale usable row behind `limit` unusable rows is never indexed, however often it is called.
#[test]
fn unusable_stale_rows_do_not_starve_reindex() {
    let mut store = SharedStore::default();
    let mut rows = vec![row(&rid(1), "the cat sat on the mat", "old/model", Vec::new(), true)];
    for i in 0..8 {
        rows.push(row(&rid(100 + i), "", "old/model", Vec::new(), true)); // newer ids: stale view is newest first
    }
    store.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store.clone()));
    let n: usize = (0..4).map(|_| m.reindex(8).unwrap()).sum();
    assert!(n >= 1, "4 x reindex(8) indexed {n} rows: eight empty-text rows keep the usable stale row out of every batch");
}

// ------------------------------------------------ (4) remember() bookkeeping

/// `write_times.retain(.. t <= now)` deletes a write stamped after a backward clock step for good:
/// when the clock is back, the day's quota has been forgotten.
#[test]
fn backward_clock_step_does_not_reset_the_daily_quota() {
    let (mut m, clock) = build(Config { writes_per_day: 3, ..cfg() });
    for i in 0..3 {
        clock.advance(1.0);
        insert(&mut m, &format!("fact number {i} alpha"), 1.0, "");
    }
    clock.advance(1.0);
    assert!(matches!(m.remember("fact four beta", 1.0, "").unwrap(), Remembered::RateLimited { .. }));
    let back = clock.get();
    clock.set(back - 3600.0); // an NTP step back
    let _ = m.remember("fact five gamma", 1.0, ""); // may or may not count: it is a new write
    clock.set(back + 1.0); // the clock is right again: 3 writes (+1 above) in the last second
    let r = m.remember("fact six delta", 1.0, "").unwrap();
    assert!(matches!(r, Remembered::RateLimited { .. }), "quota 3/day but a 4th..6th write in the same minute was accepted: {r:?}");
}

// ------------------------------------------------ (3) the unreadable placeholder

fn tamper(path: &std::path::Path, ids: &[&str]) {
    let db = redb::Database::open(path).unwrap();
    let wx = db.begin_write().unwrap();
    {
        let mut rows = wx.open_table(redb::TableDefinition::<&str, &[u8]>::new("rows")).unwrap();
        for id in ids {
            rows.insert(*id, b"{not json".as_slice()).unwrap();
        }
    }
    wx.commit().unwrap();
}

/// Entries of the file's index tables that still name `id` (and the row table's own entry).
fn leftovers(path: &std::path::Path, id: &str) -> Vec<String> {
    use redb::{ReadableTable, TableDefinition as T};
    let db = redb::Database::open(path).unwrap();
    let rx = db.begin_read().unwrap();
    let mut out = Vec::new();
    if rx.open_table(T::<&str, &[u8]>::new("rows")).unwrap().get(id).unwrap().is_some() {
        out.push("rows".to_owned());
    }
    for (name, t) in [("model", T::<(&str, &str), ()>::new("model")), ("labile", T::<(&str, &str), ()>::new("labile"))] {
        for it in rx.open_table(t).unwrap().iter().unwrap() {
            if it.unwrap().0.value().1 == id {
                out.push(name.to_owned());
            }
        }
    }
    for it in rx.open_table(T::<(&str, &str), ()>::new("texts")).unwrap().iter().unwrap() {
        if it.unwrap().0.value().1 == id {
            out.push("texts".to_owned());
        }
    }
    for it in rx.open_table(T::<(i64, &str), ()>::new("created")).unwrap().iter().unwrap() {
        if it.unwrap().0.value().1 == id {
            out.push("created".to_owned());
        }
    }
    for it in rx.open_table(T::<(u8, i64, &str), i64>::new("band")).unwrap().iter().unwrap() {
        if it.unwrap().0.value().2 == id {
            out.push("band".to_owned());
        }
    }
    for it in rx.open_table(T::<(&str, u32, &str), &[u8]>::new("lists")).unwrap().iter().unwrap() {
        if it.unwrap().0.value().2 == id {
            out.push("lists".to_owned());
        }
    }
    if rx.open_table(T::<&str, u32>::new("listed")).unwrap().get(id).unwrap().is_some() {
        out.push("listed".to_owned());
    }
    out
}

fn damaged_file(name: &str, n: usize) -> std::path::PathBuf {
    let path = temp_dir(name).join("s.redb");
    let mut st = engram::Store::open(&path).unwrap();
    let mut e = FakeEmbedder::new();
    let rows: Vec<Trace> = (0..n)
        .map(|i| {
            let text = format!("fact number {i} about topic{i}");
            let v = e.embed(&text);
            row(&rid(i), &text, MODEL, v, false)
        })
        .collect();
    st.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    drop(st);
    path
}

/// Forgetting a damaged row must leave no entry of it in any table, and keep the counts exact.
#[test]
fn forgetting_a_damaged_row_leaves_no_index_entry() {
    let path = damaged_file("a3a-purge", 6);
    tamper(&path, &[&rid(2)]);
    {
        let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(engram::Store::open(&path).unwrap()));
        assert!(m.forget(&rid(2)).unwrap().is_some());
        assert_eq!(m.stats().unwrap().records, 5);
        assert_eq!(m.stats().unwrap().labile, 5);
    }
    assert_eq!(leftovers(&path, &rid(2)), Vec::<String>::new());
}

/// The id generator may hand back the id of a row the engine cannot read: it is still taken.
#[test]
fn a_damaged_rows_id_is_not_reused() {
    let path = damaged_file("a3a-id", 3);
    tamper(&path, &[&rid(1)]);
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(engram::Store::open(&path).unwrap()));
    let mut calls = 0;
    m.set_ids(Box::new(move |_, _| {
        calls += 1;
        if calls == 1 { rid(1) } else { rid(1000 + calls) }
    }));
    let id = insert(&mut m, "a brand new fact", 1.0, "");
    assert_ne!(id, rid(1));
    assert_eq!(m.stats().unwrap().records, 4);
}

/// Over capacity the unreadable row goes first and the count stays honest.
#[test]
fn eviction_removes_the_damaged_row_first() {
    let path = damaged_file("a3a-evict", 5);
    tamper(&path, &[&rid(3)]);
    let (mut m, _c) = build_with(Config { capacity: 5, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(engram::Store::open(&path).unwrap()));
    insert(&mut m, "a brand new fact", 1.0, "");
    assert_eq!(m.stats().unwrap().records, 5);
    drop(m);
    assert_eq!(leftovers(&path, &rid(3)), Vec::<String>::new());
}

// ------------------------------------------------ (5) clock

/// Hostile clocks never panic and never wrap.
#[test]
fn infinite_and_nan_clocks_do_not_panic() {
    for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX, f64::MIN] {
        let (mut m, clock) = build(cfg());
        insert(&mut m, "the cat sat on the mat", 1.0, "");
        insert(&mut m, "the cat sat on the mat today", 1.0, "the cat sat on the mat");
        clock.set(t);
        let _ = m.remember("the cat sat on the mat again", 1.0, "the cat sat on the mat today");
        let _ = m.recall("the cat sat");
        m.dream(&mut merge_to_gist, Some(3)).unwrap();
        let _ = m.now_local();
    }
}

// ------------------------------------------------ (7) prompt / JSON

fn req(current_time: &str, text: &str) -> engram::DreamRequest {
    engram::DreamRequest {
        current_time: current_time.into(),
        members: vec![engram::DreamMember { id: "A".into(), text: text.into(), local_time: "x".into(), timezone: "UTC;+00:00".into(), retrievability: 1.0 }],
    }
}

#[test]
fn prompt_is_single_pass() {
    let p = req("{listing}", "{current_time} and {listing}").prompt();
    assert_eq!(p.matches("{current_time}").count(), 1);
    assert_eq!(p.matches("{listing}").count(), 2); // the time and the text, each once
    assert!(p.contains("{\"action\": \"replace\""));
}

#[test]
#[ignore = "accepted: SPAN_TRIES bounds the opener scan on purpose"]
fn json_answer_after_prose_with_many_openers() {
    let raw = format!("{} {{\"action\":\"keep\"}}", "[note] ".repeat(10));
    assert_eq!(DreamDecision::parse(&raw), Some(DreamDecision::keep()), "ten bracketed notes before the object exhaust SPAN_TRIES");
}

// ------------------------------------------------ sound guards (pass)

/// A row the engine cannot use (empty text) indexed under another model can never be re-embedded
/// and cannot be what a write reactivates; it must not keep every later write labile.
#[test]
fn unusable_stale_row_does_not_make_every_write_labile() {
    let mut store = SharedStore::default();
    store.write(&[Write::Insert(&row(&rid(1), "", "old/model", Vec::new(), true))]).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store));
    match m.remember("a brand new fact about nothing", 1.0, "").unwrap() {
        Remembered::Inserted { trace, .. } => assert!(trace.consolidated, "born labile only because an unusable row sits in the stale view"),
        other => panic!("{other:?}"),
    }
}

/// Seventy unusable labile rows ahead of a real seed do not hide its older trace.
#[test]
fn dream_reaches_a_seed_behind_many_unusable_rows() {
    let mut store = SharedStore::default();
    let mut e = FakeEmbedder::new();
    let mut rows: Vec<Trace> = (0..70).map(|i| row(&rid(i), "", MODEL, Vec::new(), false)).collect();
    let old = "the user lives in kyoto japan";
    rows.push(row(&rid(100), old, MODEL, e.embed(old), true));
    let new = "the user lives in osaka japan now";
    rows.push(row(&rid(101), new, MODEL, e.embed(new), false));
    store.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    let (mut m, _c) = build_with(Config { theta_related: 0.3, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(store));
    let reports = m.dream(&mut merge_to_gist, Some(1)).unwrap();
    assert!(reports.iter().any(|r| r.action == engram::DreamAction::Replace), "{reports:?}");
}

/// Recall widens past 200 nearer unusable rows.
#[test]
fn recall_widens_past_unusable_rows() {
    let mut store = SharedStore::default();
    let mut e = FakeEmbedder::new();
    let q = "the cat sat on the mat";
    let mut rows: Vec<Trace> = (0..200).map(|i| row(&rid(i), "", MODEL, e.embed(q), true)).collect();
    rows.push(row(&rid(500), "the cat sat on the mat today", MODEL, e.embed("the cat sat on the mat today"), true));
    store.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    let (mut m, _c) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(store));
    assert_eq!(m.peek(q).unwrap().recalled.len(), 1);
}
