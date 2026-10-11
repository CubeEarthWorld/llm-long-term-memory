//! Audit lane 2: capacity and eviction (SPEC §3, §4 step 5, §7). Tests that FAIL
//! document a real defect; each says which invariant (L2-n) it checks.

mod support;

use engram::{ulid_ids, Config, Remembered, Storage, Store, Trace, View, Write};
use support::{build, build_with, cfg, insert, rows, FakeEmbedder, SharedStore, DAY, START};

const MODEL: &str = "fake/token-overlap";

fn cfg_cap(capacity: usize) -> Config {
    Config { capacity, ..cfg() }
}

/// Usable rows with real vectors, to be written behind the engine's back.
struct Rows {
    ids: engram::IdGen,
    emb: FakeEmbedder,
}

impl Rows {
    fn new() -> Self {
        Self { ids: ulid_ids(), emb: FakeEmbedder::new() }
    }
    fn row(&mut self, text: &str, stability: f64, last_recall: i64, created_at: i64) -> Trace {
        let id = (self.ids)(created_at * 1000, text);
        Trace {
            id,
            text: text.to_owned(),
            created_at,
            tz: "Asia/Tokyo;+09:00".into(),
            last_recall,
            stability,
            consolidated: true,
            cue: String::new(),
            model_id: MODEL.into(),
            vector: self.emb.embed(text),
        }
    }
}

fn put(store: &SharedStore, ts: &[Trace]) {
    let mut s = store.clone();
    let w: Vec<Write<'_>> = ts.iter().map(Write::Insert).collect();
    s.write(&w).unwrap();
}

fn count(m: &engram::Engram) -> usize {
    m.store().count(View::All).unwrap()
}

fn inserted(r: Remembered) -> (Trace, Option<Trace>) {
    match r {
        Remembered::Inserted { trace, evicted, .. } => (trace, evicted),
        other => panic!("expected an insert, got {other:?}"),
    }
}

/// L2-1: the count never exceeds capacity, for tiny capacities and mixed strengths;
/// L2-2: the reported eviction agrees with what is stored.
#[test]
fn l2_1_capacity_holds_for_tiny_capacities() {
    for cap in [1usize, 2, 3, 5, 9, 10, 11, 20] {
        let (mut s, clock) = build(cfg_cap(cap));
        for i in 0..80 {
            let sal = [0.0, 0.3, 1.0, 3.0, 10.0][i % 5];
            clock.advance([0.0, 1.0, 3600.0, DAY, 5.0 * DAY][(i * 7) % 5]);
            let (trace, evicted) = inserted(s.remember(&format!("tok{i}x{cap}"), sal, "").unwrap());
            assert!(count(&s) <= cap, "cap {cap} step {i}: count {}", count(&s));
            assert_eq!(rows(&s).len(), count(&s));
            let stored = s.memory(&trace.id).unwrap().is_some();
            match evicted {
                Some(e) if e.id == trace.id => assert!(!stored, "cap {cap}: reported as evicted but stored"),
                Some(e) => {
                    assert!(stored, "cap {cap}: another trace evicted but the new one is missing");
                    assert!(s.memory(&e.id).unwrap().is_none());
                }
                None => assert!(stored),
            }
        }
    }
}

/// L2-1: equal keys (same second, same stability).
#[test]
fn l2_1_equal_keys() {
    let (mut s, _) = build(cfg_cap(3));
    for i in 0..12 {
        inserted(s.remember(&format!("same{i}k"), 1.0, "").unwrap());
        assert!(count(&s) <= 3);
    }
    assert_eq!(count(&s), 3);
}

/// L2-1/L2-2: capacity 1 with a stronger resident: the newcomer is dropped and reported.
#[test]
fn l2_2_capacity_one_new_is_weakest() {
    let (mut s, _) = build(cfg_cap(1));
    let (a, _) = inserted(s.remember("resident aaa", 10.0, "").unwrap());
    let (b, ev) = inserted(s.remember("newcomer bbb", 0.5, "").unwrap());
    assert_eq!(ev.map(|e| e.id), Some(b.id.clone()));
    assert!(s.memory(&b.id).unwrap().is_none());
    assert!(s.memory(&a.id).unwrap().is_some());
    assert_eq!(count(&s), 1);
}

/// L2-2: when several traces must go and the newcomer is one of them, `evicted` must say so
/// (Remembered::Inserted doc: "`trace` itself when it was the weakest (then it is not stored)").
/// Reachable after undo() or a lowered capacity (count > capacity beforehand).
#[test]
fn l2_2_evicted_reports_unstored_newcomer_with_excess() {
    let shared = SharedStore::default();
    let mut r = Rows::new();
    let olds: Vec<Trace> = (0..6).map(|i| r.row(&format!("old{i}z"), DAY * 5.0, START as i64, START as i64)).collect();
    put(&shared, &olds);
    let (mut s, _) = build_with(cfg_cap(4), Box::new(FakeEmbedder::new()), Box::new(shared));
    // total = 7, capacity 4: three victims; the newcomer (stability 1 s) is the weakest of all.
    let (trace, evicted) = inserted(s.remember("weak newcomer", 0.0, "").unwrap());
    assert_eq!(count(&s), 4);
    assert!(s.memory(&trace.id).unwrap().is_none(), "premise: the weakest newcomer is a victim");
    assert_eq!(
        evicted.map(|e| e.id),
        Some(trace.id),
        "caller is told another trace was evicted although the newcomer itself vanished"
    );
}

/// L2-2/L2-3: excess > 1 evicts exactly the `excess` weakest, distinct.
#[test]
fn l2_2_excess_victims_are_the_weakest_and_distinct() {
    let shared = SharedStore::default();
    let mut r = Rows::new();
    let mut ts = Vec::new();
    for i in 0..40 {
        let s_ = 100.0 * 1.5f64.powi(i % 20);
        ts.push(r.row(&format!("spread{i}q"), s_, START as i64 - 400_000 - (i as i64) * 1000, START as i64 - 10 * DAY as i64));
    }
    put(&shared, &ts);
    let (mut s, _) = build_with(cfg_cap(10), Box::new(FakeEmbedder::new()), Box::new(shared));
    let now = START;
    let rank = |t: &Trace| t.stability.log2() - (now - t.last_recall as f64).max(0.0) / t.stability;
    let mut all = ts.clone();
    all.sort_by(|a, b| rank(a).total_cmp(&rank(b)).then(a.id.cmp(&b.id)));
    let (trace, _) = inserted(s.remember("fresh newcomer", 1.0, "").unwrap());
    assert_eq!(count(&s), 10);
    assert!(s.memory(&trace.id).unwrap().is_some());
    let got: std::collections::HashSet<_> = rows(&s).iter().filter(|t| t.id != trace.id).map(|t| t.id.clone()).collect();
    let want: std::collections::HashSet<_> = all[all.len() - 9..].iter().map(|t| t.id.clone()).collect();
    assert_eq!(got.len(), 9);
    assert_eq!(want, got);
}

/// L2-3: the band head is not always the weakest of its band. Inside one half-octave band
/// the order by last_recall differs from the order by log2 S - dt/S once dt/S is large
/// (a year idle), so a weaker trace behind the first window must still be found (the window grows).
#[test]
fn l2_3_band_head_is_not_the_global_minimum() {
    let shared = SharedStore::default();
    let mut r = Rows::new();
    let t0 = START as i64;
    let mut h: Vec<Trace> = (0..10).map(|i| r.row(&format!("hi{i}h"), 92_000.0, t0, t0)).collect();
    let l = r.row("lowstab lll", 66_000.0, t0 + 100_000, t0);
    h.push(l.clone());
    assert!(h.iter().all(|t| t.band() == 32));
    put(&shared, &h);
    let (mut s, clock) = build_with(cfg_cap(11), Box::new(FakeEmbedder::new()), Box::new(shared));
    clock.advance(3.0e5); // 3–4 half-lives idle: alive, so the exact order matters
    let now = clock.get();
    let rank = |t: &Trace| t.stability.log2() - (now - t.last_recall as f64).max(0.0) / t.stability;
    let weakest = h.iter().min_by(|a, b| rank(a).total_cmp(&rank(b))).unwrap();
    assert_eq!(weakest.id, l.id, "premise: L is the global weakest");
    let (_, evicted) = inserted(s.remember("trigger newcomer", 1.0, "").unwrap());
    assert_eq!(evicted.map(|e| e.id), Some(l.id), "evicted the head of the band instead of the weakest");
}

/// L2-3: differential test against the full ranking, excess 1, 100 rows in one band whose
/// last recalls span 1e6 s a year ago (idle store). The weakest by log2 S - dt/S is the
/// lowest-stability row, not the least recently recalled one.
#[test]
fn l2_3_random_differential_against_full_ranking() {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut bad = 0;
    let trials = 20;
    for trial in 0..trials {
        let shared = SharedStore::default();
        let mut r = Rows::new();
        let ts: Vec<Trace> = (0..100)
            .map(|i| {
                let s_ = 65_600.0 * 1.41f64.powf(rnd()); // one band (32)
                let lr = START as i64 - 150_000 - (rnd() * 100_000.0) as i64; // 2–4 half-lives idle: alive
                r.row(&format!("diff{trial}n{i}"), s_, lr, START as i64 - 400 * DAY as i64)
            })
            .collect();
        assert!(ts.iter().all(|t| t.band() == 32));
        put(&shared, &ts);
        let (mut s, _) = build_with(cfg_cap(100), Box::new(FakeEmbedder::new()), Box::new(shared));
        let rank = |t: &Trace| t.stability.log2() - (START - t.last_recall as f64).max(0.0) / t.stability;
        let want = ts.iter().min_by(|a, b| rank(a).total_cmp(&rank(b)).then(a.id.cmp(&b.id))).unwrap().id.clone();
        let (_, ev) = inserted(s.remember(&format!("newcomer{trial}"), 1.0, "").unwrap());
        if ev.map(|e| e.id) != Some(want) {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad}/{trials} evictions were not the global minimum of log2 strength");
}

/// L2-5 mechanism: serde_json (no `float_roundtrip` feature) does not parse back what it printed,
/// so the band computed on insert and the band computed from the stored row can differ.
#[test]
fn l2_5_serde_json_f64_round_trip_is_exact() {
    for x in [1.9999999999999998f64, 3.9999999999999996, 127.99999999999997, 1448.1546878700485, 8191.9999999999945] {
        let s = serde_json::json!([x]).to_string();
        let y = serde_json::from_str::<serde_json::Value>(&s).unwrap()[0].as_f64().unwrap();
        assert_eq!(x.to_bits(), y.to_bits(), "{s} parsed as {y:?}");
    }
}

/// L2-3 / SPEC §2: an unusable row must die first, even when it sits behind more than
/// the first window of usable rows of its own band.
#[test]
fn l2_3_unusable_row_dies_first() {
    let shared = SharedStore::default();
    let mut r = Rows::new();
    let t0 = START as i64;
    let mut ts: Vec<Trace> = (0..10).map(|i| r.row(&format!("ok{i}u"), DAY, t0, t0)).collect();
    let mut bad = r.row("placeholder", DAY, t0 + 10, t0);
    bad.text = String::new(); // unusable: empty text
    ts.push(bad.clone());
    put(&shared, &ts);
    let (mut s, clock) = build_with(cfg_cap(11), Box::new(FakeEmbedder::new()), Box::new(shared));
    clock.advance(10.0 * DAY);
    let (_, evicted) = inserted(s.remember("trigger two", 1.0, "").unwrap());
    assert_eq!(evicted.map(|e| e.id), Some(bad.id), "a usable row was evicted while an unusable one stayed");
}

/// L2-4: the grace boundary. A weak trace created exactly `grace` ago is out of grace;
/// one second younger is protected.
#[test]
fn l2_4_grace_cutoff_boundary() {
    let run = |extra: f64| -> (Trace, Option<Trace>) {
        let (mut s, clock) = build(cfg_cap(20));
        for i in 0..19 {
            insert(&mut s, &format!("olds{i}o"), 1.0, "");
        }
        clock.advance(1000.0);
        let y = insert(&mut s, "yyy weak", 0.0, "");
        let y = s.memory(&y).unwrap().unwrap();
        clock.set(START + 1000.0 + 3.0 * DAY + extra);
        let (_, ev) = inserted(s.remember("trigger three", 1.0, "").unwrap());
        (y, ev)
    };
    let (y, ev) = run(-1.0);
    assert_ne!(ev.map(|e| e.id), Some(y.id), "protected inside the grace period");
    let (y, ev) = run(0.0);
    assert_eq!(ev.map(|e| e.id), Some(y.id), "created exactly at the cutoff counts as past grace");
}

/// L2-4: a created_at in the future is protected like a young trace (SPEC §4.5).
#[test]
fn l2_4_future_created_at_is_protected() {
    let shared = SharedStore::default();
    let mut r = Rows::new();
    let t0 = START as i64;
    let old = t0 - 5 * DAY as i64;
    let mut ts: Vec<Trace> = (0..19).map(|i| r.row(&format!("fut{i}f"), DAY, old, old)).collect();
    let mut f = r.row("future weak", 1.0, old, old);
    f.created_at = t0 + 5000;
    ts.push(f.clone());
    put(&shared, &ts);
    let (mut s, _) = build_with(cfg_cap(20), Box::new(FakeEmbedder::new()), Box::new(shared));
    let (_, ev) = inserted(s.remember("trigger four", 1.0, "").unwrap());
    assert_ne!(ev.map(|e| e.id), Some(f.id));
    assert_eq!(count(&s), 20);
}

/// L2-4 (characterisation, passes): for capacity < 10, capacity/10 == 0, so the newcomer
/// always counts as a flood and grace is void: a weak newcomer is judged against everything.
#[test]
fn l2_4_small_capacity_has_no_grace() {
    let (mut s, clock) = build(cfg_cap(5));
    for i in 0..5 {
        insert(&mut s, &format!("res{i}r"), 10.0, "");
    }
    clock.advance(1.0);
    let (b, ev) = inserted(s.remember("low salience newcomer", 0.01, "").unwrap());
    assert_eq!(ev.map(|e| e.id), Some(b.id), "the 1-second-old newcomer is dropped on arrival");
    assert_eq!(count(&s), 5);
}

/// L2-5 (accepted trade-off, see audit/lane2.md): a forward clock glitch (a recall stamped far in the future) must not freeze decay
/// until the real clock catches up. `last_recall` in the future is clamped to `now` on every
/// read, so the elapsed time stays 0 and the trace never fades.
#[test]
#[ignore = "accepted: SPEC §7 clamps last_recall to now; the engine cannot tell a glitch from the truth"]
fn l2_5_future_last_recall_does_not_make_a_trace_immortal() {
    let (mut s, clock) = build(cfg_cap(5));
    let x = insert(&mut s, "xglitch trace", 1.0, "");
    clock.set(START + 100.0 * 365.0 * DAY); // clock jumps forward 100 years
    s.recall("xglitch trace").unwrap();
    clock.set(START + DAY); // clock is repaired
    for i in 0..240 {
        clock.advance(30.0 * DAY); // 20 years of idle time with steady churn
        s.remember(&format!("churn{i}c"), 1.0, "").unwrap();
    }
    assert!(s.memory(&x).unwrap().is_none(), "a never-recalled trace outlived 20 years of churn (frozen by a future last_recall)");
}

/// L2-5: Store indexes stay in sync with the rows under random Insert/State/Index/Remove,
/// replacement of an id, and rejected (non-finite) writes.
#[test]
fn l2_5_store_indexes_stay_in_sync_random() {
    use std::collections::HashMap;
    let mut seed = 0xdead_beef_cafe_f00du64;
    let mut rnd = move |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let mut store = Store::in_memory().unwrap();
    let mut model: HashMap<String, Trace> = HashMap::new();
    let models = ["", "A", "B"];
    let vec_for = |k: u64| -> Vec<f32> {
        let a = k as f32 * 0.7;
        let v = [a.sin() + 1.5, a.cos(), (a * 1.3).sin(), 0.3];
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    };
    let stabs = [0.2, 1.0, 1.5, 65_536.0, 86_400.0, 92_000.0, 1e9, 3.2e8, 1e300];
    for step in 0..1500 {
        let mut batch: Vec<(u64, String, Trace)> = Vec::new();
        let poison = step % 97 == 5;
        for _ in 0..1 + rnd(4) {
            let id = format!("id{:02}", rnd(25));
            let m = models[rnd(3) as usize];
            let t = Trace {
                id: id.clone(),
                text: format!("text{}", rnd(6)),
                created_at: rnd(1000) as i64 - 100,
                tz: "UTC;+00:00".into(),
                last_recall: rnd(2000) as i64 - 500,
                stability: stabs[rnd(stabs.len() as u64) as usize] * (1.0 + rnd(100) as f64 / 100.0),
                consolidated: rnd(2) == 0,
                cue: String::new(),
                model_id: m.into(),
                vector: if m.is_empty() { vec![] } else { vec_for(rnd(50)) },
            };
            batch.push((rnd(4), id, t));
        }
        let ws: Vec<Write<'_>> = batch
            .iter()
            .map(|(k, id, t)| match k {
                0 => Write::Insert(t),
                1 => Write::State { id, last_recall: t.last_recall, stability: if poison { f64::NAN } else { t.stability }, consolidated: t.consolidated },
                2 => Write::Index { id, model_id: &t.model_id, vector: &t.vector },
                _ => Write::Remove(id),
            })
            .collect();
        let res = store.write(&ws);
        // A NaN State on a present row aborts the whole commit; otherwise it is a no-op.
        if res.is_ok() {
            for (k, id, t) in &batch {
                match k {
                    0 => {
                        model.insert(id.clone(), t.clone());
                    }
                    1 => {
                        if let Some(o) = model.get_mut(id) {
                            o.last_recall = t.last_recall;
                            o.stability = t.stability;
                            o.consolidated = t.consolidated;
                        }
                    }
                    2 => {
                        if let Some(o) = model.get_mut(id) {
                            o.model_id = t.model_id.clone();
                            o.vector = t.vector.clone();
                        }
                    }
                    _ => {
                        model.remove(id);
                    }
                }
            }
        }
        let n = model.len();
        assert_eq!(store.count(View::All).unwrap(), n, "step {step}: count");
        let mut band_total = 0;
        for b in 0..=255u8 {
            let c = store.count(View::Band { band: b, created_by: i64::MAX }).unwrap();
            let want = model.values().filter(|t| t.band() == b).count();
            assert_eq!(c, want, "step {step}: band {b} entries (orphan or missing)");
            band_total += c;
            if want > 0 {
                let got: Vec<String> = store.find(View::Band { band: b, created_by: i64::MAX }, usize::MAX).unwrap().into_iter().map(|t| t.id).collect();
                let mut exp: Vec<&Trace> = model.values().filter(|t| t.band() == b).collect();
                exp.sort_by(|a, c| a.last_recall.cmp(&c.last_recall).then(a.id.cmp(&c.id)));
                assert_eq!(got, exp.iter().map(|t| t.id.clone()).collect::<Vec<_>>(), "step {step}: band {b} order");
            }
        }
        assert_eq!(band_total, n);
        assert_eq!(store.count(View::Young { created_after: i64::MIN }).unwrap(), n, "step {step}: created index");
        assert_eq!(store.count(View::Stale { model_id: "zzz" }).unwrap(), n, "step {step}: model index");
        for m in models {
            let want = model.values().filter(|t| t.model_id == m && !t.consolidated).count();
            assert_eq!(store.count(View::Labile { model_id: m, after: "" }).unwrap(), want, "step {step}: labile {m:?}");
            let stale = model.values().filter(|t| t.model_id != m).count();
            assert_eq!(store.count(View::Stale { model_id: m }).unwrap(), stale, "step {step}: stale {m:?}");
            if !m.is_empty() {
                let q = vec_for(3);
                let hits = store.near(&q, m, -2.0, 1000, None).unwrap();
                let want: std::collections::BTreeSet<_> = model.values().filter(|t| t.model_id == m && !t.vector.is_empty()).map(|t| t.id.clone()).collect();
                let got: std::collections::BTreeSet<_> = hits.iter().map(|h| h.trace.id.clone()).collect();
                assert_eq!(got, want, "step {step}: similarity lists of {m}");
                assert!(hits.iter().all(|h| h.trace.model_id == m), "step {step}: orphan list entry returned under {m}");
            }
        }
        for t in 0..6 {
            let want = model.values().filter(|x| x.text == format!("text{t}")).count();
            assert_eq!(store.count(View::Text(&format!("text{t}"))).unwrap(), want, "step {step}: text index");
        }
    }
}

/// L2-5: a stability at a half-octave edge must land in the same band when the row is read back
/// from JSON (serde_json without float_roundtrip may parse 1 ULP off); an orphan band entry
/// would remain after State/Remove.
#[test]
fn l2_5_band_key_survives_json_round_trip() {
    let mut store = Store::in_memory().unwrap();
    let mut bad = Vec::new();
    for k in 0..=520u32 {
        let edge = 2f64.powf(k as f64 / 2.0);
        let mut bits = edge.to_bits() - 6;
        for _ in 0..13 {
            let x = f64::from_bits(bits);
            bits += 1;
            let t = Trace {
                id: "edge".into(),
                text: "e".into(),
                created_at: 0,
                tz: "UTC;+00:00".into(),
                last_recall: 0,
                stability: x,
                consolidated: true,
                cue: String::new(),
                model_id: String::new(),
                vector: vec![],
            };
            store.write(&[Write::Insert(&t)]).unwrap();
            store.write(&[Write::State { id: "edge", last_recall: 1, stability: x, consolidated: true }]).unwrap();
            store.write(&[Write::Remove("edge")]).unwrap();
            let left: usize = (0..=255u8).map(|b| store.count(View::Band { band: b, created_by: i64::MAX }).unwrap()).sum();
            if left != 0 {
                bad.push(x);
                store.clear().unwrap();
            }
        }
    }
    assert!(bad.is_empty(), "orphan band entries for stabilities {bad:?}");
}

/// L2-5: after dream + undo the restored rows count against capacity and the next write
/// brings the store back to capacity; indexes agree.
#[test]
fn l2_5_undo_then_remember_restores_capacity() {
    let (mut s, _) = build(cfg_cap(4));
    for t in ["trip kyoto plan schedule note", "trip kyoto hotel schedule note", "trip kyoto food schedule note"] {
        insert(&mut s, t, 1.0, "");
    }
    let reports = s.dream(&mut support::merge_to_gist, None).unwrap();
    assert_eq!(reports.len(), 1);
    for i in 0..3 {
        insert(&mut s, &format!("filler{i}f"), 1.0, "");
    }
    assert_eq!(count(&s), 4);
    s.undo(&reports[0]).unwrap();
    assert!(count(&s) > 4, "premise: undo may exceed capacity");
    inserted(s.remember("after undo", 1.0, "").unwrap());
    assert_eq!(count(&s), 4);
    let band_total: usize = (0..=255u8).map(|b| s.store().count(View::Band { band: b, created_by: i64::MAX }).unwrap()).sum();
    assert_eq!(band_total, 4);
}
