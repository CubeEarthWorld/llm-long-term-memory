//! The bundled Store: every view against a direct computation, persistence across a
//! reopen, and the IVF index keeping search quality as its lists split.

mod support;

use engram::{Storage, Store, Trace, View, Write};
use support::temp_dir;

/// A reproducible pseudo-random unit vector.
fn unit(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let v: Vec<f32> = (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        })
        .collect();
    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    v.into_iter().map(|a| a / n).collect()
}

fn trace(i: u64, dim: usize) -> Trace {
    Trace {
        id: format!("{i:06}"),
        text: format!("fact {i}"),
        created_at: 1_000 + (i as i64 % 13) * 100,
        tz: "UTC;+00:00".into(),
        last_recall: 2_000 + (i as i64 % 5) * 10,
        stability: 2f64.powi((i % 9) as i32 + 10),
        consolidated: !i.is_multiple_of(3),
        cue: String::new(),
        model_id: if i.is_multiple_of(11) { "old".into() } else { "m".into() },
        vector: unit(i, dim),
    }
}

fn ids(v: Vec<Trace>) -> Vec<String> {
    v.into_iter().map(|t| t.id).collect()
}

#[test]
fn every_view_matches_a_direct_computation() {
    let mut st = Store::in_memory().unwrap();
    let rows: Vec<Trace> = (0..200).map(|i| trace(i, 8)).collect();
    st.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    st.write(&[Write::State { id: "000005", last_recall: 9_999, stability: 1e6, consolidated: false }, Write::Remove("000007")]).unwrap();
    let live: Vec<Trace> = st.find(View::All, usize::MAX).unwrap();
    assert_eq!(live.len(), 199);
    assert_eq!(st.count(View::All).unwrap(), 199);
    let want = |f: &dyn Fn(&Trace) -> bool, key: &dyn Fn(&Trace) -> (i64, String)| {
        let mut v: Vec<&Trace> = live.iter().filter(|t| f(t)).collect();
        v.sort_by_key(|t| key(t));
        v.into_iter().map(|t| t.id.clone()).take(25).collect::<Vec<_>>()
    };
    let by_id = |t: &Trace| (0, t.id.clone());
    assert_eq!(ids(st.find(View::Labile { model_id: "m", after: "" }, 25).unwrap()), want(&|t| !t.consolidated && t.model_id == "m", &by_id));
    let band = live[0].band();
    assert_eq!(ids(st.find(View::Band { band, created_by: 1_500 }, 25).unwrap()), want(&|t| t.band() == band && t.created_at <= 1_500, &|t| (t.last_recall, t.id.clone())));
    assert_eq!(ids(st.find(View::Young { created_after: 1_800 }, 25).unwrap()), want(&|t| t.created_at > 1_800, &|t| (t.created_at, t.id.clone())));
    let mut stale = want(&|t| t.model_id != "m", &by_id);
    stale.reverse();
    let mut got = ids(st.find(View::Stale { model_id: "m" }, 25).unwrap());
    got.sort();
    stale.sort();
    assert_eq!(got, stale);
    assert_eq!(ids(st.find(View::Text("fact 9"), 1).unwrap()), ["000009"]);
    assert_eq!(st.find(View::Id("000005"), 1).unwrap()[0].stability, 1e6);
    assert!(st.find(View::Id("000007"), 1).unwrap().is_empty());
}

#[test]
fn a_reopened_file_holds_everything() {
    let path = temp_dir("store-file").join("m.redb");
    {
        let mut st = Store::open(&path).unwrap();
        st.write(&(0..50).map(|i| trace(i, 8)).collect::<Vec<_>>().iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    }
    let st = Store::open(&path).unwrap();
    assert_eq!(st.count(View::All).unwrap(), 50);
    let t = &st.find(View::Id("000012"), 1).unwrap()[0];
    assert_eq!((t.text.as_str(), t.vector.clone()), ("fact 12", unit(12, 8)));
    let hits = st.near(&unit(12, 8), "m", 0.99, 1, None).unwrap();
    assert_eq!(hits[0].trace.id, "000012");
}

#[test]
fn search_stays_accurate_as_lists_split_and_forgets_removed_traces() {
    let dim = 16;
    let mut st = Store::in_memory().unwrap();
    let rows: Vec<Trace> = (0..5_000).map(|i| Trace { model_id: "m".into(), ..trace(i, dim) }).collect();
    for chunk in rows.chunks(500) {
        st.write(&chunk.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    }
    let removed: Vec<String> = (0..5_000).step_by(2).map(|i| format!("{i:06}")).collect();
    st.write(&removed.iter().map(|id| Write::Remove(id)).collect::<Vec<_>>()).unwrap();
    let live: Vec<&Trace> = rows.iter().filter(|t| !removed.contains(&t.id)).collect();
    let mut found = 0;
    for j in 0..50 {
        let q = unit(100_000 + j, dim);
        let mut exact: Vec<(f64, &str)> = live.iter().map(|t| (t.vector.iter().zip(&q).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum(), t.id.as_str())).collect();
        exact.sort_by(|a, b| b.0.total_cmp(&a.0));
        let hits = st.near(&q, "m", -1.0, 10, None).unwrap();
        assert!(hits.iter().all(|h| !removed.contains(&h.trace.id)));
        found += exact[..10].iter().filter(|e| hits.iter().any(|h| h.trace.id == e.1)).count();
    }
    assert!(found >= 400, "recall@10 = {}%", found / 5); // approximate, but close
}

#[test]
fn clear_empties_everything() {
    let mut st = Store::in_memory().unwrap();
    st.write(&(0..20).map(|i| trace(i, 8)).collect::<Vec<_>>().iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    st.clear().unwrap();
    assert_eq!(st.count(View::All).unwrap(), 0);
    assert!(st.near(&unit(1, 8), "m", -1.0, 5, None).unwrap().is_empty());
}

#[test]
fn traces_sharing_a_text_each_stay_findable() {
    // Gists and restored traces may repeat a text: removing one must not hide the other.
    let mut st = Store::in_memory().unwrap();
    let a = Trace { text: "same".into(), ..trace(1, 8) };
    let b = Trace { text: "same".into(), ..trace(2, 8) };
    st.write(&[Write::Insert(&a), Write::Insert(&b), Write::Remove(&a.id)]).unwrap();
    assert_eq!(ids(st.find(View::Text("same"), 5).unwrap()), vec![b.id.clone()]);
    assert_eq!(st.count(View::Text("same")).unwrap(), 1);
}

#[test]
fn a_non_finite_stability_is_refused_and_changes_nothing() {
    let mut st = Store::in_memory().unwrap();
    let good = trace(1, 8);
    st.write(&[Write::Insert(&good)]).unwrap();
    for s in [f64::NAN, f64::INFINITY] {
        let bad = Trace { stability: s, ..trace(2, 8) };
        assert!(st.write(&[Write::Insert(&bad)]).is_err());
        assert!(st.write(&[Write::State { id: &good.id, last_recall: 1, stability: s, consolidated: true }]).is_err());
    }
    assert_eq!(st.find(View::All, 10).unwrap(), vec![good.clone()]);
    st.write(&[Write::Remove(&good.id)]).unwrap(); // the row stays removable
    assert_eq!(st.count(View::All).unwrap(), 0);
}

#[test]
fn any_id_is_reachable_through_every_index() {
    // Ids are whatever the IdGen returns: no sentinel character may bound a range.
    let mut st = Store::in_memory().unwrap();
    let t = Trace { id: "\u{10FFFF}\u{10FFFF}z".into(), model_id: "m".into(), ..trace(3, 8) };
    st.write(&[Write::Insert(&t)]).unwrap();
    assert_eq!(ids(st.find(View::Band { band: t.band(), created_by: i64::MAX }, 5).unwrap()), vec![t.id.clone()]);
    assert_eq!(ids(st.find(View::Stale { model_id: "other" }, 5).unwrap()), vec![t.id.clone()]);
    assert_eq!(st.near(&t.vector, "m", 0.99, 1, None).unwrap()[0].trace.id, t.id);
    st.write(&[Write::Index { id: &t.id, model_id: "n", vector: &unit(4, 8) }]).unwrap();
    assert!(st.near(&t.vector, "m", 0.0, 5, None).unwrap().is_empty());
    assert_eq!(st.near(&unit(4, 8), "n", 0.99, 1, None).unwrap()[0].trace.id, t.id);
}

#[test]
fn stale_lists_each_other_model_newest_first() {
    let mut st = Store::in_memory().unwrap();
    let rows: Vec<Trace> = [("a", 1), ("a", 2), ("z", 3), ("z", 4), ("m", 5)].iter().map(|&(m, i)| Trace { model_id: m.into(), ..trace(i, 8) }).collect();
    st.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    assert_eq!(ids(st.find(View::Stale { model_id: "m" }, 10).unwrap()), ["000004", "000003", "000002", "000001"]);
    assert_eq!(st.count(View::Stale { model_id: "m" }).unwrap(), 4);
}

#[test]
fn strengthening_keeps_the_vector_where_it_is() {
    let mut st = Store::in_memory().unwrap();
    let t = trace(1, 8);
    st.write(&[Write::Insert(&t), Write::State { id: &t.id, last_recall: 9_000, stability: 5e6, consolidated: false }]).unwrap();
    let got = st.near(&t.vector, &t.model_id, 0.99, 1, None).unwrap().remove(0).trace;
    assert_eq!((got.vector, got.last_recall, got.stability, got.consolidated), (t.vector.clone(), 9_000, 5e6, false));
    assert_eq!(ids(st.find(View::Labile { model_id: &t.model_id, after: "" }, 5).unwrap()), vec![t.id.clone()]);
}

#[test]
fn identical_vectors_beyond_a_list_stay_searchable() {
    // A list of duplicates cannot be split; it must keep accepting and answering.
    let mut st = Store::in_memory().unwrap();
    let v = unit(9, 8);
    let rows: Vec<Trace> = (0..1_200).map(|i| Trace { model_id: "m".into(), vector: v.clone(), ..trace(i, 8) }).collect();
    for chunk in rows.chunks(100) {
        st.write(&chunk.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    }
    let hits = st.near(&v, "m", 0.99, 5, None).unwrap();
    assert_eq!(ids(hits.into_iter().map(|h| h.trace).collect()), ["000000", "000001", "000002", "000003", "000004"]);
}

#[test]
fn a_file_with_the_old_text_table_is_migrated() {
    let path = temp_dir("old-text").join("m.redb");
    let t = trace(1, 8);
    {
        let st = Store::open(&path).unwrap();
        drop(st);
        let db = redb::Database::create(&path).unwrap();
        let wx = db.begin_write().unwrap();
        {
            let mut rows = wx.open_table(redb::TableDefinition::<&str, &[u8]>::new("rows")).unwrap();
            let row = serde_json::json!([t.text, t.created_at, t.tz, t.last_recall, t.stability, t.consolidated, t.cue, t.model_id]).to_string();
            rows.insert(t.id.as_str(), row.as_bytes()).unwrap();
            let mut old = wx.open_table(redb::TableDefinition::<&str, &str>::new("text")).unwrap();
            old.insert(t.text.as_str(), t.id.as_str()).unwrap();
        }
        wx.delete_table(redb::TableDefinition::<(&str, &str), ()>::new("texts")).unwrap();
        wx.commit().unwrap();
    }
    let st = Store::open(&path).unwrap();
    assert_eq!(ids(st.find(View::Text(&t.text), 5).unwrap()), vec![t.id.clone()]);
}

/// Every (id → list) assignment in a store file.
fn listed(path: &std::path::Path) -> Vec<(String, u32)> {
    let db = redb::Database::open(path).unwrap();
    let rx = db.begin_read().unwrap();
    let t = rx.open_table(redb::TableDefinition::<&str, u32>::new("listed")).unwrap();
    redb::ReadableTable::iter(&t).unwrap().map(|e| e.map(|(k, v)| (k.value().to_owned(), v.value())).unwrap()).collect()
}

#[test]
fn strengthening_never_moves_a_vector_between_lists() {
    let path = temp_dir("state-lists").join("m.redb");
    let rows: Vec<Trace> = (0..1_100).map(|i| Trace { model_id: "m".into(), ..trace(i, 8) }).collect();
    {
        let mut st = Store::open(&path).unwrap();
        st.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    }
    let before = listed(&path);
    {
        let mut st = Store::open(&path).unwrap();
        let states: Vec<Write<'_>> = rows.iter().map(|t| Write::State { id: &t.id, last_recall: 7, stability: 3e6, consolidated: true }).collect();
        st.write(&states).unwrap();
    }
    assert!(before.iter().map(|b| b.1).collect::<std::collections::HashSet<_>>().len() > 1); // the index did split
    assert_eq!(listed(&path), before);
}

#[test]
fn a_damaged_entry_costs_the_vector_not_the_store() {
    let path = temp_dir("short-entry").join("m.redb");
    let t = Trace { model_id: "m".into(), ..trace(1, 8) };
    Store::open(&path).unwrap().write(&[Write::Insert(&t)]).unwrap();
    {
        let db = redb::Database::open(&path).unwrap();
        let wx = db.begin_write().unwrap();
        {
            let mut lists = wx.open_table(redb::TableDefinition::<(&str, u32, &str), &[u8]>::new("lists")).unwrap();
            let key: Vec<(String, u32, String)> = redb::ReadableTable::iter(&lists).unwrap().map(|e| { let k = e.unwrap().0; let (a, b, c) = k.value(); (a.to_owned(), b, c.to_owned()) }).collect();
            lists.insert((key[0].0.as_str(), key[0].1, key[0].2.as_str()), [1u8, 2, 3].as_slice()).unwrap();
            let mut cents = wx.open_table(redb::TableDefinition::<(&str, u32), &[u8]>::new("centroids")).unwrap();
            cents.insert(("m", key[0].1), [9u8].as_slice()).unwrap();
        }
        wx.commit().unwrap();
    }
    // A damaged index entry costs the row its vector (it reads as unindexed); the store stays usable.
    let st = Store::open(&path).unwrap();
    assert!(st.find(View::Id(&t.id), 1).unwrap()[0].vector.is_empty());
    assert!(st.near(&t.vector, "m", 0.0, 5, None).unwrap().is_empty());
}

#[test]
fn a_non_finite_vector_is_refused() {
    let mut st = Store::in_memory().unwrap();
    let t = Trace { vector: vec![f32::NAN; 8], ..trace(1, 8) };
    assert!(st.write(&[Write::Insert(&t)]).is_err());
    let ok = trace(2, 8);
    st.write(&[Write::Insert(&ok)]).unwrap();
    assert!(st.write(&[Write::Index { id: &ok.id, model_id: "m", vector: &[f32::INFINITY; 8] }]).is_err());
    assert_eq!(st.find(View::All, 5).unwrap(), vec![ok]);
}

#[test]
fn a_repeated_migration_drops_text_entries_of_deleted_rows() {
    // New version migrates, an old version then deletes a row (and recreates `text`),
    // the new version opens again: `texts` must not keep pointing at the deleted row.
    let path = temp_dir("remigrate").join("m.redb");
    let (a, b) = (Trace { text: "same".into(), ..trace(1, 8) }, Trace { text: "same".into(), ..trace(2, 8) });
    Store::open(&path).unwrap().write(&[Write::Insert(&a), Write::Insert(&b)]).unwrap();
    {
        let db = redb::Database::open(&path).unwrap();
        let wx = db.begin_write().unwrap();
        {
            let mut rows = wx.open_table(redb::TableDefinition::<&str, &[u8]>::new("rows")).unwrap();
            rows.remove(a.id.as_str()).unwrap();
            wx.open_table(redb::TableDefinition::<&str, &str>::new("text")).unwrap();
        }
        wx.commit().unwrap();
    }
    let st = Store::open(&path).unwrap();
    assert_eq!(ids(st.find(View::Text("same"), 1).unwrap()), vec![b.id.clone()]);
}
