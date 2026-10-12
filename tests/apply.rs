//! Host-originated changes go through `Engram::apply` (one atomic commit, no rebuild), the
//! bounded count, and the store's same-vector insert path.

mod support;

use engram::{Engram, Storage, Store, Trace, View, Write};
use std::sync::{Arc, Mutex};
use support::{build_with, cfg, insert, mem, rows, FakeEmbedder, SharedStore};

fn engine() -> Engram {
    build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(SharedStore::default())).0
}

#[test]
fn applied_changes_show_without_a_rebuild() {
    let mut s = engine();
    let a = insert(&mut s, "kyoto autumn temple", 1.0, "");
    let b = insert(&mut s, "tokyo winter station", 1.0, "");
    let before = mem(&s, &a).unwrap();

    // State only: the text and the vector stay.
    s.apply(&[Write::State { id: &a, last_recall: 5, stability: 1e6, consolidated: false }]).unwrap();
    let t = mem(&s, &a).unwrap();
    assert_eq!((t.last_recall, t.stability, t.consolidated), (5, 1e6, false));
    assert_eq!((t.text.as_str(), &t.vector), (before.text.as_str(), &before.vector));

    // New text without a vector, as a synced row arrives; reindex embeds it.
    let synced = Trace { text: "osaka summer harbor".into(), model_id: String::new(), vector: Vec::new(), ..before.clone() };
    s.apply(&[Write::Insert(&synced)]).unwrap();
    assert_eq!(rows(&s).len(), 2);
    assert_eq!(mem(&s, &a).unwrap().text, "osaka summer harbor");
    assert_eq!(s.reindex(8).unwrap(), 1);
    assert!(s.recall("osaka summer harbor").unwrap().recalled.iter().any(|r| r.trace.id == a));

    s.apply(&[Write::Remove(&b)]).unwrap();
    assert!(mem(&s, &b).is_none());
    assert_eq!(rows(&s).len(), 1);
}

#[test]
fn an_applied_insert_or_removal_ends_the_citation_window_of_its_row() {
    let mut s = engine();
    let a = insert(&mut s, "kyoto autumn temple", 1.0, "");
    let b = insert(&mut s, "kyoto autumn garden", 1.0, "");
    let c = insert(&mut s, "kyoto autumn bridge", 1.0, "");
    let reply = format!("《id:{a}》《id:{b}》《id:{c}》");
    s.recall("kyoto autumn").unwrap();
    let rewritten = Trace { text: "kyoto autumn temple, rewritten".into(), ..mem(&s, &a).unwrap() };
    s.apply(&[Write::Insert(&rewritten), Write::Remove(&b)]).unwrap();
    assert_eq!(s.cite(&reply).unwrap(), [c]);
}

#[test]
fn count_up_to_is_the_count_capped() {
    let st = Arc::new(Mutex::new(Store::in_memory().unwrap()));
    let mut store = st.lock().unwrap();
    let rows: Vec<Trace> = (0..30)
        .map(|i| Trace {
            id: format!("{i:03}"),
            text: format!("fact {i}"),
            created_at: 100 + i,
            tz: "UTC;+00:00".into(),
            last_recall: 100,
            stability: 1e5,
            consolidated: true,
            cue: String::new(),
            model_id: String::new(),
            vector: Vec::new(),
        })
        .collect();
    store.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    for view in [View::All, View::Young { created_after: 114 }, View::Young { created_after: 1_000 }, View::Text("fact 3")] {
        let n = store.count(view).unwrap();
        for limit in [0, 1, 2, n, n + 1, usize::MAX] {
            assert_eq!(store.count_up_to(view, limit).unwrap(), n.min(limit), "{view:?} up to {limit}");
        }
    }
}

fn unit(seed: u64, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|d| ((seed * 31 + d as u64 * 17) % 23) as f32 - 11.0).collect();
    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    v.iter_mut().for_each(|a| *a /= n);
    v
}

#[test]
fn rewriting_a_row_with_the_same_vector_only_moves_its_state() {
    let mut st = Store::in_memory().unwrap();
    let t = |id: &str, vector: Vec<f32>| Trace {
        id: id.into(),
        text: format!("fact {id}"),
        created_at: 10,
        tz: "UTC;+00:00".into(),
        last_recall: 100,
        stability: 2f64.powi(12),
        consolidated: true,
        cue: String::new(),
        model_id: if vector.is_empty() { String::new() } else { "m".into() },
        vector,
    };
    let a = t("a", unit(1, 8));
    st.write(&[Write::Insert(&a), Write::Insert(&t("b", unit(2, 8)))]).unwrap();

    // Same text, time and vector; new state, cue and timezone.
    let moved = Trace { last_recall: 500, stability: 2f64.powi(20), consolidated: false, cue: "when".into(), tz: "Asia/Tokyo;+09:00".into(), ..a.clone() };
    st.write(&[Write::Insert(&moved)]).unwrap();
    assert_eq!(st.find(View::Id("a"), 1).unwrap(), [moved.clone()]);
    assert_eq!(st.count(View::All).unwrap(), 2);
    assert!(st.find(View::Band { band: a.band(), created_by: 10 }, 9).unwrap().iter().all(|r| r.id != "a"));
    assert_eq!(st.find(View::Band { band: moved.band(), created_by: 10 }, 9).unwrap()[0].id, "a");
    assert_eq!(st.find(View::Labile { model_id: "m", after: "" }, 9).unwrap()[0].id, "a");
    assert_eq!(st.find(View::Text("fact a"), 9).unwrap().len(), 1);
    assert_eq!(st.near(&a.vector, "m", 0.99, 5, None).unwrap()[0].trace, moved);

    // Back to consolidated: it leaves the dream queue.
    st.write(&[Write::Insert(&a)]).unwrap();
    assert!(st.find(View::Labile { model_id: "m", after: "" }, 9).unwrap().is_empty());

    // A different vector, a different text and a dropped vector take the full path.
    let revec = Trace { vector: unit(3, 8), ..a.clone() };
    st.write(&[Write::Insert(&revec)]).unwrap();
    assert_eq!(st.near(&revec.vector, "m", 0.99, 5, None).unwrap()[0].trace, revec);
    assert!(st.near(&a.vector, "m", 0.99, 5, None).unwrap().is_empty());
    let retext = Trace { text: "other".into(), ..revec.clone() };
    st.write(&[Write::Insert(&retext)]).unwrap();
    assert!(st.find(View::Text("fact a"), 9).unwrap().is_empty());
    assert_eq!(st.find(View::Text("other"), 9).unwrap().len(), 1);
    let bare = Trace { model_id: String::new(), vector: Vec::new(), ..retext };
    st.write(&[Write::Insert(&bare)]).unwrap();
    assert!(st.near(&revec.vector, "m", 0.99, 5, None).unwrap().is_empty());
    assert_eq!(st.find(View::Stale { model_id: "m" }, 9).unwrap()[0].id, "a");
}
