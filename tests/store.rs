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
    assert_eq!(ids(st.find(View::Labile { model_id: "m" }, 25).unwrap()), want(&|t| !t.consolidated && t.model_id == "m", &by_id));
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
