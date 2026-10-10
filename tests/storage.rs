//! The SQLite substrate and the parameter set.
#![cfg(feature = "sqlite")]

mod support;

use engram::{Config, SqliteStore, Store, Trace, Write};
use serde_json::json;
use support::temp_dir;

fn trace(id: &str, consolidated: bool) -> Trace {
    Trace {
        id: id.into(),
        text: "t".into(),
        created_at: 1,
        tz: "UTC;+00:00".into(),
        last_recall: 1,
        stability: 2.0,
        consolidated,
        cue: String::new(),
        model_id: "m".into(),
        vector: vec![1.0, 0.0],
    }
}

#[test]
fn store_skips_unreadable_rows_and_keeps_text_with_a_broken_vector() {
    // SQLite is dynamically typed: one hand-edited row must not abort the whole load. A
    // row without a usable stability is skipped; one whose vector is not whole float32s
    // keeps its (canonical) text and loads unindexed.
    let path = temp_dir("corrupt").join("corrupt.db");
    let mut st = SqliteStore::open(&path).unwrap();
    let (a, b, c) = (trace("a", true), trace("b", true), trace("c", true));
    st.write(&[Write::Put(&a), Write::Put(&b), Write::Put(&c)]).unwrap();
    st.connection().execute("UPDATE memory SET stability='not-a-number' WHERE id='b'", []).unwrap();
    st.connection().execute("UPDATE memory SET vector=X'010203' WHERE id='c'", []).unwrap(); // not whole float32s
    let rows = st.load_all().unwrap();
    assert_eq!(rows.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(), ["a", "c"]);
    assert!(rows[1].vector.is_empty());
}

#[test]
fn store_round_trip_is_atomic_and_snapshots_rotate() {
    let dir = temp_dir("store");
    let mut st = SqliteStore::open(&dir.join("s.db")).unwrap();
    st.write(&[Write::Put(&trace("b", false)), Write::Put(&trace("a", true))]).unwrap();
    let rows = st.load_all().unwrap();
    assert_eq!(rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["a", "b"]); // id order
    assert!(rows[0].consolidated && rows[0].vector == [1.0, 0.0] && !rows[1].consolidated);
    // A failing write (NULL text violates NOT NULL) rolls back the whole batch.
    st.connection().execute_batch("CREATE TRIGGER boom BEFORE DELETE ON memory WHEN old.id = 'b' BEGIN SELECT RAISE(ABORT, 'boom'); END;").unwrap();
    assert!(st.write(&[Write::Remove("a"), Write::Remove("b")]).is_err());
    assert_eq!(st.load_all().unwrap().len(), 2);
    st.connection().execute_batch("DROP TRIGGER boom;").unwrap();
    st.write(&[Write::Remove("a")]).unwrap();
    assert_eq!(st.load_all().unwrap().iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["b"]);
    for _ in 0..3 {
        assert!(st.backup().unwrap().is_some());
    }
    assert_eq!(std::fs::read_dir(dir.join("snapshots")).unwrap().count(), 3);
    assert!(SqliteStore::open_in_memory().unwrap().backup().unwrap().is_none());
}

#[test]
fn config_round_trip_and_validation() {
    let mut c = Config::default();
    c.merge_json(&json!({"capacity": "42", "alpha": 0.5, "max_cues": 3.9, "unknown": 1, "inject_n": "junk"}));
    assert_eq!((c.capacity, c.alpha, c.max_cues, c.inject_n), (42, 0.5, 3, 5));
    let mut d = Config::default();
    d.merge_json(&c.to_json());
    assert_eq!(d, c);
    assert_eq!(c.to_json().as_object().unwrap().keys().next().unwrap(), "capacity"); // declaration order
    assert!(Config { theta_related: 1.5, ..Config::default() }.validate().is_err());
    let mut zero = Config::default();
    zero.merge_json(&json!({"capacity": -3}));
    assert!(zero.validate().is_err());
    assert!(Config::default().validate().is_ok());
}
