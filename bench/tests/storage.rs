//! The SQLite store.

#[path = "../../tests/support/mod.rs"]
mod support;

use engram::{Store, Trace, Write};
use ltm::SqliteStore;
use support::{build_with, cfg, insert, merge_to_gist, temp_dir, FakeEmbedder};

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
fn dream_snapshots_the_store_first() {
    let dir = temp_dir("backup");
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(SqliteStore::open(&dir.join("m.db")).unwrap()));
    insert(&mut s, "trip kyoto plan schedule note", 1.0, "");
    for _ in 0..10 {
        s.dream(&mut merge_to_gist, None).unwrap();
    }
    assert_eq!(std::fs::read_dir(dir.join("snapshots")).unwrap().count(), 8); // a ring of 8
}
