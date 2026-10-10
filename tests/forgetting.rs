//! The forgetting curve and the spacing effect (SPEC §3).

mod support;

use support::mem;
use support::{build, cfg, insert, DAY};

#[test]
fn retrievability_halves_every_stability() {
    let (mut s, _) = build(cfg());
    let id = insert(&mut s, "half life", 1.0, "");
    let m = mem(&s, &id).unwrap().clone();
    let t0 = m.last_recall as f64;
    assert_eq!(s.retrievability(&m, t0), 1.0);
    assert!((s.retrievability(&m, t0 + DAY) - 0.5).abs() < 1e-9);
    assert!((s.retrievability(&m, t0 + 2.0 * DAY) - 0.25).abs() < 1e-9);
    assert_eq!(s.retrievability(&m, t0 - 1000.0), 1.0); // clock rollback
    assert_eq!(s.retrievability(&m, t0 + DAY * 100_000.0), 0.0); // underflow is exactly 0
}

#[test]
fn spaced_beats_massed() {
    let (mut s, clock) = build(cfg());
    let massed = insert(&mut s, "massed alpha", 1.0, "");
    let spaced = insert(&mut s, "spaced beta", 1.0, "");
    for _ in 0..3 {
        clock.advance(60.0);
        s.recall("massed alpha").unwrap();
    }
    clock.advance(3.0 * DAY);
    s.recall("spaced beta").unwrap();
    assert!(mem(&s, &massed).unwrap().stability < DAY * 1.05);
    assert!(mem(&s, &spaced).unwrap().stability > DAY * 2.0);
}

#[test]
fn no_immortal_memory() {
    let (mut s, clock) = build(cfg());
    let id = insert(&mut s, "rehearsed daily", 1.0, "");
    for _ in 0..400 {
        clock.advance(30.0 * DAY);
        s.recall("rehearsed daily").unwrap();
    }
    let m = mem(&s, &id).unwrap();
    assert_eq!(m.stability, cfg().max_stability);
    assert!(s.retrievability(&m, clock.get() + 100.0 * 365.0 * DAY) < 0.001);
}

#[test]
fn strength_orders_stable_above_fresh_junk() {
    let (mut s, clock) = build(cfg());
    let fact = insert(&mut s, "stable fact", 10.0, "");
    clock.advance(5.0 * DAY);
    let junk = insert(&mut s, "junk", 1.0, "");
    let now = clock.get();
    assert!(s.strength(&mem(&s, &fact).unwrap(), now) > s.strength(&mem(&s, &junk).unwrap(), now));
}

#[test]
fn dormant_but_relevant_still_competes() {
    let (mut s, clock) = build(cfg());
    insert(&mut s, "rust database project", 1.0, "");
    clock.advance(3650.0 * DAY);
    let r = &s.recall("rust database").unwrap().recalled[0];
    assert!(r.retrievability < 1e-6 && r.score > 0.25);
}

#[test]
fn peek_leaves_no_trace() {
    let (mut s, clock) = build(cfg());
    let id = insert(&mut s, "ユーザーは抹茶味のアイスクリームが好き", 1.0, "");
    clock.advance(2.0 * DAY);
    let before = mem(&s, &id).unwrap().clone();
    assert!(!s.peek("抹茶アイス").unwrap().recalled.is_empty());
    assert_eq!(mem(&s, &id).unwrap(), before);
    assert!(s.cite(&format!("《id:{id}》")).unwrap().is_empty()); // a peek opens no citation window
    s.recall("抹茶アイス").unwrap(); // a real recall does strengthen
    assert!(mem(&s, &id).unwrap().stability > before.stability);
}
