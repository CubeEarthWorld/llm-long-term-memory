//! The id generator seam: the engine never reuses an id, whatever the generator returns.

mod support;

use support::{mem, rows};
use engram::{DreamAction, DreamDecision, DreamRequest, Result};
use support::{build, cfg, insert};

#[test]
fn a_gist_that_repeats_its_seed_keeps_a_distinct_id() {
    let (mut s, _) = build(cfg());
    // Derived from text and time only (the salt grows only while the same request is
    // repeated): a gist repeating its seed's text in the same millisecond (frozen clock)
    // first gets the seed's id.
    let mut last: Option<(i64, String, u32)> = None;
    let made = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = made.clone();
    s.set_ids(Box::new(move |ms, text| {
        let salt = match &last {
            Some((m, t, n)) if *m == ms && t == text => n + 1,
            _ => 0,
        };
        last = Some((ms, text.to_owned(), salt));
        let id = format!("{ms:013}-{}-{salt}", text.replace(' ', "_"));
        log.lock().unwrap().push(id.clone());
        id
    }));
    insert(&mut s, "trip kyoto plan schedule note", 1.0, "");
    let seed = insert(&mut s, "trip kyoto plan schedule notes", 1.0, "");
    insert(&mut s, "zebra unrelated", 1.0, ""); // so the gist's request is not a repeat
    let text = mem(&s, &seed).unwrap().text.clone();
    let mut same = |r: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: vec![text.clone()], absorbed_ids: r.members[1..].iter().map(|m| m.id.clone()).collect() })
    };
    let reports = s.dream(&mut same, None).unwrap();
    assert_eq!(reports[0].action, DreamAction::Replace);
    let gist = &reports[0].after[0];
    assert_ne!(gist.id, seed);
    let made = made.lock().unwrap().clone();
    assert_eq!(made.iter().filter(|id| **id == seed).count(), 2, "{made:?}"); // the collision happened
    assert!(mem(&s, &gist.id).is_some(), "the gist was deleted with the members it replaced");
    assert_eq!(rows(&s).len(), 2);
}

#[test]
fn a_generator_stuck_on_used_ids_is_an_error_not_a_hang() {
    let (mut s, _) = build(cfg());
    s.set_ids(Box::new(|_, _| "SAME".into()));
    insert(&mut s, "first fact", 1.0, "");
    assert!(s.remember("second fact", 1.0, "").is_err());
}
