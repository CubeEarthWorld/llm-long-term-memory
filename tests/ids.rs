//! Reproducible ids (own process: the id seed is global).

mod support;

use engram::{deterministic_ids, DreamAction, DreamDecision, DreamRequest, Result};
use support::{build, cfg, insert};

#[test]
fn a_gist_that_repeats_its_seed_keeps_a_distinct_id() {
    deterministic_ids(42);
    let (mut s, _) = build(cfg());
    insert(&mut s, "trip kyoto plan schedule note", 1.0, "");
    let seed = insert(&mut s, "trip kyoto plan schedule notes", 1.0, "");
    // Same text as the seed, same millisecond (frozen clock): the derived id would collide.
    let text = s.memory(&seed).unwrap().text.clone();
    let mut same = |r: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: vec![text.clone()], absorbed_ids: r.members[1..].iter().map(|m| m.id.clone()).collect() })
    };
    let reports = s.dream(&mut same, None).unwrap();
    deterministic_ids(0);
    assert_eq!(reports[0].action, DreamAction::Replace);
    let gist = &reports[0].after[0];
    assert_ne!(gist.id, seed);
    assert!(s.memory(&gist.id).is_some(), "the gist was deleted with the members it replaced");
    assert_eq!(s.memories().len(), 1);
}
