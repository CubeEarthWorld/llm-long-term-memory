//! The sleep phase (SPEC §5).

mod support;

use support::{mem, rows};
use engram::{DreamAction, DreamDecision, DreamRequest, Engram, ManualClock, Result};
use support::{always_fail, always_keep, build, build_with, cfg, insert, merge_to_gist, BrokenEmbedder, FakeEmbedder, SharedStore, DAY};

fn seeded() -> (Engram, ManualClock) {
    let (mut s, clock) = build(cfg());
    for t in ["trip kyoto plan schedule note", "trip kyoto hotel schedule note", "trip kyoto food schedule note", "completely different topic zebra"] {
        insert(&mut s, t, 1.0, "");
    }
    (s, clock)
}

#[test]
fn replace_merges_into_gist_with_live_evidence() {
    let (mut s, _) = seeded();
    let first = rows(&s)[0].clone();
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    assert!(reports.len() == 1 && reports[0].action == DreamAction::Replace && reports[0].before.len() == 3);
    let gist = mem(&s, &reports[0].after[0].id).unwrap();
    assert!(gist.consolidated && (gist.stability - 3.0 * DAY).abs() < 1.0 && gist.last_recall == first.last_recall);
    assert!(rows(&s).len() == 2 && rows(&s).iter().all(|m| m.consolidated));
    assert!(s.dream(&mut merge_to_gist, None).unwrap().is_empty()); // settled store ⇒ no LLM calls
}

#[test]
fn keep_marks_consolidated_without_touching_strength() {
    let (mut s, _) = seeded();
    let before = rows(&s).to_vec();
    assert_eq!(s.dream(&mut always_keep, None).unwrap()[0].action, DreamAction::Keep);
    for (m, b) in rows(&s).iter().zip(&before) {
        assert!(m.consolidated && m.stability == b.stability && m.last_recall == b.last_recall);
    }
}

#[test]
fn error_leaves_seed_labile() {
    let (mut s, _) = seeded();
    let reports = s.dream(&mut always_fail, None).unwrap();
    assert_eq!((reports[0].action, reports[0].error.as_deref()), (DreamAction::Error, Some("llm down")));
    assert!(!s.clusters(None).unwrap().is_empty());
    assert_eq!(s.dream(&mut merge_to_gist, None).unwrap()[0].action, DreamAction::Replace);
}

#[test]
fn confabulation_guard() {
    let mut halluc = |_: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: vec!["zzz qqq unrelated hallucination".into()], absorbed_ids: vec![] })
    };
    let (mut s, _) = seeded();
    assert_eq!(s.dream(&mut halluc, None).unwrap()[0].action, DreamAction::Keep);
    // Names every candidate, so the gist count is the only thing left to reject.
    let mut too_many = |r: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision {
            memories: (0..=r.members.len()).map(|i| format!("trip kyoto {i}")).collect(),
            absorbed_ids: r.members[1..].iter().map(|m| m.id.clone()).collect(),
        })
    };
    let (mut s2, _) = seeded();
    assert_eq!(s2.dream(&mut too_many, None).unwrap()[0].action, DreamAction::Keep);
}

#[test]
fn budget_and_fifo_seeds() {
    let (mut s, _) = build(cfg());
    insert(&mut s, "weak pair one alpha x", 1.0, "");
    insert(&mut s, "weak pair one alpha y", 1.0, "");
    insert(&mut s, "strong pair two beta x", 5.0, "");
    insert(&mut s, "strong pair two beta y", 5.0, "");
    let reports = s.dream(&mut merge_to_gist, Some(1)).unwrap();
    assert!(reports.len() == 1 && reports[0].before[0].text.starts_with("weak"));
    assert_eq!(s.clusters(None).unwrap().len(), 1);
}

#[test]
fn correction_is_seeded_by_the_new_evidence() {
    let mut seen: Vec<String> = Vec::new();
    let mut spy = |r: &DreamRequest| -> Result<DreamDecision> {
        seen = r.members.iter().map(|m| m.id.clone()).collect();
        Ok(DreamDecision::keep())
    };
    let (mut s, clock) = build(cfg());
    let old = insert(&mut s, "user lives in tokyo city", 5.0, "");
    clock.advance(30.0 * DAY);
    let fresh = insert(&mut s, "user lives in osaka city", 1.0, "");
    s.dream(&mut spy, None).unwrap();
    assert_eq!(seen[0], fresh); // the newest evidence seeds
    assert!(seen.contains(&old));
    assert!(mem(&s, &old).unwrap().consolidated); // only offered, not rewritten
}

#[test]
fn only_named_candidates_are_absorbed() {
    let mut picky = |r: &DreamRequest| -> Result<DreamDecision> {
        if r.members[0].text != "user lives in osaka city" {
            return Ok(DreamDecision::keep());
        }
        Ok(DreamDecision {
            memories: vec!["user moved to osaka city".into()],
            absorbed_ids: r.members[1..].iter().filter(|m| m.text == "user lives in tokyo city").map(|m| m.id.clone()).collect(),
        })
    };
    let (mut s, clock) = build(cfg());
    insert(&mut s, "user lives in tokyo city", 1.0, "");
    let bystander = insert(&mut s, "user lives in tokyo city with a cat", 1.0, "");
    clock.advance(30.0 * DAY);
    insert(&mut s, "user lives in osaka city", 1.0, "");
    assert!(s.dream(&mut picky, None).unwrap().iter().any(|r| r.action == DreamAction::Replace));
    assert!(mem(&s, &bystander).is_some()); // not named ⇒ not rewritten
    let texts: Vec<String> = rows(&s).into_iter().map(|m| m.text).collect();
    assert!(texts.contains(&"user moved to osaka city".to_owned()) && !texts.contains(&"user lives in tokyo city".to_owned()));
}

#[test]
fn model_switch_reindexes_from_text() {
    let db = SharedStore::default();
    let open = || Box::new(db.clone());
    let (mut a, _) = build_with(cfg(), Box::new(FakeEmbedder::with(64, "m1")), open());
    insert(&mut a, "persisted across models", 1.0, "");
    drop(a);
    let (mut b, _) = build_with(cfg(), Box::new(FakeEmbedder::with(64, "m2")), open());
    assert!(b.recall("persisted across models").unwrap().recalled.is_empty()); // opening loads nothing
    assert!(b.dream(&mut merge_to_gist, None).unwrap().is_empty()); // the dream re-embeds a batch first
    assert_eq!(rows(&b)[0].model_id, "m2");
    assert_eq!(b.recall("persisted across models").unwrap().recalled.len(), 1);
    drop(b);
    let (mut c, _) = build_with(cfg(), Box::new(BrokenEmbedder), open());
    assert!(c.dream(&mut merge_to_gist, None).unwrap().is_empty() && rows(&c)[0].model_id == "m2");
}

#[test]
fn cue_reaches_the_version_it_supersedes() {
    let (mut s, clock) = build(cfg());
    insert(&mut s, "user lives in tokyo city", 1.0, "");
    clock.advance(30.0 * DAY);
    // The text is far from the old fact (cos ≈ −0.11 < θ_related); only the cue reaches back (≈ 0.90).
    insert(&mut s, "resident of osaka prefecture now", 1.0, "user lives in city");
    let clusters: Vec<Vec<String>> = s.clusters(None).unwrap().into_iter().map(|c| c.into_iter().map(|t| t.text).collect()).collect();
    assert_eq!(clusters, [vec!["resident of osaka prefecture now".to_owned(), "user lives in tokyo city".to_owned()]]);
}

#[test]
fn a_dream_verdict_can_be_undone() {
    let (mut s, _) = seeded();
    let before = rows(&s);
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    assert_eq!(reports[0].action, DreamAction::Replace);
    s.undo(&reports[0]).unwrap(); // the report is the journal: no snapshot needed
    assert_eq!(rows(&s), before);
}
