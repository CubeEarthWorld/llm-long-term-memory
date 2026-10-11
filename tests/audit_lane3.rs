//! Audit lane 3 — the dream (SPEC §5, §7). Public API only.
//! Each test reproduces a defect found in the audit (now fixed) or pins an invariant that
//! holds; see audit/lane3.md.

mod support;

use engram::{DreamAction, DreamDecision, DreamRequest, Engram, Result, Storage, Store, Trace, View, Write};
use support::{build, build_with, cfg, insert, mem, merge_to_gist, rows, CountingEmbedder, FakeEmbedder, SharedStore, DAY};

fn config_budget(b: usize) -> engram::Config {
    engram::Config { dream_budget: b, ..cfg() }
}

/// A raw row, written behind the engine's back.
fn raw_row(id: &str, text: &str, consolidated: bool) -> Trace {
    Trace {
        id: id.into(),
        text: text.into(),
        created_at: support::START as i64,
        tz: "Asia/Tokyo;+09:00".into(),
        last_recall: support::START as i64,
        stability: DAY,
        consolidated,
        cue: String::new(),
        model_id: "fake/token-overlap".into(),
        vector: FakeEmbedder::new().embed("junk filler row"),
    }
}

// ------------------------------------------------------------------ L3-1

/// L3-1: labile rows the engine can never use (empty text, written by an older version or
/// by hand) sit at the head of the queue; the dream pages past them to the usable seeds.
#[test]
fn unusable_labile_rows_do_not_starve_the_queue() {
    let db = SharedStore::default();
    let (mut s, _) = build_with(config_budget(1), Box::new(FakeEmbedder::new()), Box::new(db.clone()));
    let mut raw = db.clone();
    for i in 0..8 {
        raw.write(&[Write::Insert(&raw_row(&format!("0000-junk-{i}"), "", false))]).unwrap();
    }
    insert(&mut s, "pair alpha gamma one", 1.0, "");
    insert(&mut s, "pair alpha gamma two", 1.0, "");
    let reports = s.dream(&mut merge_to_gist, Some(1)).unwrap();
    assert!(reports.iter().any(|r| r.action == DreamAction::Replace), "usable seed starved by unusable labile rows: {reports:?}");
}

/// L3-1: with 16 undecidable seeds ahead (fallback-vector path: query embedder offline) the
/// cyclic cursor still reaches a decidable seed behind them.
#[test]
fn undecidable_fallback_seeds_do_not_starve_later_seeds() {
    let (e, _) = CountingEmbedder::new(true, None);
    let (mut s, _) = build_with(config_budget(1), Box::new(e), Box::new(Store::in_memory().unwrap()));
    for i in 0..16 {
        insert(&mut s, &format!("u{i}a v{i}b w{i}c"), 1.0, &format!("x{i}d y{i}e"));
    }
    insert(&mut s, "pair alpha gamma one", 1.0, "");
    insert(&mut s, "pair alpha gamma two", 1.0, "");
    let mut replaced = false;
    for _ in 0..20 {
        replaced |= s.dream(&mut merge_to_gist, Some(1)).unwrap().iter().any(|r| r.action == DreamAction::Replace);
    }
    assert!(replaced, "the decidable pair behind 16 undecidable seeds was never examined in 20 dreams");
}

/// L3-1: the same through the LLM-failure path.
#[test]
fn failing_verdicts_do_not_starve_later_seeds() {
    let (mut s, _) = build(config_budget(1));
    for i in 0..16 {
        insert(&mut s, &format!("poison{i} xa{i} xb{i} xc{i}"), 1.0, "");
        insert(&mut s, &format!("poison{i} xa{i} xb{i} xd{i}"), 1.0, "");
    }
    insert(&mut s, "good beta zeta eta", 1.0, "");
    insert(&mut s, "good beta zeta theta", 1.0, "");
    let mut picky = |r: &DreamRequest| -> Result<DreamDecision> {
        if r.members.iter().any(|m| m.text.starts_with("poison")) { Err("unparseable".into()) } else { merge_to_gist(r) }
    };
    let mut replaced = false;
    for _ in 0..40 {
        replaced |= s.dream(&mut picky, Some(1)).unwrap().iter().any(|r| r.action == DreamAction::Replace);
    }
    assert!(replaced, "the good pair behind 16 poison seeds was never examined in 40 dreams");
}

/// Holds: failures that fit the window rotate, and the 8 cued + pair case of the
/// existing suite passes with a budget larger than the config's too.
#[test]
fn seeds_window_within_cap_rotates() {
    let (e, _) = CountingEmbedder::new(true, None);
    let (mut s, _) = build_with(config_budget(1), Box::new(e), Box::new(Store::in_memory().unwrap()));
    for i in 0..8 {
        insert(&mut s, &format!("u{i}a v{i}b w{i}c"), 1.0, &format!("x{i}d y{i}e"));
    }
    insert(&mut s, "pair alpha gamma one", 1.0, "");
    insert(&mut s, "pair alpha gamma two", 1.0, "");
    assert!(s.dream(&mut merge_to_gist, Some(1)).unwrap().is_empty());
    assert_eq!(s.dream(&mut merge_to_gist, Some(1)).unwrap()[0].action, DreamAction::Replace);
}

// ------------------------------------------------------------------ L3-2

fn groups(s: &mut Engram, n: usize) {
    for i in 0..n {
        insert(s, &format!("grp{i} aa{i} bb{i} cc{i}"), 1.0, "");
        insert(s, &format!("grp{i} aa{i} bb{i} dd{i}"), 1.0, "");
    }
}

#[test]
fn llm_calls_never_exceed_budget_and_each_seed_is_judged_once() {
    let (mut s, _) = build(cfg());
    groups(&mut s, 6);
    let mut seen: Vec<String> = Vec::new();
    let mut calls = 0usize;
    let mut spy = |r: &DreamRequest| -> Result<DreamDecision> {
        calls += 1;
        seen.push(r.members[0].id.clone());
        Ok(DreamDecision::keep())
    };
    let reports = s.dream(&mut spy, Some(3)).unwrap();
    assert_eq!((reports.len(), calls), (3, 3));
    let mut uniq = seen.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), seen.len());
    // A failing adjudicator costs a call each too, but never more than the budget.
    let mut calls2 = 0usize;
    let mut fail = |_: &DreamRequest| -> Result<DreamDecision> {
        calls2 += 1;
        Err("down".into())
    };
    let r2 = s.dream(&mut fail, Some(2)).unwrap();
    assert_eq!((r2.len(), calls2), (2, 2));
}

#[test]
fn seeds_settled_without_candidates_cost_nothing() {
    let (mut s, clock) = build(cfg());
    for i in 0..3 {
        let x = insert(&mut s, &format!("topic{i} aa{i} bb{i} cc{i}"), 1.0, "");
        insert(&mut s, &format!("topic{i} aa{i} bb{i} dd{i}"), 1.0, ""); // labile: x reactivated
        s.forget(&x).unwrap(); // nothing left to find
    }
    clock.advance(1.0);
    groups(&mut s, 1);
    let mut calls = 0;
    let mut spy = |r: &DreamRequest| -> Result<DreamDecision> {
        calls += 1;
        merge_to_gist(r)
    };
    let reports = s.dream(&mut spy, Some(1)).unwrap();
    assert_eq!(calls, 1);
    assert!(reports.len() == 1 && reports[0].action == DreamAction::Replace);
    assert!(rows(&s).iter().all(|m| m.consolidated), "settled seeds are consolidated by the one final write");
}

#[test]
fn fallback_seed_costs_no_budget() {
    let (e, _) = CountingEmbedder::new(true, None);
    let (mut s, _) = build_with(cfg(), Box::new(e), Box::new(Store::in_memory().unwrap()));
    let mut cued = Vec::new();
    for i in 0..3 {
        cued.push(insert(&mut s, &format!("u{i}a v{i}b w{i}c"), 1.0, &format!("x{i}d y{i}e")));
    }
    insert(&mut s, "pair alpha gamma one", 1.0, "");
    insert(&mut s, "pair alpha gamma two", 1.0, "");
    let reports = s.dream(&mut merge_to_gist, Some(1)).unwrap();
    assert!(reports.len() == 1 && reports[0].action == DreamAction::Replace);
    assert!(cued.iter().all(|id| !mem(&s, id).unwrap().consolidated));
}

// ------------------------------------------------------------------ L3-3

#[test]
fn clusters_are_bounded_directed_and_exclude_the_seed() {
    let c = engram::Config { dream_max_members: 3, ..cfg() };
    let (mut s, clock) = build(c);
    for i in 0..7 {
        insert(&mut s, &format!("topic alpha beta gamma w{i}"), 1.0, "");
        clock.advance(DAY);
    }
    let mut reqs: Vec<Vec<String>> = Vec::new();
    let mut spy = |r: &DreamRequest| -> Result<DreamDecision> {
        reqs.push(r.members.iter().map(|m| m.id.clone()).collect());
        Ok(DreamDecision::keep())
    };
    s.dream(&mut spy, Some(20)).unwrap();
    assert!(!reqs.is_empty());
    for ids in &reqs {
        assert!(ids.len() <= 3);
        let mut u = ids.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), ids.len(), "seed is its own candidate or duplicate");
        let seed_at = mem(&s, &ids[0]).unwrap().created_at;
        for id in &ids[1..] {
            assert!(mem(&s, id).unwrap().created_at <= seed_at, "a candidate was stated after the seed");
        }
    }
    // the preview agrees
    let (mut p, clk) = build(engram::Config { dream_max_members: 3, ..cfg() });
    for i in 0..7 {
        insert(&mut p, &format!("topic alpha beta gamma w{i}"), 1.0, "");
        clk.advance(DAY);
    }
    assert!(p.clusters(Some(20)).unwrap().iter().all(|c| c.len() <= 3));
}

#[test]
fn a_gist_inherits_created_at_so_later_updates_reach_it() {
    let (mut s, clock) = build(cfg());
    insert(&mut s, "user lives in tokyo city", 1.0, "");
    clock.advance(30.0 * DAY);
    let osaka = insert(&mut s, "user lives in osaka city", 1.0, "");
    let osaka_at = mem(&s, &osaka).unwrap().created_at;
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    let gist = mem(&s, &reports[0].after[0].id).unwrap();
    assert_eq!(gist.created_at, osaka_at);
    clock.advance(DAY);
    insert(&mut s, "user lives in nagoya city", 1.0, "");
    let cl = s.clusters(None).unwrap();
    assert!(cl.len() == 1 && cl[0].iter().any(|t| t.id == gist.id));
}

// ------------------------------------------------------------------ L3-4

fn snapshot_world() -> (Engram, engram::ManualClock) {
    let (mut s, clock) = build(cfg());
    let a = insert(&mut s, "trip kyoto plan schedule note", 2.0, "");
    insert(&mut s, "trip kyoto hotel schedule note", 1.0, "");
    clock.advance(2.0 * DAY);
    s.recall("trip kyoto plan schedule note").unwrap();
    let _ = a;
    clock.advance(5.0 * DAY);
    insert(&mut s, "trip kyoto food schedule note", 1.0, "");
    insert(&mut s, "completely different topic zebra", 1.0, "");
    (s, clock)
}

#[test]
fn undo_restores_absorbed_rows_exactly_and_removes_gists() {
    let (mut s, clock) = snapshot_world();
    let before = rows(&s);
    clock.advance(DAY);
    let reports = s.dream(&mut merge_to_gist, Some(1)).unwrap();
    assert_eq!(reports[0].action, DreamAction::Replace);
    let rep = &reports[0];
    // Reinforce the gist and then undo: strengthening is lost, rows return as they were.
    assert!(matches!(s.remember(&rep.after[0].text, 1.0, "").unwrap(), engram::Remembered::Reinforced(_)));
    s.undo(rep).unwrap();
    let after: Vec<Trace> = rows(&s);
    // the seed was labile before; the other rows were consolidated; compare all fields
    let brief = |v: &[Trace]| v.iter().map(|t| (t.text.clone(), t.stability, t.last_recall, t.consolidated)).collect::<Vec<_>>();
    assert_eq!(brief(&after), brief(&before));
    assert_eq!(after, before);
    assert!(rep.after.iter().all(|g| mem(&s, &g.id).is_none()));
    // idempotent
    s.undo(rep).unwrap();
    assert_eq!(rows(&s), before);
}

#[test]
fn undo_after_the_gist_was_forgotten_still_restores() {
    let (mut s, _) = snapshot_world();
    let before = rows(&s);
    let reports = s.dream(&mut merge_to_gist, Some(1)).unwrap();
    s.forget(&reports[0].after[0].id).unwrap();
    s.undo(&reports[0]).unwrap();
    assert_eq!(rows(&s), before);
}

#[test]
fn undo_of_keep_makes_the_seed_labile_again() {
    let (mut s, _) = snapshot_world();
    let before = rows(&s);
    let reports = s.dream(&mut support::always_keep, Some(1)).unwrap();
    assert_eq!(reports[0].action, DreamAction::Keep);
    s.undo(&reports[0]).unwrap();
    assert_eq!(rows(&s), before);
    assert!(!s.clusters(None).unwrap().is_empty());
}

#[test]
fn capacity_is_restored_by_the_next_write_after_undo() {
    let c = engram::Config { capacity: 3, ..cfg() };
    let (mut s, clock) = build(c);
    insert(&mut s, "trip kyoto plan schedule note", 1.0, "");
    insert(&mut s, "trip kyoto hotel schedule note", 1.0, "");
    insert(&mut s, "trip kyoto food schedule note", 1.0, "");
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    assert_eq!(rows(&s).len(), 1);
    insert(&mut s, "zebra one unrelated words", 1.0, "");
    insert(&mut s, "yak two unrelated tokens", 1.0, "");
    assert!(rows(&s).len() <= 3);
    s.undo(&reports[0]).unwrap();
    clock.advance(10.0 * DAY);
    insert(&mut s, "newt three other letters", 1.0, "");
    assert!(rows(&s).len() <= 3, "{}", rows(&s).len());
}

#[test]
fn memories_empty_with_named_ids_is_a_keep_that_touches_no_candidate() {
    let (mut s, _) = snapshot_world();
    let before = rows(&s);
    let mut weird = |r: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: vec![], absorbed_ids: r.members[1..].iter().map(|m| m.id.clone()).collect() })
    };
    let reports = s.dream(&mut weird, Some(1)).unwrap();
    assert_eq!(reports[0].action, DreamAction::Keep);
    assert!(reports[0].absorbed.is_empty() && reports[0].after.is_empty());
    let now = rows(&s);
    assert_eq!(now.len(), before.len());
    assert!(now.iter().filter(|t| !t.consolidated).count() < before.iter().filter(|t| !t.consolidated).count());
}

#[test]
fn replace_naming_only_unknown_ids_rewrites_the_seed_alone() {
    let (mut s, _) = snapshot_world();
    let n = rows(&s).len();
    let mut d = |r: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: vec![r.members[0].text.clone() + " again"], absorbed_ids: vec!["nope".into(), r.members[0].id.clone()] })
    };
    let reports = s.dream(&mut d, Some(1)).unwrap();
    assert_eq!(reports[0].action, DreamAction::Replace);
    assert_eq!((reports[0].absorbed.len(), reports[0].after.len(), rows(&s).len()), (1, 1, n));
}

#[test]
fn gist_strength_is_conserved_and_bounded() {
    let (mut s, clock) = snapshot_world();
    clock.advance(DAY);
    let now = s.now();
    let cluster = s.clusters(None).unwrap().remove(0);
    let sum: f64 = cluster.iter().map(|t| s.strength(t, now)).sum();
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    let g = &reports[0].after[0];
    let gs = s.strength(g, now);
    assert!(g.stability >= 1.0 && g.stability <= s.config().max_stability && g.last_recall <= now as i64);
    assert!((gs - sum.min(g.stability)).abs() <= 1e-6 * sum.max(1.0) + g.stability * 1e-4, "gist {gs} vs cluster {sum}");
}

// ------------------------------------------------------------------ L3-5

#[test]
fn a_settled_seed_absorbed_by_a_later_cluster_is_not_resurrected() {
    let (mut s, clock) = build(cfg());
    let x = insert(&mut s, "kyoto trip plan alpha", 1.0, "");
    let s1 = insert(&mut s, "kyoto trip plan beta", 1.0, ""); // labile: x reactivated
    s.forget(&x).unwrap();
    clock.advance(10.0);
    let s2 = insert(&mut s, "kyoto trip plan gamma", 1.0, ""); // labile: reaches s1
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    assert!(reports.iter().all(|r| r.action != DreamAction::Error), "{reports:?}");
    assert!(mem(&s, &s1).is_none() && mem(&s, &s2).is_none());
    assert_eq!(rows(&s).len(), 1);
    assert!(rows(&s)[0].consolidated);
}

#[test]
fn seeds_that_vanish_or_settle_meanwhile_are_skipped() {
    let db = SharedStore::default();
    let (mut s, _) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(db.clone()));
    groups(&mut s, 3);
    let seeds: Vec<Trace> = rows(&s).into_iter().filter(|t| !t.consolidated).collect();
    assert_eq!(seeds.len(), 3);
    let (gone, done) = (seeds[1].clone(), seeds[2].clone());
    let mut raw = db.clone();
    let mut calls = 0;
    let mut adj = |r: &DreamRequest| -> Result<DreamDecision> {
        calls += 1;
        if calls == 1 {
            raw.write(&[Write::Remove(&gone.id), Write::State { id: &done.id, last_recall: done.last_recall, stability: done.stability, consolidated: true }]).unwrap();
        }
        { let _ = r; Ok(DreamDecision::keep()) }
    };
    let reports = s.dream(&mut adj, Some(5)).unwrap();
    assert_eq!((reports.len(), calls), (1, 1));
}

#[test]
fn unnamed_candidates_stay_untouched_even_when_the_seed_is_rewritten() {
    let (mut s, clock) = build(cfg());
    insert(&mut s, "user lives in tokyo city", 1.0, "");
    let bystander = insert(&mut s, "user lives in tokyo city with a cat", 1.0, "");
    clock.advance(30.0 * DAY);
    insert(&mut s, "user lives in osaka city", 1.0, "");
    let before = mem(&s, &bystander).unwrap();
    let mut d = |r: &DreamRequest| -> Result<DreamDecision> {
        if r.members[0].text != "user lives in osaka city" {
            return Ok(DreamDecision::keep());
        }
        Ok(DreamDecision { memories: vec!["user lives in osaka city today".into()], absorbed_ids: r.members.iter().skip(1).filter(|m| m.text == "user lives in tokyo city").map(|m| m.id.clone()).collect() })
    };
    s.dream(&mut d, None).unwrap();
    let now = mem(&s, &bystander).unwrap();
    // seeds keep/replace never rewrite a candidate that was not named (it may have been a seed itself)
    assert_eq!((now.text, now.stability, now.last_recall), (before.text, before.stability, before.last_recall));
}

/// GAP (L3-4): `undo` is only exact in reverse order. Two reports of one dream where the
/// second absorbed the first's gist: undoing them in dream order removes nothing for the
/// first (its gist is gone), re-inserts its originals, and leaves the second gist standing
/// on top of them: the same facts twice, no error.
#[test]
fn undo_all_restores_a_whole_dream() {
    let (mut s, _) = snapshot_world();
    let before = rows(&s);
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    assert!(reports.len() >= 2 && reports.iter().all(|r| r.action == DreamAction::Replace));
    s.undo_all(&reports).unwrap();
    assert_eq!(rows(&s), before, "undo_all left the later gist beside restored originals");
}

/// GAP (L3-4): the bundled prompt tells the LLM to split a seed that packs several facts
/// into independent memories (ids empty, several memories), but `|texts| > |absorbed|`
/// is read as keep (SPEC §5, the capacity guard), so the split is silently dropped and
/// the seed is consolidated unsplit. Spec and prompt disagree.
#[test]
#[ignore = "accepted: a dream never adds traces (capacity guard, SPEC §5); the prompt's split hint is only honoured when the count does not grow"]
fn gap_split_of_a_compound_seed_is_kept() {
    let (mut s, _) = snapshot_world();
    let mut split = |r: &DreamRequest| -> Result<DreamDecision> {
        Ok(DreamDecision { memories: vec![format!("{} part one", r.members[0].text), format!("{} part two", r.members[0].text)], absorbed_ids: vec![] })
    };
    assert_eq!(s.dream(&mut split, Some(1)).unwrap()[0].action, DreamAction::Replace);
}

#[test]
fn undo_in_reverse_order_is_exact() {
    let (mut s, _) = snapshot_world();
    let before = rows(&s);
    let reports = s.dream(&mut merge_to_gist, None).unwrap();
    assert!(reports.len() >= 2);
    for r in reports.iter().rev() {
        s.undo(r).unwrap();
    }
    assert_eq!(rows(&s), before);
}



#[allow(dead_code)]
fn _unused(_: &dyn Storage, _: View<'_>) {}
