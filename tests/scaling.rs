//! Bulk eviction takes the weakest traces; the budgeted cluster preview is FIFO.

mod support;

use engram::{Config, Hit, Remembered, Result, Storage, Store, Trace, View, Write};
use std::sync::{Arc, Mutex};
use support::{build_with, cfg, BrokenEmbedder, FakeEmbedder};

const NOW: i64 = 1_700_000_000;

#[allow(clippy::too_many_arguments)]
fn row(id: &str, text: &str, created_at: i64, last_recall: i64, stability: f64, consolidated: bool, model_id: &str, vector: Vec<f32>) -> Trace {
    Trace { id: id.into(), text: text.into(), created_at, tz: "UTC;+00:00".into(), last_recall, stability, consolidated, cue: String::new(), model_id: model_id.into(), vector }
}

/// Records every removed id, in order.
struct Recording(Store, Arc<Mutex<Vec<String>>>);

impl Storage for Recording {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        self.0.find(view, limit)
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        self.0.count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        self.0.near(q, model_id, min_cos, k, stated_by)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        for w in writes {
            if let Write::Remove(id) = w {
                self.1.lock().unwrap().push((*id).to_owned());
            }
        }
        self.0.write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        self.0.clear()
    }
}

#[test]
fn bulk_eviction_takes_the_weakest() {
    for capacity in [1, 10, 40, 119, 120] {
        for mode in ["mixed", "young", "old"] {
            let mut store = Store::in_memory().unwrap();
            // Ids sort in insertion order, as ULIDs do.
            let rows: Vec<Trace> = (0..120)
                .map(|i| {
                    let young = mode == "young" || (mode == "mixed" && i % 3 != 0);
                    row(&format!("{i:03}"), &format!("row {i}"), NOW - if young { 10 } else { 200 }, NOW - (i % 5) * 100, 2f64.powi((i % 8) as i32), true, "", vec![])
                })
                .collect();
            for r in &rows {
                store.write(&[Write::Insert(r)]).unwrap();
            }
            let removed = Arc::new(Mutex::new(Vec::new()));
            let config = Config { capacity, grace_period: 100.0, ..cfg() };
            let (mut s, clock) = build_with(config.clone(), Box::new(BrokenEmbedder), Box::new(Recording(store, removed.clone())));
            clock.set(NOW as f64);
            // Independent reference: past the grace period, weakest first; the young
            // (and the new row) only in a flood or when too few older rows exist.
            let mut left: Vec<Trace> = rows.clone();
            let new = row("new", "new row", NOW, NOW, config.initial_stability, false, "", vec![]);
            let excess = (left.len() + 1).saturating_sub(capacity);
            let is_young = |m: &Trace| m.created_at > NOW - 100;
            let n_young = left.iter().filter(|m| is_young(m)).count() + 1;
            let n_old = left.len() + 1 - n_young;
            let mut pool: Vec<Trace> = if n_young <= capacity / 10 && n_old >= excess {
                left.iter().filter(|m| !is_young(m)).cloned().collect()
            } else {
                left.iter().cloned().chain([new.clone()]).collect()
            };
            let rank = |t: &Trace| t.stability.log2() - ((NOW - t.last_recall).max(0) as f64) / t.stability;
            pool.sort_by(|a, b| rank(a).total_cmp(&rank(b)).then(a.id.cmp(&b.id)));
            let expected: Vec<String> = pool.iter().take(excess).map(|m| m.text.clone()).collect();
            left.push(new);
            left.retain(|m| !expected.contains(&m.text));
            let Remembered::Inserted { evicted, .. } = s.remember("new row", 1.0, "").unwrap() else { panic!() };
            let texts_of = |ids: &[String]| ids.iter().map(|id| rows.iter().find(|r| &r.id == id).map_or("new row".to_owned(), |r| r.text.clone())).collect::<Vec<_>>();
            assert_eq!(texts_of(&removed.lock().unwrap()), expected, "capacity {capacity} {mode}");
            assert_eq!(evicted.map(|t| t.text), expected.last().cloned());
            let mut survivors: Vec<String> = support::rows(&s).into_iter().map(|m| m.text).collect();
            let mut want: Vec<String> = left.iter().map(|m| m.text.clone()).collect();
            survivors.sort();
            want.sort();
            assert_eq!(survivors, want);
        }
    }
}

#[test]
fn clusters_preview_the_seeds_dream_scans_first_in_first_out() {
    let mut fake = FakeEmbedder::new();
    let v = fake.embed("same vector");
    let mut store = Store::in_memory().unwrap();
    for i in 0..100 {
        let t = row(&format!("{i:03}"), &format!("row {i}"), NOW - i % 7, NOW, 86_400.0 + (i % 3) as f64, false, "fake/token-overlap", v.clone());
        store.write(&[Write::Insert(&t)]).unwrap();
    }
    let (mut s, _) = build_with(cfg(), Box::new(fake), Box::new(store));
    let seeds = |cs: Vec<Vec<Trace>>| cs.into_iter().map(|c| c[0].id.clone()).collect::<Vec<_>>();
    let fifo: Vec<String> = (0..100).map(|i| format!("{i:03}")).collect();
    assert_eq!(s.config().dream_budget, 5);
    assert_eq!(seeds(s.clusters(None).unwrap()), fifo[..40]);
    assert!(seeds(s.clusters(Some(0)).unwrap()).is_empty());
    assert_eq!(seeds(s.clusters(Some(2)).unwrap()), fifo[..16]);
    assert_eq!(seeds(s.clusters(Some(100)).unwrap()), fifo);
}
