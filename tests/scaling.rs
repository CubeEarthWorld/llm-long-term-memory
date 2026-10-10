//! Bulk eviction and the budgeted cluster preview stay linear and exact.

mod support;

use engram::{Config, MemoryStore, Remembered, Result, Store, Trace, Write};
use std::sync::{Arc, Mutex};
use support::{build_with, cfg, BrokenEmbedder, FakeEmbedder};

const NOW: i64 = 1_700_000_000;

#[allow(clippy::too_many_arguments)]
fn row(id: &str, text: &str, created_at: i64, last_recall: i64, stability: f64, consolidated: bool, model_id: &str, vector: Vec<f32>) -> Trace {
    Trace { id: id.into(), text: text.into(), created_at, tz: "UTC;+00:00".into(), last_recall, stability, consolidated, cue: String::new(), model_id: model_id.into(), vector }
}

/// Records every removed id, in order.
struct Recording(MemoryStore, Arc<Mutex<Vec<String>>>);

impl Store for Recording {
    fn load_all(&self) -> Result<Vec<Trace>> {
        self.0.load_all()
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
fn bulk_eviction_matches_repeated_scan() {
    for capacity in [1, 10, 40, 119, 120] {
        for mode in ["mixed", "young", "old"] {
            let mut store = MemoryStore::default();
            // Ids sort in insertion order, as ULIDs do.
            let rows: Vec<Trace> = (0..120)
                .map(|i| {
                    let young = mode == "young" || (mode == "mixed" && i % 3 != 0);
                    row(&format!("{i:03}"), &format!("row {i}"), NOW - if young { 10 } else { 200 }, NOW - (i % 5) * 100, 2f64.powi((i % 8) as i32), true, "", vec![])
                })
                .collect();
            for r in &rows {
                store.write(&[Write::Put(r)]).unwrap();
            }
            let removed = Arc::new(Mutex::new(Vec::new()));
            let config = Config { capacity, grace_period: 100.0, ..cfg() };
            let (mut s, clock) = build_with(config.clone(), Box::new(BrokenEmbedder), Box::new(Recording(store, removed.clone())));
            clock.set(NOW as f64);
            // Independent reference: the original full scan after every removal.
            let mut left: Vec<Trace> = rows.clone();
            left.push(row("new", "new row", NOW, NOW, config.initial_stability, false, "", vec![]));
            let mut expected = Vec::new();
            while left.len() > capacity {
                let young: Vec<bool> = left.iter().map(|m| (0..100).contains(&(NOW - m.created_at))).collect();
                let n_young = young.iter().filter(|&&y| y).count();
                let all = n_young == left.len() || n_young > capacity / 10;
                let rank = |m: &Trace| m.stability.ln() / std::f64::consts::LN_2 - ((NOW - m.last_recall).max(0) as f64) / m.stability;
                let victim = (0..left.len()).filter(|&i| all || !young[i]).min_by(|&a, &b| rank(&left[a]).total_cmp(&rank(&left[b])).then(young[a].cmp(&young[b])).then(a.cmp(&b))).unwrap(); // ties: old first
                expected.push(left.remove(victim).text);
            }
            let Remembered::Inserted { evicted, .. } = s.remember("new row", 1.0, "").unwrap() else { panic!() };
            let texts_of = |ids: &[String]| ids.iter().map(|id| rows.iter().find(|r| &r.id == id).map_or("new row".to_owned(), |r| r.text.clone())).collect::<Vec<_>>();
            assert_eq!(texts_of(&removed.lock().unwrap()), expected, "capacity {capacity} {mode}");
            assert_eq!(evicted.map(|t| t.text), expected.last().cloned());
            let survivors: Vec<&str> = s.memories().iter().map(|m| m.text.as_str()).collect();
            assert_eq!(survivors, left.iter().map(|m| m.text.as_str()).collect::<Vec<_>>());
        }
    }
}

#[test]
fn clusters_preview_the_seeds_dream_scans_first_in_first_out() {
    let mut fake = FakeEmbedder::new();
    let v = fake.embed("same vector");
    let mut store = MemoryStore::default();
    for i in 0..100 {
        let t = row(&format!("{i:03}"), &format!("row {i}"), NOW - i % 7, NOW, 86_400.0 + (i % 3) as f64, false, "fake/token-overlap", v.clone());
        store.write(&[Write::Put(&t)]).unwrap();
    }
    let (mut s, _) = build_with(cfg(), Box::new(fake), Box::new(store));
    let seeds = |cs: Vec<Vec<Trace>>| cs.into_iter().map(|c| c[0].id.clone()).collect::<Vec<_>>();
    let fifo: Vec<String> = (0..100).map(|i| format!("{i:03}")).collect();
    assert_eq!(s.config().dream_budget, 5);
    assert_eq!(seeds(s.clusters(None)), fifo[..40]);
    assert!(seeds(s.clusters(Some(0))).is_empty());
    assert_eq!(seeds(s.clusters(Some(2))), fifo[..16]);
    assert_eq!(seeds(s.clusters(Some(100))), fifo);
}
