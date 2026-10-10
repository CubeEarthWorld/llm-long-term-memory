//! The seeded workload behind `tests/equivalence/golden.json`: remember with cues,
//! recall, cite, clusters, dream with both verdicts, eviction.

use super::rows;
use super::{always_keep, merge_to_gist, FakeEmbedder, START};
use engram::{Config, Engram, ManualClock, Recall, Remembered, Store};
use serde_json::{json, Value};

const TOPICS: [[&str; 6]; 6] = [
    ["kyoto", "trip", "hotel", "train", "temple", "autumn"],
    ["doctor", "clinic", "allergy", "pollen", "medicine", "spring"],
    ["piano", "lesson", "teacher", "recital", "practice", "chopin"],
    ["salary", "bank", "rent", "budget", "savings", "loan"],
    ["cat", "vet", "food", "litter", "toy", "kitten"],
    ["python", "dart", "engine", "memory", "vector", "test"],
];

pub fn config() -> Config {
    Config { cosine_floor: 0.0, capacity: 48, grace_period: 86_400.0, dream_budget: 3, relative_score: 0.3, ..Config::default() }
}

/// Runs the workload on a fresh in-RAM engine; `after_step` sees the engine after every step.
pub fn run(mut after_step: impl FnMut(&Engram)) -> Vec<Value> {
    let clock = ManualClock::new(START);
    let mut m = Engram::with_clock(Box::new(Store::in_memory().unwrap()), Box::new(FakeEmbedder::new()), config(), "Asia/Tokyo", clock.clock()).unwrap();
    let mut seed: u64 = 0x5EED;
    let mut next = |n: u64| {
        seed = (seed * 1_103_515_245 + 12_345) & 0x7FFF_FFFF;
        seed % n
    };
    let pack = |r: &Recall| {
        let mut out = String::new();
        for line in r.pack_text.split_inclusive('\n') {
            let cut = line.find("《id:").unwrap();
            out.push_str(&line[..cut]);
            out.push_str("《id》\n");
        }
        out
    };
    let mut trace = Vec::new();
    let mut last: Option<Recall> = None;
    for _ in 0..220 {
        clock.advance((600 + next(12 * 3600)) as f64);
        let op = next(100);
        let topic = TOPICS[next(6) as usize];
        let words = |n: usize, next: &mut dyn FnMut(u64) -> u64| (0..n).map(|_| topic[next(6) as usize]).collect::<Vec<_>>().join(" ");
        if op < 55 {
            let w = words(3, &mut next);
            let other = TOPICS[next(6) as usize][next(6) as usize];
            let text = format!("{w} {other} n{}", next(40));
            let cue = if next(3) == 0 { String::new() } else { words(2, &mut next) };
            let salience = (1 + next(4)) as f64;
            let row = match m.remember(&text, salience, &cue).unwrap() {
                Remembered::Inserted { cosine, evicted, .. } => json!({"action": "inserted", "cosine": cosine, "evicted": evicted.map(|t| t.text)}),
                Remembered::Reinforced(_) => json!({"action": "reinforced", "cosine": null, "evicted": null}),
                Remembered::Rejected => json!({"action": "rejected", "cosine": null, "evicted": null}),
                Remembered::RateLimited { .. } => json!({"action": "rateLimited", "cosine": null, "evicted": null}),
            };
            trace.push(json!({"op": "remember", "action": row["action"], "cosine": row["cosine"], "evicted": row["evicted"]}));
        } else if op < 75 {
            let q = format!("{}、{}", words(2, &mut next), words(2, &mut next));
            let r = m.recall(&q).unwrap();
            let recalled: Vec<Value> = r.recalled.iter().map(|c| json!([c.trace.text, c.score, c.cosine, c.retrievability])).collect();
            trace.push(json!({"op": "recall", "recalled": recalled, "pack": pack(&r)}));
            last = Some(r);
        } else if op < 80 {
            let reply: String = last.iter().flat_map(|r| &r.recalled).map(|c| format!("《id:{}》", c.trace.id)).collect();
            trace.push(json!({"op": "cite", "cited": m.cite(&reply).unwrap().len()}));
        } else if op < 90 {
            let clusters: Vec<Vec<String>> = m.clusters(Some(1 + next(4) as usize)).unwrap().iter().map(|c| c.iter().map(|t| t.text.clone()).collect()).collect();
            trace.push(json!({"op": "clusters", "clusters": clusters}));
        } else {
            let merge = next(2) == 0;
            let budget = Some(1 + next(3) as usize);
            let reports = if merge { m.dream(&mut merge_to_gist, budget) } else { m.dream(&mut always_keep, budget) }.unwrap();
            let reports: Vec<Value> = reports
                .iter()
                .map(|rep| {
                    json!({
                        "action": rep.action.as_str(),
                        "before": rep.before.iter().map(|b| b.text.clone()).collect::<Vec<_>>(),
                        "after": rep.after.iter().map(|g| json!([g.text, g.stability, g.last_recall])).collect::<Vec<_>>(),
                    })
                })
                .collect();
            trace.push(json!({"op": "dream", "reports": reports}));
        }
        after_step(&m);
    }
    let memories: Vec<Value> = rows(&m).iter().map(|t| json!([t.text, t.stability, t.last_recall, t.consolidated])).collect();
    trace.push(json!({"op": "final", "memories": memories}));
    trace
}
