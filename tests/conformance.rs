//! Scripted scenario: replays `tests/conformance/scenario.json` and must reproduce the
//! committed trace (texts, actions, stabilities, scores at 6 decimals).

mod support;

use support::rows;
use engram::{Config, Remembered, Store};
use serde_json::{json, Value};
use std::path::PathBuf;
use support::{merge_to_gist, FakeEmbedder};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/conformance")
}

fn r6(v: f64) -> String {
    format!("{v:.6}")
}

fn masked(pack: &str) -> String {
    pack.split_inclusive('\n').map(|l| format!("{}《id》\n", &l[..l.find("《id:").unwrap()])).collect()
}

#[test]
fn scripted_scenario_reproduced() {
    let read = |f: &str| -> Value { serde_json::from_str(&std::fs::read_to_string(dir().join(f)).expect(f)).unwrap() };
    let (scenario, expected) = (read("scenario.json"), read("trace.json"));
    let c = &scenario["config"];
    let cfg = Config {
        capacity: c["capacity"].as_u64().unwrap() as usize,
        grace_period: c["gracePeriod"].as_f64().unwrap(),
        writes_per_day: c["writesPerDay"].as_u64().unwrap() as usize,
        min_score: c["minScore"].as_f64().unwrap(),
        cosine_floor: 0.0,
        ..Config::default()
    };
    let dim = scenario["dimension"].as_u64().unwrap() as usize;
    let (mut s, clock) = support::build_with(cfg, Box::new(FakeEmbedder::with(dim, "fake/token-overlap")), Box::new(Store::in_memory().unwrap()));
    let mut trace = Vec::new();
    for step in scenario["steps"].as_array().unwrap() {
        let op = step["op"].as_str().unwrap();
        let mut row = json!({"op": op});
        match op {
            "advance" => clock.advance(step["seconds"].as_f64().unwrap()),
            "remember" => {
                let r = s.remember(step["text"].as_str().unwrap(), step["salience"].as_f64().unwrap_or(1.0), "").unwrap();
                let (action, t, evicted) = match r {
                    Remembered::Inserted { trace, evicted, .. } => ("inserted", Some(trace), evicted.map(|e| e.text)),
                    Remembered::Reinforced(t) => ("reinforced", Some(t), None),
                    Remembered::Rejected => ("rejected", None, None),
                    Remembered::RateLimited { .. } => ("rateLimited", None, None),
                };
                row["action"] = json!(action);
                row["text"] = json!(t.as_ref().map(|t| t.text.clone()));
                row["stability"] = json!(t.as_ref().map(|t| r6(t.stability)));
                row["consolidated"] = json!(t.as_ref().map(|t| t.consolidated));
                row["evicted"] = json!(evicted);
            }
            "recall" => {
                let r = s.recall(step["query"].as_str().unwrap()).unwrap();
                row["texts"] = json!(r.recalled.iter().map(|c| c.trace.text.clone()).collect::<Vec<_>>());
                row["scores"] = json!(r.recalled.iter().map(|c| r6(c.score)).collect::<Vec<_>>());
                row["pack"] = json!(masked(&r.pack_text));
            }
            "cite" => {
                let texts: Vec<&str> = step["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
                let ids: Vec<String> = rows(&s).iter().filter(|m| texts.contains(&m.text.as_str())).map(|m| format!("《id:{}》", m.id)).collect();
                row["cited"] = json!(s.cite(&ids.join(" ")).unwrap().len());
            }
            "dream" => {
                let reports = s.dream(&mut merge_to_gist, step["budget"].as_u64().map(|b| b as usize)).unwrap();
                row["reports"] = json!(reports
                    .iter()
                    .map(|rep| json!({
                        "action": rep.action.as_str(),
                        "before": rep.before.iter().map(|b| b.text.clone()).collect::<Vec<_>>(),
                        "after": rep.after.iter().map(|a| a.text.clone()).collect::<Vec<_>>(),
                        "stability": rep.after.iter().map(|a| r6(a.stability)).collect::<Vec<_>>(),
                    }))
                    .collect::<Vec<_>>());
            }
            "forget" => {
                let id = rows(&s).iter().find(|m| m.text == step["text"].as_str().unwrap()).map(|m| m.id.clone());
                row["forgot"] = json!(id.is_some_and(|id| s.forget(&id).unwrap().is_some()));
            }
            other => panic!("unknown op {other}"),
        }
        let now = clock.get();
        row["state"] = json!(rows(&s)
            .iter()
            .map(|m| json!({"text": m.text, "S": r6(m.stability), "R": r6(s.retrievability(m, now)), "c": m.consolidated}))
            .collect::<Vec<_>>());
        trace.push(row);
    }
    if std::env::var_os("BLESS").is_some() {
        std::fs::write(dir().join("trace.json"), serde_json::to_string_pretty(&trace).unwrap() + "
").unwrap();
        return;
    }
    let expected = expected.as_array().unwrap();
    assert_eq!(trace.len(), expected.len());
    for (i, (a, b)) in trace.iter().zip(expected).enumerate() {
        assert_eq!(a, b, "step {i} ({}) differs", a["op"]);
    }
}
