//! Same-algorithm regression: the seeded workload must reproduce
//! `tests/equivalence/golden.json`. Regenerate (only when the algorithm is meant to
//! change) with `BLESS=1 cargo test --test equivalence`.

mod support;

use std::path::Path;

#[test]
fn seeded_workload_reproduces_the_golden_outcome() {
    let trace = serde_json::Value::Array(support::workload::run(|_| {}));
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/equivalence/golden.json");
    if std::env::var_os("BLESS").is_some() {
        std::fs::write(&golden, serde_json::to_string_pretty(&trace).unwrap() + "\n").unwrap();
    }
    let expected: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(golden).unwrap()).unwrap();
    if let Err(diff) = support::same_json(&trace, &expected, "trace") {
        panic!("{diff}");
    }
}
