//! Same-algorithm regression: the seeded workload must reproduce
//! `tests/equivalence/golden.json` — recorded by the Dart package, whose arithmetic
//! (f32 vectors, f64 accumulation in dimension order) this crate shares — exactly.
//! Regenerate (only when the algorithm is meant to change) with
//! `EQUIVALENCE_WRITE=1 cargo test --test equivalence`.

mod support;

use std::path::Path;

#[test]
fn seeded_workload_reproduces_the_golden_outcome_bit_for_bit() {
    let trace = serde_json::Value::Array(support::workload::run(|_| {}));
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/equivalence/golden.json");
    if std::env::var("EQUIVALENCE_WRITE").as_deref() == Ok("1") {
        std::fs::write(&golden, serde_json::to_string_pretty(&trace).unwrap() + "\n").unwrap();
    }
    let expected: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(golden).unwrap()).unwrap();
    if let Err(diff) = support::same_json(&trace, &expected, "trace") {
        panic!("{diff}");
    }
}
