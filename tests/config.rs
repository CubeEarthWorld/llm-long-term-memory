//! The parameter set: tolerant JSON form and validation.

use engram::Config;
use serde_json::json;

#[test]
fn config_round_trip_and_validation() {
    let mut c = Config::default();
    c.merge_json(&json!({"capacity": "42", "alpha": 0.5, "max_cues": 3.9, "unknown": 1, "inject_n": "junk"}));
    assert_eq!((c.capacity, c.alpha, c.max_cues, c.inject_n), (42, 0.5, 3, 5));
    let mut d = Config::default();
    d.merge_json(&c.to_json());
    assert_eq!(d, c);
    assert_eq!(c.to_json().as_object().unwrap().keys().next().unwrap(), "capacity"); // declaration order
    assert!(Config { theta_related: 1.5, ..Config::default() }.validate().is_err());
    let mut zero = Config::default();
    zero.merge_json(&json!({"capacity": -3}));
    assert!(zero.validate().is_err());
    assert!(Config::default().validate().is_ok());
}
