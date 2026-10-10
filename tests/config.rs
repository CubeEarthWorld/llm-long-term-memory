//! The parameter set: defaults are valid, broken invariants are rejected.

use engram::Config;

#[test]
fn defaults_validate_and_broken_values_do_not() {
    assert!(Config::default().validate().is_ok());
    assert!(Config { theta_related: 1.5, ..Config::default() }.validate().is_err());
    assert!(Config { capacity: 0, ..Config::default() }.validate().is_err());
}
