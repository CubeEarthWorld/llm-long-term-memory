//! Millennia of virtual time (SPEC §8): bounded state, no NaN/overflow, key facts survive
//! while noise is forgotten, clock faults and decade-long silences absorbed.
//! Slow in debug builds: `cargo test --release --test longevity -- --ignored`.

mod support;

use support::rows;
use engram::{fmt_local, Config, Store};
use support::{build_with, cfg, merge_to_gist, FakeEmbedder, DAY};

#[test]
#[ignore = "3000 virtual years; run with --ignored (seconds in release)"]
fn three_thousand_years() {
    // theta_related is calibrated per model; the token-overlap fake scores unrelated text far
    // higher than a real model, so the band is raised to keep the same selectivity.
    let config = Config { capacity: 300, writes_per_day: 50, theta_related: 0.75, ..cfg() };
    let (mut s, clock) = build_with(config.clone(), Box::new(FakeEmbedder::with(32, "fake/token-overlap")), Box::new(Store::in_memory().unwrap()));
    let mut rng: u64 = 7;
    let mut rand = |n: u64| {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (rng >> 33) % n
    };
    let facts = ["user likes matcha ice cream", "user was born in kyoto japan", "user has a cat named tama"];
    for f in facts {
        s.remember(f, 2.0, "").unwrap();
    }
    let start = clock.get();
    let mut d = 0u64;
    while d < 3000 * 365 {
        clock.advance(3.0 * DAY);
        if rand(1000) == 0 {
            clock.advance(10.0 * 365.0 * DAY); // a decade of silence
        }
        s.remember(&format!("note {} topic{}", rand(1 << 30), d % 97), 1.0, "").unwrap();
        if d % 7 < 3 {
            let f = facts[rand(3) as usize];
            assert!(s.recall(f).unwrap().recalled.iter().any(|c| c.trace.text == f), "day {d}");
        }
        if d % 30 < 3 {
            s.dream(&mut merge_to_gist, Some(2)).unwrap();
        }
        if d == 1500 * 365 {
            let t = clock.get(); // clock fault: +200 years, then back
            clock.set(t + 200.0 * 365.0 * DAY);
            s.recall(facts[0]).unwrap();
            clock.set(t);
        }
        d += 3;
    }
    let now = clock.get();
    assert!(rows(&s).len() <= config.capacity);
    for m in rows(&s) {
        assert!((1.0..=config.max_stability).contains(&m.stability));
        assert!((1..=config.text_max).contains(&m.text.chars().count()));
        assert!((0.0..=1.0).contains(&s.retrievability(&m, now))); // a NaN already fails this
    }
    for f in facts {
        let top = &s.recall(f).unwrap().recalled[0];
        assert!(top.trace.text == f && top.trace.stability > 365.0 * DAY);
    }
    assert!(fmt_local(now as i64, "UTC;+00:00")[..4].parse::<i64>().unwrap() > 5000);
    assert!((now - start) / (365.0 * DAY) >= 3000.0);
    assert_eq!(s.stats().unwrap().records, rows(&s).len());
}
