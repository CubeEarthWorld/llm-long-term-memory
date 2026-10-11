//! Audit lane 1: forgetting curve and clamping (L1-1 .. L1-5). Public API only.
//! A FAILING test here is a confirmed defect; see audit/lane1.md.

mod support;

use engram::{Config, Engram, ManualClock, Remembered, Storage, Trace, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use support::*;

const MODEL: &str = "fake/token-overlap";

fn row(id: &str, text: &str, created: i64, last_recall: i64, stability: f64, consolidated: bool) -> Trace {
    Trace {
        id: id.into(),
        text: text.into(),
        created_at: created,
        tz: "UTC;+00:00".into(),
        last_recall,
        stability,
        consolidated,
        cue: String::new(),
        model_id: MODEL.into(),
        vector: FakeEmbedder::new().embed(text),
    }
}

fn check_rows(m: &Engram, now: f64, ctx: &str) {
    let smax = m.config().max_stability;
    for t in rows(m) {
        assert!(t.stability.is_finite() && t.stability >= 1.0 && t.stability <= smax, "{ctx}: stored stability {} outside [1,{smax}]", t.stability);
        assert!(t.last_recall as f64 <= now.max(0.0) + 1.0 || t.last_recall <= now as i64, "{ctx}: last_recall {} > now {now}", t.last_recall);
        let r = m.retrievability(&t, now);
        assert!((0.0..=1.0).contains(&r), "{ctx}: R={r}");
    }
}

// ---------------------------------------------------------------- L1-1

/// `Engram::retrievability` is public and `store().find()` hands out raw, unchecked rows,
/// so it must be total: R in [0,1] for every stability / time. (Engine paths pre-clamp.)
#[test]
fn l1_1_retrievability_is_total_on_public_api() {
    let (m, _) = build(cfg());
    let stabs = [f64::NAN, 0.0, -0.0, -1.0, -1e300, f64::NEG_INFINITY, 5e-324, 1e-310, 1e-300, 0.5, 1.0, 1e300, f64::MAX, f64::INFINITY];
    let lrs: [i64; 5] = [i64::MIN, 0, 1_700_000_000, 1_800_000_000, i64::MAX];
    let nows = [f64::NAN, f64::NEG_INFINITY, -1e300, 0.0, 1_700_000_000.0, 1e300, f64::INFINITY];
    let mut bad = Vec::new();
    for &s in &stabs {
        for &lr in &lrs {
            for &now in &nows {
                let t = row("X", "x", 0, lr, s, true);
                let r = m.retrievability(&t, now);
                if !(0.0..=1.0).contains(&r) {
                    bad.push(format!("S={s:e} lr={lr} now={now:e} -> R={r}"));
                }
                let st = m.strength(&t, now);
                if st.is_nan() && !r.is_nan() {
                    bad.push(format!("S={s:e} lr={lr} now={now:e} -> strength NaN"));
                }
            }
        }
    }
    assert!(bad.is_empty(), "{} violations, e.g.\n{}", bad.len(), bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
}

/// Hand-written rows (allowed by the store: only non-finite stability is refused) must
/// never leak a stability outside [1,S_max] or an R outside [0,1] through recall/cite/remember.
#[test]
fn l1_1_l1_2_weird_stored_rows_are_clamped_by_every_path() {
    let db = SharedStore::default();
    let c = Config { max_stability: 1e9, capacity: 6, ..cfg() };
    let (mut m, clock) = build_with(c, Box::new(FakeEmbedder::new()), Box::new(db.clone()));
    let now = clock.get() as i64;
    let mut w = db.clone();
    let weird = [
        (0.0, now), (-5.0, now), (1e-320, now), (0.5, now - 10), (1e300, now - 5), (f64::MAX, i64::MAX), (3.0, i64::MIN), (1e9 * 2.0, now + 10_000),
    ];
    for (i, (s, lr)) in weird.iter().enumerate() {
        let t = row(&format!("W{i}"), &format!("weird{i} common"), now - 100, *lr, *s, true);
        w.write(&[Write::Insert(&t)]).unwrap();
    }
    assert!(w.write(&[Write::Insert(&row("NAN", "n", 0, 0, f64::NAN, true))]).is_err());
    assert!(w.write(&[Write::Insert(&row("INF", "n", 0, 0, f64::INFINITY, true))]).is_err());
    let rec = m.recall("weird0 weird1 weird2 weird3 weird4 weird5 weird6 weird7 common").unwrap();
    for r in &rec.recalled {
        assert!((0.0..=1.0).contains(&r.retrievability), "R={} for {}", r.retrievability, r.trace.id);
        assert!(r.trace.stability >= 1.0 && r.trace.stability <= 1e9, "S={} for {}", r.trace.stability, r.trace.id);
        // a stored time is not rewritten on read: a future one reads as no time elapsed (R above)
    }
    let reply: String = rec.recalled.iter().map(|r| format!("《id:{}》", r.trace.id)).collect();
    m.cite(&reply).unwrap();
    // After all writes the engine itself wrote: every row it touched is in range.
    for t in rows(&m) {
        if rec.recalled.iter().any(|r| r.trace.id == t.id) {
            assert!(t.stability >= 1.0 && t.stability <= 1e9, "{}: stored S={}", t.id, t.stability);
            assert!(t.last_recall <= now, "{}: stored last_recall {} > now", t.id, t.last_recall);
        }
    }
    // Eviction over a store full of weird rows must not panic and must keep the capacity.
    for i in 0..4 {
        clock.advance(1.0);
        insert(&mut m, &format!("filler{i} unique{i}"), 1.0, "");
    }
    assert!(rows(&m).len() <= 6);
}

// ---------------------------------------------------------------- L1-2

#[test]
fn l1_2_extreme_configs_keep_stability_in_range() {
    let configs = [
        ("smax=1", Config { max_stability: 1.0, initial_stability: 1.0, ..cfg() }),
        ("smax=MAX", Config { max_stability: f64::MAX, initial_stability: 1e308, spacing_gain: f64::MAX, ..cfg() }),
        ("s0=subnormal", Config { initial_stability: 5e-324, ..cfg() }),
        ("gain=0", Config { spacing_gain: 0.0, ..cfg() }),
        ("gain=huge", Config { spacing_gain: 1e300, ..cfg() }),
        ("grace=MAX", Config { grace_period: f64::MAX, capacity: 3, ..cfg() }),
    ];
    for (name, c) in configs {
        let (mut m, clock) = build(c);
        let mut ids = Vec::new();
        for i in 0..5 {
            clock.advance(3600.0);
            ids.push(insert(&mut m, &format!("alpha{i} shared topic words"), if i % 2 == 0 { 10.0 } else { 0.0 }, ""));
        }
        for _ in 0..4 {
            clock.advance(30.0 * DAY);
            let rec = m.recall("alpha1 alpha2 shared topic words").unwrap();
            let reply: String = rec.recalled.iter().map(|r| format!("《id:{}》", r.trace.id)).collect();
            m.cite(&reply).unwrap();
            m.remember("alpha1 shared topic words", 1.0, "").unwrap();
            check_rows(&m, clock.get(), name);
        }
        let _ = m.dream(&mut merge_to_gist, None).unwrap();
        check_rows(&m, clock.get(), name);
    }
}

// ---------------------------------------------------------------- L1-3

#[test]
fn l1_3_retrieval_monotone_and_fixed_points() {
    let (mut m, clock) = build(cfg());
    let id = insert(&mut m, "monotone fact", 1.0, "");
    let s0 = mem(&m, &id).unwrap().stability;
    // R = 1 (same instant): rehearsal leaves stability unchanged.
    m.remember("monotone fact", 1.0, "").unwrap();
    assert_eq!(mem(&m, &id).unwrap().stability, s0);
    // R < 1: grows by exactly 1 + gain (1 - R) for a = 1.
    clock.advance(DAY);
    m.remember("monotone fact", 1.0, "").unwrap();
    let s1 = mem(&m, &id).unwrap().stability;
    assert!((s1 / s0 - (1.0 + 3.0 * 0.5)).abs() < 1e-9, "ratio {}", s1 / s0);
    // monotone over a sweep of gaps
    let mut prev = s1;
    for k in [1.0, 10.0, 100.0, 1e4, 1e8] {
        clock.advance(k);
        m.remember("monotone fact", 1.0, "").unwrap();
        let s = mem(&m, &id).unwrap().stability;
        assert!(s >= prev, "stability decreased {prev} -> {s} after {k}s");
        prev = s;
    }
    // gain = 0 leaves stability unchanged at any R
    let (mut z, zc) = build(Config { spacing_gain: 0.0, ..cfg() });
    let zid = insert(&mut z, "zero gain", 1.0, "");
    zc.advance(10.0 * DAY);
    z.remember("zero gain", 1.0, "").unwrap();
    assert_eq!(mem(&z, &zid).unwrap().stability, DAY);
}

/// cite's `S * full / partial` multiplies before dividing: with a legal (if extreme) config the
/// product overflows to +inf although the true result is 1e305 << S_max = f64::MAX.
#[test]
fn l1_3_cite_product_overflow_order() {
    // Probe the cosine of the query against the trace under the zero-floor config.
    let cos = {
        let (mut p, _) = build(cfg());
        insert(&mut p, "red green blue yellow", 1.0, "");
        p.peek("red green purple orange").unwrap().recalled[0].cosine
    };
    let a_target = 2e-5;
    let floor = (cos - a_target) / (1.0 - a_target);
    let c = Config {
        cosine_floor: floor,
        min_score: 0.0,
        relative_score: 0.0,
        initial_stability: 1e295,
        max_stability: f64::MAX,
        spacing_gain: 2e10,
        ..cfg()
    };
    let (mut m, clock) = build(c);
    let id = insert(&mut m, "red green blue yellow", 1.0, "");
    clock.advance(1e295); // one half-life: R = 1/2
    let rec = m.recall("red green purple orange").unwrap();
    assert_eq!(rec.recalled.len(), 1);
    let r = rec.recalled[0].retrievability;
    assert!((r - 0.5).abs() < 1e-9);
    let s_pre = 1e295;
    m.cite(&format!("《id:{id}》")).unwrap();
    let s = mem(&m, &id).unwrap().stability;
    let expected = s_pre * (1.0 + 2e10 * (1.0 - r)); // full activation from the original stability
    assert!((s / expected - 1.0).abs() < 1e-3, "cite gave {s:e}, a full recall would give {expected:e}");
}

// ---------------------------------------------------------------- L1-4

#[test]
fn l1_4_band_is_monotone_and_u8_for_default_smax() {
    let mut prev = 0u8;
    let mut s = 1.0f64;
    let smax = Config::default().max_stability;
    while s <= smax {
        let b = Trace { stability: s, ..row("B", "b", 0, 0, s, true) }.band();
        assert!(b >= prev, "band decreased at S={s}");
        prev = b;
        s *= 1.01;
    }
    assert!(prev < 255, "default S_max reaches band {prev}");
}

/// floor(2 log2 S) at octave boundaries (log2 rounding may put 2^k - 1ulp one band high: benign).
#[test]
fn l1_4_band_is_exact_at_boundaries() {
    let band = |s: f64| row("B", "b", 0, 0, s, true);
    let mut off = Vec::new();
    for k in 1..60u32 {
        let p = 2f64.powi(k as i32);
        for (s, want) in [(p, 2 * k), (f64::from_bits(p.to_bits() - 1), 2 * k - 1)] {
            let got = u32::from(Trace { ..band(s) }.band());
            // 2^k exactly must be band 2k. One ulp below, f64 log2 rounds up to k, so the band is
            // 2k instead of 2k-1: benign (S differs by 1 ulp), so tolerate +1 only.
            let ok = if s == p { got == want } else { got == want || got == want + 1 };
            if !ok {
                off.push(format!("S=2^{k}{} band {got} want {want}", if s == p { "" } else { " - ulp" }));
            }
        }
    }
    assert!(off.is_empty(), "{off:?}");
}

/// `Config::validate` accepts any max_stability >= 1, but band is `as u8` after clamp(0,255):
/// above 2^127.5 s every stability shares band 255, so the "head by oldest last_recall" of that
/// band is not the weakest. Real defect for configs with S_max > ~2.4e38.
#[test]
fn l1_4_huge_smax_collapses_bands_and_evicts_the_wrong_trace() {
    let c = Config { capacity: 10, grace_period: 0.0, initial_stability: 1e40, max_stability: 1e60, ..cfg() };
    assert!(c.validate().is_ok(), "validate rejects it: fine, no defect");
    let (mut m, clock) = build(c);
    let mut strong = Vec::new();
    for i in 0..9 {
        clock.advance(1.0);
        strong.push(insert(&mut m, &format!("strong{i}"), 10.0, ""));
    }
    clock.advance(1.0);
    let weak = insert(&mut m, "weak one", 1.0, "");
    assert_eq!(mem(&m, &weak).unwrap().band(), mem(&m, &strong[0]).unwrap().band());
    clock.advance(1.0);
    let Remembered::Inserted { evicted, .. } = m.remember("newcomer", 1.0, "").unwrap() else { panic!() };
    assert_eq!(evicted.map(|t| t.id), Some(weak), "evicted a strong (S=1e41) trace instead of the weak (S=1e40) one");
}

/// Within a band the claim "oldest last_recall = weakest" is false: ratio < sqrt2 but the
/// decay terms differ. S_B=70000 (recalled 3 half-lives ago), S_A=91000 (older: 3.1 B-half-lives,
/// but only 2.38 own half-lives): rank_A > rank_B, so B is weaker, yet A is the band head.
/// Nine such A's push B out of the `excess + BAND_DEPTH` window.
#[test]
fn l1_4_oldest_in_band_is_not_always_weakest() {
    let n = START + 4e5;
    let c = Config { capacity: 10, grace_period: 0.0, initial_stability: 70_000.0, max_stability: 1e9, ..cfg() };
    let (mut m, clock) = build(c);
    for i in 0..9 {
        clock.set(n - 3.1 * 70_000.0 + f64::from(i));
        insert(&mut m, &format!("olda{i}"), 1.3, "");
    }
    clock.set(n - 3.0 * 70_000.0);
    let b = insert(&mut m, "newerb", 1.0, "");
    clock.set(n);
    let all = rows(&m);
    assert!(all.iter().all(|t| t.band() == all[0].band()), "setup: one band");
    let rank = |t: &Trace| t.stability.log2() - (n - t.last_recall as f64) / t.stability;
    let weakest = all.iter().min_by(|x, y| rank(x).total_cmp(&rank(y))).unwrap().id.clone();
    assert_eq!(weakest, b, "setup: B is the true weakest");
    let Remembered::Inserted { evicted, .. } = m.remember("newcomer", 1.0, "").unwrap() else { panic!() };
    assert_eq!(evicted.map(|t| t.id), Some(b), "evicted the band head, not the weakest trace");
}

// ---------------------------------------------------------------- L1-5

fn dream_pair(c: Config, now: f64, a: Trace, b: Trace) -> Result<(Vec<engram::DreamReport>, Engram, ManualClock), String> {
    let db = SharedStore::default();
    let (mut m, clock) = build_with(c, Box::new(FakeEmbedder::new()), Box::new(db.clone()));
    clock.set(now);
    let mut w = db.clone();
    w.write(&[Write::Insert(&a), Write::Insert(&b)]).unwrap();
    let rep = catch_unwind(AssertUnwindSafe(|| m.dream(&mut merge_to_gist, None))).map_err(|e| format!("PANIC: {:?}", e.downcast_ref::<String>().cloned().or(e.downcast_ref::<&str>().map(|s| s.to_string()))))?;
    Ok((rep.map_err(|e| e.to_string())?, m, clock))
}

#[test]
fn l1_5_gist_invariants_on_extreme_rows() {
    let now = START;
    let n = now as i64;
    let cases: Vec<(&str, f64, i64, f64, i64)> = vec![
        ("both above S_max, future", 1e300, n + 1000, 1e300, n + 5),
        ("tiny and dead", 1.0, 0, 1.0, i64::MIN),
        ("huge S old", 1e9, i64::MIN / 2, 1e9, 0),
        ("stability<1", 0.0, n, -3.0, n),
    ];
    for (name, s1, lr1, s2, lr2) in cases {
        let c = Config { max_stability: 1e9, ..cfg() };
        let a = row("A", "gist base words here", n - 1000, lr1, s1, true);
        let mut b = row("B", "gist base words here too", n - 10, lr2, s2, false);
        b.consolidated = false;
        let (reps, m, clock) = dream_pair(c, now, a, b).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(reps.iter().any(|r| r.action == engram::DreamAction::Replace), "{name}: {reps:?}");
        for g in reps.iter().flat_map(|r| r.after.iter()) {
            assert!(g.stability >= 1.0 && g.stability <= 1e9, "{name}: gist S={}", g.stability);
            assert!(g.last_recall <= n, "{name}: gist last_recall {} > now", g.last_recall);
            let r = m.retrievability(g, clock.get());
            assert!((0.0..=1.0).contains(&r), "{name}: gist R={r}");
        }
    }
}

/// `now as i64 - round(stability * elapsed) as i64` is unchecked. With the clock before 1970
/// (negative) and a row whose last_recall is i64::MIN the shift saturates to i64::MAX and the
/// subtraction overflows (debug: panic, release: wraps to a far-future last_recall).
#[test]
fn l1_5_gist_last_recall_subtraction_cannot_overflow() {
    let c = Config { max_stability: 1e19, initial_stability: 1e18, ..cfg() };
    let a = row("A", "gist base words here", -3_000_000, i64::MIN, 1e18, true);
    let b = row("B", "gist base words here too", -2_000_000, i64::MIN, 1e18, false);
    let (reps, _m, _c) = dream_pair(c, -1_000_000.0, a, b).unwrap_or_else(|e| panic!("{e}"));
    for g in reps.iter().flat_map(|r| r.after.iter()) {
        assert!(g.last_recall <= -1_000_000, "wrapped: last_recall={}", g.last_recall);
    }
}
