# Lane 1 audit: forgetting curve and clamping

Files: `audit/lean/Lane1.lean` (compiles with Lean 4.34.1 core, zero `sorry`), `tests/audit_lane1.rs`
(copy in `audit/tests/`). Run: `CARGO_TARGET_DIR=C:/t/l1 cargo test --test audit_lane1 -j 3`
(7 pass, 4 fail = 4 confirmed defects below).

| Inv | Status |
|---|---|
| L1-1 | engine paths proved/tested OK; **defect D1** on the public `retrievability`/`strength` |
| L1-2 | OK. `validate` rejects S_max < 1, so the lower bound holds. Tested at S_max=1, S_max=MAX, S0 subnormal, gain 0/1e300/MAX, rows with S in {0,-5,1e-320,1e300,MAX}. **D4** (cite overflow order) is the only leak, and it is a precision loss that still stays in range |
| L1-3 | OK in model (`retrieval_monotone`, `retrieval_fixed_when_x_zero`, `cite_part_ge_one`, `cite_part_le_full`) and in tests. cite divisor >= 1 always, so "tiny partial" cannot occur. **D4** is the multiplication order |
| L1-4 | band monotone and u8-safe below 2^128: proved. **D2** (S_max above 2^127.5) and **D3** (sqrt2 claim false) are real |
| L1-5 | stability <= S_max, R in [0,1], last_recall <= now: proved/tested. Overflow only at an f64 rounding edge (see G1) |

## Real defects (all have failing tests)

**D1. `Engram::retrievability` / `strength` are not total** (`l1_1_retrievability_is_total_on_public_api`, 138 violations).
They divide by the raw `t.stability`. NaN stability gives R=NaN; stability 0 with dt=0 gives 0/0=NaN;
negative stability gives R>1 (exponent flips sign, `neg_stability_exponent_positive`); stability=inf with dt=inf
gives NaN. Engine-internal callers only pass rows that went through `checked`, so nothing stored leaks; but
`engine.store().find()` returns raw rows and the doc says R in [0,1]. Fix: `clamp_stability(t.stability)` inside `retrievability`.
Repro: `m.retrievability(&Trace{stability: f64::NAN, ..t}, now)`. Severity: low (API hygiene).

**D2. S_max > 2^127.5 collapses all bands into 255 and eviction picks the wrong trace**
(`l1_4_huge_smax_collapses_bands_and_evicts_the_wrong_trace`).
`Config::validate` allows any `max_stability >= 1`. `band()` is `floor(2 log2 S).clamp(0,255) as u8`, so
`band(2^128)=256` (`band_2_128`) saturates. With `initial_stability=1e40, max_stability=1e60, capacity=10, grace=0`:
9 traces at S=1e41, then 1 at S=1e40 (weakest), then an 11th write evicts a 1e41 trace because the band head
(oldest last_recall) window of `excess+8 = 9` rows never reaches the weak one. Fix: validate `max_stability <= 2^127`
(band <= 253), or widen the band key. Severity: low (absurd config), but it is a validated config.

**D3. The sqrt2 / "oldest last_recall in a band is the weakest" claim is false** (`l1_4_oldest_in_band_is_not_always_weakest`;
Lean `older_but_stronger`). Same band only gives S_hi/S_lo < sqrt2 (`same_band_ratio`); the weakness is
`log2 S - dt/S`, and the dt/S terms differ. Minimal counter: S_B=4 (dt 12), S_A=5 (dt 15): same band, A is older,
strength A = 5/8 > B = 4/8. So the head is not the weakest. The engine masks this by reading `excess + BAND_DEPTH(8)`
rows per band, but SPEC section 3 states the result equals a full comparison ("全件比較と同じ結果"). With 9 older,
slightly-more-stable traces in one band, `remember` evicts the band head (strength ratio about 1.8x too strong at
3 half-lives) instead of the true weakest. Test: S_B=70000, S_A=91000, nine A's. Error is unbounded in log2 terms
but only matters at moderate decay (when dt/S is large both are nearly dead). Fix: weaken the SPEC claim to "approximate,
depth-8 window", or key the band index by a time-stable quantity. Severity: low-medium (spec claim wrong, behavior approximate).

**D4. cite computes `t.stability * full / partial` product-first** (`l1_3_cite_product_overflow_order`).
If `S*full` overflows f64 to +inf while the true result is below S_max, cite returns S_max. Repro: `max_stability=f64::MAX,
initial=1e295, gain=2e10, a=2e-5, R=0.5`: expected 1.0000000001e305, got 1.797e308. Fix: `t.stability * (full / partial)`.
Needs S_max near f64::MAX, so low severity, but trivial to fix.

## Gaps and residual risks

* **G1 (not reproduced).** `last_recall = now as i64 - round(stability*elapsed) as i64` is an unchecked i64 subtraction.
  With now >= 0 it cannot overflow (`gist_no_overflow_nonneg`). With now < 0 and shift saturated at i64::MAX it does
  (`gist_overflow_negative_now`, debug panic / release wrap). But shift is about min(dt, 64 S) <= now - last_recall, so
  overflow needs `last_recall < i64::MIN`, impossible; only f64 rounding noise at `last_recall == i64::MIN` can tip it
  (`gist_shift_bounded_by_dt`). The test `l1_5_gist_last_recall_subtraction_cannot_overflow` passed. Cheap hardening:
  `now_i.saturating_sub(shift)`.
* **G2 (benign).** f64 `log2` rounds `2^k - 1ulp` up to `k`, so such an S lands one band high (test tolerates +1). Order is
  still monotone (`l1_4_band_is_monotone_and_u8_for_default_smax`), and the Lean model uses exact integer log.
* **G3 (out of lane, noticed).** `checked` validates finiteness of vector components but not unit norm. A hand-written row with a
  large non-unit vector gives a cosine far above 1; `select` clamps `dot().min(1.0)` so it scores as a perfect match
  (and `NaN.min(1.0) == 1.0`). Worth a look in the vector lane.
* Float model assumptions (Lean): floor(2 log2 S) = `Nat.log2 (S*S)` for integer S; R in [0,1] is reduced to "exponent >= 0";
  f64 overflow in D4 is modelled as an explicit cap M. The Lean file does not model rounding.

## Proved in Lean (`audit/lean/Lane1.lean`)
`band_mono`, `band_lt_256_of_lt`, `band_2_128`, `band_default_smax` (=56), `same_band_ratio`, `older_but_stronger`,
`retrieval_monotone`, `retrieval_fixed_when_x_zero`, `cite_part_ge_one`, `cite_part_le_full`, `cite_product_first_overflow`,
`gist_no_overflow_nonneg`, `gist_overflow_negative_now`, `gist_shift_bounded_by_dt`, `gist_last_recall_le_now`,
`clamp_in_range`, `clamp_fails_if_smax_lt_one`, `exponent_nonneg`, `neg_stability_exponent_positive`.
