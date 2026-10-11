/-
  Lane 1 -- forgetting curve and clamping (engram v2.2)
  Lean 4 core only (no Mathlib).  Floats are modelled by Nat / Int; every place where
  a float fact is ASSUMED is marked `-- ASSUME`.
  Band model: band(S) = floor(2*log2 S) = Nat.log2 (S*S) for integer S >= 1
  (exact for integer S; the Rust code computes it with f64 log2, see lane1.md for the
  rounding remark).
-/

def band (s : Nat) : Nat := Nat.log2 (s * s)

/-! ## L1-4a  band is monotone -/

theorem band_mono {s t : Nat} (h : s ≤ t) : band s ≤ band t := by
  unfold band
  by_cases hs : s * s = 0
  · rw [hs]; simp
  · have hst : s * s ≤ t * t := Nat.mul_le_mul h h
    have ht : t * t ≠ 0 := by omega
    exact (Nat.le_log2 ht).2 (Nat.le_trans (Nat.log2_self_le hs) hst)

/-! ## L1-4b  fits u8 only below 2^128 (Rust: `clamp(0,255) as u8`, so it SATURATES) -/

theorem band_lt_256_of_lt {s : Nat} (hs : 1 ≤ s) (h : s < 2 ^ 128) : band s < 256 := by
  unfold band
  have h2 : s * s < 2 ^ 256 := by
    have := Nat.mul_lt_mul'' h h
    rw [← Nat.pow_add] at this; exact this
  have hne : s * s ≠ 0 := Nat.mul_ne_zero (by omega) (by omega)
  exact (Nat.log2_lt hne).2 h2

/-- Counterexample: S = 2^128 (about 3.4e38 s, allowed by `Config::validate`,
    which only requires max_stability >= 1) has band 256, which `as u8` cannot hold;
    the Rust code clamps to 255, so every S >= 2^127.5 shares band 255. -/
theorem band_2_128 : band (2 ^ 128) = 256 := by
  unfold band
  have : (2 ^ 128 : Nat) * 2 ^ 128 = 2 ^ 256 := by rw [← Nat.pow_add]
  rw [this, Nat.log2_two_pow]

/-- With the default S_max = 10 years (315 360 000 s) the band is far below 255. -/
theorem band_default_smax : band 315360000 = 56 := by decide

/-! ## L1-4c  same band => ratio < sqrt 2 (t^2 < 2 s^2).  So the sqrt2 claim is about the
  RATIO only; it says nothing about the dt/S term (see L1-4d). -/

theorem same_band_ratio {s t : Nat} (hs : 1 ≤ s) (h : band s = band t) :
    t * t < 2 * (s * s) := by
  unfold band at h
  have hne : s * s ≠ 0 := Nat.mul_ne_zero (by omega) (by omega)
  have h1 := Nat.log2_self_le hne
  have h2 := @Nat.lt_log2_self (t * t)
  rw [← h, Nat.pow_succ] at h2
  omega

/-! ## L1-4d  the "oldest in band is weakest" claim is FALSE (counterexample)
  S_B = 4, S_A = 5 : same band (log2 16 = log2 25 = 4).
  B recalled dt_B = 12 s ago (3 half-lives), A recalled dt_A = 15 s ago (3 half-lives).
  A is OLDER (head of the band index) but STRONGER:
    strength_B = 4 * 2^-3 = 4/8, strength_A = 5 * 2^-3 = 5/8.
  We encode strength*8 (exact because dt is a whole number of half-lives). -/

def strengthEighths (s dt : Nat) : Nat := s * 8 / (2 ^ (dt / s))

theorem band_same_4_5 : band 4 = band 5 := by decide
theorem older_but_stronger :
    band 4 = band 5 ∧ (12 : Nat) < 15 ∧ strengthEighths 4 12 < strengthEighths 5 15 := by decide

/-! ## L1-3  retrieval never decreases stability (pre-clamp) -/

/-- S*(D + X)/D >= S for D>0 where X = D*gain*a*(1-R) >= 0 (all scaled to naturals). -/
theorem retrieval_monotone (S D X : Nat) (hD : 0 < D) : S ≤ S * (D + X) / D := by
  have : S * D ≤ S * (D + X) := Nat.mul_le_mul_left S (Nat.le_add_right D X)
  exact (Nat.le_div_iff_mul_le hD).2 this

theorem retrieval_fixed_when_x_zero (S D : Nat) (hD : 0 < D) : S * (D + 0) / D = S := by
  simp [Nat.mul_div_cancel _ hD]

/-! ## L1-3b  cite: 1 <= part <= full (so ratio >= 1 and the divisor is >= 1)
  Let u = (1-R) in [0,1] as un/ud, a in [0,1] as an/ad, gain g >= 0.
  part = 1 + g*(a/2)*u, full = 1 + g*u.  Scaled by 2*ad*ud. -/

theorem cite_part_ge_one (g an ad un ud : Nat) :
    2 * ad * ud ≤ 2 * ad * ud + g * an * un := Nat.le_add_right _ _

theorem cite_part_le_full (g an ad un ud : Nat) (ha : an ≤ ad) :
    2 * ad * ud + g * an * un ≤ 2 * ad * ud + 2 * g * ad * un := by
  have h1 : g * an * un ≤ g * ad * un := Nat.mul_le_mul_right un (Nat.mul_le_mul_left g ha)
  have h2 : g * ad * un ≤ 2 * g * ad * un := by
    have : 2 * g * ad * un = g * ad * un + g * ad * un := by
      rw [Nat.mul_assoc 2 g ad, Nat.mul_assoc 2 (g*ad) un]; omega
    omega
  omega

/-! ## L1-3c  `t.stability * full / part` evaluates the product first.
  ASSUME: f64 overflow of S*full to +inf happens when S*full > M (M = f64::MAX),
  and inf/part = inf, then clamp gives S_max.  Counterexample in the model:
  true result fits (<= M) but the intermediate product does not. -/

theorem cite_product_first_overflow :
    let M := 60; let S := 10; let full := 10; let part := 5
    S * full > M ∧ S * full / part ≤ M := by decide

/-! ## L1-5  gist last_recall = now - shift, with shift saturating at i64::MAX (2^63-1)
  and `now_i - shift` in plain i64 arithmetic. -/

def i64min : Int := -(2 ^ 63)
def i64max : Int := 2 ^ 63 - 1

/-- For now >= 0 no overflow is possible: now - shift >= -(2^63) when shift <= 2^63-1. -/
theorem gist_no_overflow_nonneg (now shift : Int) (h0 : 0 ≤ now) (hs : 0 ≤ shift) (hm : shift ≤ i64max) :
    i64min ≤ now - shift := by
  unfold i64min i64max at *; omega

/-- For now < 0 and a saturated shift it overflows (debug: panic, release: wraps). -/
theorem gist_overflow_negative_now :
    ∃ now shift : Int, now < 0 ∧ 0 ≤ shift ∧ shift ≤ i64max ∧ now - shift < i64min :=
  ⟨-5, i64max, by decide, by decide, by decide, by decide⟩

/-- Why this does not fire in practice: elapsed <= 64 and elapsed ~ dt/S, so shift ~ min(dt, 64 S)
  and dt <= now - last_recall.  Overflow needs now - last_recall > now + 2^63, i.e.
  last_recall < -2^63 which is impossible; only f64 rounding at last_recall = i64::MIN. -/
theorem gist_shift_bounded_by_dt (now lr shift : Int) (hlr : i64min ≤ lr) (hsh : shift ≤ now - lr) :
    i64min ≤ now - shift := by
  unfold i64min at *; omega

/-- last_recall <= now because shift >= 0 (stability >= 1, elapsed >= 0, round >= 0). -/
theorem gist_last_recall_le_now (now shift : Int) (h : 0 ≤ shift) : now - shift ≤ now := by omega

/-! ## L1-2  clamp_stability(s) in [1, S_max] given S_max >= 1 (validate), NaN -> 1.
  Model: the output is min (max s 1) smax; NaN mapped to 1 first. -/

def clampS (s smax : Nat) : Nat := min (max s 1) smax

theorem clamp_in_range (s smax : Nat) (h : 1 ≤ smax) : 1 ≤ clampS s smax ∧ clampS s smax ≤ smax := by
  unfold clampS; omega

/-- If validate allowed S_max < 1 the lower bound would fail: it does not (validate rejects it). -/
theorem clamp_fails_if_smax_lt_one : ¬ (1 ≤ clampS 5 0) := by decide

/-! ## L1-1  R = 2^-(max(0,dt)/S) in (0,1] provided S > 0 (and not NaN).
  Exponent e = max 0 dt / S is >= 0, so 2^-e <= 1.  The public `Engram::retrievability`
  does NOT clamp S: S = 0 with dt = 0 gives 0/0 = NaN; S < 0 gives exponent > 0 => R > 1. -/

theorem exponent_nonneg (dt : Int) (S : Nat) (hS : 0 < S) : 0 ≤ (max 0 dt) / (S : Int) := by
  have : 0 ≤ max 0 dt := Int.le_max_left 0 dt
  exact Int.ediv_nonneg this (by omega)

/-- negative stability: -dt/S with S<0 is a positive exponent, so R = 2^(+) > 1 -/
theorem neg_stability_exponent_positive : ∃ (dt S : Int), 0 < dt ∧ S < 0 ∧ 0 < (-dt) / S := by
  exact ⟨1, -1, by decide, by decide, by decide⟩
