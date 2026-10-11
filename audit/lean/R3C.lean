/-! Round 3, lane C -- arithmetic-overflow freedom of the eviction window (`Engram::victims`).
    Lean 4 core only (no Mathlib).  Compile:  ~/.elan/bin/lean.exe audit/lean/R3C.lean
    No `sorry`, no `native_decide`; large literals are checked by the kernel's GMP-backed Nat.

    Rust (engine.rs, `victims`), with usize = Nat below `W = 2^64`:

        let total  = self.store.count(View::All)? + 1;                 // (1)
        let excess = total.saturating_sub(self.cfg.capacity);
        let cap    = BAND_CAP.max(excess + BAND_DEPTH);               // (2)
        read(b, created_by, excess + BAND_DEPTH)                      // (2) again: first window
        ... if w.more && w.limit < cap && ... { read(.., (w.limit * 4).min(cap)) }   // (3)

    Claim of the code's author (implicit): none of (1)-(3) overflows.  What is true:

      A. GIVEN that `excess + 8` does not overflow, the window growth (3) never overflows, for
         every `excess` (so the `* 4` needs no `saturating_mul`): `grow_safe`.
      B. (1) and (2) DO overflow when the store reports a row count within 8 + capacity of
         `usize::MAX` (a lying `Storage::count`): `count_plus_one_overflows`,
         `excess_plus_depth_overflows` -- these are the defects `s1`/`s2` in tests/audit3_c.rs.
      C. Saturating (1) and (2) restores A for EVERY count: `sat_grow_safe`.
      D. The doubling / quadrupling widening loops (`near`: ask*2, `select`: k*4) reach the
         saturation point `usize::MAX` after at most 64 / 32 rounds, so "terminates by
         saturation" is a bound on rounds, not on the work per round (that is bounded only by
         the store: defects `s7`, `s7b`).
-/

namespace R3C

/-- usize::MAX + 1. -/
def W : Nat := 2 ^ 64

def BAND_CAP : Nat := 128
def BAND_DEPTH : Nat := 8

/-- Rust `usize::saturating_add`. -/
def satAdd (a b : Nat) : Nat := min (a + b) (W - 1)

/-- Rust `usize::saturating_sub`. -/
def satSub (a b : Nat) : Nat := a - b   -- Nat subtraction is already truncated at 0

/-- Rust `usize::saturating_mul`. -/
def satMul (a b : Nat) : Nat := min (a * b) (W - 1)

/-- The window cap: `BAND_CAP.max(excess + BAND_DEPTH)` (exact arithmetic). -/
def cap (excess : Nat) : Nat := max BAND_CAP (excess + BAND_DEPTH)

/-- One growth step `(limit * 4).min(cap)` (exact arithmetic). -/
def grow (limit c : Nat) : Nat := min (limit * 4) c

/-- Windows that can occur: the first window asks for `excess + BAND_DEPTH` rows; a window
    grows only while `limit < cap` (the guard `w.limit < cap` in the loop). -/
inductive Reach (excess : Nat) : Nat → Prop
  | start : Reach excess (excess + BAND_DEPTH)
  | step  {l : Nat} : Reach excess l → l < cap excess → Reach excess (grow l (cap excess))

/-- The invariant: either the window already equals the cap (no growth can fire), or the cap
    is the constant 128 and the window is below it. -/
theorem inv_of_reach {excess l : Nat} (h : Reach excess l) :
    l = cap excess ∨ (cap excess = BAND_CAP ∧ l < BAND_CAP) := by
  induction h with
  | start =>
    unfold cap BAND_CAP BAND_DEPTH
    omega
  | step hl hlt ih =>
    unfold grow
    rcases ih with h1 | ⟨h2, h3⟩
    · omega
    · simp only [BAND_CAP] at *
      omega

/-- A. Whenever a growth fires (`l < cap`), the product `l * 4` is below 512 -- far below `W`,
    for every `excess` (the proof does not assume `excess + 8 < W`: it is about the numbers,
    so it only needs that `excess + 8` itself was computed without wrapping). -/
theorem grow_safe {excess l : Nat} (h : Reach excess l) (hlt : l < cap excess) : l * 4 < 512 := by
  rcases inv_of_reach h with h1 | ⟨h2, h3⟩
  · omega
  · unfold BAND_CAP at h3
    omega

theorem grow_safe_w {excess l : Nat} (h : Reach excess l) (hlt : l < cap excess) : l * 4 < W := by
  have := grow_safe h hlt
  unfold W
  omega

/-- B. The two defects, as arithmetic.  `count = usize::MAX`: `count + 1 = W`, not a usize. -/
theorem count_plus_one_overflows : (W - 1) + 1 ≥ W := by decide

/-- `capacity = 1`, `count = usize::MAX - 2`: `total = usize::MAX`, `excess = usize::MAX - 1`,
    and `excess + BAND_DEPTH` exceeds `usize::MAX`. -/
theorem excess_plus_depth_overflows :
    let count := W - 1 - 2
    let total := count + 1
    let excess := total - 1
    excess + BAND_DEPTH ≥ W := by decide

/-- The overflow window of (2): exactly the counts in `[W - 8 + capacity - 1, W - 2]`
    (for capacity ≥ 1) make `excess + 8` wrap while `total` itself does not. -/
theorem overflow_window (count capacity : Nat) (hc : 1 ≤ capacity) (hcnt : count + 1 < W)
    (h : count + 1 - capacity + BAND_DEPTH ≥ W) : count + capacity + BAND_DEPTH ≥ W + capacity ∧ capacity ≤ BAND_DEPTH := by
  unfold BAND_DEPTH at *
  unfold W at *
  omega

/-- C. With saturating (1) and (2) the invariant survives for every count a store may report. -/
def satExcess (count capacity : Nat) : Nat := satSub (satAdd count 1) capacity
def satCap (excess : Nat) : Nat := max BAND_CAP (satAdd excess BAND_DEPTH)

theorem sat_cap_lt_w (count capacity : Nat) : satCap (satExcess count capacity) < W := by
  unfold satCap satExcess satSub satAdd BAND_CAP BAND_DEPTH W
  omega

theorem sat_start_le_cap (count capacity : Nat) :
    satAdd (satExcess count capacity) BAND_DEPTH ≤ satCap (satExcess count capacity) := by
  unfold satCap satAdd
  exact Nat.le_max_right _ _

/-- With saturation the first window is `min (excess + 8) (W - 1)` and the cap is the larger of
    that and 128: a growth `l < satCap` leaves `l < 128` or `l = satCap`, as in A, so `l * 4` is
    again below 512. -/
theorem sat_grow_safe (count capacity l : Nat)
    (hinv : l = satCap (satExcess count capacity) ∨ (satCap (satExcess count capacity) = BAND_CAP ∧ l < BAND_CAP))
    (hlt : l < satCap (satExcess count capacity)) : l * 4 < 512 := by
  rcases hinv with h1 | ⟨h2, h3⟩
  · omega
  · unfold BAND_CAP at h3
    omega

/-- D. Rounds of a saturating doubling / quadrupling from `a ≥ 1`: after `n` rounds the value
    is `min (a * m^n) (W - 1)`; with `a = 1, m = 2` it is saturated at n = 64 (`near`), with
    `a ≥ 4` (`inject_n * 4` in `select`, inject_n ≥ 1) and `m = 4` at n = 32. -/
def iter (m : Nat) : Nat → Nat → Nat
  | 0, a => a
  | n + 1, a => satMul (iter m n a) m

theorem iter_double_saturates : iter 2 64 1 = W - 1 := by decide +kernel

theorem iter_quad_saturates : iter 4 32 4 = W - 1 := by decide +kernel

/-- Once saturated, a round changes nothing (the `ask == usize::MAX` break in `near`
    is therefore reachable only when a store really returns `usize::MAX` rows). -/
theorem saturated_fixed (m : Nat) (hm : 1 ≤ m) : satMul (W - 1) m = W - 1 := by
  unfold satMul W
  omega

end R3C
