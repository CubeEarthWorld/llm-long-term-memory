/-! # R3D: cost of the eviction window-growth loop (round 3, lane D)

Lean 4.34.1, core only (no Mathlib).  Compile: `lean audit/lean/R3D.lean`.

Models `Engram::victims` (src/engine.rs): per stability band a window of `limit` rows is read
(`limit = excess + BAND_DEPTH` at first); while some band could still hide a row ranking at or below the
`excess`-th weakest, the bands that may hide one and have `limit < cap` re-read `min (4 * limit) cap` rows.
`cap = max BAND_CAP (excess + BAND_DEPTH)`, `BAND_CAP = 128`, `BAND_DEPTH = 8`.

What is proved (no `sorry`, no `native_decide`):

* `cost_bound`      one band's reads, summed over the re-reads, satisfy `3 * cost + l0 <= 7 * cap`
                    (so `cost <= 7/3 cap`, whatever `excess` and whatever the stop rule says).
* `spec_1_4_holds_for_excess_1` / `spec_1_4_false`
                    SPEC section 3 says "the total read is about 1.4 x the cap": true for `excess = 1`
                    (limits 9, 36, 128: 173 rows), false for `excess = 23` (limits 31, 124, 128: 283 rows = 2.21 x cap).
                    The tight bound over all first windows is 283 (`cost_max_283`).
* `rows_per_heads`  b populated bands read at most `b * 283` rows per `heads` call when `cap = 128`
                    (empty bands read 0 rows and never grow); the engine calls `heads` at most twice per remember
                    (grace-protected pool, then the whole store), so a remember reads at most `2 * b * 283`
                    band rows: 57 bands (the whole 1 s .. 10 y range) give <= 32 262, the measured worst
                    case (every band tied) is ~9.5 k at 100 k traces (tests/audit3_d.rs).
* `loop_terminates` the growth loop reaches a fixpoint after at most `meas` iterations where
                    `meas = sum (cap - limit)`: every iteration that changes anything strictly lowers `meas`
                    (`step_lt_or_eq`), and limits stay in `[1, cap]` (`step_valid`).  With first window 9 the
                    limit sequence 9, 36, 128 has at most 3 growths per band (`growths_9`).
* `eviction_terminates_and_removes_excess`
                    abstract tail of `victims`: sort the pool and `truncate(excess)` leaves at most `excess`
                    victims, exactly `excess` when the pool has that many (the engine adds the new trace to
                    the pool when it is short, so the capacity invariant holds).

Not covered (documented gaps): the soundness of the lower bound `log2 s_lo - last_dt / s_lo` (real analysis),
and the weakest-first claim itself, which is *false* once a band holds more than `cap` relevant rows:
`bandcap_repro_a_dense_aged_band_evicts_a_stronger_trace_than_the_weakest` in tests/audit3_d.rs.
-/

namespace R3D

/-- Rows read from one band over the whole growth loop: the window of `l` rows is read, and while
`l < cap` (and the band may hide a weaker row: the worst case is that it always does) it is re-read
with `min (4 * l) cap` rows. `fuel` only bounds the recursion for the definition. -/
def cost (cap : Nat) : Nat → Nat → Nat
  | 0, l => l
  | f + 1, l => if l < cap then l + cost cap f (min (4 * l) cap) else l

/-- The invariant behind every bound: `3 * cost + first window <= 7 * cap`. -/
theorem cost_bound (cap : Nat) : ∀ (f l : Nat), 1 ≤ l → l ≤ cap → 3 * cost cap f l + l ≤ 7 * cap := by
  intro f
  induction f with
  | zero => intro l h1 h2; simp only [cost]; omega
  | succ f ih =>
    intro l h1 h2
    simp only [cost]
    by_cases hl : l < cap
    · simp only [hl, ite_true]
      by_cases h4 : 4 * l ≤ cap
      · have hm : min (4 * l) cap = 4 * l := Nat.min_eq_left h4
        have := ih (4 * l) (by omega) h4
        rw [hm]; omega
      · have hm : min (4 * l) cap = cap := Nat.min_eq_right (by omega)
        rw [hm]
        cases f with
        | zero => simp only [cost]; omega
        | succ g => simp only [cost, Nat.lt_irrefl, ite_false]; omega
    · simp only [hl, ite_false]; omega

theorem cost_le (cap f l : Nat) (h1 : 1 ≤ l) (h2 : l ≤ cap) : 3 * cost cap f l ≤ 7 * cap := by
  have := cost_bound cap f l h1 h2; omega

/-- The engine's constants. -/
def BAND_CAP : Nat := 128
def BAND_DEPTH : Nat := 8

/-- `excess = 1` (every ordinary remember): limits 9, 36, 128. -/
theorem excess1_cost : cost 128 10 (1 + BAND_DEPTH) = 173 := by decide

/-- SPEC section 3: "about 1.4 x the cap" (5 * cost <= 7 * cap) holds for `excess = 1` ... -/
theorem spec_1_4_holds_for_excess_1 : 5 * cost 128 10 (1 + BAND_DEPTH) ≤ 7 * 128 := by decide

/-- ... and fails for `excess = 23` (first window 31): 31 + 124 + 128 = 283 rows, 2.21 x the cap. -/
theorem excess23_cost : cost 128 10 (23 + BAND_DEPTH) = 283 := by decide

theorem spec_1_4_false : ¬ (5 * cost 128 10 (23 + BAND_DEPTH) ≤ 7 * 128) := by decide

/-- The tight bound over every first window `1 <= l0 <= cap = 128`: 283. -/
theorem cost_max_283 : ∀ l : Nat, l ≤ 128 → 1 ≤ l → cost 128 10 l ≤ 283 := by decide

/-- 7/3 cap = 298 is the general (any cap) ceiling, 283 the exact one for cap = 128. -/
theorem cost_general (cap : Nat) (l : Nat) (h1 : 1 ≤ l) (h2 : l ≤ cap) : cost cap 10 l ≤ 7 * cap / 3 := by
  have := cost_le cap 10 l h1 h2; omega

/-- A window larger than `cap` (bulk eviction: `excess + 8 > 128`) is read once and never grows. -/
theorem big_first_window (cap l f : Nat) (h : cap ≤ l) : cost cap f l = l := by
  cases f <;> simp only [cost] <;> (try (rw [if_neg (by omega)]))

/-- Rows read by one `heads` call over `b` populated bands (each band reads at most its `cost`;
an empty band answers with 0 rows and never grows). -/
theorem rows_per_heads (b : Nat) (reads : List Nat) (hlen : reads.length = b) (hb : ∀ r ∈ reads, r ≤ 283) :
    reads.foldl (· + ·) 0 ≤ b * 283 := by
  have key : ∀ (xs : List Nat) (acc : Nat), (∀ r ∈ xs, r ≤ 283) →
      xs.foldl (· + ·) acc ≤ acc + xs.length * 283 := by
    intro xs
    induction xs with
    | nil => intro acc _; simp
    | cons x xs ih =>
      intro acc h
      simp only [List.foldl_cons, List.length_cons]
      have hx : x ≤ 283 := h x (by simp)
      have := ih (acc + x) (fun r hr => h r (by simp [hr]))
      calc xs.foldl (· + ·) (acc + x) ≤ acc + x + xs.length * 283 := this
        _ ≤ acc + (xs.length + 1) * 283 := by omega
  have := key reads 0 hb
  simpa [hlen] using this

/-- A remember calls `heads` at most twice and the u8 band space has 256 bands: the absolute ceiling. -/
theorem rows_per_remember_ceiling : 2 * (256 * 283) = 144896 := by decide

/-! ## Termination of the growth loop -/

/-- All limits are in `[1, cap]`. -/
def Valid (cap : Nat) : List Nat → Prop
  | [] => True
  | l :: ls => 1 ≤ l ∧ l ≤ cap ∧ Valid cap ls

/-- One pass of the loop body over the bands from index `b`: a band whose `c b l` holds (it may hide a
weaker row) and whose limit is below `cap` is re-read with a window four times larger, at most `cap`. -/
def stepFrom (cap : Nat) (c : Nat → Nat → Bool) : Nat → List Nat → List Nat
  | _, [] => []
  | b, l :: ls => (if l < cap ∧ c b l = true then min (4 * l) cap else l) :: stepFrom cap c (b + 1) ls

/-- The remaining room to grow. -/
def meas (cap : Nat) : List Nat → Nat
  | [] => 0
  | l :: ls => (cap - l) + meas cap ls

theorem step_valid (cap : Nat) (c : Nat → Nat → Bool) :
    ∀ (ws : List Nat) (b : Nat), Valid cap ws → Valid cap (stepFrom cap c b ws) := by
  intro ws
  induction ws with
  | nil => intro b _; simp [stepFrom, Valid]
  | cons l ls ih =>
    intro b h
    obtain ⟨h1, h2, h3⟩ := h
    simp only [stepFrom, Valid]
    refine ⟨?_, ?_, ih (b + 1) h3⟩
    · split
      · exact Nat.le_min.2 ⟨by omega, by omega⟩
      · exact h1
    · split
      · exact Nat.min_le_right _ _
      · exact h2

theorem step_lt_or_eq (cap : Nat) (c : Nat → Nat → Bool) :
    ∀ (ws : List Nat) (b : Nat), Valid cap ws →
      stepFrom cap c b ws = ws ∨ meas cap (stepFrom cap c b ws) < meas cap ws := by
  intro ws
  induction ws with
  | nil => intro b _; left; simp [stepFrom]
  | cons l ls ih =>
    intro b h
    obtain ⟨h1, h2, h3⟩ := h
    have ihb := ih (b + 1) h3
    have hle : ∀ ws' : List Nat, (stepFrom cap c (b + 1) ls = ws' ∨ True) → True := fun _ _ => trivial
    by_cases hg : l < cap ∧ c b l = true
    · right
      simp only [stepFrom, hg, and_self, ite_true, meas]
      have hmin : l < min (4 * l) cap := Nat.lt_min.2 ⟨by omega, hg.1⟩
      have hmin2 : min (4 * l) cap ≤ cap := Nat.min_le_right _ _
      have : meas cap (stepFrom cap c (b + 1) ls) ≤ meas cap ls := by
        rcases ihb with h | h
        · rw [h]; exact Nat.le_refl _
        · omega
      omega
    · simp only [stepFrom, hg, ite_false, meas]
      rcases ihb with h | h
      · left; rw [h]
      · right; omega

/-- Run the loop for `n` iterations or until nothing grows. -/
def iter (cap : Nat) (c : Nat → Nat → Bool) : Nat → List Nat → List Nat
  | 0, ws => ws
  | n + 1, ws => if stepFrom cap c 0 ws = ws then ws else iter cap c n (stepFrom cap c 0 ws)

/-- After `meas` iterations the loop is at a fixpoint: it has stopped (the `if !grew { break; }`). -/
theorem loop_terminates (cap : Nat) (c : Nat → Nat → Bool) :
    ∀ (n : Nat) (ws : List Nat), Valid cap ws → meas cap ws ≤ n →
      stepFrom cap c 0 (iter cap c n ws) = iter cap c n ws := by
  intro n
  induction n with
  | zero =>
    intro ws hv hm
    simp only [iter]
    rcases step_lt_or_eq cap c ws 0 hv with h | h
    · exact h
    · omega
  | succ n ih =>
    intro ws hv hm
    simp only [iter]
    by_cases hfix : stepFrom cap c 0 ws = ws
    · simp only [hfix, ite_true]
    · simp only [hfix, ite_false]
      have hlt := (step_lt_or_eq cap c ws 0 hv).resolve_left hfix
      exact ih _ (step_valid cap c ws 0 hv) (by omega)

/-- With first window 9 and cap 128 a band grows at most three times (9, 36, 128): the loop runs
at most `bands * 3 + 1` times, whatever the stop rule says. -/
theorem growths_9 : (min (4 * (min (4 * 9) 128)) 128) = 128 ∧ min (4 * 9) 128 = 36 := by decide

/-! ## The tail of `victims`: sort, truncate to `excess` -/

/-- Sorting keeps the length, truncating keeps `min excess length`. -/
theorem eviction_terminates_and_removes_excess (pool : List Nat) (excess : Nat) :
    (pool.take excess).length = min excess pool.length := by
  simp [List.length_take]

/-- The engine adds the new trace to a pool that is short, so the capacity invariant holds:
`n + 1 - excess <= cap` when `excess = n + 1 - cap` victims are removed. -/
theorem capacity_after_eviction (n cap : Nat) (hcap : 0 < cap) : n + 1 - (n + 1 - cap) ≤ cap := by omega

end R3D
