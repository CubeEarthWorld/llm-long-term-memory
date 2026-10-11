/-! Lane 2 (capacity and eviction): models of `Engram::victims` and the Store band index.
    Core Lean only (no Mathlib). Real log2 strength is modelled by `Int` ranks (scaled);
    floats are assumed to behave as exact reals except where a theorem says otherwise. -/

namespace Lane2

/-! ## L2-1  count after remember -/

/-- `count + 1 - capacity` traces must go. -/
def excess (count cap : Nat) : Nat := (count + 1) - cap

/-- Store size after Insert(new) followed by `v` distinct removals that exist. -/
def after (count v : Nat) : Nat := count + 1 - v

/-- If victims returns at least `excess` distinct existing traces (it truncates to exactly
    `excess`), the store ends within capacity. -/
theorem l2_1_after_within_cap (count cap : Nat) (hcap : 1 ≤ cap) :
    after count (excess count cap) ≤ cap := by
  unfold after excess; omega

/-- It ends at exactly `min (count+1) cap`: no over-eviction. -/
theorem l2_1_exact (count cap : Nat) :
    after count (excess count cap) = min (count + 1) cap := by
  unfold after excess; omega

/-- Pool size in the fallback (`heads(MAX)` + new): per band at most `e + 8` entries are read.
    The pool is never smaller than `min total e`, so `truncate(excess)` yields `excess`
    victims whenever `excess ≤ total` (always true for cap ≥ 1). -/
theorem sum_min_ge (e : Nat) : ∀ ns : List Nat,
    min ns.sum e ≤ (ns.map (fun n => min n (e + 8))).sum := by
  intro ns
  induction ns with
  | nil => simp
  | cons a s ih => simp only [List.sum_cons, List.map_cons]; omega

theorem l2_2_pool_big_enough (ns : List Nat) (cap : Nat) (hcap : 1 ≤ cap) :
    let total := ns.sum + 1
    let e := total - cap
    e ≤ (ns.map (fun n => min n (e + 8))).sum + 1 := by
  intro total e
  have := sum_min_ge e ns
  omega

/-! ## L2-2  write set -/

/-- Net effect of `[Insert new] ++ victims.map Remove` on membership of `new`:
    stored iff `new` is not a victim. -/
def stored (victims : List Nat) (new : Nat) : Bool := !(victims.contains new)

example : stored [7, 3] 7 = false := by decide

/-- `Remembered::Inserted.evicted = victims.last()`. Victims are sorted weakest first, so the
    newcomer, when it is the weakest and `excess > 1`, is NOT the last victim:
    the report names another trace although the newcomer vanished (real bug, see test
    `l2_2_evicted_reports_unstored_newcomer_with_excess`). -/
theorem l2_2_evicted_misreports :
    ∃ victims : List Nat, ∃ new : Nat,
      victims.contains new = true ∧ victims.getLast? ≠ some new := by
  exact ⟨[9, 4, 5], 9, by decide, by decide⟩

/-! ## L2-3  band heads vs the global minimum -/

/-- A row: (last_recall, rank) where rank = log2 S − dt/S scaled to an Int. `none` rank is
    the unusable row (−inf). -/
abbrev Rk := Option Int

def le' : Rk → Rk → Bool
  | none, _ => true
  | some _, none => false
  | some a, some b => decide (a ≤ b)

/-- If inside a band the index order (head first) is rank-monotone, the head of the band is the
    band minimum: this is what SPEC §3 claims. -/
def Monotone : List Rk → Prop
  | [] => True
  | x :: xs => (∀ y ∈ xs, le' x y = true) ∧ Monotone xs

theorem head_is_min (x : Rk) (xs : List Rk) (h : Monotone (x :: xs)) :
    ∀ y ∈ x :: xs, le' x y = true := by
  intro y hy
  cases hy with
  | head => cases x <;> simp [le']
  | tail _ hy' => exact h.1 y hy'

/-- Counterexample to "head = min" (L2-3): one band, index order by last_recall, ten rows with
    rank −326 (older last_recall, higher S) then one row with rank −446 (newer last_recall,
    lower S).  With depth `excess + 8 = 9` the window is the ten... first nine, whose minimum is
    −326, while the global minimum is −446.  Numbers come from SPEC formulae at
    now = last_recall + 3.15e7 s (see lane2.md). -/
def band32 : List Rk := List.replicate 10 (some (-326)) ++ [some (-446)]

def minRk : List Rk → Rk
  | [] => some 0
  | x :: xs => if le' x (minRk xs) then x else minRk xs

theorem l2_3_depth_window_misses_min :
    minRk (band32.take 9) ≠ minRk band32 := by decide

theorem l2_3_window_covers_when_small (xs : List Rk) (e : Nat) (h : xs.length ≤ e + 8) :
    xs.take (e + 8) = xs := List.take_of_length_le h

/-- Unusable rows (rank −inf) are subject to the same window: ten usable rows ahead hide it. -/
def bandU : List Rk := List.replicate 10 (some (-1)) ++ [none]

theorem l2_3_unusable_hidden : minRk (bandU.take 9) ≠ minRk bandU := by decide

/-! ## L2-4  grace / flood arithmetic -/

/-- Non-flood branch requires `young ≤ capacity / 10`, and `young` always counts the newcomer
    (`created_at > cutoff` when grace > 0 and the clock has not stopped): for cap < 10 the
    branch is unreachable, i.e. grace is void. -/
theorem l2_4_small_cap_void (cap young : Nat) (hc : cap < 10) (hy : 1 ≤ young) :
    ¬ (young ≤ cap / 10) := by omega

/-- Young (`created_at > cutoff`) and Band-eligible (`created_at ≤ cutoff`) partition the rows:
    no row is both protected and a candidate, none is neither. -/
theorem l2_4_partition (created cutoff : Int) :
    (created > cutoff) ≠ (created ≤ cutoff) := by
  intro h
  have : (created > cutoff) ↔ (created ≤ cutoff) := by rw [h]
  omega

/-! ## L2-5  immortality and index consistency -/

/-- `checked` clamps `last_recall` to `now`; retrievability uses `max 0 (now − lr)`.
    For a stored `lr` in the future, the effective elapsed time is 0 at every `now ≤ lr`:
    the trace does not decay until the clock reaches `lr`. -/
def elapsed (now lr : Int) : Int := max 0 (now - min lr now)

theorem l2_5_future_lr_frozen (now lr : Int) (h : now ≤ lr) : elapsed now lr = 0 := by
  unfold elapsed; omega

theorem l2_5_past_lr_decays (now lr : Int) (h : lr < now) : 0 < elapsed now lr := by
  unfold elapsed; omega

/-- A band index entry is `(band, last_recall, id)`.  State does
    `erase (band (parse S_old), …)` then `insert (band S_new, …)`. If `parse` moves `S_old`
    across a band edge, the erase misses and the old entry stays (orphan). -/
def stateWrite (idx : List (Nat × Nat)) (bandOfParsed bandNew lr : Nat) : List (Nat × Nat) :=
  (bandNew, lr) :: idx.erase (bandOfParsed, lr)

/-- Entry inserted under `bandAtInsert`; the row later reads back with `bandOfParsed`. -/
theorem l2_5_orphan_when_band_differs (lr bandAtInsert bandOfParsed bandNew : Nat)
    (h : bandAtInsert ≠ bandOfParsed) :
    (bandAtInsert, lr) ∈ stateWrite [(bandAtInsert, lr)] bandOfParsed bandNew lr := by
  unfold stateWrite
  simp [List.erase_cons]
  exact Or.inr h

/-- Conversely, with an exact round trip (`bandOfParsed = bandAtInsert`) and the row moving to
    another entry, the index has exactly one entry for the row. -/
theorem l2_5_sync_when_exact (lr b b' : Nat) :
    (stateWrite [(b, lr)] b b' lr).length = 1 := by
  simp [stateWrite]

/-- Insert / State / Index / Remove each keep "one band entry per row" provided the parse is
    exact (GAP: full induction over arbitrary op sequences is not mechanised here; the Rust
    test `l2_5_store_indexes_stay_in_sync_random` covers it empirically). -/
theorem l2_5_ops_preserve_sync : True := by
  trivial -- GAP: model of the redb tables not mechanised

end Lane2
