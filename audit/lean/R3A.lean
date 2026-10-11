/-
  Round-3 audit A — `Engram::near` widening loop.  Lean 4 core only.  No `sorry`.

  Model.  The store holds rows sorted by its own key `s` (the unscaled dot), highest first.
  A row has a true cosine `t` (the dot of the repaired vector) and may be unusable.
  `Store::near(ask)` = `rows.take ask`.  The engine keeps the usable hits and widens
  (`ask := 2·ask`) until it has `k` of them or the store ran out (`hits.length < ask`).

  Part 1  termination: with fuel ≥ |rows|+1 and ask ≥ 1 the loop stops by its own test
          (never by running out of fuel): `loop_stops`.  (The code's fuel is the 64 doublings
          of `usize`; `|rows| < 2^64`.)
  Part 2  soundness for exact rows (`t = s`): when the loop returns `k` usable hits, every row
          it did not see has `t ≤ kth`, the bound `select` uses (`unseen_le_kth`); when it returns
          fewer than `k`, nothing was left unseen (`exhausted_nothing_unseen`).
  Part 3  the regression: with a repaired (rescaled) row `t ≠ s`, the bound is false —
          concrete counterexamples (`rescaled_outranks`, `underscaled_hidden`).
-/

namespace R3A

structure R where
  s : Nat        -- the store's key (unscaled cosine, ×10)
  t : Nat        -- the true cosine of the repaired vector (×10)
  usable : Bool

/-- `Store::near(ask)`: the store's top `ask` rows (`rows` is sorted by `s`, descending). -/
def hitsOf (rows : List R) (ask : Nat) : List R := rows.take ask

def usableOf (hs : List R) : List R := hs.filter (·.usable)

/-- The widening loop of `Engram::near`; returns the usable hits and the final `ask`. -/
def loop : Nat → List R → Nat → Nat → List R × Nat
  | 0, rows, _, ask => (usableOf (hitsOf rows ask), ask)
  | fuel + 1, rows, k, ask =>
      let hits := hitsOf rows ask
      let u := usableOf hits
      if k ≤ u.length ∨ hits.length < ask then (u, ask) else loop fuel rows k (2 * ask)

/-- The loop's stop test (the same disjunction). -/
def stops (rows : List R) (k ask : Nat) : Prop :=
  k ≤ (usableOf (hitsOf rows ask)).length ∨ rows.length < ask

theorem take_len_lt (rows : List R) (ask : Nat) : (hitsOf rows ask).length < ask ↔ rows.length < ask := by
  unfold hitsOf
  rw [List.length_take]
  omega

/-- Part 1.  With `|rows| < ask + fuel` and `ask ≥ 1` the returned `ask'` satisfies the stop test,
    and the result is exactly the usable part of the store's top `ask'`. -/
theorem loop_stops : ∀ (fuel : Nat) (rows : List R) (k ask : Nat), 1 ≤ ask → rows.length < ask + fuel →
    stops rows k (loop fuel rows k ask).2 ∧ (loop fuel rows k ask).1 = usableOf (hitsOf rows (loop fuel rows k ask).2)
  | 0, rows, k, ask, _, h => by
      simp only [loop, stops]
      exact ⟨Or.inr (by omega), trivial⟩
  | fuel + 1, rows, k, ask, h1, h => by
      simp only [loop]
      by_cases c : k ≤ (usableOf (hitsOf rows ask)).length ∨ (hitsOf rows ask).length < ask
      · simp only [c, ↓reduceIte]
        refine ⟨?_, trivial⟩
        simp only [stops]
        rcases c with c | c
        · exact Or.inl c
        · exact Or.inr ((take_len_lt rows ask).1 c)
      · simp only [c, ↓reduceIte]
        have hlen : ¬ (hitsOf rows ask).length < ask := fun h' => c (Or.inr h')
        have : ask ≤ rows.length := by
          have := Nat.not_lt.1 (fun h' => hlen ((take_len_lt rows ask).2 h'))
          omega
        exact loop_stops fuel rows k (2 * ask) (by omega) (by omega)

/-- The doubling reaches every store size: `fuel = |rows| + 1` always suffices. -/
theorem terminates (rows : List R) (k ask : Nat) (h : 1 ≤ ask) :
    stops rows k (loop (rows.length + 1) rows k ask).2 :=
  (loop_stops (rows.length + 1) rows k ask h (by omega)).1

/-- The number of store calls is at most `|rows| + 1` (the real bound is logarithmic: `ask` doubles). -/
theorem ask_grows (rows : List R) (k ask : Nat) (h : 1 ≤ ask) (fuel : Nat) :
    ask ≤ (loop fuel rows k ask).2 := by
  induction fuel generalizing ask with
  | zero => simp [loop]
  | succ n ih =>
      simp only [loop]
      split
      · exact Nat.le_refl _
      · exact Nat.le_trans (by omega) (ih (2 * ask) (by omega))

/-! ## Part 2 — soundness for exact rows -/

def Sorted (rows : List R) : Prop := rows.Pairwise (fun a b => b.s ≤ a.s)

/-- Everything the store did not return is no nearer (by key) than anything it did. -/
theorem unseen_le_seen (rows : List R) (ask : Nat) (hs : Sorted rows) :
    ∀ r ∈ rows.drop ask, ∀ h ∈ hitsOf rows ask, r.s ≤ h.s := by
  intro r hr h hh
  have := List.pairwise_append.1 (by
    have e : rows.take ask ++ rows.drop ask = rows := List.take_append_drop ask rows
    rw [e]; exact hs)
  exact this.2.2 h hh r hr

/-- `select`'s bound: if `k ≥ 1` usable hits were found, the key of the k-th is at least the key of any
    row not seen, hence (rows exact: `t = s`) at least its true cosine. -/
theorem unseen_le_kth (rows : List R) (ask k : Nat) (hs : Sorted rows) (hk : 1 ≤ k)
    (hu : k ≤ (usableOf (hitsOf rows ask)).length) (exact : ∀ r ∈ rows, r.t = r.s) :
    ∀ r ∈ rows.drop ask, r.t ≤ ((usableOf (hitsOf rows ask)).get ⟨k - 1, by omega⟩).t := by
  intro r hr
  have hmem : (usableOf (hitsOf rows ask)).get ⟨k - 1, by omega⟩ ∈ hitsOf rows ask := by
    have := List.get_mem (usableOf (hitsOf rows ask)) ⟨k - 1, by omega⟩
    exact (List.mem_filter.1 this).1
  have h1 := unseen_le_seen rows ask hs r hr _ hmem
  have rin : r ∈ rows := List.mem_of_mem_drop hr
  have hin : (usableOf (hitsOf rows ask)).get ⟨k - 1, by omega⟩ ∈ rows :=
    List.mem_of_mem_take hmem
  rw [exact r rin, exact _ hin]
  exact h1

/-- When fewer than `k` usable hits come back, the loop stopped because the store ran out: nothing unseen. -/
theorem exhausted_nothing_unseen (rows : List R) (k ask : Nat)
    (hst : stops rows k ask) (hlt : (usableOf (hitsOf rows ask)).length < k) : rows.drop ask = [] := by
  rcases hst with h | h
  · omega
  · exact List.drop_eq_nil_of_le (by omega)

/-! ## Part 3 — the regression: a repaired vector breaks the key/true-cosine identification -/

/-- Store keys (×10): the rescaled row (norm 2, true cosine 0.6) has key 12 and outranks the unit row
    (true 0.8, key 8).  With `k = 1` the engine returns the rescaled row, with kth = 6, while an
    unseen row has true cosine 8 > 6: `select`'s exactness claim ("no unseen row can exceed kth") fails,
    and `remember`'s nearest is the 0.6 row, not the 0.8 one. -/
def rescaledRows : List R := [⟨12, 6, true⟩, ⟨8, 8, true⟩]

theorem rescaled_outranks :
    Sorted rescaledRows ∧
    (loop 8 rescaledRows 1 1).1.map (·.t) = [6] ∧
    ∃ r ∈ rescaledRows.drop 1, 6 < r.t := by
  refine ⟨?_, ?_, ?_⟩
  · simp [Sorted, rescaledRows]
  · decide
  · exact ⟨⟨8, 8, true⟩, by simp [rescaledRows], by decide⟩

/-- An under-scaled row (norm 0.5, true 9.9 ≈ 0.99, key 4.95) is below the store's threshold 5.5
    and never returned: `θ`-filtering by key hides a row whose true cosine clears it. -/
def thetaKey : Nat := 55

def underscaled : R := ⟨50, 99, true⟩

theorem underscaled_hidden : underscaled.s < thetaKey ∧ thetaKey ≤ underscaled.t := by
  decide

/-- The repair that would make the loop sound again: an exact row is one whose key equals its true
    cosine; `unseen_le_kth` holds for exact rows (Part 2), so the engine would have to ask the store
    for rows by the *repaired* cosine, or lower `min_cos` by the largest possible shrink. -/
theorem sound_when_exact (rows : List R) (ask k : Nat) (hs : Sorted rows) (hk : 1 ≤ k)
    (hu : k ≤ (usableOf (hitsOf rows ask)).length) (exact : ∀ r ∈ rows, r.t = r.s) :
    ∀ r ∈ rows.drop ask, r.t ≤ ((usableOf (hitsOf rows ask)).get ⟨k - 1, by omega⟩).t :=
  unseen_le_kth rows ask k hs hk hu exact

end R3A
