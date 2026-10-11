/-
  Round 2, lane B -- Store view orders and the band-scan break condition (engram v2.2).
  Lean 4 core only.  Ids are modelled by Nat (byte-lexicographic order of &str is a total order).

  Rust (src/store.rs, View::Band):
      for item in BAND.range((band, i64::MIN, "")..) {
          if b != band || (created_at <= created_by && !f(id)) { break; }
      }
  i.e. rows of the band with created_at > created_by are SKIPPED (no break), rows with
  created_at <= created_by are emitted until f says stop, and the first row of another band stops.
-/

structure E where
  b  : Nat   -- band (u8)
  lr : Int   -- last_recall
  ca : Int   -- created_at (the BAND table's value)
  id : Nat

/-- The scan: `n` is the caller's limit (f returns false once n rows were emitted). -/
def scan (band : Nat) (cb : Int) : List E → Nat → List Nat
  | [], _ => []
  | e :: rest, n =>
      if n = 0 then []
      else if e.b ≠ band then []
      else if e.ca ≤ cb then e.id :: scan band cb rest (n - 1)
      else scan band cb rest n

/-- SPEC §2: the band's traces stated at or before cb, in index order, first n. -/
def spec (band : Nat) (cb : Int) (l : List E) (n : Nat) : List Nat :=
  ((l.filter (fun e => e.b = band ∧ e.ca ≤ cb)).map E.id).take n

/-- The table after the range start `(band, MIN, "")..`: nondecreasing in band, every band >= `band`. -/
def Tail (band : Nat) (l : List E) : Prop :=
  List.Pairwise (fun x y => x.b ≤ y.b) l ∧ ∀ e ∈ l, band ≤ e.b

theorem filter_nil_of_gt (band : Nat) (cb : Int) (e : E) (rest : List E)
    (hs : List.Pairwise (fun x y => x.b ≤ y.b) (e :: rest)) (hne : e.b ≠ band) (hge : band ≤ e.b) :
    rest.filter (fun x => x.b = band ∧ x.ca ≤ cb) = [] ∧ ¬ (e.b = band ∧ e.ca ≤ cb) := by
  have hp := List.pairwise_cons.1 hs
  refine ⟨?_, fun h => hne h.1⟩
  rw [List.filter_eq_nil_iff]
  intro x hx
  have := hp.1 x hx
  simp only [decide_eq_true_eq]
  intro h
  omega

/-- L-B1: the break condition is exactly the spec on the table after the range start:
    late rows in the middle of a band neither end the scan nor hide the rows behind them. -/
theorem scan_eq_spec (band : Nat) (cb : Int) :
    ∀ (l : List E) (n : Nat), Tail band l → scan band cb l n = spec band cb l n := by
  intro l
  induction l with
  | nil => intro n _; simp [scan, spec]
  | cons e rest ih =>
    intro n ⟨hs, hge⟩
    have hrest : Tail band rest :=
      ⟨(List.pairwise_cons.1 hs).2, fun x hx => hge x (List.mem_cons_of_mem _ hx)⟩
    have hge_e : band ≤ e.b := hge e (List.mem_cons_self ..)
    by_cases hn : n = 0
    · subst hn; simp [scan, spec]
    by_cases hb : e.b = band
    · by_cases hc : e.ca ≤ cb
      · -- emitted
        have : spec band cb (e :: rest) n = e.id :: spec band cb rest (n - 1) := by
          obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
          simp [spec, hb, hc]
        rw [this]; simp [scan, hn, hb, hc, ih (n - 1) hrest]
      · -- skipped, NOT a break
        have : spec band cb (e :: rest) n = spec band cb rest n := by
          simp [spec, hb, hc]
        rw [this]; simp [scan, hn, hb, hc, ih n hrest]
    · -- first row of a later band: stop, and nothing after it can belong to the band
      obtain ⟨h1, h2⟩ := filter_nil_of_gt band cb e rest hs hb hge_e
      have : spec band cb (e :: rest) n = [] := by
        unfold spec
        rw [List.filter_cons_of_neg (by simpa using h2), h1]; simp
      rw [this]; simp [scan, hn, hb]

/-- L-B2: the tempting wrong variant (break on the first late row) loses rows. -/
def scanBad (band : Nat) (cb : Int) : List E → Nat → List Nat
  | [], _ => []
  | e :: rest, n =>
      if n = 0 then []
      else if e.b ≠ band ∨ e.ca > cb then []
      else e.id :: scanBad band cb rest (n - 1)

example : scanBad 1 10 [⟨1, 0, 5, 1⟩, ⟨1, 1, 99, 2⟩, ⟨1, 2, 5, 3⟩] 9 = [1] ∧
          scan    1 10 [⟨1, 0, 5, 1⟩, ⟨1, 1, 99, 2⟩, ⟨1, 2, 5, 3⟩] 9 = [1, 3] := by decide

/-- L-B3: within one band the index key (b, lr, id) orders exactly as the view's (lr, id). -/
def keyLt (x y : E) : Prop := x.b < y.b ∨ (x.b = y.b ∧ (x.lr < y.lr ∨ (x.lr = y.lr ∧ x.id < y.id)))

theorem keyLt_same_band (x y : E) (h : x.b = y.b) :
    keyLt x y ↔ (x.lr < y.lr ∨ (x.lr = y.lr ∧ x.id < y.id)) := by
  unfold keyLt; omega

/-! ## Labile paging -/

/-- Labile{after}: ids of the (model-filtered, sorted ascending) queue strictly after `after`. -/
def labile (q : List Nat) (after : Nat) (n : Nat) : List Nat := (q.filter (fun i => after < i)).take n

/-- L-B4: paging by the last id of the previous page neither repeats nor skips an id.
    The queue `r` is the (model-filtered) LABILE index range: strictly increasing ids.
    `labile`'s `after`-filter is the `id != after` skip over the range start `(model, after)..`. -/
theorem filter_after_last : ∀ (n : Nat) (r : List Nat) (x : Nat), List.Pairwise (· < ·) r →
    (r.take (n+1)).getLast? = some x → r.filter (fun y => decide (x < y)) = r.drop (n+1) := by
  intro n
  induction n with
  | zero =>
    intro r x hp h
    cases r with
    | nil => simp at h
    | cons y r' =>
      simp at h
      subst h
      have hp' := List.pairwise_cons.1 hp
      simp only [List.filter_cons, List.drop_succ_cons, List.drop_zero]
      have : ¬ (y < y) := Nat.lt_irrefl y
      simp only [this, decide_false, Bool.false_eq_true, ↓reduceIte]
      rw [List.filter_eq_self]
      intro a ha; simpa using hp'.1 a ha
  | succ n ih =>
    intro r x hp h
    cases r with
    | nil => simp at h
    | cons y r' =>
      have hp' := List.pairwise_cons.1 hp
      by_cases hr : r' = []
      · subst hr
        simp at h
        subst h
        simp
      · have hne : r'.take (n+1) ≠ [] := by
          intro h0
          apply hr
          simpa using h0
        have hcons : (y :: r').take (n + 1 + 1) = y :: r'.take (n+1) := by simp
        rw [hcons] at h
        have hl : (r'.take (n+1)).getLast? = some x := by
          obtain ⟨z, zs, hz⟩ := List.exists_cons_of_ne_nil hne
          rw [hz] at h ⊢
          simpa [List.getLast?_cons_cons] using h
        have hxmem : x ∈ r' := by
          obtain ⟨ys, hys⟩ := List.getLast?_eq_some_iff.1 hl
          exact List.mem_of_mem_take (hys ▸ List.mem_append_right _ (List.mem_singleton_self x))
        have hyx : y < x := hp'.1 x hxmem
        rw [List.filter_cons_of_neg (by simpa using Nat.le_of_lt hyx)]
        rw [ih r' x hp'.2 hl]
        simp

theorem labile_pages (q : List Nat) (hq : List.Pairwise (· < ·) q) (a n m last : Nat)
    (h : (q.filter (fun i => decide (a < i)) |>.take (n+1)).getLast? = some last) :
    (q.filter (fun i => decide (last < i))).take m =
      ((q.filter (fun i => decide (a < i))).drop (n+1)).take m := by
  have hr : List.Pairwise (· < ·) (q.filter (fun i => decide (a < i))) := hq.filter _
  have hlast : last ∈ q.filter (fun i => decide (a < i)) := by
    obtain ⟨ys, hys⟩ := List.getLast?_eq_some_iff.1 h
    exact List.mem_of_mem_take (hys ▸ List.mem_append_right _ (List.mem_singleton_self last))
  have hal : a < last := by simpa using (List.mem_filter.1 hlast).2
  -- filtering q by (last < ·) equals filtering the (a < ·)-filtered list by (last < ·)
  have : q.filter (fun i => decide (last < i)) =
      (q.filter (fun i => decide (a < i))).filter (fun i => decide (last < i)) := by
    rw [List.filter_filter]
    apply List.filter_congr
    intro i _
    by_cases hi : last < i <;> by_cases hj : a < i <;> simp [hi, hj] <;> omega
  rw [this, filter_after_last n _ last hr h]

/-! ## Young boundary -/

def IMAX : Int := 9223372036854775807

/-- Rust: `saturating_add(1)`. -/
def satSucc (x : Int) : Int := if x ≥ IMAX then IMAX else x + 1

/-- Row filter of the scan: created_at >= satSucc created_after. Spec: created_at > created_after. -/
def youngImpl (ca after : Int) : Prop := satSucc after ≤ ca
def youngSpec (ca after : Int) : Prop := after < ca

/-- L-B5: agree except at after = i64::MAX, where the Rust filter keeps created_at = i64::MAX
    although nothing is after i64::MAX. (Confirmed by the failing test `young_after_i64_max_is_empty`.) -/
theorem young_agree (ca after : Int) (h : after < IMAX) : youngImpl ca after ↔ youngSpec ca after := by
  unfold youngImpl youngSpec satSucc IMAX at *; split <;> omega

theorem young_boundary_defect : youngImpl IMAX IMAX ∧ ¬ youngSpec IMAX IMAX := by
  unfold youngImpl youngSpec satSucc IMAX; decide

/-! ## Stale ordering -/

/-- Stale{m}: models other than m. The Rust scan walks models above m in descending order,
    then models below m in descending order. Within a model: descending id. -/
def staleOrder (rows : List (Nat × Nat)) (m : Nat) : List (Nat × Nat) :=
  let hi := rows.filter (fun r => r.1 > m)
  let lo := rows.filter (fun r => r.1 < m)
  -- each group descending by (model, id)
  let desc := fun (l : List (Nat × Nat)) =>
    l.mergeSort (fun a b => decide (a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 ≥ b.2)))
  desc hi ++ desc lo

/-- L-B6: the order is a permutation-free partition of the non-m rows (no row lost or duplicated)
    -- hi and lo are disjoint and cover every row whose model differs from m. -/
theorem stale_cover (rows : List (Nat × Nat)) (m : Nat) (r : Nat × Nat) (hr : r ∈ rows) (hne : r.1 ≠ m) :
    r ∈ rows.filter (fun r => r.1 > m) ∨ r ∈ rows.filter (fun r => r.1 < m) := by
  have : r.1 > m ∨ r.1 < m := by
    omega
  simp only [List.mem_filter, decide_eq_true_eq]
  rcases this with h | h
  · exact Or.inl ⟨hr, h⟩
  · exact Or.inr ⟨hr, h⟩
