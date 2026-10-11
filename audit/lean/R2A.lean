/-
  Round-2 audit A — the dream cursor cycle and the eviction window growth.
  Lean 4 core only (no Mathlib).  No `sorry`.

  Part 1  `Engram::seeds` / `dream` cursor:  rot L c = (ids after c) ++ (ids up to c),
          seeds = take want (rot L c), the cursor moves to the last seed examined.
          Facts: |seeds| <= want; no duplicate between the two passes; every labile id is
          reached when want >= |L|; the cursor may name a deleted/consolidated id;
          after the cursor moves, rot is exactly the rotation of the old cycle
          (`rot_step`), hence every seed is examined within ceil(n/k) dreams when each
          dream examines k >= 1 seeds (`fairness`).
          (The real code reads pages; `pages_eq` shows the page loop = take of the filter.)
  Part 2  `Engram::victims` window growth: soundness of the lower bound (rank is monotone
          in S and antitone in elapsed time; unread rows have S >= s_lo and dt <= last_dt,
          also for last_recall in the future, S < 1, S > S_max), termination, and the
          arithmetic of the 9 -> 36 -> 144 -> 512 reads (701 rows > BAND_CAP).
-/

namespace R2A

/-! ## Part 1 — the cursor cycle -/

/-- ids after the cursor, then (wrapping) those up to it: the two passes of `seeds`. -/
def rot (L : List Nat) (c : Nat) : List Nat :=
  L.filter (fun x => decide (c < x)) ++ L.filter (fun x => decide (x ≤ c))

def seeds (want : Nat) (L : List Nat) (c : Nat) : List Nat := (rot L c).take want

theorem seeds_len (want : Nat) (L : List Nat) (c : Nat) : (seeds want L c).length ≤ want := by
  unfold seeds; exact List.length_take_le _ _

theorem rot_perm (L : List Nat) (c : Nat) : (rot L c).Perm L := by
  unfold rot
  have h : (L.filter (fun x => decide (x ≤ c))) = L.filter (fun x => !decide (c < x)) := by
    apply List.filter_congr
    intro x _
    by_cases h : x ≤ c <;> simp [h] <;> omega
  rw [h]
  exact List.filter_append_perm _ L

/-- No id is examined twice in one dream (nothing duplicated between the two passes). -/
theorem seeds_nodup (want : Nat) (L : List Nat) (c : Nat) (h : L.Nodup) : (seeds want L c).Nodup := by
  unfold seeds
  exact (((rot_perm L c).nodup_iff.2 h).sublist (List.take_sublist _ _))

/-- With `want >= |L|` the whole queue is examined: nothing is skipped by the wrap. -/
theorem seeds_all (want : Nat) (L : List Nat) (c : Nat) (h : L.length ≤ want) :
    (seeds want L c).Perm L := by
  unfold seeds
  rw [List.take_of_length_le (by rw [(rot_perm L c).length_eq]; exact h)]
  exact rot_perm L c

/-- Every seed comes from the queue (cursor id deleted or not: `c` is only a number). -/
theorem seeds_sub (want : Nat) (L : List Nat) (c : Nat) (x : Nat) (h : x ∈ seeds want L c) : x ∈ L :=
  (rot_perm L c).subset (List.mem_of_mem_take h)

/-- For a sorted queue the two filters split it: `L = (ids <= c) ++ (ids > c)`. -/
theorem split_sorted (c : Nat) : ∀ L : List Nat, L.Pairwise (· < ·) →
    L.filter (fun x => decide (x ≤ c)) ++ L.filter (fun x => decide (c < x)) = L := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons x xs ih =>
    intro h
    rw [List.pairwise_cons] at h
    by_cases hx : x ≤ c
    · have hn : ¬ c < x := by omega
      simp [List.filter_cons, hx, hn, ih h.2]
    · have hx' : c < x := by omega
      have h0 : xs.filter (fun y => decide (y ≤ c)) = [] := by
        rw [List.filter_eq_nil_iff]
        intro a ha
        have := h.1 a ha
        simp; omega
      have h1 : xs.filter (fun y => decide (c < y)) = xs := by
        rw [List.filter_eq_self]
        intro a ha
        have := h.1 a ha
        simp; omega
      simp [List.filter_cons, hx, hx', h0, h1]

/-- `L = P ++ Q` sorted and the cursor is the last of `P`: the cycle starts at `Q`
    and ends with `P`. -/
theorem rot_split (P Q : List Nat) (hs : (P ++ Q).Pairwise (· < ·)) (c : Nat)
    (hc : P.getLast? = some c) : rot (P ++ Q) c = Q ++ P := by
  rw [List.pairwise_append] at hs
  obtain ⟨hP, hQ, hPQ⟩ := hs
  unfold rot
  have hcP : c ∈ P := List.mem_of_getLast? hc
  have hPle : ∀ a ∈ P, a ≤ c := by
    intro a ha
    obtain ⟨ys, hys⟩ := List.getLast?_eq_some_iff.1 hc
    rw [hys] at ha hP
    rw [List.pairwise_append] at hP
    rcases List.mem_append.1 ha with ha | ha
    · have := hP.2.2 a ha c (by simp)
      omega
    · simp at ha; omega
  have hQgt : ∀ b ∈ Q, c < b := fun b hb => hPQ c hcP b hb
  rw [List.filter_append, List.filter_append]
  have e1 : P.filter (fun x => decide (c < x)) = [] := by
    rw [List.filter_eq_nil_iff]; intro a ha; have := hPle a ha; simp; omega
  have e2 : Q.filter (fun x => decide (c < x)) = Q := by
    rw [List.filter_eq_self]; intro a ha; have := hQgt a ha; simp; omega
  have e3 : P.filter (fun x => decide (x ≤ c)) = P := by
    rw [List.filter_eq_self]; intro a ha; have := hPle a ha; simp; omega
  have e4 : Q.filter (fun x => decide (x ≤ c)) = [] := by
    rw [List.filter_eq_nil_iff]; intro a ha; have := hQgt a ha; simp; omega
  rw [e1, e2, e3, e4]
  simp

/-- One dream that examines `k` seeds and parks the cursor on the last one rotates the cycle. -/
def step (k : Nat) (R : List Nat) : List Nat := R.drop k ++ R.take k

theorem rot_step (L : List Nat) (hs : L.Pairwise (· < ·)) (c0 : Nat) (k : Nat)
    (_hk1 : 1 ≤ k) (hkn : k ≤ L.length) (c : Nat)
    (hc : (seeds k L c0).getLast? = some c) : rot L c = step k (rot L c0) := by
  have _ := hkn
  -- L = B ++ A with B = ids <= c0, A = ids > c0; the old cycle is A ++ B
  have hL : L.filter (fun x => decide (x ≤ c0)) ++ L.filter (fun x => decide (c0 < x)) = L := split_sorted c0 L hs
  generalize hB : L.filter (fun x => decide (x ≤ c0)) = B at hL
  generalize hA : L.filter (fun x => decide (c0 < x)) = A at hL
  have hR : rot L c0 = A ++ B := by unfold rot; rw [hA, hB]
  have hlen : B.length + A.length = L.length := by rw [← hL]; simp
  unfold seeds at hc
  rw [hR] at hc
  unfold step
  rw [hR]
  by_cases hka : k ≤ A.length
  · have e1 : (A ++ B).take k = A.take k := List.take_append_of_le_length hka
    rw [e1] at hc
    have eL : L = (B ++ A.take k) ++ A.drop k := by
      rw [List.append_assoc, List.take_append_drop, hL]
    have hc' : (B ++ A.take k).getLast? = some c := by
      rw [List.getLast?_append, hc]; simp
    have hs' : ((B ++ A.take k) ++ A.drop k).Pairwise (· < ·) := by rw [← eL]; exact hs
    have key := rot_split (B ++ A.take k) (A.drop k) hs' c hc'
    rw [show rot L c = rot ((B ++ A.take k) ++ A.drop k) c by rw [← eL], key]
    rw [List.drop_append_of_le_length hka, List.take_append_of_le_length hka]
    simp [List.append_assoc]
  · have hj : 1 ≤ k - A.length := by omega
    have hjB : k - A.length ≤ B.length := by omega
    have e1 : (A ++ B).take k = A ++ B.take (k - A.length) := by
      rw [List.take_append, List.take_of_length_le (by omega)]
    rw [e1] at hc
    have hne : B.take (k - A.length) ≠ [] := by
      intro h0
      have := List.length_take_of_le hjB
      rw [h0] at this; simp at this; omega
    have hc' : (B.take (k - A.length)).getLast? = some c := by
      rw [List.getLast?_append] at hc
      cases h : (B.take (k - A.length)).getLast? with
      | some y => rw [h] at hc; simpa using hc
      | none => exact absurd (List.getLast?_eq_none_iff.1 h) hne
    have eL : L = B.take (k - A.length) ++ (B.drop (k - A.length) ++ A) := by
      rw [← List.append_assoc, List.take_append_drop, hL]
    have hs' : (B.take (k - A.length) ++ (B.drop (k - A.length) ++ A)).Pairwise (· < ·) := by rw [← eL]; exact hs
    have key := rot_split (B.take (k - A.length)) (B.drop (k - A.length) ++ A) hs' c hc'
    rw [show rot L c = rot (B.take (k - A.length) ++ (B.drop (k - A.length) ++ A)) c by rw [← eL], key]
    have d1 : A.drop k = [] := List.drop_of_length_le (by omega)
    have t1 : A.take k = A := List.take_of_length_le (by omega)
    rw [List.drop_append, List.take_append, d1, t1]
    simp [List.append_assoc]

/-- The n-th cycle: `n` dreams of `k` seeds each, the cycle rotating every time. -/
def cyc (k : Nat) : Nat → List Nat → List Nat
  | 0, R => R
  | n + 1, R => cyc k n (step k R)

/-- Whatever sits in the first `m·k` places of the cycle is examined within `m` dreams. -/
theorem fairness_aux (k : Nat) : ∀ (m : Nat) (R : List Nat) (x : Nat),
    x ∈ R.take (m * k) → ∃ j, j < m ∧ x ∈ (cyc k j R).take k := by
  intro m
  induction m with
  | zero => intro R x h; simp at h
  | succ m ih =>
    intro R x h
    by_cases hx : x ∈ R.take k
    · exact ⟨0, by omega, by simpa [cyc] using hx⟩
    · have h2 : R.take ((m + 1) * k) = R.take k ++ (R.drop k).take (m * k) := by
        rw [show (m + 1) * k = k + m * k by rw [Nat.add_mul]; omega, List.take_add]
      rw [h2] at h
      rcases List.mem_append.1 h with h | h
      · exact absurd h hx
      · have h3 : x ∈ (step k R).take (m * k) := by
          unfold step
          have hp : ((R.drop k).take (m * k)) <+: ((R.drop k ++ R.take k).take (m * k)) := by
            rw [List.take_append]
            exact List.prefix_append _ _
          exact hp.subset h
        obtain ⟨j, hj, hjx⟩ := ih (step k R) x h3
        exact ⟨j + 1, by omega, by simpa [cyc] using hjx⟩

/-- ceil(n/k) dreams of `k` seeds reach every id of a cycle of length `n`. -/
theorem fairness (k : Nat) (hk : 1 ≤ k) (R : List Nat) (x : Nat) (hx : x ∈ R) :
    ∃ j, j < (R.length + k - 1) / k ∧ x ∈ (cyc k j R).take k := by
  apply fairness_aux k ((R.length + k - 1) / k) R x
  have hq := Nat.div_add_mod (R.length + k - 1) k
  have hr := Nat.mod_lt (R.length + k - 1) (by omega : 0 < k)
  rw [Nat.mul_comm]
  have hge : R.length ≤ k * ((R.length + k - 1) / k) := by omega
  have hp : R.take R.length <+: R.take (k * ((R.length + k - 1) / k)) := List.take_prefix_take_left hge
  rw [List.take_of_length_le (Nat.le_refl _)] at hp
  exact hp.subset hx

/-- The cursor after a dream that examined `k` seeds (the last one; unchanged if none). -/
def cursorAfter (k : Nat) (L : List Nat) (c : Nat) : Nat := ((seeds k L c).getLast?).getD c

/-- `j` dreams, each parking the cursor on the last seed it examined. -/
def curs (k : Nat) (L : List Nat) : Nat → Nat → Nat
  | 0, c => c
  | j + 1, c => curs k L j (cursorAfter k L c)

theorem curs_rot (k : Nat) (L : List Nat) (hs : L.Pairwise (· < ·)) (hk : 1 ≤ k) (hkn : k ≤ L.length) :
    ∀ (j c : Nat), rot L (curs k L j c) = cyc k j (rot L c) := by
  intro j
  induction j with
  | zero => intro c; rfl
  | succ j ih =>
    intro c
    have hlen : (seeds k L c).length = k := by
      unfold seeds; rw [List.length_take, (rot_perm L c).length_eq]; omega
    have hsome : ∃ y, (seeds k L c).getLast? = some y := by
      cases h : (seeds k L c).getLast? with
      | some y => exact ⟨y, rfl⟩
      | none => have := List.getLast?_eq_none_iff.1 h; rw [this] at hlen; simp at hlen; omega
    obtain ⟨y, hy⟩ := hsome
    have hc : cursorAfter k L c = y := by unfold cursorAfter; rw [hy]; rfl
    show rot L (curs k L j (cursorAfter k L c)) = cyc k j (step k (rot L c))
    rw [ih, hc, rot_step L hs c k hk hkn y hy]

/-- **Fairness of the cursor.**  With `k >= 1` seeds examined per dream (the LLM never
    decides, so every seed stays labile), the queue `L` (sorted, any starting cursor, which
    may name a deleted id) is fully examined within ceil(n/k) dreams. -/
theorem cursor_fair (k : Nat) (L : List Nat) (hs : L.Pairwise (· < ·)) (hk : 1 ≤ k) (hkn : k ≤ L.length)
    (c0 : Nat) (x : Nat) (hx : x ∈ L) :
    ∃ j, j < (L.length + k - 1) / k ∧ x ∈ seeds k L (curs k L j c0) := by
  have hR : x ∈ rot L c0 := (rot_perm L c0).symm.subset hx
  obtain ⟨j, hj, hxj⟩ := fairness k hk (rot L c0) x hR
  refine ⟨j, ?_, ?_⟩
  · rw [(rot_perm L c0).length_eq] at hj; exact hj
  · unfold seeds; rw [curs_rot k L hs hk hkn]; exact hxj

/-! The page loop of `seeds`: `rest` = the ids after `after` (sorted), `u` = usable.
    Each page asks for `want - |out|` rows, appends the usable ones, and moves `after` to the
    last row of a full page.  It stops on a short page, so it terminates, and its result is
    the first `want` usable ids — rows that are unusable never hide the ones behind them. -/
def pages (u : Nat → Bool) (want : Nat) : Nat → List Nat → List Nat → List Nat
  | 0, _, out => out
  | f + 1, rest, out =>
    if want ≤ out.length then out
    else
      let asked := want - out.length
      let page := rest.take asked
      let out' := out ++ page.filter u
      if page.length = asked then pages u want f (rest.drop asked) out' else out'

theorem pages_eq (u : Nat → Bool) (want : Nat) :
    ∀ (f : Nat) (rest out : List Nat), rest.length < f → out.length ≤ want →
      pages u want f rest out = (out ++ rest.filter u).take want := by
  intro f
  induction f with
  | zero => intro rest out h; omega
  | succ f ih =>
    intro rest out hf ho
    unfold pages
    by_cases hw : want ≤ out.length
    · have h1 : out.length = want := by omega
      rw [if_pos hw, List.take_append_of_le_length hw, List.take_of_length_le (by omega)]
    · rw [if_neg hw]
      dsimp only
      by_cases hp : (rest.take (want - out.length)).length = want - out.length
      · rw [if_pos hp]
        rw [List.length_take] at hp
        have hlen : (out ++ (rest.take (want - out.length)).filter u).length ≤ want := by
          have := List.length_filter_le u (rest.take (want - out.length))
          rw [List.length_append]
          have h2 := List.length_take_le (want - out.length) rest
          omega
        have hr : (rest.drop (want - out.length)).length < f := by
          rw [List.length_drop]; omega
        rw [ih _ _ hr hlen, List.append_assoc, ← List.filter_append, List.take_append_drop]
      · rw [if_neg hp]
        rw [List.length_take] at hp
        have hs : rest.length < want - out.length := by omega
        have e : rest.take (want - out.length) = rest := List.take_of_length_le (by omega)
        rw [e]
        symm
        apply List.take_of_length_le
        rw [List.length_append]
        have := List.length_filter_le u rest
        omega

/-! ## Part 2 — `victims`: the lower bound that stops a band window growing

  rank(S, dt) = log2 S - dt / S   (the engine's `rank`, compared in the log domain).
  Reals are not in core Lean, so rank is abstract here and only the two properties the
  proof uses are assumed; both are the sign of the partial derivatives
      d rank / dS  = 1/(S ln 2) + dt/S^2  > 0         d rank / d dt = -1/S < 0
  and `frac_*` below shows them on the underlying fractions by cross-multiplication
  (dt/S <= dt'/S' iff dt*S' <= dt'*S).  Floating-point rounding is covered by the engine's
  `s_lo * (1 - 1e-9)` slack, which only lowers the bound (lower bound => grows more). -/

/-- dt/S' <= dt/S when S <= S' (dt >= 0, S > 0). -/
theorem frac_den_anti (d s s' : Int) (hd : 0 ≤ d) (_hs : 0 < s) (h : s ≤ s') : d * s ≤ d * s' :=
  Int.mul_le_mul_of_nonneg_left h hd

/-- dt'/S <= dt/S when dt' <= dt (S > 0). -/
theorem frac_num_mono (d d' s : Int) (hs : 0 < s) (h : d' ≤ d) : d' * s ≤ d * s :=
  Int.mul_le_mul_of_nonneg_right h (Int.le_of_lt hs)

/-- `Engram::checked` on stability: NaN and below 1 read as 1; above S_max read as S_max. -/
def clampS (smax s : Int) : Int := min (max s 1) smax

/-- Elapsed time as the engine reads it: `last_recall` is clamped to now, dt >= 0. -/
def dtOf (now lr : Int) : Int := max 0 (now - min lr now)

/-- Index order is raw `last_recall` ascending; the engine's dt is antitone in it, also when
    some `last_recall` lies in the future (dt = 0 there). -/
theorem dt_antitone (now lr lr' : Int) (h : lr ≤ lr') : dtOf now lr' ≤ dtOf now lr := by
  unfold dtOf; omega

theorem dt_nonneg (now lr : Int) : 0 ≤ dtOf now lr := by unfold dtOf; omega

/-- A row of band `lo` (`lo = 2^(b/2) >= 1`): raw S >= lo — or, for band 0 only, any
    S < 1 / NaN / garbage (`Trace::band` sends them to band 0 where lo = 1). -/
def inBand (lo raw : Int) : Prop := lo ≤ raw ∨ (lo = 1 ∧ raw < 1)

/-- Whatever the raw stability (below 1, above S_max, NaN as -inf), the stability the engine
    ranks with is at least `min lo S_max` — the `s_lo` of the code. -/
theorem slo_sound (smax lo raw : Int) (h1 : 1 ≤ lo) (hs : 1 ≤ smax) (hb : inBand lo raw) :
    min lo smax ≤ clampS smax raw := by
  unfold inBand at hb; unfold clampS; omega

section
variable (rank : Int → Int → Int)
  (hS : ∀ s s' d, 1 ≤ s → s ≤ s' → rank s d ≤ rank s' d)
  (hD : ∀ s d d', 0 ≤ d' → d' ≤ d → rank s d ≤ rank s d')
include hS hD

/-- **Soundness of the window bound.**  Rows are read in `last_recall` order; `lastlr` is the
    last row read, so every unread row has `lastlr <= lr`.  Every unread row ranks at least
    `bound = rank (min lo S_max) (dt lastlr)`, clamped or not, future `last_recall` included. -/
theorem unread_rank_ge (now smax lo lastlr lr raw : Int) (h1 : 1 ≤ lo) (hs : 1 ≤ smax)
    (hb : inBand lo raw) (hord : lastlr ≤ lr) :
    rank (min lo smax) (dtOf now lastlr) ≤ rank (clampS smax raw) (dtOf now lr) := by
  have a := slo_sound smax lo raw h1 hs hb
  have b := dt_antitone now lastlr lr hord
  have c := dt_nonneg now lr
  have e1 : rank (min lo smax) (dtOf now lastlr) ≤ rank (min lo smax) (dtOf now lr) :=
    hD _ _ _ c b
  have e2 : rank (min lo smax) (dtOf now lr) ≤ rank (clampS smax raw) (dtOf now lr) :=
    hS _ _ _ (by omega) a
  exact Int.le_trans e1 e2

/-- The stopping rule: if `w < bound` (the code's `bound > weakest`), the `excess` rows chosen
    from the window (all with rank <= w) are weaker than every unread usable row, so the
    window already contains the true `excess` weakest of the band. -/
theorem stop_is_sound (now smax lo lastlr : Int) (h1 : 1 ≤ lo) (hs : 1 ≤ smax) (w : Int)
    (hw : w < rank (min lo smax) (dtOf now lastlr))
    (chosen unread : List (Int × Int))
    (hc : ∀ r ∈ chosen, rank (clampS smax r.1) (dtOf now r.2) ≤ w)
    (hu : ∀ u ∈ unread, inBand lo u.1 ∧ lastlr ≤ u.2) :
    ∀ r ∈ chosen, ∀ u ∈ unread,
      rank (clampS smax r.1) (dtOf now r.2) < rank (clampS smax u.1) (dtOf now u.2) := by
  intro r hr u hu'
  have := unread_rank_ge rank hS hD now smax lo lastlr u.2 u.1 h1 hs (hu u hu').1 (hu u hu').2
  have := hc r hr
  omega
end

/-- The window growth terminates: each growth step strictly raises the limit until `cap`
    (so at most log_4 steps), also when `excess + 8` already exceeds BAND_CAP
    (`cap = max 512 (excess + 8)` is never below the first window). -/
theorem grow_progress (limit cap : Nat) (h1 : 1 ≤ limit) (h : limit < cap) : limit < min (4 * limit) cap := by
  omega

theorem first_window_fits (excess : Nat) : excess + 8 ≤ max 512 (excess + 8) := by omega

/-- Total rows one band costs, growing 4x from `l` to `cap` and re-reading from the head each
    time (the window is read afresh: `read(b, created_by, limit)`). -/
def reads (cap : Nat) : Nat → Nat → Nat
  | 0, l => l
  | f + 1, l => if l < cap then l + reads cap f (min (4 * l) cap) else l

/-- BAND_CAP is 512, but one dense band whose bound can never be proven (the typical
    idle store: all rows fresh in one band) costs 9 + 36 + 144 + 512 = 701 rows per remember. -/
theorem dense_band_costs_701 : reads 512 8 9 = 701 := by decide

theorem dense_band_exceeds_cap : 512 < reads 512 8 9 := by decide

/-- ... and with `excess = 1` that holds for every band that cannot be proven: 41 bands in the
    measured store (tests/audit2_a.rs, `measure_victims_cost`) read 28 742 rows. -/
theorem cost_per_unproven_band (b : Nat) : b * 701 = reads 512 8 9 * b := by
  rw [dense_band_costs_701]; exact Nat.mul_comm _ _

end R2A
