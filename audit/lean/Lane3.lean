/-
  Audit lane 3 — the dream (SPEC §5, §7).  Lean 4 core only (no Mathlib).
  The LLM is an arbitrary function; the store is a function `Nat → Option Nat`
  (id ↦ abstract row state).  `sorry` only with a `-- GAP:` comment.
-/

namespace Lane3

/-! ## L3-1  seeds(): FIFO window, failed seeds last, bound, starvation -/

/-- `Engram::seeds` over the labile queue `rows` (FIFO by id).
    `u`: the row survives `checked` (unusable rows are dropped AFTER the limit);
    `failed`: ids deferred by earlier dreams; `want = 8 * budget`. -/
def seeds (want : Nat) (failed rows : List Nat) (u : Nat → Bool) : List Nat :=
  let win := (rows.take (want + failed.length)).filter u
  ((win.filter (fun x => !failed.contains x)) ++ (win.filter (fun x => failed.contains x))).take want

/-- |seeds| ≤ 8·budget (§5, §7). -/
theorem seeds_bound (want : Nat) (failed rows : List Nat) (u : Nat → Bool) :
    (seeds want failed rows u).length ≤ want := by
  unfold seeds
  exact List.length_take_le _ _

/-- Every seed comes from the first `want + |failed|` labile rows: the window is *rows*,
    not *usable rows*. -/
theorem seeds_in_window (want : Nat) (failed rows : List Nat) (u : Nat → Bool) (x : Nat)
    (h : x ∈ seeds want failed rows u) : x ∈ rows.take (want + failed.length) := by
  unfold seeds at h
  have h1 := List.mem_of_mem_take h
  simp only [List.mem_append, List.mem_filter] at h1
  rcases h1 with ⟨hx, _⟩ | ⟨hx, _⟩ <;> exact hx.1

/-- REAL BUG (L3-1a): if the first `want + |failed|` rows are all unusable, no seed is
    produced, however many usable rows follow.  (`bug_unusable_labile_rows_starve_the_queue`) -/
theorem unusable_head_starves (want : Nat) (failed rows : List Nat) (u : Nat → Bool)
    (h : ∀ x ∈ rows.take (want + failed.length), u x = false) :
    seeds want failed rows u = [] := by
  unfold seeds
  have : (rows.take (want + failed.length)).filter u = [] := by
    rw [List.filter_eq_nil_iff]
    intro a ha
    simp [h a ha]
  rw [this]
  simp

/-- `Engram::defer`: move `id` to the back and keep the newest `cap` entries
    (`cap = 8 * cfg.dream_budget`, NOT the budget passed to `dream`). -/
def defer (cap : Nat) (failed : List Nat) (id : Nat) : List Nat :=
  let f := failed.filter (fun x => x != id) ++ [id]
  f.drop (f.length - cap)

theorem defer_bound (cap : Nat) (failed : List Nat) (id : Nat) (hc : 1 ≤ cap) :
    (defer cap failed id).length ≤ cap := by
  unfold defer
  simp only [List.length_drop, List.length_append, List.length_singleton]
  omega

def deferAll (cap : Nat) (failed : List Nat) : List Nat → List Nat
  | [] => failed
  | s :: ss => deferAll cap (defer cap failed s) ss

theorem deferAll_bound (cap : Nat) (hc : 1 ≤ cap) (ss : List Nat) :
    ∀ failed : List Nat, failed.length ≤ cap → (deferAll cap failed ss).length ≤ cap := by
  induction ss with
  | nil => intro f h; simpa [deferAll] using h
  | cons s ss ih =>
    intro f _
    simp only [deferAll]
    exact ih _ (defer_bound cap f s hc)

/-- Consistency of the two bounds (§8: failed ≤ 8·dreamBudget):  when `dream(Some b)` is
    called with `b` above `cfg.dream_budget`, one dream can defer `8b` seeds but only
    `8·cfg` survive: the oldest deferrals are forgotten *inside the same dream*. -/
theorem bound_mismatch (b c : Nat) (h : c < b) : 8 * c < 8 * b := by omega

/-- Starvation theorem (L3-1b).  Whatever the verdicts are, a seed always lies in the
    first `want + cap` labile rows as long as `|failed| ≤ cap`; `deferAll_bound` keeps
    that invariant.  Hence a decidable seed at queue position `≥ want + cap` is never
    examined while the rows ahead of it stay labile (poison verdicts, fallback deferrals,
    unusable rows).  Position 16 for `budget = 1`, 80 for the default 5. -/
theorem seeds_within_want_plus_cap (want cap : Nat) (failed rows : List Nat) (u : Nat → Bool)
    (hf : failed.length ≤ cap) (x : Nat) (h : x ∈ seeds want failed rows u) :
    x ∈ rows.take (want + cap) := by
  have h1 := seeds_in_window want failed rows u x h
  have h2 : rows.take (want + failed.length) = (rows.take (want + cap)).take (want + failed.length) := by
    rw [List.take_take]
    congr 1
    omega
  rw [h2] at h1
  exact List.mem_of_mem_take h1

/-! Concrete witness of the starvation (budget 1: want = cap = 8, 16 permanently failing
    seeds at positions 0‥15 and a decidable seed at 16).  Seeds are positions. -/

def stepAllFail (want cap : Nat) (total : Nat) (failed : List Nat) : List Nat :=
  deferAll cap failed (seeds want failed (List.range total) (fun _ => true))

def iter (want cap total : Nat) : Nat → List Nat → List Nat
  | 0, f => f
  | n + 1, f => iter want cap total n (stepAllFail want cap total f)

/-- After any number of dreams in which every examined seed fails, position 16 has
    still not been examined (checked for 40 rounds; the general statement is
    `seeds_within_want_plus_cap`). -/
example : (seeds 8 (iter 8 8 20 40 []) (List.range 20) (fun _ => true)).contains 16 = false := by
  decide +kernel

/-! ## L3-2  LLM calls ≤ budget, each seed at most once, `left` accounting -/

/-- Seeds are given as "has candidates (needs a verdict)".  A seed without candidates
    costs nothing; the loop stops when `left = 0`. -/
def calls : List Bool → Nat → Nat
  | [], _ => 0
  | b :: t, l => if l = 0 then 0 else if b then 1 + calls t (l - 1) else calls t l

theorem calls_le_budget : ∀ (ss : List Bool) (l : Nat), calls ss l ≤ l := by
  intro ss
  induction ss with
  | nil => intro l; simp [calls]
  | cons b t ih =>
    intro l
    simp only [calls]
    split
    · omega
    · split
      · have := ih (l - 1); omega
      · exact ih l

theorem calls_le_seeds : ∀ (ss : List Bool) (l : Nat), calls ss l ≤ ss.length := by
  intro ss
  induction ss with
  | nil => intro l; simp [calls]
  | cons b t ih =>
    intro l
    simp only [calls, List.length_cons]
    split
    · omega
    · split
      · have := ih (l - 1); omega
      · have := ih l; omega

/-- Seeds with no candidates are free: dropping them does not change the call count
    (while budget remains). -/
theorem settled_seeds_are_free (t : List Bool) (l : Nat) (hl : 0 < l) :
    calls (false :: t) l = calls t l := by
  simp [calls]; omega

/-! ## L3-3  direction, cluster size, seed ≠ candidate, gist created_at -/

structure Tr where
  id : Nat
  at_ : Nat
deriving DecidableEq

def candidates (maxMembers : Nat) (seed : Tr) (hits : List Tr) : List Tr :=
  ((hits.filter (fun t => decide (t.at_ ≤ seed.at_))).filter (fun t => t.id != seed.id)).take (maxMembers - 1)

theorem cand_direction (m : Nat) (seed : Tr) (hits : List Tr) (c : Tr)
    (h : c ∈ candidates m seed hits) : c.at_ ≤ seed.at_ := by
  unfold candidates at h
  have := List.mem_of_mem_take h
  simp only [List.mem_filter, decide_eq_true_eq] at this
  exact this.1.2

theorem cand_not_seed (m : Nat) (seed : Tr) (hits : List Tr) (c : Tr)
    (h : c ∈ candidates m seed hits) : c.id ≠ seed.id := by
  unfold candidates at h
  have := List.mem_of_mem_take h
  simp only [List.mem_filter, bne_iff_ne, ne_eq] at this
  exact this.2

theorem cluster_size (m : Nat) (hm : 1 ≤ m) (seed : Tr) (hits : List Tr) :
    (seed :: candidates m seed hits).length ≤ m := by
  unfold candidates
  simp only [List.length_cons, List.length_take]
  omega

/-- `gists`: `newest = reduce (fun a b => if b.created_at > a.created_at then b else a)`
    over `absorbed = seed :: named candidates`.  With the direction invariant the
    gist inherits the seed's `created_at` (ties keep the first = seed). -/
def newest : Nat → List Nat → Nat
  | a, [] => a
  | a, b :: t => newest (if b > a then b else a) t

theorem newest_is_seed (seedAt : Nat) (cs : List Nat) (h : ∀ c ∈ cs, c ≤ seedAt) :
    newest seedAt cs = seedAt := by
  induction cs with
  | nil => rfl
  | cons c t ih =>
    simp only [newest]
    have hc := h c (List.mem_cons_self ..)
    have : ¬ c > seedAt := by omega
    rw [if_neg this]
    exact ih (fun x hx => h x (List.mem_cons_of_mem _ hx))

/-! ## L3-4  atomic replace / undo -/

abbrev Store := Nat → Option Nat

def inL (l : List Nat) (i : Nat) : Bool := l.contains i

/-- replace: remove `absorbed`, insert gists with values `g`. -/
def replace (s : Store) (absorbed gists : List Nat) (g : Nat → Nat) : Store :=
  fun i => if inL absorbed i then none else if inL gists i then some (g i) else s i

/-- undo (Replace): remove the gists, re-insert the absorbed rows as recorded (`orig`). -/
def undo (s : Store) (absorbed gists : List Nat) (orig : Nat → Option Nat) : Store :=
  fun i => if inL gists i then none else if inL absorbed i then orig i else s i

theorem undo_replace (s : Store) (absorbed gists : List Nat) (g : Nat → Nat)
    (fresh : ∀ i, inL gists i = true → s i = none) :       -- new ids are unused (new_id)
    undo (replace s absorbed gists g) absorbed gists s = s := by
  funext i
  unfold undo replace
  by_cases hg : inL gists i = true
  · simp [hg, fresh i hg]
  · have hg' : inL gists i = false := by simpa using hg
    by_cases ha : inL absorbed i = true
    · simp [hg', ha]
    · have ha' : inL absorbed i = false := by simpa using ha
      simp [hg', ha']

/-- Capacity (§7): a replace never raises the count if `|gists| ≤ |absorbed|`
    (`texts.len() > absorbed.len()  ⇒  keep`). -/
theorem replace_no_growth (n a g : Nat) (ha : a ≤ n) (hg : g ≤ a) : n - a + g ≤ n := by omega

/-- Settle write on an id absent from the store (seed absorbed by a later cluster in the
    same dream): `Store::apply(State)` is `get → None → return Ok`, i.e. a no-op. -/
def stateWrite (s : Store) (id : Nat) (v : Nat) : Store :=
  fun i => if i = id then (match s i with | none => none | some _ => some v) else s i

theorem state_on_removed_is_noop (s : Store) (id v : Nat) (h : s id = none) :
    stateWrite s id v = s := by
  funext i
  unfold stateWrite
  by_cases hi : i = id
  · subst hi; simp [h]
  · simp [hi]

/-- Reports of one dream, undone in DREAM order (not reverse): counter-example.
    Store = {1↦1, 2↦2, 3↦3}; r1 absorbs {1,2} into gist 10, r2 absorbs {10,3} into gist 11.
    Evaluated on ids 0‥12. -/
def s0 : Store := fun i => if i = 1 ∨ i = 2 ∨ i = 3 then some i else none
def r1 (s : Store) := replace s [1, 2] [10] (fun _ => 10)
def r2 (s : Store) := replace s [10, 3] [11] (fun _ => 11)
def snap (s : Store) : List (Option Nat) := (List.range 13).map s

-- reverse order restores exactly
example : snap (undo (undo (r2 (r1 s0)) [10, 3] [11] (r1 s0)) [1, 2] [10] s0) = snap s0 := by decide
-- GAP (design, tested as `gap_undo_in_dream_order_double_counts`): dream order leaves row 10
example : snap (undo (undo (r2 (r1 s0)) [1, 2] [10] s0) [10, 3] [11] (r1 s0)) ≠ snap s0 := by decide

/-! ## L3-5  keep + undo ; verdict guard -/

/-- Keep marks the seed consolidated; undo of a Keep makes it labile again; neither touches
    stability / last_recall (modelled by a pair `(stab, cons)`). -/
def keep (r : Nat × Bool) : Nat × Bool := (r.1, true)
def undoKeep (r : Nat × Bool) : Nat × Bool := (r.1, false)
theorem keep_undo (r : Nat × Bool) (h : r.2 = false) : undoKeep (keep r) = r := by
  cases r; simp_all [keep, undoKeep]

/-- `verdict`: replace only if `|texts| ≤ |absorbed|` (else keep).  A compound seed that
    the prompt tells the LLM to split (ids = [], ≥ 2 texts) has `|absorbed| = 1 < |texts|`. -/
def isReplace (nTexts nAbsorbed : Nat) : Bool := decide (0 < nTexts ∧ nTexts ≤ nAbsorbed)
-- GAP (spec/prompt conflict, `gap_split_of_a_compound_seed_is_kept`):
example : isReplace 2 1 = false := by decide

end Lane3
