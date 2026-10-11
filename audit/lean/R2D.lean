/-! # R2D: the whole engine as one transition system (round 2, lane D)

Lean 4.34.1, core only (no Mathlib).  Compile: `lean audit/lean/R2D.lean`.

State = the trace rows + the dream cursor + the engine's bounded memory (`expo`, `wt`)
+ two ghost fields (`next`: the id counter the `IdGen` contract promises, `gone`: every id
ever removed).  Steps = remember / recall / cite / forget / dream (+ stutter for the
operations that change nothing: rejected, rate limited, peek, errors that roll back).

Abstractions (each is a documented *gap*, see the end of the file and `audit/round2_d.md`):
* floats, vectors, text and the similarity search are not modelled, only `stab : Nat`
  clamped to `[1, smax]`, the labile flag and the id;
* eviction is "any victims that leave exactly `min (n+1) cap` rows" (weakest-first optimality
  is a floating-point ranking; it is checked against a brute-force oracle by the fuzz harness);
* ids are fresh by the `IdGen` contract (`t.id = next`); the engine itself only refuses an id
  that is *live*, so no-resurrection is a theorem about contract-keeping generators;
* undo is proved only for "dream, then immediately undo_all" (`dream_undo_restores`).
  Late / repeated undo is outside the model (witness: `note_stale_undo_resurrects_forgotten_trace`).

Invariants (SPEC section 7 and 8) and their status:

| #  | invariant                                                    | status |
|----|--------------------------------------------------------------|--------|
| 1  | count <= capacity after remember / dream                     | PROVED  `Inv.cap` |
| 2  | 1 <= stability <= S_max (new, recalled, cited, gisted rows)  | PROVED  `Inv.range` |
| 3  | ids unique among live rows                                   | PROVED  `Inv.nodup` |
| 4  | no resurrection of a removed id                              | PROVED  `Inv.nores` (needs fresh ids) |
| 5  | exposures name live traces only, at most inject_n            | PROVED  `Inv.expo_sub`, `Inv.expo_len` |
| 6  | rate window never holds more than writes_per_day entries     | PROVED  `Inv.wt_len` |
| 7  | recall / cite / peek never change the id set                 | PROVED  `recall_ids`, `cite_ids` |
| 8  | recall only rewrites the named rows                          | PROVED  `recall_frames` |
| 9  | the labile flag is cleared only by keep / replace / settle   | PROVED  `remember_keeps_labile`, `recall_keeps_labile`, `forget_keeps_labile` |
| 10 | recall and cite never lower stability                        | PROVED  `recall_monotone`, `cite_monotone` |
| 11 | a dream never increases the count                            | PROVED  `dream_len_le` |
| 12 | LLM verdicts per dream <= budget                             | PROVED  `dream_calls_le_budget` |
| 13 | at most one verdict per seed                                 | PROVED  `dream_seed_once` |
| 14 | every removal by a dream is in a report                      | PROVED  `dream_removals_reported` |
| 15 | undo of one verdict restores the rows exactly                | PROVED  `undo_replace_rows`, `undo_keep_rows`; chains: fuzz `I7-undo-not-exact` |
| 16 | recall pack within budget (first line always fits)           | PROVED  `pack_within_budget` |
| 17 | 0 <= R <= 1, vectors unit norm, single model, text rules     | GAP (floats / strings): fuzz `I1-*`, `I2-*`, `I5-*` |
| 18 | eviction removes the weakest eligible traces                 | GAP (float ranking): fuzz oracle `I4-eviction-not-weakest` |
| 19 | determinism, time translation invariance                     | GAP (not a safety property of this model): `fuzz_*`, `fuzz_time_translation` |
| 20 | no trace is ever lost to a failed write (atomicity)          | GAP (store contract): fuzz `I-err-but-applied` finds DEFECT D1 |
-/

namespace R2D

structure Trace where
  id   : Nat
  stab : Nat
  lab  : Bool
  deriving DecidableEq, Repr

structure Params where
  cap   : Nat
  smax  : Nat
  hcap  : 1 ≤ cap
  hsmax : 1 ≤ smax

def clampS (P : Params) (x : Nat) : Nat := max 1 (min x P.smax)

theorem clampS_range (P : Params) (x : Nat) : 1 ≤ clampS P x ∧ clampS P x ≤ P.smax := by
  have := P.hsmax; unfold clampS; omega

theorem clampS_ge (P : Params) {s x : Nat} (h1 : 1 ≤ s) (h2 : s ≤ P.smax) (h : s ≤ x) : s ≤ clampS P x := by
  unfold clampS; omega

structure State where
  rows   : List Trace
  cursor : Nat
  next   : Nat
  gone   : List Nat
  expo   : List Nat
  wt     : List Nat

def ids (l : List Trace) : List Nat := l.map Trace.id

theorem ids_append (a b : List Trace) : ids (a ++ b) = ids a ++ ids b := by simp [ids]

theorem mem_ids {x : Nat} {l : List Trace} : x ∈ ids l ↔ ∃ r ∈ l, r.id = x := by simp [ids]

theorem ids_perm {a b : List Trace} (h : a.Perm b) : (ids a).Perm (ids b) := List.Perm.map _ h

/-- The row updated by a retrieval: a named row gets a clamped new stability. -/
def upd (P : Params) (named : Trace → Bool) (g : Trace → Nat) (r : Trace) : Trace :=
  if named r then { r with stab := g r |> clampS P } else r

theorem upd_id (P : Params) (named : Trace → Bool) (g : Trace → Nat) (r : Trace) : (upd P named g r).id = r.id := by
  unfold upd; split <;> rfl

theorem upd_lab (P : Params) (named : Trace → Bool) (g : Trace → Nat) (r : Trace) : (upd P named g r).lab = r.lab := by
  unfold upd; split <;> rfl

theorem ids_map_upd (P : Params) (named : Trace → Bool) (g : Trace → Nat) (l : List Trace) :
    ids (l.map (upd P named g)) = ids l := by
  simp only [ids, List.map_map]
  exact List.map_congr_left (fun r _ => upd_id P named g r)

/-- Clearing the labile flag of one row (keep / settle). -/
def clearLab (sd : Nat) (r : Trace) : Trace := if r.id = sd then { r with lab := false } else r

def setLab (sd : Nat) (r : Trace) : Trace := if r.id = sd then { r with lab := true } else r

theorem clearLab_id (sd : Nat) (r : Trace) : (clearLab sd r).id = r.id := by
  unfold clearLab; split <;> rfl

theorem ids_map_clear (sd : Nat) (l : List Trace) : ids (l.map (clearLab sd)) = ids l := by
  simp only [ids, List.map_map]
  exact List.map_congr_left (fun r _ => clearLab_id sd r)

/-! ## the invariant -/

structure Inv (P : Params) (N W : Nat) (s : State) : Prop where
  cap      : s.rows.length ≤ P.cap
  range    : ∀ r ∈ s.rows, 1 ≤ r.stab ∧ r.stab ≤ P.smax
  nodup    : (ids s.rows).Nodup
  fresh    : ∀ r ∈ s.rows, r.id < s.next
  gfresh   : ∀ g ∈ s.gone, g < s.next
  nores    : ∀ g ∈ s.gone, g ∉ ids s.rows
  expo_sub : ∀ e ∈ s.expo, e ∈ ids s.rows
  expo_len : s.expo.length ≤ N
  wt_len   : s.wt.length ≤ W

def init : State := { rows := [], cursor := 0, next := 0, gone := [], expo := [], wt := [] }

theorem init_inv (P : Params) (N W : Nat) : Inv P N W init :=
  { cap := by simp [init]
    range := by simp [init]
    nodup := by simp [init, ids]
    fresh := by simp [init]
    gfresh := by simp [init]
    nores := by simp [init]
    expo_sub := by simp [init]
    expo_len := by simp [init]
    wt_len := by simp [init] }

/-! ## dream (big step over the seed queue) -/

inductive Kind | keep | replace | error
  deriving DecidableEq, Repr

structure Report where
  seed     : Nat
  kind     : Kind
  absorbed : List Trace
  gists    : List Trace

/-- One report undone: a replace removes its gists and restores what it absorbed; a keep makes the seed labile again. -/
def undoRows (rep : Report) (rows : List Trace) : List Trace :=
  match rep.kind with
  | .replace => rep.absorbed ++ rows.filter (fun r => decide (r.id ∉ ids rep.gists))
  | .keep    => rows.map (setLab rep.seed)
  | .error   => rows

/-- `Dream P left seeds s s' rs`: the dream may still judge `left` verdicts and walks `seeds`
    (distinct ids, oldest first); `rs` are its reports (oldest first). -/
inductive Dream (P : Params) : Nat → List Nat → State → State → List Report → Prop
  /-- budget spent or queue exhausted -/
  | stop (left : Nat) (seeds : List Nat) (s : State) : Dream P left seeds s s []
  /-- the seed vanished or is already consolidated: skipped, no verdict -/
  | skip {left sd rest s s' rs} :
      (sd ∉ ids s.rows ∨ ∀ r ∈ s.rows, r.id = sd → r.lab = false) →
      Dream P left rest s s' rs → Dream P left (sd :: rest) s s' rs
  /-- the seed's cue reactivates nothing: settled with no verdict and no budget -/
  | settle {left sd rest s s' rs} :
      (∀ r ∈ s.rows, r.id = sd → r.lab = true) → sd ∈ ids s.rows →
      Dream P left rest { s with rows := s.rows.map (clearLab sd) } s' rs →
      Dream P left (sd :: rest) s s' rs
  /-- keep: the seed alone becomes consolidated; one verdict -/
  | keep {left sd rest s s' rs} :
      (∀ r ∈ s.rows, r.id = sd → r.lab = true) → sd ∈ ids s.rows →
      Dream P left rest { s with rows := s.rows.map (clearLab sd) } s' rs →
      Dream P (left + 1) (sd :: rest) s s' ({ seed := sd, kind := .keep, absorbed := [], gists := [] } :: rs)
  /-- the LLM / embedder failed: nothing changes, the seed stays labile; the verdict is spent -/
  | error {left sd rest s s' rs} :
      Dream P left rest s s' rs →
      Dream P (left + 1) (sd :: rest) s s' ({ seed := sd, kind := .error, absorbed := [], gists := [] } :: rs)
  /-- replace: `absorbed` (the seed and the named candidates) leave, at most as many fresh gists enter -/
  | replace {left sd rest s s' rs} (absorbed rest' gists : List Trace) :
      s.rows.Perm (absorbed ++ rest') →
      sd ∈ ids absorbed →
      gists.length ≤ absorbed.length →
      (ids gists).Nodup →
      (∀ g ∈ gists, s.next ≤ g.id ∧ g.id < s.next + gists.length) →
      (∀ g ∈ gists, 1 ≤ g.stab ∧ g.stab ≤ P.smax ∧ g.lab = false) →
      Dream P left rest
        { s with rows := rest' ++ gists, next := s.next + gists.length, gone := s.gone ++ ids absorbed,
                 expo := s.expo.filter (fun e => decide (e ∈ ids (rest' ++ gists))) }
        s' rs →
      Dream P (left + 1) (sd :: rest) s s' ({ seed := sd, kind := .replace, absorbed := absorbed, gists := gists } :: rs)

/-- The `Dream` constructors that judge consume budget; compare with `left + 1`. -/
theorem dream_calls_le_budget {P : Params} {left seeds s s' rs} (h : Dream P left seeds s s' rs) :
    rs.length ≤ left := by
  induction h with
  | stop => simp
  | skip _ _ ih => exact ih
  | settle _ _ _ ih => exact ih
  | keep _ _ _ ih => simp; omega
  | error _ ih => simp; omega
  | replace _ _ _ _ _ _ _ _ _ _ ih => simp; omega

/-- Every report is for a distinct seed of the queue: at most one LLM call per seed. -/
theorem dream_seed_once {P : Params} {left seeds s s' rs} (h : Dream P left seeds s s' rs) :
    (rs.map Report.seed).Sublist seeds := by
  induction h with
  | stop => simp
  | skip _ _ ih => exact ih.cons _
  | settle _ _ _ ih => exact ih.cons _
  | keep _ _ _ ih => exact ih.cons_cons _
  | error _ ih => exact ih.cons_cons _
  | replace _ _ _ _ _ _ _ _ _ _ ih => exact ih.cons_cons _

theorem dream_seed_nodup {P : Params} {left seeds s s' rs} (h : Dream P left seeds s s' rs)
    (hn : seeds.Nodup) : (rs.map Report.seed).Nodup :=
  (dream_seed_once h).nodup hn

/-- A dream never increases the count (SPEC 5: a gist set is at most as large as what it absorbs). -/
theorem dream_len_le {P : Params} {left seeds s s' rs} (h : Dream P left seeds s s' rs) :
    s'.rows.length ≤ s.rows.length := by
  induction h with
  | stop => exact Nat.le_refl _
  | skip _ _ ih => exact ih
  | settle _ _ _ ih => simpa using ih
  | keep _ _ _ ih => simpa using ih
  | error _ ih => exact ih
  | @replace left sd rest s s' rs absorbed rest' gists hperm _ hlen _ _ _ _ ih =>
      have := hperm.length_eq
      simp only [List.length_append] at this ih ⊢
      omega

/-- Every id a dream removes is named by a report (no trace vanishes unreported). -/
theorem dream_removals_reported {P : Params} {left seeds s s' rs} (h : Dream P left seeds s s' rs) :
    ∀ g ∈ s'.gone, g ∈ s.gone ∨ ∃ r ∈ rs, g ∈ ids r.absorbed := by
  induction h with
  | stop => intro g hg; exact Or.inl hg
  | skip _ _ ih => exact ih
  | settle _ _ _ ih => exact ih
  | keep _ _ _ ih =>
      intro g hg
      rcases ih g hg with h | ⟨r, hr, hm⟩
      · exact Or.inl h
      · exact Or.inr ⟨r, List.mem_cons_of_mem _ hr, hm⟩
  | error _ ih =>
      intro g hg
      rcases ih g hg with h | ⟨r, hr, hm⟩
      · exact Or.inl h
      · exact Or.inr ⟨r, List.mem_cons_of_mem _ hr, hm⟩
  | @replace left sd rest s s' rs absorbed rest' gists _ _ _ _ _ _ _ ih =>
      intro g hg
      rcases ih g hg with h | ⟨r, hr, hm⟩
      · rcases List.mem_append.1 h with h | h
        · exact Or.inl h
        · exact Or.inr ⟨_, List.mem_cons_self .., h⟩
      · exact Or.inr ⟨r, List.mem_cons_of_mem _ hr, hm⟩

/-- The invariant survives a whole dream. -/
theorem dream_inv {P : Params} {N W left seeds s s' rs} (h : Dream P left seeds s s' rs) :
    Inv P N W s → Inv P N W s' := by
  induction h with
  | stop => exact id
  | skip _ _ ih => exact ih
  | settle _ _ _ ih =>
      intro hi
      refine ih ?_
      refine { cap := ?_, range := ?_, nodup := ?_, fresh := ?_, gfresh := hi.gfresh, nores := ?_,
               expo_sub := ?_, expo_len := hi.expo_len, wt_len := hi.wt_len }
      · simpa using hi.cap
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        unfold clearLab; split
        · exact hi.range r0 hr0
        · exact hi.range r0 hr0
      · simpa [ids_map_clear] using hi.nodup
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        rw [clearLab_id]; exact hi.fresh r0 hr0
      · intro g hg; simpa [ids_map_clear] using hi.nores g hg
      · intro e he; simpa [ids_map_clear] using hi.expo_sub e he
  | keep _ _ _ ih =>
      intro hi
      refine ih ?_
      refine { cap := ?_, range := ?_, nodup := ?_, fresh := ?_, gfresh := hi.gfresh, nores := ?_,
               expo_sub := ?_, expo_len := hi.expo_len, wt_len := hi.wt_len }
      · simpa using hi.cap
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        unfold clearLab; split
        · exact hi.range r0 hr0
        · exact hi.range r0 hr0
      · simpa [ids_map_clear] using hi.nodup
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        rw [clearLab_id]; exact hi.fresh r0 hr0
      · intro g hg; simpa [ids_map_clear] using hi.nores g hg
      · intro e he; simpa [ids_map_clear] using hi.expo_sub e he
  | error _ ih => exact ih
  | @replace left sd rest s s' rs absorbed rest' gists hperm _ hlen hgn hgb hgr _ ih =>
      intro hi
      refine ih ?_
      have hids : (ids s.rows).Perm (ids absorbed ++ ids rest') := by
        simpa [ids_append] using ids_perm hperm
      have hnd : (ids absorbed ++ ids rest').Nodup := hids.nodup_iff.1 hi.nodup
      have hnd' := List.nodup_append.1 hnd
      have hlenp := hperm.length_eq
      simp only [List.length_append] at hlenp
      have hmemrows : ∀ r, r ∈ s.rows ↔ r ∈ absorbed ∨ r ∈ rest' := by
        intro r; rw [hperm.mem_iff, List.mem_append]
      refine { cap := ?_, range := ?_, nodup := ?_, fresh := ?_, gfresh := ?_, nores := ?_,
               expo_sub := ?_, expo_len := ?_, wt_len := hi.wt_len }
      · simp only [List.length_append]; have := hi.cap; omega
      · intro r hr
        rcases List.mem_append.1 hr with h | h
        · exact hi.range r ((hmemrows r).2 (Or.inr h))
        · have := hgr r h; exact ⟨this.1, this.2.1⟩
      · -- ids of rest' ++ gists are distinct: rest' ids come from old rows (< next), gists ≥ next
        show (ids (rest' ++ gists)).Nodup
        rw [ids_append, List.nodup_append]
        refine ⟨hnd'.2.1, hgn, ?_⟩
        intro a ha b hb hab
        subst hab
        obtain ⟨r, hr, rfl⟩ := mem_ids.1 ha
        obtain ⟨g, hg, hgid⟩ := mem_ids.1 hb
        have h1 := hi.fresh r ((hmemrows r).2 (Or.inr hr))
        have h2 := (hgb g hg).1
        omega
      · intro r hr
        show r.id < s.next + gists.length
        rcases List.mem_append.1 hr with h | h
        · have := hi.fresh r ((hmemrows r).2 (Or.inr h)); omega
        · exact (hgb r h).2
      · intro g hg
        show g < s.next + gists.length
        rcases List.mem_append.1 hg with h | h
        · have := hi.gfresh g h; omega
        · obtain ⟨r, hr, rfl⟩ := mem_ids.1 h
          have := hi.fresh r ((hmemrows r).2 (Or.inl hr)); omega
      · intro g hg
        show g ∉ ids (rest' ++ gists)
        rw [ids_append]
        intro hmem
        rcases List.mem_append.1 hg with h | h
        · rcases List.mem_append.1 hmem with h2 | h2
          · exact hi.nores g h (by
              obtain ⟨r, hr, hrg⟩ := mem_ids.1 h2
              exact mem_ids.2 ⟨r, (hmemrows r).2 (Or.inr hr), hrg⟩)
          · obtain ⟨gg, hgg, hid⟩ := mem_ids.1 h2
            have := hi.gfresh g h
            have := (hgb gg hgg).1
            omega
        · -- g is an absorbed id: disjoint from rest' (nodup) and from the gists (fresh)
          rcases List.mem_append.1 hmem with h2 | h2
          · exact hnd'.2.2 g h g h2 rfl
          · obtain ⟨gg, hgg, hid⟩ := mem_ids.1 h2
            obtain ⟨a, ha, rfl⟩ := mem_ids.1 h
            have := hi.fresh a ((hmemrows a).2 (Or.inl ha))
            have := (hgb gg hgg).1
            omega
      · intro e he
        have := (List.mem_filter.1 he).2
        simpa using this
      · exact Nat.le_trans (List.length_filter_le _ _) hi.expo_len

/-! ## the other operations -/

/-- One step of the engine's wake phase and dream.  `N` = inject_n, `W` = writes_per_day. -/
inductive Step (P : Params) (N W : Nat) : State → State → Prop
  /-- peek, rejected, rate limited, a failed operation that rolled back: nothing changes -/
  | stutter (s : State) : Step P N W s s
  /-- insert a fresh trace and evict down to `min (n+1) cap` (the victims leave) -/
  | remember {s : State} (now : Nat) (t : Trace) (victims rows' : List Trace) :
      (s.wt.filter (fun x => decide (now ≤ x + 86400 ∧ x ≤ now))).length < W →
      t.id = s.next → 1 ≤ t.stab → t.stab ≤ P.smax →
      (t :: s.rows).Perm (victims ++ rows') →
      rows'.length = min (s.rows.length + 1) P.cap →
      Step P N W s
        { s with rows := rows', next := s.next + 1, gone := s.gone ++ ids victims,
                 expo := s.expo.filter (fun e => decide (e ∈ ids rows')),
                 wt := s.wt.filter (fun x => decide (now ≤ x + 86400 ∧ x ≤ now)) ++ [now] }
  /-- recall: the named rows (at most `inject_n`) get a retrieval update, never lower -/
  | recall {s : State} (named : Trace → Bool) (g : Trace → Nat) :
      (s.rows.filter named).length ≤ N →
      (∀ r ∈ s.rows, named r = true → r.stab ≤ g r) →
      Step P N W s
        { s with rows := s.rows.map (upd P named g), expo := ids (s.rows.filter named) }
  /-- cite: the cited exposures are completed and spent -/
  | cite {s : State} (cited : Trace → Bool) (g : Trace → Nat) :
      (∀ r ∈ s.rows, cited r = true → r.id ∈ s.expo ∧ r.stab ≤ g r) →
      Step P N W s
        { s with rows := s.rows.map (upd P cited g),
                 expo := s.expo.filter (fun e => decide (e ∈ ids (s.rows.filter (fun r => !cited r)))) }
  /-- forget by id: only rows with that id leave -/
  | forget {s : State} (id : Nat) (victims rows' : List Trace) :
      s.rows.Perm (victims ++ rows') →
      (∀ v ∈ victims, v.id = id) →
      Step P N W s
        { s with rows := rows', gone := s.gone ++ ids victims,
                 expo := s.expo.filter (fun e => decide (e ∈ ids rows')) }
  /-- dream; the cursor moves to a seed it examined (or stays) -/
  | dream {s : State} (s' : State) (seeds : List Nat) (rs : List Report) (c : Nat) :
      Dream P N seeds s s' rs → (c = s.cursor ∨ c ∈ seeds) →
      Step P N W s { s' with cursor := c }

theorem upd_range (P : Params) (named : Trace → Bool) (g : Trace → Nat) (r : Trace)
    (h : 1 ≤ r.stab ∧ r.stab ≤ P.smax) : 1 ≤ (upd P named g r).stab ∧ (upd P named g r).stab ≤ P.smax := by
  unfold upd; split
  · exact clampS_range P _
  · exact h

theorem step_inv {P : Params} {N W s s'} (hi : Inv P N W s) (h : Step P N W s s') : Inv P N W s' := by
  cases h with
  | stutter => exact hi
  | remember now t victims rows' hwt htid ht1 ht2 hperm hlen =>
      have hids : ((t.id :: ids s.rows)).Perm (ids victims ++ ids rows') := by
        simpa [ids, ids_append] using ids_perm hperm
      have hnot : t.id ∉ ids s.rows := by
        intro hm
        obtain ⟨r, hr, hrid⟩ := mem_ids.1 hm
        have := hi.fresh r hr; omega
      have hnd : (t.id :: ids s.rows).Nodup := List.nodup_cons.2 ⟨hnot, hi.nodup⟩
      have hnd' := List.nodup_append.1 (hids.nodup_iff.1 hnd)
      have hmem : ∀ r, r ∈ rows' → r = t ∨ r ∈ s.rows := by
        intro r hr
        have : r ∈ t :: s.rows := hperm.mem_iff.2 (List.mem_append_right _ hr)
        simpa using this
      refine { cap := ?_, range := ?_, nodup := hnd'.2.1, fresh := ?_, gfresh := ?_, nores := ?_,
               expo_sub := ?_, expo_len := ?_, wt_len := ?_ }
      · have := P.hcap; show rows'.length ≤ P.cap; omega
      · intro r hr
        rcases hmem r hr with rfl | h
        · exact ⟨ht1, ht2⟩
        · exact hi.range r h
      · intro r hr
        show r.id < s.next + 1
        rcases hmem r hr with rfl | h
        · omega
        · have := hi.fresh r h; omega
      · intro g hg
        show g < s.next + 1
        rcases List.mem_append.1 hg with h | h
        · have := hi.gfresh g h; omega
        · obtain ⟨v, hv, rfl⟩ := mem_ids.1 h
          have : v ∈ t :: s.rows := hperm.mem_iff.2 (List.mem_append_left _ hv)
          rcases List.mem_cons.1 this with rfl | h2
          · omega
          · have := hi.fresh v h2; omega
      · intro g hg hgm
        obtain ⟨r, hr, hrg⟩ := mem_ids.1 hgm
        rcases List.mem_append.1 hg with h | h
        · have hg' := hi.gfresh g h
          rcases hmem r hr with rfl | h2
          · omega
          · exact hi.nores g h (mem_ids.2 ⟨r, h2, hrg⟩)
        · exact hnd'.2.2 g h g (mem_ids.2 ⟨r, hr, hrg⟩) rfl
      · intro e he; exact by simpa using (List.mem_filter.1 he).2
      · exact Nat.le_trans (List.length_filter_le _ _) hi.expo_len
      · show (List.filter _ s.wt ++ [now]).length ≤ W
        simp only [List.length_append, List.length_singleton]; omega
  | recall named g hnamed hmono =>
      refine { cap := ?_, range := ?_, nodup := ?_, fresh := ?_, gfresh := hi.gfresh, nores := ?_,
               expo_sub := ?_, expo_len := ?_, wt_len := hi.wt_len }
      · simpa using hi.cap
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        exact upd_range P named g r0 (hi.range r0 hr0)
      · simpa [ids_map_upd] using hi.nodup
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        rw [upd_id]; exact hi.fresh r0 hr0
      · intro g hg; simpa [ids_map_upd] using hi.nores g hg
      · intro e he
        show e ∈ ids (s.rows.map (upd P named g))
        rw [ids_map_upd]
        obtain ⟨r, hr, rfl⟩ := mem_ids.1 he
        exact mem_ids.2 ⟨r, (List.mem_filter.1 hr).1, rfl⟩
      · show (ids (s.rows.filter named)).length ≤ N
        simpa [ids] using hnamed
  | cite cited g hc =>
      refine { cap := ?_, range := ?_, nodup := ?_, fresh := ?_, gfresh := hi.gfresh, nores := ?_,
               expo_sub := ?_, expo_len := ?_, wt_len := hi.wt_len }
      · simpa using hi.cap
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        exact upd_range P cited g r0 (hi.range r0 hr0)
      · simpa [ids_map_upd] using hi.nodup
      · intro r hr
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.1 hr
        rw [upd_id]; exact hi.fresh r0 hr0
      · intro g hg; simpa [ids_map_upd] using hi.nores g hg
      · intro e he
        show e ∈ ids (s.rows.map (upd P cited g))
        rw [ids_map_upd]
        have := (List.mem_filter.1 he).2
        obtain ⟨r, hr, rfl⟩ := mem_ids.1 (by simpa using this)
        exact mem_ids.2 ⟨r, (List.mem_filter.1 hr).1, rfl⟩
      · exact Nat.le_trans (List.length_filter_le _ _) hi.expo_len
  | forget id victims rows' hperm hv =>
      have hids : (ids s.rows).Perm (ids victims ++ ids rows') := by
        simpa [ids_append] using ids_perm hperm
      have hnd' := List.nodup_append.1 (hids.nodup_iff.1 hi.nodup)
      have hsub : ∀ r, r ∈ rows' → r ∈ s.rows := fun r hr => hperm.mem_iff.2 (List.mem_append_right _ hr)
      refine { cap := ?_, range := ?_, nodup := hnd'.2.1, fresh := ?_, gfresh := ?_, nores := ?_,
               expo_sub := ?_, expo_len := ?_, wt_len := hi.wt_len }
      · have := hperm.length_eq
        simp only [List.length_append] at this
        have := hi.cap; show rows'.length ≤ P.cap; omega
      · intro r hr; exact hi.range r (hsub r hr)
      · intro r hr; exact hi.fresh r (hsub r hr)
      · intro g hg
        rcases List.mem_append.1 hg with h | h
        · exact hi.gfresh g h
        · obtain ⟨v, hvm, rfl⟩ := mem_ids.1 h
          exact hi.fresh v (hperm.mem_iff.2 (List.mem_append_left _ hvm))
      · intro g hg hgm
        obtain ⟨r, hr, hrg⟩ := mem_ids.1 hgm
        rcases List.mem_append.1 hg with h | h
        · exact hi.nores g h (mem_ids.2 ⟨r, hsub r hr, hrg⟩)
        · exact hnd'.2.2 g h g (mem_ids.2 ⟨r, hr, hrg⟩) rfl
      · intro e he; simpa using (List.mem_filter.1 he).2
      · exact Nat.le_trans (List.length_filter_le _ _) hi.expo_len
  | dream s1 seeds rs c hd hc =>
      have h1 := dream_inv (N := N) (W := W) hd hi
      exact { h1 with }

/-! ## reachable states keep the invariant -/

inductive Reach (P : Params) (N W : Nat) : State → Prop
  | init : Reach P N W init
  | step {s s'} : Reach P N W s → Step P N W s s' → Reach P N W s'

theorem reach_inv {P : Params} {N W s} (h : Reach P N W s) : Inv P N W s := by
  induction h with
  | init => exact init_inv P N W
  | step _ hs ih => exact step_inv ih hs

/-- Capacity after any history of remember / recall / cite / forget / dream. -/
theorem capacity_always {P : Params} {N W s} (h : Reach P N W s) : s.rows.length ≤ P.cap := (reach_inv h).cap

/-- Stability range after any history. -/
theorem stability_always {P : Params} {N W s} (h : Reach P N W s) :
    ∀ r ∈ s.rows, 1 ≤ r.stab ∧ r.stab ≤ P.smax := (reach_inv h).range

/-- No removed id (evicted, forgotten or absorbed by a dream) is ever live again. -/
theorem no_resurrection {P : Params} {N W s} (h : Reach P N W s) :
    ∀ g ∈ s.gone, ∀ r ∈ s.rows, r.id ≠ g := by
  intro g hg r hr hrg
  exact (reach_inv h).nores g hg (mem_ids.2 ⟨r, hr, hrg⟩)

/-! ## transition-level properties -/

theorem recall_ids (P : Params) (named : Trace → Bool) (g : Trace → Nat) (rows : List Trace) :
    ids (rows.map (upd P named g)) = ids rows := ids_map_upd P named g rows

theorem cite_ids (P : Params) (cited : Trace → Bool) (g : Trace → Nat) (rows : List Trace) :
    ids (rows.map (upd P cited g)) = ids rows := ids_map_upd P cited g rows

/-- Recall only rewrites the named rows: an un-named row is carried over unchanged. -/
theorem recall_frames (P : Params) (named : Trace → Bool) (g : Trace → Nat) (rows : List Trace) :
    ∀ r ∈ rows, named r = false → upd P named g r = r := by
  intro r _ h; unfold upd; simp [h]

/-- Recall and cite never lower stability, and never touch the labile flag. -/
theorem recall_monotone (P : Params) (named : Trace → Bool) (g : Trace → Nat) (r : Trace)
    (h1 : 1 ≤ r.stab) (h2 : r.stab ≤ P.smax) (hg : named r = true → r.stab ≤ g r) :
    r.stab ≤ (upd P named g r).stab := by
  unfold upd; split
  · rename_i hn; exact clampS_ge P h1 h2 (hg hn)
  · exact Nat.le_refl _

theorem recall_keeps_labile (P : Params) (named : Trace → Bool) (g : Trace → Nat) (r : Trace) :
    (upd P named g r).lab = r.lab := upd_lab P named g r

/-- The cite update `s * f / p` of a stability `s` with `1 <= p <= f`. -/
theorem cite_monotone (P : Params) {s f p : Nat} (h1 : 1 ≤ s) (h2 : s ≤ P.smax) (hp : 0 < p) (hpf : p ≤ f) :
    s ≤ clampS P (s * f / p) := by
  apply clampS_ge P h1 h2
  exact (Nat.le_div_iff_mul_le hp).2 (Nat.mul_le_mul_left s hpf)

/-- remember never clears a labile flag: a surviving row keeps `lab = true` (surviving rows are old rows). -/
theorem remember_keeps_labile {rows victims rows' : List Trace} {t : Trace}
    (hperm : (t :: rows).Perm (victims ++ rows')) (hnd : (ids (t :: rows)).Nodup)
    {r r' : Trace} (hr : r ∈ rows) (hr' : r' ∈ rows') (hid : r'.id = r.id) : r' = r := by
  have hids : (ids (t :: rows)).Perm (ids victims ++ ids rows') := by
    simpa [ids_append] using ids_perm hperm
  have hn := hids.nodup_iff.1 hnd
  have hm : r' ∈ t :: rows := hperm.mem_iff.2 (List.mem_append_right _ hr')
  have hm2 : r ∈ t :: rows := List.mem_cons_of_mem _ hr
  -- ids are injective on a list with nodup ids
  have inj : ∀ {l : List Trace}, (ids l).Nodup → ∀ {a b : Trace}, a ∈ l → b ∈ l → a.id = b.id → a = b := by
    intro l
    induction l with
    | nil => intro _ a _ ha; cases ha
    | cons x xs ih =>
      intro hn a b ha hb hab
      simp only [ids, List.map_cons, List.nodup_cons] at hn
      rcases List.mem_cons.1 ha with rfl | ha' <;> rcases List.mem_cons.1 hb with rfl | hb'
      · rfl
      · exact absurd (List.mem_map.2 ⟨b, hb', hab.symm⟩) hn.1
      · exact absurd (List.mem_map.2 ⟨a, ha', hab⟩) hn.1
      · exact ih (by simpa [ids] using hn.2) ha' hb' hab
  have hnd2 : (ids (t :: rows)).Nodup := hnd
  exact inj hnd2 hm hm2 hid

theorem forget_keeps_labile (rows victims rows' : List Trace) (hperm : rows.Perm (victims ++ rows'))
    {r' : Trace} (hr' : r' ∈ rows') : r' ∈ rows := hperm.mem_iff.2 (List.mem_append_right _ hr')

/-! ## undo exactness, one verdict at a time (SPEC 5: "取り消せる")

`undo_all` undoes the reports newest first; each step is the atomic lemma below, so a chain
restores the rows exactly *provided each undo is applied to the state its verdict produced*.
Interleaved operations, settle flags and repeated undo are outside the model: they are the
documented gap G5 (`I7-undo-not-exact` fuzz kind, `note_stale_undo_resurrects_forgotten_trace`). -/

theorem setLab_clearLab (sd : Nat) (r : Trace) (h : r.id = sd → r.lab = true) : setLab sd (clearLab sd r) = r := by
  unfold setLab clearLab
  by_cases hid : r.id = sd
  · have hl := h hid
    cases r
    simp_all
  · simp [hid]

/-- undo of a keep restores the seed's labile flag (the seed was labile when judged). -/
theorem undo_keep_rows (sd : Nat) (rows : List Trace) (h : ∀ r ∈ rows, r.id = sd → r.lab = true) :
    undoRows { seed := sd, kind := .keep, absorbed := [], gists := [] } (rows.map (clearLab sd)) = rows := by
  unfold undoRows
  rw [List.map_map]
  calc List.map (setLab sd ∘ clearLab sd) rows = List.map id rows :=
        List.map_congr_left (fun r hr => by simpa using setLab_clearLab sd r (h r hr))
    _ = rows := List.map_id _

/-- undo of a replace: gists out, absorbed back -- the multiset of rows is the pre-verdict one. -/
theorem undo_replace_rows (absorbed rest' gists : List Trace) (sd : Nat)
    (hdisj : ∀ r ∈ rest', r.id ∉ ids gists) :
    (undoRows { seed := sd, kind := .replace, absorbed := absorbed, gists := gists } (rest' ++ gists)).Perm (absorbed ++ rest') := by
  unfold undoRows
  simp only [List.filter_append]
  have h1 : List.filter (fun r => decide (r.id ∉ ids gists)) rest' = rest' :=
    List.filter_eq_self.2 (fun a ha => by simpa using hdisj a ha)
  have h2 : List.filter (fun r => decide (r.id ∉ ids gists)) gists = [] :=
    List.filter_eq_nil_iff.2 (fun a ha => by simpa using mem_ids.2 ⟨a, ha, rfl⟩)
  rw [h1, h2]
  simp

/-! ## recall pack (S16): greedy packing within the character budget -/

/-- Lines of cost `c` are taken in order; a line that would overflow the budget is skipped, except
    that the first line always fits (SPEC 4: 先頭 1 行は必ず入れる). `(used, taken)` so far. -/
def pack (budget : Nat) : Nat → Nat → List Nat → Nat × Nat
  | used, taken, [] => (used, taken)
  | used, taken, c :: cs =>
      if taken ≠ 0 ∧ budget < used + c then pack budget used taken cs
      else pack budget (used + c) (taken + 1) cs

theorem pack_inv (budget : Nat) : ∀ (cs : List Nat) (used taken : Nat),
    (used ≤ budget ∨ taken ≤ 1) → (pack budget used taken cs).1 ≤ budget ∨ (pack budget used taken cs).2 ≤ 1 := by
  intro cs
  induction cs with
  | nil => intro used taken h; simpa [pack] using h
  | cons c cs ih =>
    intro used taken h
    unfold pack
    split
    · exact ih _ _ h
    · rename_i hc
      apply ih
      by_cases h0 : taken = 0
      · right; omega
      · left; have : ¬ budget < used + c := fun hl => hc ⟨h0, hl⟩
        omega

theorem pack_within_budget (budget : Nat) (cs : List Nat) :
    (pack budget 0 0 cs).1 ≤ budget ∨ (pack budget 0 0 cs).2 ≤ 1 :=
  pack_inv budget cs 0 0 (Or.inr (by omega))

/-! ## documented gaps with witnesses (all in `tests/audit2_d.rs`)

* G1  eviction optimality (weakest first): `eviction_oracle` in the fuzz harness checks every
      eviction against a brute-force ranking; no Lean statement (needs `Real.log`).
* G2  atomicity on `Err`: defect D1 `r2d_remember_reinforce_returns_err_after_commit`.
* G3  backward clock vs the dream's `stated_by` bound: defect D2
      `r2d_backward_clock_hides_older_candidates_from_the_dream` (a candidate's raw `created_at`
      is compared with the seed's *clamped* one; the seed settles unjudged).
* G4  quota burn: defect D3 `r2d_failed_writes_burn_the_daily_quota` (the write window is pushed before the write is known to succeed).
* G5  late or repeated undo: `note_stale_undo_resurrects_forgotten_trace`.
* G6  the IdGen contract: `note_idgen_may_reuse_a_removed_id` (the engine refuses live ids only;
      `Inv.nores` holds only for generators that never repeat an id).
-/

end R2D
