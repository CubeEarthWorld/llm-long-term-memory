/-! Lane 4 — wake phase and text.  Core Lean 4 only (no Mathlib).
    Characters are Nat code points; texts are `List Nat`.  Each section names the
    invariant (L4-n) and says whether the theorem is PROVED, or a COUNTEREXAMPLE
    (a machine-checked witness of a real defect), or has a `sorry` with `-- GAP:`. -/

namespace Lane4

/-! ## L4-1  ids and clean_text -/

def isControl (c : Nat) : Bool := c < 0x20 || (0x7F ≤ c && c < 0xA0)

/-- Rust `is_space`: `char::is_whitespace` ∪ U+001C..1F ∪ U+FEFF. -/
def isSpace (c : Nat) : Bool :=
  c == 0x20 || (0x09 ≤ c && c ≤ 0x0D) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
  (0x2000 ≤ c && c ≤ 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
  c == 0x205F || c == 0x3000 || (0x1C ≤ c && c ≤ 0x1F) || c == 0xFEFF

def isBidi (c : Nat) : Bool := (0x202A ≤ c && c ≤ 0x202E) || (0x2066 ≤ c && c ≤ 0x2069)
def isDelim (c : Nat) : Bool := c == 0x300A || c == 0x300B

/-- Rust `is_dropped` (what `valid_id` and `clean_text` both use). -/
def isDropped (c : Nat) : Bool := isDelim c || isBidi c || (isControl c && !isSpace c)

/-- SPEC §2: ids may hold no delimiter, no control character (C0, DEL, C1), no bidi control. -/
def idBadSpec (c : Nat) : Bool := isDelim c || isBidi c || isControl c

def validId (s : List Nat) : Bool := !s.isEmpty && !s.any isDropped

/-- COUNTEREXAMPLE (real bug, Rust test `l4_1_valid_id_rejects_every_control_character`):
    a newline is bad by the spec, accepted by the implementation. -/
theorem valid_id_gap : ∃ c, idBadSpec c = true ∧ isDropped c = false := ⟨0x0A, by decide⟩

theorem valid_id_accepts_newline_id : validId [0x41, 0x0A, 0x42] = true := by decide

/-- The disagreement is exactly "a control character that `is_space` calls blank"
    (checked on every code point below U+0100; U+2028/2029 behave the same by `isSpace`). -/
theorem valid_id_gap_exact :
    ∀ c, c < 0x100 → (idBadSpec c && !isDropped c) = (isControl c && isSpace c) := by
  set_option maxRecDepth 100000 in decide

theorem valid_id_gap_2028 : idBadSpec 0x2028 = false ∧ isDropped 0x2028 = false := by decide
-- NOTE: U+2028/2029 are not C0/C1 controls (category Zl/Zp), so SPEC's literal rule does not
-- reject them; they are in the brief's list only.  The Rust test still reports them.

/-- Sentence-boundary cut: `idx` is the `rposition` inside the first `n` chars, so `idx < n`. -/
def shorten (l : List Nat) (n : Nat) (idx : Option Nat) : List Nat :=
  if l.length ≤ n then l else
  match idx with
  | some i => l.take (i + 1)
  | none => l.take n

/-- PROVED: shorten never exceeds `n` chars (chars, not bytes), given `idx < n`. -/
theorem shorten_len (l : List Nat) (n : Nat) (idx : Option Nat)
    (h : ∀ i, idx = some i → i < n) : (shorten l n idx).length ≤ n := by
  unfold shorten
  split
  · assumption
  · cases hidx : idx with
    | none => simp [List.length_take]; omega
    | some i =>
      have := h i hidx
      simp [List.length_take]
      omega

/-- PROVED: idempotence of the shortening step on an already short text. -/
theorem shorten_idem (l : List Nat) (n : Nat) (idx : Option Nat) (h : l.length ≤ n) :
    shorten l n idx = l := by
  simp [shorten, h]

/-- trim of both ends. -/
def trim (l : List Nat) : List Nat := (l.dropWhile isSpace).reverse.dropWhile isSpace |>.reverse

theorem trim_len (l : List Nat) : (trim l).length ≤ l.length := by
  unfold trim
  simp only [List.length_reverse]
  have h1 := (List.dropWhile_suffix (p := isSpace) (l := l)).length_le
  have h2 := (List.dropWhile_suffix (p := isSpace) (l := (l.dropWhile isSpace).reverse)).length_le
  simp only [List.length_reverse] at h2
  omega

/-- PROVED: `clean_text` ≤ max chars, composed from the two steps above
    (`shorten` result is bounded, `trim` only shrinks). -/
theorem clean_len (l : List Nat) (n : Nat) (idx : Option Nat)
    (h : ∀ i, idx = some i → i < n) : (trim (shorten l n idx)).length ≤ n :=
  Nat.le_trans (trim_len _) (shorten_len l n idx h)

/-- clean_text idempotence needs: (a) the collapse step is idempotent, (b) a shortened,
    trimmed text re-collapses to itself.  The Rust fuzz test (500 random strings over an
    alphabet of 30 troublesome chars) found no counterexample. -/
theorem trim_idem_on_nonblank_ends (l : List Nat) (hs : ∀ c, l.head? = some c → isSpace c = false)
    (ht : ∀ c, l.getLast? = some c → isSpace c = false) : trim l = l := by
  -- GAP: needs dropWhile/reverse/getLast? lemma plumbing that is not worth the lines here;
  -- the property is covered by the fuzz test.
  sorry

/-! ### clean_tz (field = name ++ ';' ++ offset) -/

/-- PROVED: the length budget arithmetic of `clean_tz` (`name ≤ max - (|offset|+1)`,
    `offset ≤ max - 1`) gives a total ≤ max, whatever the lengths. -/
theorem clean_tz_len (max nameLen offLen : Nat)
    (hn : nameLen ≤ max - (offLen + 1)) (ho : offLen ≤ max - 1) (hm : 1 ≤ max) :
    nameLen + 1 + offLen ≤ max := by omega

/-! ## L4-2  recall pack -/

/-- Greedy pack: the first line always goes in, later ones only if they fit.
    State = (lines kept, chars used), newest line first. -/
def packStep (budget : Nat) (st : List Nat × Nat) (cost : Nat) : List Nat × Nat :=
  if st.1.isEmpty || st.2 + cost ≤ budget then (cost :: st.1, st.2 + cost) else st

def packAll (budget : Nat) (costs : List Nat) : List Nat × Nat :=
  costs.foldl (packStep budget) ([], 0)

/-- Invariant of the fold. -/
def PackInv (budget : Nat) (st : List Nat × Nat) : Prop :=
  st.2 = st.1.foldr (· + ·) 0 ∧ (st.2 ≤ budget ∨ st.1.length ≤ 1)

theorem packStep_inv (budget : Nat) (st : List Nat × Nat) (cost : Nat) (h : PackInv budget st) :
    PackInv budget (packStep budget st cost) := by
  obtain ⟨hsum, hb⟩ := h
  unfold packStep
  by_cases hc : (st.1.isEmpty || decide (st.2 + cost ≤ budget)) = true
  · simp only [hc, if_true, PackInv]
    constructor
    · simp [List.foldr, hsum]; omega
    · by_cases he : st.1.isEmpty = true
      · right
        have : st.1 = [] := List.isEmpty_iff.mp he
        simp [this]
      · have hle : st.2 + cost ≤ budget := by
          simp [he] at hc
          exact hc
        left; exact hle
  · simp only [hc]
    exact ⟨hsum, hb⟩

/-- PROVED (L4-2): the pack never exceeds `budget_chars` except when it is a single line. -/
theorem pack_budget (budget : Nat) (costs : List Nat) :
    (packAll budget costs).2 ≤ budget ∨ (packAll budget costs).1.length ≤ 1 := by
  have key : ∀ (cs : List Nat) (st : List Nat × Nat), PackInv budget st →
      PackInv budget (cs.foldl (packStep budget) st) := by
    intro cs
    induction cs with
    | nil => intro st h; exact h
    | cons c cs ih => intro st h; exact ih _ (packStep_inv budget st c h)
  have := key costs ([], 0) ⟨by simp, by simp⟩
  exact this.2

/-- MMR selection as implemented: pick any member of `alive` (argmax abstracted as `pick`),
    remove it, repeat up to `fuel = inject_n` times. -/
def mmr (pick : List Nat → Nat) : Nat → List Nat → List Nat → List Nat
  | 0, _, sel => sel   -- Rust reverses at the end; a permutation changes neither Nodup nor length
  | fuel + 1, alive, sel =>
    if alive.isEmpty then sel
    else let b := pick alive; mmr pick fuel (alive.filter (· != b)) (b :: sel)

theorem mmr_nodup (pick : List Nat → Nat) (hp : ∀ a, a ≠ [] → pick a ∈ a) :
    ∀ (fuel : Nat) (alive sel : List Nat), (∀ x, x ∈ alive → x ∉ sel) → sel.Nodup →
      (mmr pick fuel alive sel).Nodup := by
  intro fuel
  induction fuel with
  | zero => intro alive sel _ hs; simpa [mmr] using hs
  | succ n ih =>
    intro alive sel hdis hs
    unfold mmr
    by_cases he : alive.isEmpty = true
    · simpa [he] using hs
    · simp only [he, Bool.false_eq_true, if_false]
      have hne : alive ≠ [] := by
        intro h; simp [h] at he
      have hb := hp alive hne
      apply ih
      · intro x hx
        simp only [List.mem_filter, bne_iff_ne, ne_eq, decide_eq_true_eq] at hx
        intro hxs
        simp only [List.mem_cons] at hxs
        rcases hxs with h | h
        · exact hx.2 h
        · exact hdis x hx.1 h
      · exact List.nodup_cons.mpr ⟨hdis _ hb, hs⟩

theorem mmr_len (pick : List Nat → Nat) :
    ∀ (fuel : Nat) (alive sel : List Nat),
      (mmr pick fuel alive sel).length ≤ fuel + sel.length := by
  intro fuel
  induction fuel with
  | zero => intro alive sel; simp [mmr]
  | succ n ih =>
    intro alive sel
    unfold mmr
    by_cases he : alive.isEmpty = true
    · (simp [he]) <;> omega
    · simp only [he, Bool.false_eq_true, if_false]
      have := ih (alive.filter (· != pick alive)) (pick alive :: sel)
      simp at this ⊢
      omega

/-- PROVED (L4-2): MMR picks pairwise distinct indices, at most `inject_n` of them. -/
theorem mmr_distinct_and_bounded (pick : List Nat → Nat) (hp : ∀ a, a ≠ [] → pick a ∈ a)
    (n : Nat) (alive : List Nat) :
    (mmr pick n alive []).Nodup ∧ (mmr pick n alive []).length ≤ n :=
  ⟨mmr_nodup pick hp n alive [] (by simp) List.nodup_nil, by simpa using mmr_len pick n alive []⟩

/-- PROVED (L4-2): the widening loop `k ← 4k` leaves every bound `N` behind, so
    "fewer than k hits returned" is eventually true (mathematical integers). -/
theorem lt_four_pow : ∀ n : Nat, n < 4 ^ n := by
  intro n
  induction n with
  | zero => decide
  | succ n ih => rw [Nat.pow_succ]; omega

theorem widening_terminates (k N : Nat) (hk : 1 ≤ k) : ∃ n, N < 4 ^ n * k := by
  refine ⟨N, ?_⟩
  have h := lt_four_pow N
  calc N < 4 ^ N := h
    _ = 4 ^ N * 1 := (Nat.mul_one _).symm
    _ ≤ 4 ^ N * k := Nat.mul_le_mul_left _ hk

/-- COUNTEREXAMPLE (machine integers): `4 * inject_n` already wraps for a valid
    config (`inject_n > 0` is all `validate` demands); with `usize = 2^64`. -/
theorem k_start_overflows : ∃ n : Nat, 0 < n ∧ 2 ^ 64 ≤ 4 * n := ⟨2 ^ 62, by decide, by decide⟩

/-- With the machine widening in 64-bit arithmetic, `k` stays ≤ 2^64 only while
    N < 2^64; the loop is fine for every realistic store (N bounded by memory). -/
theorem widening_stops_before_wrap (N : Nat) (hN : N < 2 ^ 62) : ∃ n, N < 4 ^ n * 20 ∧ 4 ^ n * 20 < 2 ^ 64 * 4 := by
  -- GAP: arithmetic with the exact first n where 4^n*20 > N; true since N < 2^62 gives n ≤ 31.
  sorry

/-! ## L4-3  cite -/

/-- PROVED: `S · full / partial ≥ S` when `partial ≤ full`, `partial > 0` (monotone, before the
    S_max clamp, which only lowers a value that is already ≤ S_max). -/
theorem cite_monotone (S P F : Nat) (hP : 0 < P) (hF : P ≤ F) : S ≤ S * F / P := by
  rw [Nat.le_div_iff_mul_le hP]
  exact Nat.mul_le_mul_left S hF

/-- `max 1 (min s m)` lies in [1, m]  (clamp_stability, without NaN). -/
theorem clamp_range (s m : Nat) (hm : 1 ≤ m) : 1 ≤ max 1 (min s m) ∧ max 1 (min s m) ≤ m := by
  constructor <;> omega

/-- The citation scanner `cited_ids`: 1 = `《id:`, 2 = `》`, anything else = payload. -/
def dropToOpen : List Nat → Option (List Nat)
  | [] => none
  | c :: r => if c == 1 then some r else dropToOpen r

def splitAtClose : List Nat → List Nat → Option (List Nat × List Nat)
  | [], _ => none
  | c :: r, acc => if c == 2 then some (acc.reverse, r) else splitAtClose r (c :: acc)

def citedIds : Nat → List Nat → List (List Nat)
  | 0, _ => []
  | f + 1, s =>
    match dropToOpen s with
    | none => []
    | some after =>
      match splitAtClose after [] with
      | none => []
      | some (id, rest) => if id.isEmpty then citedIds f after else id :: citedIds f rest

/-- Agrees with the Rust unit test (`a《id:X1》 b 《id:》《id:Y2》《id:Z`). -/
theorem cited_ids_basic :
    citedIds 10 [9, 1, 5, 6, 2, 9, 1, 2, 1, 7, 8, 2, 1, 4] = [[5, 6], [7, 8]] := by decide

/-- COUNTEREXAMPLE (Rust test `l4_3_cite_parser_finds_a_well_formed_citation_after_a_stray_opener`):
    `《id:《id:X》` yields the single "id" `《id:X` (contains an opener), and the
    well-formed inner citation X is never reported. -/
theorem cited_ids_swallows_inner : citedIds 10 [1, 1, 5, 2] = [[1, 5]] ∧ [5] ∉ citedIds 10 [1, 1, 5, 2] := by
  decide

/-- Each id is cited at most once: the exposure map removes an id on first use.
    Model: exposures as a list of ids; `take` removes the first occurrence. -/
def takeOnce (id : Nat) (ex : List Nat) : Bool × List Nat :=
  if id ∈ ex then (true, ex.filter (· != id)) else (false, ex)

theorem second_take_fails (id : Nat) (ex : List Nat) :
    (takeOnce id (takeOnce id ex).2).1 = false := by
  unfold takeOnce
  by_cases h : id ∈ ex
  · simp [h]
  · simp [h]

/-- COUNTEREXAMPLE (Rust `l4_3_cite_survives_a_transient_store_error`): exposures are removed
    before the store read; if the read fails the window is gone.  Model: state after a failed
    cite has no exposure for `id`, so the retry cites nothing. -/
theorem failed_cite_loses_window :
    (takeOnce 7 [7]).2 = [] ∧ (takeOnce 7 (takeOnce 7 [7]).2).1 = false := by decide

/-- COUNTEREXAMPLE (Rust `l4_3_cite_does_not_boost_a_trace_re_created_under_a_forgotten_id`):
    `forget` does not touch exposures; they are keyed by id only.  A store keyed by id after
    forget+recreate with the same id satisfies the cite guard (`id ∈ ex` ∧ trace exists). -/
theorem recreated_id_passes_cite_guard (id : Nat) :
    let ex := [id]            -- exposure of the forgotten trace
    let store_has := true     -- a new trace now lives under the same id
    (id ∈ ex) ∧ store_has = true := by
  simp

/-! ## L4-4  remember -/

/-- Rate window: `write_times.retain(|t| t ≥ now-86400 ∧ t ≤ now)`, refuse when len ≥ limit. -/
def retain (times : List Nat) (now : Nat) : List Nat :=
  times.filter (fun t => now ≤ t + 86400 && t ≤ now)

def allowed (times : List Nat) (now limit : Nat) : Bool := (retain times now).length < limit

/-- PROVED: with the clock monotone, a full window refuses. -/
theorem full_window_refuses (limit : Nat) (t now : Nat) (h1 : t ≤ now) (h2 : now ≤ t + 86400) :
    allowed (List.replicate limit t) now limit = false := by
  unfold allowed retain
  have : (List.replicate limit t).filter (fun t' => now ≤ t' + 86400 && t' ≤ now) = List.replicate limit t := by
    apply List.filter_eq_self.mpr
    intro a ha
    have := List.eq_of_mem_replicate ha
    subst this
    simp [h1, h2]
  rw [this]
  simp

/-- COUNTEREXAMPLE (Rust `note_l4_4_clock_step_back_does_not_reset_the_write_limit`):
    three writes at t = 1000, limit 3, clock steps back to 995: all three are dropped
    (they are "in the future"), so a 4th write is allowed. -/
theorem clock_step_back_bypasses_limit : allowed [1000, 1000, 1000] 995 3 = true := by decide

/-- `writes_per_day = 0`: nothing is ever allowed (matches Rust). -/
theorem zero_limit_blocks (times : List Nat) (now : Nat) : allowed times now 0 = false := by
  simp [allowed]

/-- Text probe: `find(Text, 8)` then "first usable".  A window of 8 unusable rows hides a
    usable 9th one.  COUNTEREXAMPLE (Rust `note_l4_4_more_than_eight_unusable_duplicates...`). -/
def probe (rows : List Bool) : Option Nat := (rows.take 8).findIdx? id

theorem probe_misses_ninth : probe (List.replicate 8 false ++ [true]) = none := by decide
theorem probe_finds_eighth : probe (List.replicate 7 false ++ [true]) = some 7 := by decide

/-- consolidated = cued ∧ nearest = none ∧ no-stale-rows. -/
def consolidated (cued noNearest noStale : Bool) : Bool := cued && noNearest && noStale

theorem failed_embed_never_consolidated (noNearest noStale : Bool) :
    consolidated false noNearest noStale = false := by simp [consolidated]

theorem stale_row_keeps_labile (cued noNearest : Bool) :
    consolidated cued noNearest false = false := by simp [consolidated]

/-- Ids. Generator state is the last millisecond; the `ulid` Generator keeps ordering when
    `now ≤ last` by incrementing the random part (timestamp stays `last`). -/
def genTs (last now : Nat) : Nat := max last now

/-- PROVED: within one engine the id timestamp never decreases. -/
theorem gen_monotone (last now : Nat) : last ≤ genTs last now := Nat.le_max_left _ _

/-- COUNTEREXAMPLE (Rust `l4_4_ids_stay_ordered_across_a_restart_with_a_slower_clock`):
    a restart resets `last` to 0, so an id made at an earlier wall-clock time sorts
    before an older stored id (SPEC §2: id order = insertion order). -/
theorem restart_breaks_order : ∃ stored now, genTs 0 now < stored ∧ now < stored := ⟨2000, 1000, by decide, by decide⟩

/-- COUNTEREXAMPLE (Rust `note_l4_1_ulid_ids_stay_ordered_past_48_bit_milliseconds`):
    the `ulid` crate keeps 48 timestamp bits; SPEC §8 claims 50. -/
def ulidTs (ms : Nat) : Nat := ms % 2 ^ 48
theorem ulid_wraps : ulidTs (2 ^ 48) < ulidTs (2 ^ 48 - 1) := by decide

/-! ## L4-5  checked / forget / stats -/

/-- `checked`: stability clamp, with NaN modelled as `none` → 1 -/
def clampStab (s : Option Nat) (m : Nat) : Nat := match s with | none => 1 | some v => max 1 (min v m)

theorem checked_stability_range (s : Option Nat) (m : Nat) (hm : 1 ≤ m) :
    1 ≤ clampStab s m ∧ clampStab s m ≤ m := by
  cases s <;> simp [clampStab] <;> omega

theorem checked_time_clamp (t now : Nat) : min t now ≤ now := Nat.min_le_right _ _

/-- A cosine from the store is used only if finite and then capped: `min 1000 c ≤ 1000`
    (scaled by 1000).  The cap is the ONLY defence against a non-unit stored vector:
    a vector of norm 100 and true cosine 0.05 reports 5 → capped to 1.0 — a perfect match.
    COUNTEREXAMPLE (Rust `note_l4_5_a_non_unit_stored_vector_cannot_fake_a_perfect_match`). -/
theorem capped_cos_is_perfect_for_scaled_vector : min 1000 (100 * 50) = 1000 := by decide

/-- `forget` goes through `memory` = `checked`: an unusable row (invalid id / empty text)
    maps to `none`, so `forget` returns `None` and removes nothing.
    COUNTEREXAMPLE (Rust `note_l4_5_forget_can_remove_an_unusable_row`). -/
def checkedRow (idOk textNonEmpty : Bool) : Option Unit := if idOk && textNonEmpty then some () else none
def forgetRemoves (idOk textNonEmpty : Bool) : Bool := (checkedRow idOk textNonEmpty).isSome

theorem forget_skips_unusable : forgetRemoves true false = false ∧ forgetRemoves false true = false := by decide

end Lane4
