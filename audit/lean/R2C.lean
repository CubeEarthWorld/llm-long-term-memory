/-! Round 2, lane C -- peripheral modules and boundary validation.
    Lean 4 core only (no Mathlib).  Compile:  ~/.elan/bin/lean.exe audit/lean/R2C.lean
    Floats are not modelled; every float fact that is ASSUMED is marked `-- ASSUME`.
    `decide +kernel` is the kernel evaluator; `native_decide` is used exactly once (the exhaustive
    146 097-day era check in A, minutes in the kernel, milliseconds compiled).  No `sorry`.

    Contents
      A. `time.rs::civil_from_days` is the proleptic Gregorian calendar for EVERY i64 day count
         (so "pure arithmetic beyond year 9999" is true), and `fmt_local` cannot overflow i64.
      B. `parse_offset` / `fmt_offset`: exact round trip for real offsets, a made-up rendering
         for out-of-range ones (defect C6), minutes of a historic LMT offset are truncated.
      C. `ulid_ids`: the Windows `SystemTime` limit (defect C3) and the 48-bit year.
      D. `Config::validate` => no usize overflow in the counts the engine multiplies (A set of
         parameters is bounded by 2^20) and the one place the argument needs a clock >= 0 (gist
         `last_recall`, defect C9, theoretical).
      E. `DreamRequest::prompt`: two sequential `replace` calls re-scan inserted text (defect C1);
         a single pass cannot. -/

namespace R2C

/-! ## A. calendar -/

/-- Hinnant `civil_from_days`, written exactly as `time.rs` (Euclidean `/` and `%` on Int,
    like `div_euclid` / `rem_euclid`).  `finish era doe` is everything after the era split. -/
def finish (era doe : Int) : Int × Int × Int :=
  let yoe := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
  let doy := doe - (365 * yoe + yoe / 4 - yoe / 100)
  let mp := (5 * doy + 2) / 153
  let d := doy - (153 * mp + 2) / 5 + 1
  let m := if mp < 10 then mp + 3 else mp - 9
  (yoe + era * 400 + (if m ≤ 2 then 1 else 0), m, d)

def civil (days : Int) : Int × Int × Int :=
  finish ((days + 719468) / 146097) ((days + 719468) % 146097)

/-- Hinnant `days_from_civil` (the independent inverse). -/
def fromCivil (y m d : Int) : Int :=
  let y' := if m ≤ 2 then y - 1 else y
  let era := y' / 400
  let yoe := y' - era * 400
  let doy := (153 * (if m > 2 then m - 3 else m + 9) + 2) / 5 + d - 1
  let doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
  era * 146097 + doe - 719468

def isLeap (y : Int) : Bool := y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
def dim (y m : Int) : Int :=
  if m == 2 then (if isLeap y then 29 else 28)
  else if m == 4 || m == 6 || m == 9 || m == 11 then 30 else 31

/-- The window check for one `doe` in `[0, 146097)`: valid month and day, and `fromCivil` returns
    the day. -/
def windowOk (i : Nat) : Bool :=
  match finish 0 (i : Int) with
  | (y, m, d) => 1 ≤ m && m ≤ 12 && 1 ≤ d && d ≤ dim y m && fromCivil y m d == (i : Int) - 719468

def allOk : Nat → Bool
  | 0 => true
  | n + 1 => windowOk n && allOk n

theorem allOk_spec : ∀ n, allOk n = true → ∀ i, i < n → windowOk i = true := by
  intro n
  induction n with
  | zero => intro _ i hi; omega
  | succ n ih =>
    intro h i hi
    simp only [allOk, Bool.and_eq_true] at h
    by_cases e : i = n
    · subst e; exact h.1
    · exact ih h.2 i (by omega)

/-- One full 400-year era (146 097 days) checked exhaustively.  `native_decide` (compiled
    evaluation; the kernel route takes minutes): this is the only trusted-compiler step. -/
theorem window_exhaustive : allOk 146097 = true := by native_decide

/-- Shifting the year by `400 e` shifts the day count by exactly `146097 e`. -/
theorem fromCivil_shift (y m d e : Int) : fromCivil (y + 400 * e) m d = fromCivil y m d + 146097 * e := by
  simp only [fromCivil]
  split <;> split <;> omega

theorem finish_shift (era doe : Int) :
    finish era doe = ((finish 0 doe).1 + 400 * era, (finish 0 doe).2.1, (finish 0 doe).2.2) := by
  simp only [finish]
  refine Prod.ext ?_ (Prod.ext rfl rfl)
  simp only []
  omega

/-- Round trip for EVERY day count (all of i64, indeed all Int), not just years 1..9999. -/
theorem fromCivil_civil (days : Int) :
    fromCivil (civil days).1 (civil days).2.1 (civil days).2.2 = days := by
  have hw := allOk_spec _ window_exhaustive ((days + 719468) % 146097).toNat
    (by have : (days + 719468) % 146097 < 146097 := by omega
        omega)
  have h0 : 0 ≤ (days + 719468) % 146097 := by omega
  have hc : (((days + 719468) % 146097).toNat : Int) = (days + 719468) % 146097 := Int.toNat_of_nonneg h0
  simp only [windowOk, hc, Bool.and_eq_true, beq_iff_eq] at hw
  have hf := finish_shift ((days + 719468) / 146097) ((days + 719468) % 146097)
  have hS := fromCivil_shift (finish 0 ((days + 719468) % 146097)).1 (finish 0 ((days + 719468) % 146097)).2.1
    (finish 0 ((days + 719468) % 146097)).2.2 ((days + 719468) / 146097)
  have hr := hw.2
  simp only [civil]
  rw [hf]
  rw [hS, hr]
  omega

/-- The calendar fields are real dates for every day count: month 1..12, day within the month
    (leap years included: the leap rule has period 400). -/
theorem civil_valid (days : Int) :
    1 ≤ (civil days).2.1 ∧ (civil days).2.1 ≤ 12 ∧ 1 ≤ (civil days).2.2 ∧
      (civil days).2.2 ≤ dim (civil days).1 (civil days).2.1 := by
  have hw := allOk_spec _ window_exhaustive ((days + 719468) % 146097).toNat
    (by have : (days + 719468) % 146097 < 146097 := by omega
        omega)
  have h0 : 0 ≤ (days + 719468) % 146097 := by omega
  have hc : (((days + 719468) % 146097).toNat : Int) = (days + 719468) % 146097 := Int.toNat_of_nonneg h0
  simp only [windowOk, hc, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hw
  have hf := finish_shift ((days + 719468) / 146097) ((days + 719468) % 146097)
  have hleap : ∀ (y e : Int), isLeap (y + 400 * e) = isLeap y := by
    intro y e
    have a4 : (y + 400 * e) % 4 = y % 4 := by omega
    have a100 : (y + 400 * e) % 100 = y % 100 := by omega
    have a400 : (y + 400 * e) % 400 = y % 400 := by omega
    simp only [isLeap, a4, a100, a400]
  have hdim : ∀ (y m e : Int), dim (y + 400 * e) m = dim y m := by
    intro y m e
    simp only [dim, hleap]
  simp only [civil]
  rw [hf]
  simp only [hdim]
  exact ⟨hw.1.1.1.1, hw.1.1.1.2, hw.1.1.2, hw.1.2⟩

/-! ### `fmt_local` cannot overflow i64 -/

def i64min : Int := -(2 ^ 63)
def i64max : Int := 2 ^ 63 - 1
def sat (x : Int) : Int := if x < i64min then i64min else if x > i64max then i64max else x

/-- Largest magnitude `parse_offset` accepts: `+99:99`. -/
def offMax : Int := 99 * 3600 + 99 * 60

theorem sat_in_range (x : Int) : i64min ≤ sat x ∧ sat x ≤ i64max := by
  unfold sat i64min i64max
  split <;> (try split) <;> omega

/-- Every intermediate of `fmt_local` (days, z, era*400, the year) is an i64 for ANY `local` in
    i64 (and `local = saturating_add(unix, off)` always is, by `sat_in_range`).  The Rust uses
    plain `+`/`*` for these. -/
theorem fmt_local_no_overflow (loc : Int) (h0 : i64min ≤ loc) (h1 : loc ≤ i64max) :
    let days := loc / 86400
    let z := days + 719468
    let era := z / 146097
    let doe := z % 146097
    let yoe := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
    i64min ≤ days ∧ days ≤ i64max ∧ i64min ≤ z ∧ z ≤ i64max ∧
    i64min ≤ era * 400 ∧ era * 400 ≤ i64max ∧
    i64min ≤ yoe + era * 400 + 1 ∧ yoe + era * 400 + 1 ≤ i64max ∧
    0 ≤ yoe ∧ yoe < 400 := by
  intro days z era doe yoe
  simp only [days, z, era, doe, yoe, i64min, i64max] at *
  omega

/-! ## B. offsets -/

/-- `parse_offset` on `sign HH:MM` (both parts 1-2 / 2 digits, so `h <= 99`, `m <= 99`). -/
def parseOff (neg : Bool) (h m : Int) : Int := (if neg then -1 else 1) * (h * 3600 + m * 60)

/-- `fmt_offset`: (sign, hours, minutes). -/
def fmtOff (s : Int) : Bool × Int × Int :=
  let a := if s < 0 then -s else s
  (decide (s < 0), a / 3600, a % 3600 / 60)

theorem offset_roundtrip (neg : Bool) (h m : Int) (h0 : 0 ≤ h) (hh : h < 24) (m0 : 0 ≤ m) (mm : m < 60) :
    (fmtOff (parseOff neg h m)).2 = (h, m) ∨ (h = 0 ∧ m = 0) := by
  unfold fmtOff parseOff
  cases neg <;> simp <;> omega

/-- Defect C6: an accepted but impossible offset is rendered as a different, made-up offset. -/
theorem out_of_range_offsets_are_rewritten :
    (fmtOff (parseOff false 99 99)).2 = (100, 39) ∧ (fmtOff (parseOff false 9 75)).2 = (10, 15) := by
  decide +kernel

/-- Seconds of a historic LMT offset (Asia/Tokyo before 1888: +09:18:59) are dropped when the
    offset is frozen as `+HH:MM`: the stored field is off by `off % 60` (< 60 s) -- a
    cosmetic loss, noted not counted. -/
theorem lmt_seconds_are_dropped : parseOff false (fmtOff 33539).2.1 (fmtOff 33539).2.2 = 33539 - 59 := by
  decide +kernel

/-! ## C. ulid_ids -/

/-- Windows `SystemTime` is a signed 64-bit count of 100 ns ticks since 1601-01-01; the
    Unix epoch sits 116 444 736 000 000 000 ticks in.  `UNIX_EPOCH + Duration::from_millis(ms)`
    panics once `ms * 10 000` pushes the count above i64 max. -/
def winEpochTicks : Int := 116444736000000000
def winMaxMillis : Int := (i64max - winEpochTicks) / 10000

theorem win_max_millis : winMaxMillis = 910692730085477 := by decide +kernel

/-- `millis.max(0) as u64` is unclamped above: the engine's `now * 1000.0 as i64` reaches
    1e15 ms at a clock of 1e12 s (year 33 658) and saturates to i64::MAX for any larger
    clock (including +inf) -- every one of these overflows on Windows. -/
theorem ulid_overflow_on_windows :
    (1000000000000000 : Int) > winMaxMillis ∧ (200000000000000 : Int) ≤ winMaxMillis := by
  decide +kernel

/-- A 48-bit ULID timestamp wraps after this many whole years (SPEC §8 says 10 889). -/
theorem ulid_48_bit_years : (2 ^ 48 : Int) / 1000 / 31556952 = 8919 := by decide +kernel

/-! ## D. Config::validate and usize overflow -/

def COUNT_MAX : Nat := 2 ^ 20
def usizeMax : Nat := 2 ^ 64 - 1

/-- Every product the engine takes of a validated count is far below usize max:
    `4*inject_n`, `8*dream_budget` (SEEDS_PER_BUDGET), `64*dream_budget` (REINDEX_BATCH),
    `dream_max_members - 1`, and even four rounds of `k *= 4` before saturating. -/
theorem counts_cannot_overflow (n : Nat) (h : n ≤ COUNT_MAX) :
    4 * n ≤ usizeMax ∧ 8 * n ≤ usizeMax ∧ 64 * n ≤ usizeMax ∧ 4 * (4 * (4 * (4 * (4 * n)))) ≤ usizeMax := by
  unfold COUNT_MAX at h; unfold usizeMax; omega

/-- The window growth `(w.limit * 4).min(cap)` runs only while `w.limit < cap`, and
    `cap = max 512 (excess + 8)` is bounded by the row count (ASSUME rows < 2^40: a 2^40-row
    store is beyond any disk), so `limit * 4` cannot overflow. -/
theorem band_window_growth (limit cap : Nat) (hlt : limit < cap) (hcap : cap ≤ 2 ^ 40 + 8) :
    limit * 4 ≤ usizeMax := by
  unfold usizeMax; omega

/-- `last_recall = now_i - (stability*elapsed).round() as i64`, where the float -> i64 cast
    saturates at i64::MAX.  For a clock at or after 1970 the subtraction never overflows: -/
theorem gist_last_recall_ok_for_nonnegative_clock (now x : Int) (h0 : 0 ≤ now) (h1 : now ≤ i64max)
    (hx0 : 0 ≤ x) (hx1 : x ≤ i64max) : i64min ≤ now - x ∧ now - x ≤ i64max := by
  unfold i64min i64max at *; omega

/-- ... but not for a clock before 1970 with a large (valid) `max_stability`: with
    S_max >= 1.4e17 s the product `S * 64` reaches 2^63 and `now - x` leaves i64 (a debug
    overflow panic / release wrap).  Theoretical: needs now < 0 AND S_max > 1e17. -/
theorem gist_last_recall_overflows_for_negative_clock :
    ∃ now x : Int, now < 0 ∧ 0 ≤ x ∧ x ≤ i64max ∧ now - x < i64min :=
  ⟨-2, i64max, by decide, by decide, by decide, by decide⟩

/-- With the default configuration (S_max = 10 years, x <= 64 * S_max) the subtraction is safe
    for any clock above about -9.2e18 s. -/
theorem gist_last_recall_default_safe (now x : Int) (hn : i64min + 2 ^ 35 ≤ now) (hn1 : now ≤ i64max)
    (hx0 : 0 ≤ x) (hx1 : x ≤ 64 * 315360000) : i64min ≤ now - x ∧ now - x ≤ i64max := by
  unfold i64min i64max at *; omega

/-! ## E. DreamRequest::prompt substitution -/

/-- Strings over an alphabet where `ct` / `ls` are the literal tokens `{current_time}` /
    `{listing}` (as indivisible symbols: neither token overlaps another symbol). -/
inductive Sym | ct | ls | ch (c : Nat)
  deriving DecidableEq, Repr

/-- `str::replace(token, r)` over symbol lists. -/
def replaceTok (tok : Sym) (r : List Sym) : List Sym → List Sym
  | [] => []
  | s :: rest => if s = tok then r ++ replaceTok tok r rest else s :: replaceTok tok r rest

/-- The template (current time first, listing last), as `DREAM_INSTRUCTION`. -/
def template : List Sym := [.ch 1, .ct, .ch 2, .ls, .ch 3]

/-- Rust: `replace("{listing}", listing).replace("{current_time}", time)`. -/
def prompt (listing time : List Sym) : List Sym :=
  replaceTok .ct time (replaceTok .ls listing template)

/-- A single pass: every template symbol is substituted once, inserted text is never rescanned. -/
def promptOnce (listing time : List Sym) : List Sym :=
  template.flatMap fun s => match s with | .ct => time | .ls => listing | s => [s]

/-- Replacing a token inside a list rewrites every occurrence, including one in the middle:
    this is the lemma behind defect C1 -- the second `replace` walks into the listing the
    first one inserted. -/
theorem replaceTok_hit (tok : Sym) (r : List Sym) (a b : List Sym) :
    replaceTok tok r (a ++ tok :: b) = replaceTok tok r a ++ r ++ replaceTok tok r b := by
  induction a with
  | nil => simp [replaceTok]
  | cons x xs ih =>
    simp only [List.cons_append, replaceTok]
    split <;> simp [ih]

/-- Concrete instance with the engine's time text `[t1, t2]`. -/
theorem second_replace_concrete :
    prompt [.ch 7, .ct, .ch 8] [.ch 40, .ch 41] =
      [.ch 1, .ch 40, .ch 41, .ch 2, .ch 7, .ch 40, .ch 41, .ch 8, .ch 3] := by
  decide

theorem single_pass_keeps_listing :
    promptOnce [.ch 7, .ct, .ch 8] [.ch 40, .ch 41] =
      [.ch 1, .ch 40, .ch 41, .ch 2, .ch 7, .ct, .ch 8, .ch 3] := by
  decide

/-- The two agree exactly when the listing holds no `{current_time}` token. -/
theorem agree_when_listing_has_no_token :
    prompt [.ch 7, .ch 8] [.ch 40] = promptOnce [.ch 7, .ch 8] [.ch 40] := by decide

end R2C
