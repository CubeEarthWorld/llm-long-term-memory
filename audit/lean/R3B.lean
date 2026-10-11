/-! Round 3, lane B -- SPEC §3: "1 帯の窓は 128 件が上限 (窓は 4 倍ずつ広げて読み直すので、
    読み出しの総量は上限の約 1.4 倍まで)".   Lean 4 core only.
    Compile:  ~/.elan/bin/lean.exe audit/lean/R3B.lean

    Model of `Engram::victims` (src/engine.rs:496-523) for ONE band whose window keeps growing
    (worst case: the unread-row bound never lets it stop before the cap):
      first read  = excess + BAND_DEPTH (= excess + 8) rows,
      each regrow = min (4 * limit) cap rows -- a FRESH read of that many rows (the store is
                    asked for the whole prefix again) -- while limit < cap,
      cap         = max BAND_CAP (excess + BAND_DEPTH), BAND_CAP = 128.
    `total excess` = rows the store returns for the band over the whole call. -/

namespace R3B

def reads : Nat → Nat → Nat → Nat
  | 0, l, _ => l
  | fuel + 1, l, cap => if l < cap then l + reads fuel (min (4 * l) cap) cap else l

def cap (excess : Nat) : Nat := max 128 (excess + 8)
def total (excess : Nat) : Nat := reads 8 (excess + 8) (cap excess)

/-- The worst case is excess = 23: reads 31, 124, then a re-read of 128 (the last regrow step
    adds 4 rows but costs a whole 128-row read): 283 rows = 2.21 x BAND_CAP. -/
theorem worst_case : total 23 = 283 := by decide +kernel

/-- No excess in 1..119 (the only ones whose cap is the constant 128 and that can grow) reads more. -/
theorem never_above_283 : ∀ e, e < 120 → 1 ≤ e → total e ≤ 283 := by decide +kernel

/-- From excess 120 the first read already is the cap: no regrow, ratio exactly 1. -/
theorem big_excess_no_regrow : ∀ e, e < 1000 → 120 ≤ e → total e = cap e := by decide +kernel

/-- So "about 1.4 x the cap" (<= 1.5 x = 192 rows) is FALSE for excess = 23 (and for many more). -/
theorem spec_1_4_is_false : ¬ (total 23 * 2 ≤ 3 * cap 23) := by decide +kernel

/-- It does hold for the typical single eviction (excess = 1: 9 + 36 + 128 = 173 = 1.35 x). -/
theorem excess_one : total 1 = 173 := by decide +kernel

/-- Exactly which excesses 1..119 stay within 1.5 x the cap (the rest are worse): 1..4 and 24..56. -/
theorem within_1_5 : ∀ e, e < 120 → 1 ≤ e →
    (total e * 2 ≤ 3 * cap e ↔ (e ≤ 4 ∨ (24 ≤ e ∧ e ≤ 56))) := by decide +kernel

end R3B
