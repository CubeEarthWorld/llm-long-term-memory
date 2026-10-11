# Lane 2 audit: capacity and eviction

Files: `audit/lean/Lane2.lean` (compiles, no `sorry`; one `True` placeholder marked GAP),
`tests/audit_lane2.rs` (copy in `audit/tests/`). Run: `CARGO_TARGET_DIR=C:/t/l2 cargo test --test audit_lane2 -j 3`.
Result: 9 pass, 7 fail (16 tests) (each failure is a defect or spec weakness below).

## L2-1 count <= capacity after remember
Proved in model (`l2_1_after_within_cap`, `l2_1_exact`, `l2_2_pool_big_enough`, `sum_min_ge`): excess = count+1-cap,
victims truncates to excess, the fallback pool (>= min(total, excess)) is always big enough for cap >= 1, so the final
count = min(count+1, cap). Tests `l2_1_*`, `l2_5_undo_*` pass (caps 1..20, equal keys, undo overshoot). No defect.
Not covered: unusable rows missing from `Store::rows` (damaged JSON) would shrink the pool; Store::apply Remove would
also error on an unreadable row (`Tx::get` -> "unreadable row"), i.e. a damaged row can never be evicted (not reproduced; needs a damaged file).

## L2-2 victims distinct / write set consistent
Distinct and exact: pass (`l2_2_excess_victims_are_the_weakest_and_distinct`, `l2_2_capacity_one_new_is_weakest`);
Insert(new)+Remove(new) nets to "not stored" and the Store count (+1/-1) stays right.
**BUG (low, API): `evicted` mis-reports when excess > 1.** `evicted = victims.last()` and victims are sorted weakest first,
so when the newcomer is the weakest of several victims (reachable after `undo()` or a lowered capacity) `evicted` names the
strongest victim, not the newcomer, although the newcomer was never stored. Contradicts the doc of `Remembered::Inserted`.
Lean: `l2_2_evicted_misreports`. Test: `l2_2_evicted_reports_unstored_newcomer_with_excess` (FAILS).
Fix: report the newcomer when it is among victims (or return all victims).

## L2-3 band head == global minimum of log2 strength
Proved in model only under a monotonicity hypothesis (`head_is_min`). The hypothesis is false in general:
SPEC §3 says "within a band decay speed is nearly equal", but after long idle (dt/S in the hundreds) a factor of sqrt2 in S
dominates last_recall order. Exact failure condition: a band holds more than `excess + BAND_DEPTH(8)` rows AND the true
minimum of `log2 S - dt/S` is not among the first `excess+8` rows by (last_recall, id).
Minimal repro (`l2_3_band_head_is_not_the_global_minimum`, FAILS): capacity 11, ten rows S=92000 last_recall=t0 plus one row
S=66000 last_recall=t0+1e6 (all band 32), clock t0+3.15e7 s: ranks -326 vs -446; the engine evicts a head, not the weaker one.
Frequency: `l2_3_random_differential_against_full_ranking` (FAILS, 13/20 trials) with 100 rows in one band after a year idle.
Impact is bounded (all candidates have strength about 2^-300, i.e. effectively dead), but "same result as full comparison"
(SPEC §3, §8) is false, and the window size is the only thing that decides.
**BUG (medium, spec §2/§4.5 "unusable rows die first"):** the same window hides an unusable row (ranked -inf) that sits
behind >= excess+8 usable rows of its band (`l2_3_unusable_row_dies_first`, FAILS: a usable row is evicted, the broken one stays).
Lean: `l2_3_depth_window_misses_min`, `l2_3_unusable_hidden`, `l2_3_window_covers_when_small` (the window is exact when band size <= excess+8).
Fix options: store an `unusable`/-inf flag as a separate band (e.g. band 256), or read bands until the head rank bound
`log2 S_min - dt/S_min` proves nothing later can be lower.
Other sub-conditions checked and fine: ties (id order both sides), `created_by` filter (Band skips young rows without
consuming the limit), `checked` clamps last_recall to now (future rows rank as dt=0 and sit last in index order: consistent).

## L2-4 grace / flood
Boundary exact: `created_at == now - grace` is old, +1 s is young (`l2_4_grace_cutoff_boundary`, passes); Young/Band partition
rows (`l2_4_partition`); future `created_at` is protected (`l2_4_future_created_at_is_protected`, passes).
Observation (spec-literal, not a code bug): for capacity < 10, `capacity/10 == 0`, the newcomer always counts as young, so
the non-flood branch is unreachable and grace is void (`l2_4_small_cap_void`, `l2_4_small_capacity_has_no_grace` passes):
a weak 1-second-old newcomer is evicted on arrival. `pool.len() < excess` fallback is reachable only if excess > 0.9*capacity
with consistent indexes (shown by count: old >= 0.9 cap + excess).

## L2-5 no immortal memory, index sync
Strength is bounded (S <= S_max) and the random Store model test `l2_5_store_indexes_stay_in_sync_random` (1500 batches of
Insert/replace/State/Index/Remove, NaN-poisoned batches, 3 models; checks band counts and order, created, labile, model, text,
IVF lists/orphans) PASSES: no orphan or missing index entry from the ops themselves. undo() restores rows and the next write
re-fits capacity (`l2_5_undo_then_remember_restores_capacity`).
**DEFECT (medium, design): a future `last_recall` makes a trace immortal until the wall clock reaches it.** `checked` clamps
`last_recall` to `now` on every read, so dt stays 0 for any stored last_recall > now (Lean `l2_5_future_lr_frozen`).
A single forward clock glitch followed by a recall stamps `last_recall = now_glitch`; after the clock is repaired the trace
never decays for as long as the glitch was ahead (test: +100 years glitch, 20 years of churn at capacity 5: a never-recalled
trace survives; `l2_5_future_last_recall_does_not_make_a_trace_immortal` FAILS). Conflicts with the 3000-year/clock-fault claim
(§8) and with "no immortal memory". Fix: re-stamp or penalise rows with `last_recall > now` (e.g. treat as `now` AND write it
back lowered in `State`, or rank them with dt = last_recall - now).
**DEFECT (low, latent): `Store` band key is not stable across the JSON round trip.** `serde_json` is built without
`float_roundtrip`; `meta()` prints stability with ryu and `trace()` parses it back lossy (1.9999999999999998 -> 2.0). The band
is computed from the in-memory value on Insert/State (`put_state`) and from the parsed value in `drop_state`, so a stability within
an ULP below a half-octave edge leaves an orphan `(band,last_recall,id)` entry after any State/Remove (stale entry inflates
band scans and can duplicate the row in two bands, so `victims` may return the same trace twice, shorting the eviction when excess >= 2).
Test: `l2_5_band_key_survives_json_round_trip` lists the 7 offending stabilities (FAILS); mechanism `l2_5_serde_json_f64_round_trip_is_exact` (FAILS).
Lean: `l2_5_orphan_when_band_differs`, `l2_5_sync_when_exact`. Fix: enable serde_json `float_roundtrip`, or compute the band from the
same parsed value everywhere (parse after serialising in `put_state`), or store stability as bits (u64).
Probability for organically computed stabilities is about 1e-13 per write: latent.

## GAPs
`l2_5_ops_preserve_sync` is a placeholder (redb tables not modelled); covered empirically by the random test.
Distinctness of the pool (bands disjoint) is argued, not proved in Lean.
