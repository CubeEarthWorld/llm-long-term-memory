# Round-3 audit A: regression review of the round-2 fixes

Scope: `Engram::near` widening, `checked` repair, the `unreadable` placeholder, `purge`, `commit` gate,
`remember` bookkeeping, clock, future times, prompt/JSON.
Artifacts: `tests/audit3_a.rs`
(`CARGO_TARGET_DIR=C:/t/r3a cargo test --test audit3_a -j 3`: 9 failing = confirmed, 6 passing guards, 1 `#[ignore]` accepted).

## Confirmed defects (failing tests)

| # | tests | root cause | origin |
|---|-------|-----------|--------|
| R1 | `overscaled_row_below_theta_is_not_reported_as_related`, `rescaled_row_does_not_outrank_a_nearer_unit_row`, `underscaled_row_is_found_after_repair`, `underscaled_row_is_recalled` | `near()` repairs a vector (`renormalized`) and recomputes its cosine **after** the store has ranked, thresholded (`min_cos`) and truncated by the unscaled dot. The engine never re-applies `min_cos`, never re-sorts, and the kth bound uses the recomputed value. Norm 2 / true cos 0.3 is returned as "related" (born labile); norm 2 / true 0.6 outranks a unit row at 0.8; norm 0.5 / true 0.99 (key 0.495 < 0.55) and norm 0.05 under recall's min_cos 0.1 are never returned at all. | **round-2 regression** |
| R2 | `damaged_vector_is_healed_by_a_fresh_engines_reindex` | The damaged-vector queue is in-memory and filled only when a row is *read* by `find`/`near`. A zero vector (current model id) is not in `View::Stale`, never matches `near`, and a consolidated row is never a dream seed: a fresh engine's `reindex` never finds it. | **round-2 fix incomplete** (A3's heal path works only inside one process after an incidental read) |
| R3 | `wrong_dimension_vector_is_reindexed` | A unit vector of another dimension under the *current* model id passes `checked` (`unit`), is dropped by `near`/`select` length filters and by the store's `raw.len()` filter, is not stale and not queued as damaged: lost to search forever. | pre-existing gap, sibling of round-2 A3 (not covered by it) |
| R4 | `unusable_stale_rows_do_not_starve_reindex` | `reindex(limit)` takes `limit` ids of `View::Stale` and only then `checked` drops unusable ones; `limit` unusable rows newer than a usable stale row starve it permanently (dream passes 64 x budget). Same shape as A1/A2, not fixed for reindex. | pre-existing |
| R5 | `unusable_stale_row_does_not_make_every_write_labile` | `remember` decides `consolidated` with a raw `store.find(Stale, 1)`; one unusable row (empty text, other model, can never be indexed) keeps `Stale` non-empty so every later write is born labile. | pre-existing |
| R6 | `backward_clock_step_does_not_reset_the_daily_quota` | `write_times.retain(.. t <= now)` *deletes* writes stamped after a backward clock step; when the clock is right again those writes no longer count (quota bypass). Low severity; `filter` instead of `retain` would keep them. | pre-existing (present at HEAD) |

**Shared structural areas.** R1+R2+R3: *read-time repair of a vector is applied after, or outside, the index that decided to
return the row*; the engine's notion of "usable vector" (finite, unit, right dimension, queued) and the store's notion (dot >=
min_cos on raw bytes, model id) are two different predicates. R4+R5 (+ A1/A2 of round 2): *"does a usable X exist / take the
first N X" asked of a raw store view that contains unusable rows*; every such site needs the widening/paging treatment `near`
and `seeds` got. R6 is isolated (engine bookkeeping).

Regressions of round-2 fixes: **R1 (certain)**, **R2 (fix incomplete, behaviour wrong across restart)**. R3 is a gap the round-2
unit-norm rule left open. R4, R5, R6 are pre-existing (R6 verified present in the original commit). Honest count: 2 of 6.

## Examined and found sound (passing guards or analysis)

* Widening loop: terminates, cost is
  bounded by `PROBE_VECTORS` per iteration x log2 iterations; for exact (unit) rows the kth bound is sound. Only repaired rows break it (R1). `recall_widens_past_unusable_rows` (200 junk rows) passes.
* `Store::rows` placeholder: `new_id` still refuses a damaged row's id (`a_damaged_rows_id_is_not_reused`), eviction ranks it
  `-inf` and removes it, `forget` removes it, the dream skips it and pages past 70 of them
  (`dream_reaches_a_seed_behind_many_unusable_rows`). Its `created_at = 0`, `last_recall = 0`, `consolidated = true` are never read
  as data: `ranked` keeps only the id; `last_dt` from a placeholder at the end of a window is huge, which only makes the lower bound
  more conservative (windows grow to the cap: cost, not correctness).
* `purge`: after forgetting/evicting a damaged row no table keeps its key (`forgetting_a_damaged_row_leaves_no_index_entry`,
  `eviction_removes_the_damaged_row_first` inspect rows, model, labile, texts, created, band, lists, listed with raw redb);
  counts stay exact. Cost is six full-table sweeps per damaged row: O(N) per purge, acceptable only because damage is rare
  (note: a store with M damaged rows being evicted one per write costs M x N).
* `commit` gate: every `Remove` clears its exposure after the write lands; `undo` restores without an exposure. No leak found.
* Clock: NaN, +-inf, +-MAX never panic through remember/recall/dream/now_local (`infinite_and_nan_clocks_do_not_panic`);
  `as i64` saturates; the `now as i64 - round(stability*elapsed)` subtraction in `gists` cannot overflow because at `now = MIN`
  every R is 1 so `elapsed = 0`; ulid millis are clamped. Immortality after a far-future stamp is the already-accepted trade-off.
* Future `created_at`/`last_recall`: all uses go through `max(0, dt)`, the grace test counts them as young, `last_dt` is 0 for
  them (bound stays sound), `stated_by` takes the stored value. Nothing wrong found.
* Prompt: single pass is exact; text containing `{listing}`/`{current_time}` is never re-expanded (`prompt_is_single_pass`).
  `json_texts` string-only rule consistent with `parse`. `relaxed_json` stops after 8 openers (`#[ignore]`d
  `json_answer_after_prose_with_many_openers`: ten `[note]` before the object parse to `None`; accepted bound).
  `ids` of non-string type are stringified (`5` -> `"5"`) while memories are string-only: harmless.

## Notes, not tests

* `Store::rows` still returns `Err("damaged list entry")` for a truncated `LISTS` entry, so `memory`, `forget` and any
  eviction window containing that row fail forever, while a JSON-damaged row is tolerated; SPEC says damaged entries are errors,
  but the two kinds of damage are treated inconsistently.
* Empty `commit(&[])` still opens a write transaction (every recall with no hit, every dream's settle write); present at HEAD.
* `TEXT_PROBE = 8`: eight unusable rows sharing a text (invalid ids) in front of a usable one make `remember` insert a duplicate.
* `Remembered::Inserted.evicted` can be the empty placeholder when an unreadable row was the victim.

