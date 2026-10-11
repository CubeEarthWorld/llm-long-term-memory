# Round-2 audit A: dream cursor, eviction windows, commit/cite/forget, unit-norm rule

Scope: the round-1 mechanisms in `src/engine.rs` (cyclic dream cursor, adaptive band windows in `victims`,
`commit` exposure gate, `cite` retry, `forget` of unusable rows, `checked` unit-norm rule).
Artifacts: `audit/lean/R2A.lean` (Lean 4.34.1 core, compiles, no `sorry`), `tests/audit2_a.rs`
(`CARGO_TARGET_DIR=C:/t/r2a cargo test --test audit2_a -j 3`; 6 failing = confirmed defects, the rest `#[ignore]`d
as `sound`, `accepted` or `measurement`).

## Confirmed defects (failing tests)

| # | test | defect | root cause |
|---|------|--------|-----------|
| A1 | `unusable_nearest_row_does_not_hide_the_related_trace` | `remember`: if the nearest row is unusable (empty text / bad id / off-norm vector) the usable related trace behind it is missed: `cosine = 0`, born `consolidated = true`, never dreamed. | `near(.., k = 1)` takes the top k then `checked` drops rows (`engine.rs` `near()`); no widening, unlike `select`. |
| A2 | `unusable_rows_do_not_hide_dream_candidates` | `dream`: with `dream_max_members` (8) unusable rows nearer than the real candidate, `candidates()` is empty, the seed is *settled as consolidated* without meeting its older trace (silent loss of a consolidation). Control without the junk rows replaces. | same: top-k before validation in `candidates()`. |
| A3 | `off_norm_row_is_reindexed_not_lost` | A row whose stored vector is off unit length (`|‖v‖²−1| > 1e-3`) is "unindexed" for the engine but the store still says it is indexed under the current model, so `Stale` never lists it, `reindex`/`dream` never repair it: permanently invisible to recall (and, as a labile seed with no cue, never settled). | `checked()` blanks `model_id`/`vector` in memory only; healing path (`View::Stale`) is keyed on the stored `model_id`. |
| A4 | `int8_quantised_store_keeps_its_rows_searchable` | A store that quantises to int8 and returns dequantised vectors (SPEC §2: "quantisation is the store's freedom") loses 28 of 60 rows (47 %) from search at dim 64; any store whose norm error exceeds 1e-3 does. 3072-d f32 rows are fine (ignored test `engine_normalised_3072_dim_vectors_are_never_discarded` passes: error ~1e-6). | `unit()` discards instead of renormalising; with A3 the loss is permanent. A1/A2 amplify it (a dropped nearest row hides its neighbours). |
| A5 | `reopened_engine_still_reaches_every_seed` | The cursor is engine memory only. A host that reopens the engine per dream (CLI run, app start) restarts at the head every time: with 20 undecidable seeds and budget 1, 25 dreams examine 1 seed. Same failure as the pre-round-1 `failed` list; the cycle fixes it only for a long-lived engine. | `cursor: String` is not persisted; `Storage` has no place for it. Low severity / design gap. |
| A6 | `one_band_read_stays_within_the_stated_cap` | SPEC §3 says one band is read to at most 512 rows; the window regrows 9 -> 36 -> 144 -> 512 and each step re-reads from the head: 701 rows (+1 other find) per remember (Lean `dense_band_costs_701`). Low severity. | `read()` restarts at the head per growth step; `Store` has no offset. |

Shared structural areas: **A1, A2, A3, A4 = "validate after the store's top-k / discard instead of repair"** (`near()`+`checked()`); A5 = cursor lifetime; A6 = window re-read.

## Probed and sound (no defect)

* **Cursor fairness / wrap / duplicates / deleted cursor**: `cursor_cycle_matches_the_model` (200 rounds of random arrivals, forgets incl. the cursor seed, budgets 1-4) equals the model `rot`. Lean: `seeds_len`, `seeds_nodup` (the two passes never overlap), `seeds_all` (want >= n reaches everything), `rot_split`/`rot_step` (cursor on the last examined seed rotates the cycle exactly, for any cursor id, deleted or not), `cursor_fair` (every id examined within ceil(n/k) dreams for k >= 1 seeds per dream), `pages_eq` (page loop terminates and equals `take want (filter usable)`; unusable rows never hide later seeds). Note the fairness bound is ceil(n/k) with k = seeds that cost an LLM call per dream (<= budget), not ceil(n/want); free (settled) seeds leave the queue. `clusters()` is `&self`-free of cursor writes (preview does not move it); ids compare bytewise like redb's `str` keys; a custom `Storage` must sort `Labile` bytewise or the wrap pass can stop early or loop (contract, not engine, bug).
* **Window soundness** (Lean `slo_sound`, `unread_rank_ge`, `stop_is_sound`, `dt_antitone`): the bound `log2(s_lo) - last_dt/s_lo` is a true lower bound for every unread usable row for raw S < 1, NaN (read as 1), S > S_max (clamped, `s_lo = min(2^(b/2), S_max)`), `last_recall` in the future (dt = 0, order preserved), `created_by` filtering (a filter does not reorder), and band 255. `1 - 1e-9` slack only lowers the bound. Growth terminates (`grow_progress`); `excess > cap` is handled (`first_window_fits`). Randomised check against brute force (`eviction_matches_brute_force_within_the_band_cap`: wild stabilities 2^-1..2^31, future last_recall, young rows, excess 1 and 31) agrees on every case.
* **Documented approximation** (`#[ignore] accepted`): BAND_CAP = 512: a dense band whose first 512 rows are stronger than rows behind them (S = 1.41 lo vs lo, 15-day idle, 100 s spacing) evicts the wrong row (rank gap about 6 bits).
* **commit / cite / forget**: `cite_retry_boosts_once` (store error then retry: boosted once, duplicates counted once, spent afterwards), `removed_trace_leaves_no_exposure_for_a_reused_id`, `forget_removes_unusable_rows` (empty text, `《` in id, sub-1 stability) all pass.

## Cost of the eviction windows (idle store at capacity, one `remember`, in-memory redb; rows handed out by `find`)

| store | rows read | find calls | time |
|---|---|---|---|
| 1k / 10k / 100k, one band (S = 1 day) | 702 / 702 / 702 | 262 | 2-4 ms |
| 100k, 20 stabilities | 873 | 262 | 7 ms |
| 100k, 56 half-octave bands | 505 | 259 | 5 ms |
| 100k, 41 bands all equally weak (rank about -10) | 28 742 | 382 | 117 ms |

Independent of N (bounded by 701 x unproven bands), so no blow-up in N, but every band whose bound sits at or below the weakest regrows to 512; in a mature idle store that is dozens of bands (about 350 MB per remember at 3072-d f32 if vectors are materialised) and 256 `find` calls even when bands are empty. `measure_victims_cost` reproduces it (`--ignored --nocapture`).
Seed paging over unusable rows (`measure_seed_paging_over_unusable_rows`): once the page has shrunk to `asked = 1`, each unusable labile row costs one `find` (U = 5000 -> 5001 finds, 40 ms): O(U) queries per cycle pass, no growth in page size.

## Not defects
* Floating-point at band edges: log2 rounding is ~1e-16 relative, covered by the 1e-9 slack.
* 3072-d f32 `unit()` error ~1e-6 vs threshold 1e-3: safe.
