# Round 3, lane D: long horizons, scale, performance regressions

Files: `tests/audit3_d.rs` (22 tests, 6 of them `#[ignore]` measurements). `src/` and existing tests untouched.
Run: `CARGO_TARGET_DIR=C:/t/r3d cargo test --release --test audit3_d -j 3` (about 90 s; 14 pass, 2 fail by design).
Measurements: `... -- --ignored --nocapture --test-threads=1` (about 5 min, 100k x 768-d random unit vectors, in-memory redb unless noted).
Numbers come from a shared 16-core Windows machine while other agents compiled: treat them as +-20 %.
`src/` changed under me during the audit (`View::After`, the reindex scrub, widening `Engram::find`, no more time clamping in `checked`);
everything below was re-run against the final tree.

## Failing tests (real findings)

| test | severity | one-line root cause |
|---|---|---|
| `stale_unusable_rows_are_not_rescanned_by_every_cued_remember_and_dream` | medium | the new widening `Engram::find` (used by `remember`'s `find(Stale, 1)` and by `reindex`'s `find(Stale, limit)`) doubles its read until it finds a usable row or the view ends; empty-text rows stay "stale" forever (they cannot be embedded), so every cued remember and every dream re-reads all of them: O(junk) per operation (10k junk = 45 ms, 100k junk = 400-490 ms; before: 1 row). |
| `bandcap_repro_a_dense_aged_band_evicts_a_stronger_trace_than_the_weakest` | low | `BAND_CAP = 128`: when one band holds more relevant rows than the window and the order by `last_recall` disagrees with the order by rank (lower S recalled later, dt/S ~ 45), the true weakest sits behind the window; the engine evicts a trace up to 4 to 5 log2-strength stronger (SPEC §3 first says a capped band "is approximate", then says the result equals the full comparison for usable traces: self-contradictory; it is the first that is true). |

Existing test `tests/engine.rs::rows_read_are_clamped` fails on the current tree (it expects `last_recall` clamped on read; `checked` no longer does and the
SPEC says it must not): stale test, not mine.

## 1. The 3000-year simulation at reduced scale (SPEC §8)

One harness, `Spy` store (wraps `Store`, keeps a shadow of every row's `(created_at, last_recall, stability)` from the writes it sees, counts reads)
plus a brute-force eviction oracle run on the pre-state of every write that inserts: expected victims = the `excess` weakest by `log2 S - dt/S` among
traces past the grace period (young ones too when more than `capacity/10` are young, or too few old ones exist); clock = ManualClock read the way `Engram::now` reads it.
Operations: remember (60 %, 25 % near duplicates, 20 % with cue, salience 0..10), recall + cite, dream (scripted adjudicator: Err, keep, split in two, merge; budget 2),
forget, peek, reindex; five key facts recalled whenever R < 0.7 and every 11 steps; bursts of 5-155 writes within seconds (floods); 10-year silences (1 per 400 steps),
NTP steps back of 1-30 days, forward excursions of 1-30 years undone. Every 997 steps and at the end: all §7 invariants read from the store (count <= capacity,
`stats().records` equals the rows, 1 <= S <= S_max, unit vectors with a model id, text <= 170 chars without delimiters, ids unique, R in [0,1] at now / now+10y / now-40d,
shadow equals store row for row, every key fact still stored); at the end every key fact is the top recall.

| scenario | capacity | steps | virtual years | remembers (full) | dreams / replaces | band rows per full remember (mean / max) | evictions checked | not the weakest | worst gap (log2 strength) |
|---|---|---|---|---|---|---|---|---|---|
| cap50 | 50 | 90 000 | 2 260 | 61 579 (60 547) | 7 087 / 3 395 | 45.7 / 95 | 61 579 | 0 | 0 |
| cap120 | 120 | 90 000 | 2 373 | 62 124 (58 565) | 6 919 / 7 371 | 89.5 / 165 | 62 124 | 0 | 0 |
| cap500 | 500 | 90 000 | 2 361 | 61 379 (56 122) | 7 149 / 8 627 | 104 / 341 | 61 379 | 467 (0.76 %) | +0.36 (alive traces) |
| saturated200 (195 ten-year facts + keys) | 200 | 60 000 | 1 510 | 35 916 (32 764) | 6 032 / 6 506 | 80 / 203 | 35 916 | 0 | 0 |
| saturated500 | 500 | 60 000 | 1 749 | 41 171 (38 448) | 4 519 / 5 296 | 124 / 431 | 41 171 | 17 (0.04 %) | +4.74 (strength 2^-25, practically dead) |

Result: no §7 invariant is ever violated; key facts recalled periodically survive every silence, step back and flood; no young trace is ever evicted out of turn
(no negative gap in any run); eviction equals the oracle exactly in the three small-band scenarios. Deviations appear only when one band holds more than 128 rows
(cap 500): see the `bandcap_repro` failing test. The sims assert the generous bounds (<= 2 % deviating, |gap| <= 8) and that a remember never reads more than 60 x 200 band rows.

Other long-horizon tests (all pass):
* `saturated_store_new_memory_lives_grace_then_oldest_fact_yields`: 100 ten-year facts recalled 1..100 months ago; eight new memories (6 h apart) each evict exactly the
  least recently recalled fact, none dies inside its grace period; afterwards unrecalled notes are the weakest rows and go before any fact.
  Note for SPEC §8: "after the grace period the least recently recalled old fact yields" is only true while grace-protected notes are all that is left to evict;
  an unrecalled new memory (S = 1 day) is far weaker than every ten-year fact and is the first to go once it is 3 days old. Key facts in the saturated sims therefore
  had to be ten-year facts themselves.
* `ulid_48_bit_horizon_...`: clock at 2^48 ms - 10 s, 60 remembers crossing the horizon and 1e9 s past it: ids stay 26 chars, strictly increasing, unique; recall/dream work.
* `clock_extremes_never_panic_and_keep_the_invariants`: clocks 0, -1, NaN, +-inf, f64::MAX/MIN, +-2^63, 2^53, year 10000, jiff's min/max, 1e300 with eviction (cap 6),
  cued remembers, recall, cite, dream: no panic, S in range, R in [0,1], count <= capacity.
* `fmt_local_i64_edges`: i64::MIN/MAX and odd timezone fields.
* `eviction_still_picks_the_weakest_in_a_tied_store`: 3 000 rows with equal rank across 57 bands (the adversary of the window bound), 40 evictions: all exact.
* `young_count_cost_is_flat_...`: `count(Young)` with nothing young is 2.3 us at 2k and 20k.

## 2. Cost table (release, 768-d, 100k rows = 293 MB of vectors)

v2.2 reference (earlier session, approximate): recall 10k 2.1-9 ms, 100k ~11 ms; remember 2 ms; dream at 100k 43 ms.
"rows/op" counts rows returned by `Storage::find`; `near` reads vectors inside the store (not counted, see its latency); "band rows" is the part read by eviction.

Basic operations (store not full):

| operation | N = 1k | 10k | 100k | verdict |
|---|---|---|---|---|
| remember | 0.6 ms | 6.2 ms | 7.0 ms | flat above 8 192 vectors (the `PROBE_VECTORS` bound); the near search is ~90 % of it; v2.2 "2 ms" was a smaller store |
| remember with cue | 0.5 | 6.1 | 7.7 | same |
| recall, 1 cue | 0.4 | 5.9 | 7.0 | v2.2: 2.1-9 / 11 ms: same or better |
| recall, 8 cues | 3.1 | 41.6 | 55.2 | 8 near searches; flat in N |
| peek, 1 cue | 0.4 | 5.1 | 6.5 | |
| stats() | 0.01 | 0.01 | 0.01 | |
| dream, queue empty (reindex scrub page of 64 rows + labile page) | 0.2 | 0.3 | 0.35 | v2.2 43 ms was with labile seeds |
| dream, 40 labile seeds (budget 5, keep): 40 near searches | 16.8 | 209.6 | 248.8 | **> 100 ms**: 8 x budget searches of ~6 ms; bounded by the 8 192-vector probe, not by N |
| reindex(320), every row stale (model switch) | | 20.5 | 26.8 | 5 batches, flat in N |
| remember while every row is stale | | 0.6 | 1.0 | |

Eviction (store exactly full, every remember evicts one; includes the ~6 ms near search; "idle 1y" = clock one year after the rows were written):

| layout | N | ms mean / max | band rows read | `find` calls |
|---|---|---|---|---|
| 1 populated band (S near S_max), idle 1y | 1k / 10k / 100k | 1.6 / 7.6 / 9.2 | 173 / 173 / 173 | 261 |
| 10 populated bands | 1k / 10k / 100k | 1.8 / 7.9 / 10.9 | 216 / 290 / 418 | 261 |
| 41 populated bands | 1k / 10k / 100k | 2.4 / 9.4 / 10.9 | 369 / 369 / 369 | 259 |
| 57 populated bands | 1k / 10k / 100k | 3.0 / 9.6 / 12.3 | 512 / 513 / 513 | 259 |
| all traces just recalled, 57 bands | 1k / 10k / 100k | 3.1 / 9.4 / 12.9 | 512 / 545 / 651 | 260 |
| **tied ranks in all 57 bands (adversary)** | 1k / 10k / 100k | 4.8 / 45 / 81 (max 94) | 905 / 6 961 / 9 515 | 280-369 |
| flood: every row young (grace period) | 1k / 10k / 100k | 1.5 / 7.9 / 16.8 | 173 / 173 / 173 | 261 |

* Rows read per remember do not grow with N in the realistic layouts (173 to 650 band rows); `find` calls are 256 bands + a few (the engine probes all 256 `u8` bands,
  199 of them empty: ~1 ms). The tied-rank adversary reads 9.5k rows (all 57 bands grow to the 128 cap: 9 + 36 + 128 rows each, twice when the grace pool is short) and is the only
  eviction case near 100 ms (94 ms max at 100k, 81 mean). It is bounded: gives <= b x 283 rows per `heads` call (57 bands: 16 131; ceiling for 256 bands x 2 calls: 144 896).
* Flood: `count(View::Young)` walks every young row on every full remember (O(young), allowed up to the whole store when a flood fills it): 0.7 ms at 10k young, 8.2 ms at 100k
  (the 16.8 vs 9-12 ms above). Linear in N, still far below 100 ms at 100k; at 1M young rows it would be ~80 ms.

Structural cases (this session's changes):

| case | N | ms | rows / near calls | verdict |
|---|---|---|---|---|
| peek with 100 / 1 000 / 5 000 / 20 000 related traces (hot cluster at cos 0.97; `select` widens k x4 per round) | 100k | 19.6 / 35 / 78 / **189** | 3 / 4 / 5 / 6 near calls | each round re-scans 8 192 vectors and loads k rows: > 100 ms once ~10k traces are near the query (same at 10k: 158 ms) |
| peek behind 100 / 1 000 / 10 000 closer unusable rows (empty text) | 100k | 29 / 57 / **174** | 4 / 7 / 10 near calls | `Engram::near` doubles `ask` and re-runs the whole near search; at 10k junk (> 8 192) the answer can be missed at 10k (found at 100k) |
| reindex(5) / cued remember / dream behind 1 000 / 10 000 / 100 000 stale unusable rows | 10-20k | 3.6 / 30 / **437** (reindex); 8.8 / 44 / **392** (remember); 3.0 / 30 / **446** (dream) | 2 285 / 20 245 / 263 845 rows | **the failing test**: O(junk) per operation |
| dream(budget 1) behind 10 000 unusable labile rows (`seeds()` paging) | 10k / 100k | 31 / 36 | 10 064 rows, 159 finds | linear in the unusable queue, every dream (cursor does not skip them); bounded by what was planted, ~3 us/row |
| stats() with 10 000 unusable labile rows | 10k / 100k | 1.3 / 0.9 | 2 count calls | `count(Labile)` is a scan of the labile index |
| peek / remember over N identical vectors (an indivisible IVF list is read whole) | 20k / 100k | 12 / **63** | 1 near call | `PROBE_VECTORS` is checked between lists; a list that cannot split is unbounded: O(N) |
| `Store.count(Young)`, 100k young rows | 100k | 8.2 | | O(young) |
| `Store.count(Band 32)`, 100k rows in the band | 100k | 8.6 | | O(band) (not used by the engine) |
| `purge()` of one damaged row (one full pass over 6 tables) | 10k / 100k (file) | 10.5 / **89-104** | | O(N); heal-only, as documented; never on the hot path |

Flagged operations whose cost grows with N (SPEC §8 claims O(log N) index queries): `count(Young)` in a flood (O(young)), `count(Labile)` in `stats()` (O(labile)),
`purge()` (O(N)), `near()` over an indivisible list (O(list)), `seeds()` over unusable labile rows, `find(Stale)` over unusable stale rows (the failing test), and
`Tx::lists_of` loading every centroid of the model at each write (O(N/512) rows of 3 KB; invisible below 1M traces: remember is 6.2 ms at 10k and 7.0 ms at 100k).
Worst cases above ~100 ms: dream with a full labile queue (250 ms), recall with 20 000 related traces (190 ms) or 10 000 closer junk rows (174 ms), eviction in the tied adversary (94 ms),
stale-junk operations, purge.

## 3. Memory

| measure | value |
|---|---|
| `Engram` struct | 360 bytes; heap: parameters, `write_times` (<= `writes_per_day` f64s: 8 KB at 1 000/day), <= `inject_n` exposures, two id strings (cursor, scrub): independent of N |
| Store centroid cache | ~N/340 lists x (768 x 4 B + overhead): ~1 MB at 100k (private, estimated) |
| in-memory redb, 100k x 768-d | RSS 1 130 MB (raw vectors 293 MB: 3.9x; `InMemoryBackend` growth + page overhead), +3 MB after 30 remember + recall |
| file store, 100k x 768-d | file 515 MB; RSS 617 MB right after the load (write cache), 322 MB after reopen + queries (cache is capped at 1 GiB, so a large store can grow to 1 GiB + the OS cache) |
| file store query latency, 100k | cold peek 60 ms, warm peek 45 ms (in-memory: 6.5 ms): the file backend re-reads pages the 1 GiB cache does not keep warm for a scan this wide |

## 4. Probe quality / forgetting benchmark

Skipped: reproducing "overwrite forgetting is limited by theta_related capture (31/120)" needs the real EmbeddingGemma vectors and the Haiku cache of `bench/`, not a cheap proxy.
What this session changed that could move it: `Engram::near` now trusts the store's cosine (no re-normalisation, drift tolerance 0.05 on ‖v‖²-1) and widens `ask`; with the
bundled `Store` (exact f32 vectors, unit length by construction) both are neutral. Existing proxies run on the final tree: `forgetting`, `equivalence`, `conformance`, `dream`,
`scaling`, `batching`, `longevity` (212 s, 3000 virtual years) all pass; `engine::rows_read_are_clamped` fails (stale expectation, see above).

## 5

* `cost_bound`: for one band's window schedule `l, min(4l, cap), ...` (stop at `cap`), `3 * cost + l0 <= 7 * cap`; hence `cost <= 7/3 cap` for any `excess` (`cost_general`).
* The SPEC claim "total read is about 1.4 x the cap" is true only for `excess <= 2` (173 / 178 rows of 128); `spec_1_4_false`: `excess = 23` reads 31 + 124 + 128 = 283 rows (2.21 x);
  `cost_max_283`: 283 is the exact worst case over every first window at cap 128. A first window above the cap is read once (`big_first_window`).
* `rows_per_heads`: b populated bands read <= b x 283 rows per `heads` call; two calls per remember; 256 bands: <= 144 896.
* `loop_terminates` (+ `step_valid`, `step_lt_or_eq`): the growth loop is at a fixpoint after at most `sum (cap - limit)` iterations because every iteration that changes anything
  strictly lowers that sum; with first window 9 a band grows at most 9 -> 36 -> 128 (`growths_9`).
* Tail of `victims`: truncating to `excess` leaves `min excess pool` victims, and `n + 1 - excess <= cap` (`capacity_after_eviction`).
* Gaps: soundness of the lower bound `log2 s_lo - last_dt/s_lo` (real analysis), and weakest-first itself, which is false past the cap (the failing repro).

## Suggestions (not applied)

1. `Engram::find`: do not widen without bound; cap the extra reads (e.g. one widening of 8 x limit) or have `reindex`/`remember` ask the store for "stale AND usable" (or let the scrub purge rows
   with empty text), so unusable rows cost O(1) per operation. `reindex` could also advance a cursor past unusable stale rows like `seeds()` does.
2. SPEC §3: replace "about 1.4 x" by "at most 7/3 x (283 rows at 128)" and drop "equals the full comparison" next to "approximate at the cap"; or raise the window cap with the band size.
3. `select`: widen by re-using the first scan (ask the store for k x 4 in one pass) instead of re-running a full 8 192-vector near search per round; same for `Engram::near`'s junk widening.
4. `Store::near`: count `PROBE_VECTORS` inside a list, not only between lists, so an indivisible list is bounded.
