# Lane 4 — wake phase and text

Files: `audit/tests/audit_lane4.rs` (copy at `tests/audit_lane4.rs`; run
`CARGO_TARGET_DIR=C:/t/l4 cargo test --test audit_lane4 -j 3`).
Result: 32 tests, 17 pass, 15 fail. `note_*` tests pin design trade-offs / ambiguous spec points.

## Real defects (tests fail)

| id | test | root cause | repro |
|---|---|---|---|
| D1 (med) | `l4_1_valid_id_rejects_every_control_character`, `l4_1_pack_has_one_line_per_trace_even_for_an_odd_stored_id` | `valid_id` = `!any(is_dropped)` and `is_dropped` exempts `is_space` chars, so ids containing `\t \n \v \f \r U+001C-1F U+0085 U+2028 U+2029` are valid. SPEC §2 forbids all control chars in ids. A newline id splits the pack line and is adopted from an `IdGen`. | store row id `"A\nB"`, `recall` returns it; `set_ids(\|_,_\| "A\nB")` then `remember` succeeds. |
| D2 (low) | `l4_2_absurd_inject_n_is_rejected_or_harmless` | `Config::validate` accepts any `inject_n > 0`; `select` computes `4 * inject_n` (and `k *= 4`) unchecked, so `inject_n = usize::MAX` panics (debug) / wraps (release). | `inject_n = usize::MAX`, `recall`. |
| D3 (low) | `l4_3_cite_survives_a_transient_store_error` | `cite` calls `exposures.remove(id)` before `memory(id)?`; a store error loses the exposure, retry cites nothing. | fail `find` during `cite`, then retry. |
| D4 (low) | `l4_3_cite_does_not_boost_a_trace_re_created_under_a_forgotten_id` | `forget` does not drop the exposure; exposures are keyed by id only, and `new_id` only avoids ids currently stored, so a text/time-derived `IdGen` can reuse the id and the new trace inherits the old boost. Fix: `self.exposures.remove(id)` in `forget`. | IdGen returning a fixed id; recall, forget, remember, cite. |
| D5 (low) | `l4_3_cite_parser_finds_a_well_formed_citation_after_a_stray_opener` | `cited_ids` reads from the first `《id:` to the first `》`, so `《id:《id:X》` yields the id `《id:X` (never valid) and the well-formed inner citation is lost. Fix: after an opener, restart at the last opener before the closer. | `cite("《id:《id:X》")`. |
| D6 (low-med) | `l4_4_ids_stay_ordered_across_a_restart_with_a_slower_clock` | `ulid_ids()` makes a new generator per engine; the monotone guarantee covers one process only. After a restart whose clock is behind the stored ids (NTP step, VM restore), new ids sort before old ones and break "id order = insertion order" (FIFO seeds, tie-breaks). Within one engine the clock going backwards is safe (test `l4_4_ids_follow_creation_order_when_the_clock_runs_backwards` passes). | two engines on one store, second clock 1000 s behind. |

## Design notes / spec deviations (`note_*` tests fail)

* `note_l4_1_ulid_ids_stay_ordered_past_48_bit_milliseconds`: SPEC §8 says 50-bit ms (year 37,000); the `ulid` crate masks to 48 bits (year 10,889): ids wrap and stop sorting. Doc fix or custom encoding.
* `note_l4_4_rehearsal_obeys_the_write_limit`: SPEC §4 step 1 imposes the daily limit before the duplicate check; the engine reinforces (a store write) first, so repeated identical `remember` calls are unlimited.
* `note_l4_4_clock_step_back_does_not_reset_the_write_limit`: `retain(t <= now)` drops the writes stamped after a backwards step, so a step back re-opens the window.
* `note_l4_4_more_than_eight_unusable_duplicates_...`: `TEXT_PROBE = 8`; 8 unusable same-text rows ahead of the usable one create a duplicate. SPEC says "先頭数件", so within spec; a paging loop is cheap.
* `note_l4_4_rehearsal_finds_a_row_stored_with_an_older_cleaning`: `find(Text)` matches raw stored text, so a row written by an older cleaner is never rehearsed; the cleaned text is stored again.
* `note_l4_5_forget_can_remove_an_unusable_row`: `forget` goes through `memory` = `checked`, so an unusable row (invalid id / empty text) cannot be removed by id, only by eviction or `reset`.
* `note_l4_5_stats_labile_counts_text_only_traces`: `Stats.labile` = `count(Labile{model})` = labile AND indexed, so text-only labile traces are not counted.
* `note_l4_5_a_non_unit_stored_vector_cannot_fake_a_perfect_match`: `checked` only tests finiteness; a stored vector of norm 100 gets `dot().min(1.0)` = 1, a perfect match for an unrelated query. Normalise or drop non-unit vectors on read.

## Per invariant

* **L4-1 clean_text / clean_tz / valid_id.** Length bound proved in the model (`shorten_len`, `clean_len`, `clean_tz_len`); `trim` idempotence on a trimmed text is not proved but covered by a 500-string fuzz over a 30-char alphabet (delimiters, C0/C1, bidi, U+2028, U+FEFF, astral, combining marks, ideographic space) that checks: 1..170 chars (counted as chars: 300 x U+1F600 gives 170 chars/680 bytes), no dropped/control/non-space whitespace char, no edge/double blanks, idempotence via `Reinforced` of the stored text, same for `cue`. Sentence-boundary cut never gives an empty or trailing-blank text. `clean_tz` keeps `;+HH:MM` and stays <= 64 for 6 hostile zone names. U+200B/U+200E/U+200F pass through (not in SPEC's list). **valid_id disagrees with SPEC: D1.**
* **L4-2 recall pack.** Proved: pack budget invariant (`pack_budget`), MMR distinct and <= inject_n (`mmr_distinct_and_bounded`), widening loop terminates (`widening_terminates`; NaN/inf cosines are mapped to 1.0 in `near`, so `unseen_cos` is never NaN). Test: budget 10/120/400/1024, line shape, tie by id, and `recall == brute force` (80 traces, 60 random one/two-cue queries, MMR replayed, widening exercised since pools exceed 4 x inject_n) all pass. Only D2 (config overflow).
* **L4-3 cite.** Passes: only the last recall's exposure, once per id, exact boost (S x 3.25/2.125 after 2 days), `last_recall` unchanged, forgotten id not resurrected, monotone (`cite_monotone`). Fails: D3, D4, D5.
* **L4-4 remember.** Passes: rate window and `writes_per_day = 0`, salience NaN/inf/negative, ids monotone with a backwards clock inside an engine, consolidated truth table (offline embedder, cue embedding failure, stale row, unindexed then reindexed), failed embedder then `reindex` then recall. Fails: D6 and the `note_` items above.
* **L4-5 forget / stats / reset / checked.** Passes: `forget`/`stats`/`reset` consistency (reset also clears exposures and the write window), rows from an older version (stability 0, negative, 1e30, last_recall = i64::MAX, created_at = i64::MAX, dirty text/tz/cue, unindexed with vector) are read with the write-time rules through `memory`, `recall`, `cite`, `reindex` and capacity eviction; NaN/inf stability, NaN/short vectors and bogus store hits (cos NaN, -inf, 2.5) never reach a result and the engine writes only finite state. Fails: `note_` items for unusable rows, stats and non-unit vectors.

## Not found (checked by reading, no failing test)

`select` widening loop is exact for stores that return all hits (also across multiple cues: unseen cos bounded by the max of per-cue k-th cosines); `near` drops wrong-length and non-finite hits before `dot` (which asserts on mismatch); `retrieved`/`cite` clamp NaN; `reindex` bad vectors stay stale (an embedder returning an invalid vector for the newest `limit` traces would starve older ones, only when it is persistently invalid).
