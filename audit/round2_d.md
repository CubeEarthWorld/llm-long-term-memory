# Round 2, lane D: whole-system, model-based, stateful testing

Files: `tests/audit2_d.rs` (harness, 17 tests). `src/` untouched.

Run: `CARGO_TARGET_DIR=C:/t/r2d cargo test --test audit2_d -j 3`
Result (debug profile, ~2 min): **9 pass, 7 fail, 1 ignored**. The 7 failures are 4 confirmed defects (4 hand-minimal
repros + 3 delta-debugged fuzz replays). `AUDIT2_SEEDS=n` scales the sweeps (default 60 seeds per group x 4 groups).

## The harness

* One engine over the bundled `Store` wrapped in `Flaky` (random `Err` on write / find / count / near / clear; a failed write
  changes nothing), a `FuzzEmbedder` (offline, wrong count, wrong dimension, zero, NaN, inf, 1e30, 1e-30, one bad vector in a batch,
  model switches between 3 models of dimension 32/24/40) and a `FuzzAdjudicator` (Err, keep, replace of joined / split / self-rewrite /
  junk / blank / oversized / duplicate / delimiter-laden texts, ids naming non-members, the seed itself, ids with newlines).
  Optionally a faulty `IdGen` (invalid, empty, live ids). Virtual clock: seconds to 30 years forward, up to 40 days back, far jumps.
  Config randomised: capacity 3..40, S_max 1 s..100 y, text_max 8..170, thetas, floors, alpha, MMR, budgets, rate limit.
* Operations are data (`Op`), 200..900 per seed: remember / recall / peek / peek-then-recall / cite / forget / dream (+ immediate
  undo_all) / stale undo / reindex / clusters / reset / model switch / clock / fault settings.
* After every step the store is read directly and checked: SPEC section 7 (count <= capacity except between an undo and the next
  insert; 1 <= S <= S_max; unit vectors, one dimension per model, model/vector coupled; text/cue/id rules; tz; no future stamps; R in [0,1];
  `stats()` equals the rows), plus a shadow model: exact row diff per operation (only the named row changes, `Err` leaves the store unchanged,
  peek writes nothing, forget removes exactly one row, no id ever returns, new ids sort after all earlier ones), rate-window mirror,
  exposure/cite bookkeeping (cite returns exactly the cited-and-exposed ids, spent once, never lowers S, formula checked),
  brute-force oracles for eviction (grace, flood, rank), reactivation cosine and the labile flag at insert, recall (score, floor, top, and
  full pool + MMR + packing, 1830 of 2598 faultless recalls compared exactly per 40 seeds, the rest within float noise of a tie),
  reindex counts and newest-first order, `clusters()` shape and equality with the dream's first cluster, and for every dream:
  reports <-> LLM calls 1:1, calls <= budget, <= 1 per seed, absorbed = seed + named candidates, gist formulas (S, last_recall, inherited
  time/tz/cue, strength not inflated, confabulation guard), the store diff equals what the reports say and nothing else is rewritten,
  settled / kept flags, seeds with no candidates are settled and seeds with candidates are judged when budget is left,
  and dream+undo_all restores the rows (index and settled flags aside).
* Determinism: every seed runs twice and the transcripts (with a state digest per step) must be identical. `fuzz_time_translation`
  shifts the whole timeline by 123456789 s and demands the same transcript.
* Failures are shrunk by delta debugging to `(seed, kept op indices)` and printed. Known kinds (`KNOWN`) have their own named test and are
  recorded as "soft" in the sweeps, everything else fails the sweep.
* Teeth (mutation check in a scratch copy of the crate, not in the repo): eviction short by one, cite lowering S, recall using a full
  instead of half update, gist stability inflated, undo of keep not re-labilising, reversed eviction order, rate limit off by one,
  forget removing a neighbour, exposure window never cleared, candidate search without `stated_by`, dream cursor never advanced: all caught
  (by `I3-count-after-insert`, `I-cite-lowered`, `I-recall-stability`, `I7-gist-stability-formula`, `I7-undo-not-exact`,
  `I4-eviction-not-weakest`, `I-ratelimit-ignored`, `I-forget-collateral`, `I-cite-result`, `I7-candidate-newer`, and the cursor test).

Coverage per 20 seeds (group 1): 2.4k remembers (~1.2k with eviction), 3.4k recalls/peeks, 1.2k dreams, 233 replace, 91 keep, 415 error
verdicts, 270 dream+undo, 900 failed remembers, ~100 stale undos. About 2400 seeds x ~550 steps (1.3M steps) swept in the optimised profile
with no violation besides the four defects below.

## Confirmed failing tests

| test | one-line root-cause hypothesis |
|---|---|
| `repro::r2d_remember_reinforce_returns_err_after_commit` (+ `r2d_replay_seed43_remember_err_but_applied`) | `remember` of an existing text commits the rehearsal, then calls `self.memory(&t.id)?` to build the return value; a failing read returns `Err` after the write landed (a retry rehearses twice). |
| `repro::r2d_failed_writes_burn_the_daily_quota` | `write_times.push(now)` happens before the embedding, near search, eviction and commit; every failed `remember` (store outage) consumes a slot of `writes_per_day` and the engine is rate limited for a day after recovery with nothing stored. |
| `repro::r2d_backward_clock_hides_older_candidates_from_the_dream` (+ `r2d_replay_seed52_...`) | the dream reads the seed through `checked`, which clamps `created_at` to now; that clamped value is the `stated_by` bound of `Storage::near`, which compares it with the candidates' RAW `created_at`; after a clock step back, candidates stamped in the "future" window are filtered out, the seed "reactivates nothing" and is settled without the LLM. |
| `repro2::r2d_replace_gist_keeps_the_seeds_stated_time_after_backward_clock` (+ `r2d_replay_seed2_...`) | same clamp: `gists()` inherits `created_at` from the clamped seed, so a gist is stamped with the earlier clock reading instead of when the fact was stated (SPEC 5 says the gist keeps the statement time). |

**Shared structural area.** All four are the same pattern: *state that is only valid inside one operation leaks across the
operation boundary*. (a) D1/D3: engine bookkeeping (`write_times`) and the return value are produced on the wrong side of the
commit, so the operation is not transactional with respect to its own side effects. (b) D2/D4: the read-time normalisation
`checked()` (clamp to now) is applied to rows that are then used as a *comparison bound for the store* and as *source of a persisted
field*, so a clock step is written into the memory or silently changes which rows are compared. The write-backs of the clamped
`last_recall` (cite, keep, settle lower a stored future `last_recall` to now) are the same leak and are accepted
(`audit_lane2.rs` ignore note), as is the immortality-after-glitch trade-off.

Severity: D1 low (needs a read failing exactly after a commit), D3 medium (data loss window during any store outage with retrying hosts),
D2 medium (silent skip of an update's adjudication after an NTP step; the seed is *consolidated*, i.e. the update never merges), D4 low.

## Notes (pass; they document behaviour)

* `note_stale_undo_resurrects_forgotten_trace`: `undo` is unversioned; undoing the same reports twice after the host forgot a restored
  trace brings it back. The host owns report lifetimes (SPEC 5). Not provable no-resurrection when undo is in the system.
* `note_idgen_may_reuse_a_removed_id`: the engine refuses an id only while it is live (`new_id` checks `find(Id)`), so an `IdGen` that
  repeats ids gets removed ids back; "no resurrection" assumes the IdGen contract (fresh ids).
* `dream_cursor_cycles_through_undecidable_seeds` passes (and fails under the cursor mutation).
* After the budget is spent the dream stops at the next seed, so seeds with no candidates behind the last judged seed are settled one dream
  later (not a violation; the harness oracles only demand settling while budget is left).
* Batch-failure behaviour of the embedder inside `reindex` is "stop at the first bad batch": rows behind a bad vector stay stale until a
  later call (`I-reindex-order` is therefore only enforced with a faultless embedder).

