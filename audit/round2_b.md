# Round 2, lane B: Storage contract and the bundled Store

Files: `tests/audit2_b.rs` (conformance suite + reproductions).
Run: `CARGO_TARGET_DIR=C:/t/r2b cargo test --test audit2_b -j 3` (about 30 s; `AUDIT2B_SEEDS=80` widens the random search, 80 extra seeds passed).
`src/` was not modified.

## Reusable suite
`conformance::run(storage, reopen, seed, batches, label)` drives random batches (Insert new / replace with another model, vector,
dimension, band, text, created_at, consolidated; State; Index incl. missing ids; Remove incl. missing ids; batches poisoned with a
NaN stability or non-finite vector, which must be rejected and change nothing; occasional `clear`; periodic reopen) and after every
few batches compares `count` and `find` (limits 0, 1, 3, MAX) of every view (All, Id, Text, Labile incl. a full paging walk,
Stale per-model order and prefix closure, Band over every band x many `created_by` incl. i64::MIN/MAX, Young over many cut-offs) and
`near` (two models, dimensions 4/5/6/7, duplicate / antipodal / near-zero / zero vectors, min_cos, k, stated_by) with an oracle
(`Model`). It runs against `RefStorage` (trivially correct, sanity check of the suite), against `Store` in memory and against
`Store` on a file with reopens. A deliberately broken storage (Band ignores created_by) is shown to be caught.
`near` is compared against brute force with a 2e-5 cosine tolerance: set, order (cos desc, tie id asc), completeness at the k cut-off.
Bulk IVF scenarios (3000 clustered, duplicates + near duplicates, antipodal + zero + 1e-20 norm, two models + mixed dimensions with
removals and replacements, and exactly 8192 vectors): `near` is exact in all of them.

## No defect found (checked)
* Band scan break condition: late rows (created_at > created_by) in the middle of a band are skipped, never terminate the scan,
  and the scan ends only at the next band; breaking on the first late row would be the
  tempting wrong version. Test `band_scan_skips_late_rows_in_the_middle`.
* Labile paging has no duplicates and no gaps; Stale: per-model newest-first, no row lost (model order is
  unspecified by SPEC; the Store walks models above then below the current one, each descending).
* Atomicity: a rejected Insert replacing an existing id, State, or Index leaves rows, indexes, lists and count untouched (redb rollback;
  the `lists` cache is only updated after commit).
* Re-Insert of an existing id with another model/vector/dimension/band/text leaves no residue in any index.
* near exactness up to 8192: the probe stops only once 8192 vectors have been read before starting a list, so with N <= 8192 the last list
  is always reached. Mixed dimensions are separated by centroid length (`dot` is never called across dimensions in near/list).
* count() META counter stays equal to the row count through replace, missing Remove, rejected batches and reopen.
* `Store` is `Send + Sync`; concurrent readers are fine (writes need `&mut`, so the reader/writer cache race cannot occur);
  lock poisoning is tolerated.

## Confirmed failing tests (defects)
1. `young_after_i64_max_is_empty` -- `View::Young{created_after: i64::MAX}` returns rows with created_at == i64::MAX.
   Cause: `created_after.saturating_add(1)` saturates to the bound itself, turning "> MAX" into ">= MAX".
   Low severity (engine passes now - grace).
2. `near_accepts_k_of_usize_max` -- `near(k = usize::MAX)` panics "capacity overflow". Cause: `Vec::with_capacity(k)` in `Store::near`
   (`find` accepts usize::MAX for "all"; the same type in `near` should too). Fix: cap the capacity at `best.len()`.
3. `a_wrong_dimension_list_entry_does_not_panic_the_split` -- a list entry whose float count differs from the others (valid 8-byte
   header, 3 floats) makes the 513th insert panic in `split()` via `dot`'s `assert_eq!` (SPEC: damaged entries are errors, not panics).
   Cause: `split` trusts `split_entry`'s payload length; `near`/`rows` filter or validate, `split` does not. Fix: skip or Err on `vector.len()` mismatch.
4. `a_row_with_unreadable_json_can_be_removed`, `a_row_with_unreadable_json_can_be_overwritten`,
   `count_and_find_agree_when_a_row_is_damaged` -- a row whose JSON does not parse is (a) skipped silently by every `find`
   (`rows()` does `continue`), (b) still counted by every `count`, (c) impossible to Remove or Insert over: `Tx::get` returns
   `Err("unreadable row")`, so the whole batch fails. Consequence in the engine: `forget(id)` says "no such memory"
   (`find(Id)` is empty), the row permanently occupies a capacity slot (`count(All)` includes it, `victims` can never select it), and
   any batch containing a Remove of it is rejected. Cause: reads treat an undecodable row as absent, writes treat it as fatal, counts treat it
   as present. Fix: decode failures in `rows()` should be Err (consistent with damaged list entries), and `Tx::get` should let Remove/Insert
   proceed (best-effort index cleanup, or keep band/labile keys derivable without the row).

## Structural area shared
Items 3 and 4 (and the pre-existing lane-2 float round-trip finding) are the same weakness: the Store validates decoded data
inconsistently across its three read paths (`rows()` skips, `scan`/`count` trust the index, `Tx::get`/`split` fail or panic).
A single `decode` boundary (row -> Result, list entry -> Result with length check) used by find, count, near, get and split removes all
four. Items 1 and 2 are independent edge arithmetic (saturating_add, with_capacity).

## Observations, not failing tests
* Split-attempt churn: a list of identical vectors retries the 2-means split each time its size passes 513, 1025, 2049, 4097, 8193;
  an insert+remove loop oscillating at the boundary pays O(list) per write (release build: 0.28 ms per insert+remove at 512 vs 0.15 ms at 700,
  0.9 ms at 2048, 1.3 ms at 4096). Amortisation by doubling holds only for monotone growth. Performance only.
* `State`/`Index` of a missing id with a non-finite value is an accepted no-op (the SPEC rejects such rows; nothing is changed either way).
* A trace stored with a non-empty `model_id` but an empty vector is in Labile/Stale (model index) but not searchable; consistent with the
  SPEC text but worth knowing for third-party Storage implementers.
* The model cache `lists` gains an empty entry for every model string ever searched (unbounded only under hostile model ids).
