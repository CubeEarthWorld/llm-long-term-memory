# Round 3, lane C: hostile environments and panic-freedom

Files: `tests/audit3_c.rs` (48 tests).
Run: `CARGO_TARGET_DIR=C:/t/r3c cargo test --test audit3_c -j 3` (about 9 min in debug: the `r*` tests run 20 000 operations each;
every test takes a global lock so the allocator-based byte counts are not disturbed; `--skip r` gives the 45 s version).
Result on the snapshot below: 48 tests, 32 pass, 16 fail. `src/` and existing tests were not modified.

Snapshot note: `src/engine.rs`, `store.rs`, `storage.rs`, `text.rs` were edited by someone else while this lane ran (07:04-07:35).
Tests that failed earlier and pass now: `e3` (empty `model_id` never re-indexed, fixed by the new scrub in `reindex`), `d2` (a damaged list entry
made the row unforgettable), `s7c` (the usable-rows widening in `Engram::find` is now capped by `JUNK_READ`). The 16 below fail on the
07:35 `engine.rs` / 07:04 `store.rs`. Tests named `g_*` are regression guards that pass.

## Real defects (tests fail)

| id | sev | tests | root cause (one line) | area |
|---|---|---|---|---|
| H1 | med (hostile store) | `s1`, `s2`, `s3` | `victims`: `count(All) + 1`, `count(Young) + usize::from(..)` and `excess + BAND_DEPTH` are plain `usize` adds; a store that reports a count within 9 of `usize::MAX` panics every `remember` in a debug build (a release build wraps: `count+1` becomes 0, eviction silently stops). Fix: `saturating_add` in the three places. | Storage trust |
| H2 | med (hostile store) | `s5a`, `s5b`, `s5c` | Row identity is never checked. `memory(id)` and `forget(id)` take the LAST row of `find(View::Id(id), 1)` and use `raw.id`; `remember` rehearses the first row of `find(View::Text(text))` without comparing texts. A store that ignores the view or the limit makes `forget(a)` delete another trace and `remember("beta")` rehearse "alpha". Fix: filter `t.id == id` / `t.text == text` (and `take(limit)`). | Storage trust |
| H3 | low (hostile store) | `s6` | `candidates` does not de-duplicate `near()` hits: a store that repeats a hit puts the same trace in a cluster several times; the LLM sees it repeated and `gists` sums its strength once per copy (`absorbed` also repeats it). Fix: dedup by id in `Engram::near` (the pool of `select` already does through its `BTreeMap`). | Storage trust |
| H4 | low (hostile store) | `s7`, `s7b` | The widening loops are bounded only by the store running out: `Engram::near` doubles `ask` while every hit is unusable, `select` quadruples `k` while the k-th cosine stays above the floor. A store that never runs out (or lies about it) is asked for 2^16, 2^17 ... `usize::MAX` rows (max `k` seen: 131 072 with 3 rows in the store) / `iter_quad_saturates`: saturation needs 64 / 32 rounds, but each round allocates `ask` hits. Fix: clamp `ask` and `k` to `count(View::All)` (plus a constant). `Engram::find` got that bound (`JUNK_READ`, test `s7c` passes now), these two did not. | Storage trust |
| H5 | low (hostile store) | `s9b` | A hit whose vector is unit keeps the store's cosine unchecked: a store that reports `-0.9` for a hit the engine asked to be at least `theta_related` still counts as the reactivated trace (`Inserted.cosine = -0.9`, the write is born labile). `stated_by` and `k` are not re-checked either. Fix: drop hits with `cos < min_cos` (and `created_at > stated_by`) in `Engram::near`; the vector is there to recompute. | Storage trust |
| E1 | med (hostile embedder) | `e1` | `Engram::vector` reads `embedder.dimension()` once per vector; an embedder whose answer changes between calls yields query vectors of different lengths in one recall, and `select` calls `dot(&t.vector, q)` for every `q` of the pool, so `vector.rs:30` (`assert_eq!`) panics. Backtrace: `Engram::select -> dot`. Fix: read the dimension once per operation, or require `qs.iter().all(len == qs[0].len())` (and `unwrap` it into `None`). | Embedder snapshot |
| I3 | med | `i3` | Ids have no length bound (`valid_id` only checks characters). A custom `IdGen` (or a store) with 1 MB ids stores them in every index (about 8 copies per trace) and one pack line is 1 048 642 characters for `budget_chars = 1024` (the "first always fits" rule lets it through). The doc of `Recall::pack_text` says "at most `budget_chars`". Fix: refuse ids longer than a constant (say 256 bytes) in `valid_id`; the same for `model_id`, which is stored in every row and key (no test: a 100 kB model id works, `g_e4`). | Identifier bounds |
| D1 | low (damaged file) | `d1a`, `d1b` | `store.rs` does unchecked u64 arithmetic on sizes read from the file: `lists[at].size += 1` (line 421) and `read += l.size` in `near` (line 632) panic in a debug build for a centroid size of `u64::MAX` (a damaged centroid). The module promises "a damaged file is an error, never a panic". Also `lists[i].size -= moved` in `split` underflows when a stored size is smaller than the member count. Fix: `saturating_*`, or validate the size against the member count at load. | Damaged file |
| D3 | low (design) | `d3` | A centroid entry too short to decode makes `lists_of` fail for good: every `near`, and every write that indexes a vector, returns `Err("damaged centroid entry")`; `remember` and `recall` are dead and the only way out is `reset` (all data). The index is derived data. Fix: skip the undecodable list in `near` (the rows stay findable) and let `reindex` rebuild it, or drop the entry at open. | Damaged file |
| D4 | med (dependency) | `d4` | `Store::open` on a truncated database file: redb 2.6.4 panics (`page_manager.rs:243: assertion failed: storage.raw_file_len()? >= header.layout().len()`) for every cut between 512 bytes and one byte short, and loops forever on files of about 64-100 bytes (`Database::create` never returns; reproduced with plain `redb::Database::create`, no cache setting). A partial copy or a crash of a file system without journaling reaches it. Fix: wrap `Database::create` in `catch_unwind` and read the first 4 kB header to refuse a file shorter than its layout before handing it to redb (the hang needs the pre-check), or move to a redb release that validates. | Dependency |

Structural areas: **Storage answers are trusted** (H1-H5: identity, count, limit, duplicates, cosine, exhaustion; one `fn guarded_find` / `fn guarded_near` wrapper
in the engine would close all five); **per-operation snapshot of the embedder's answers** (E1); **unbounded identifiers** (I3); **the bundled store's
file-level robustness** (D1, D3, D4).

## Panic audit (non-test code of `src/`)

| site | verdict |
|---|---|
| `dream.rs` `expect("JSON values serialise")` | `serde_json::to_string_pretty` of a `Value` made by `json!`: keys are strings; a non-finite `retrievability` becomes `null` in `json!`. Unreachable. |
| `engine.rs` `gists`: two `expect("seed")` on `reduce` | `absorbed` always contains index 0 of a non-empty cluster. Unreachable (also with duplicated near hits: H3 repeats, never empties). |
| `engine.rs` `usable[k.max(1) - 1]` | guarded by `usable.len() >= k`; every caller passes `k >= 1` (`inject_n * 4`, 1, `dream_max_members >= 2`, all validated). |
| `engine.rs` `qs[0]`, `cluster[0]`, `cluster[..]`, `pool[j]`, `alive[i]`, `max_sim[i]` | `qs` has `parts.len() >= 1` entries; `pool` indexes come from `mmr` over the same slice. Unreachable. |
| `vector.rs` `assert_eq!(a.len(), b.len())` in `dot` | the one reachable panic: `select` with query vectors of different lengths (E1). Every other call site guards the length (`near`, `gists`, `Store::list`, `Store::near`, `split`, `mmr` over a pool filtered to `qs[0].len()`). |
| `text.rs` `parts[i * n / max_cues..(i + 1) * n / max_cues]` | `max_cues >= 1`; `i * n <= 2^20 * n` fits `usize` on 64-bit (on a 32-bit target `n > 4 000` could overflow: note, no test). |
| `text.rs` `cut[..max_chars]`, `chars[..max_chars]` | taken only after `chars.len() > max_chars`. |
| `time.rs` `&t[1..]`, `rest.split_at(len - 2)` | after an ASCII sign / an `is_ascii` check, length 3..=4. `civil_from_days`: proved no i64 overflow. `fmt_offset` `abs()` of at most 86 399. |
| `store.rs` `expect("just inserted")`, `expect("members")`, `expect("listed")` | each follows the insert / the `first()` guard / `save_list` of the same list in the same transaction; a damaged file cannot remove the entry between them. Unreachable. |
| `store.rs` `trace()` `v[0]`...`v[7]` on `serde_json::Value` | `Index` on a non-array returns `Null`, never panics; a short row is `None` and read as damaged. |
| `store.rs` `size += 1`, `read += l.size`, `size -= moved` | reachable from a damaged file (D1). |
| `store.rs` redb `?` paths | all return `Err`; redb itself panics on a truncated file (D4). |
| `Mutex` poisoning | The engine no longer holds a `Mutex` (the old `damaged: Mutex<Vec<_>>` became the `scrub: String` cursor). The only lock is `Store.lists: RwLock`, always taken with `unwrap_or_else(into_inner)` and written only after the redb commit succeeded, so a panic inside `write` leaves the cache as the committed state. A panic in a user callback (embedder, adjudicator, id generator, storage) cannot poison anything; `g_e9`, `g_i2`, `g_s12`, `g_a3` check the engine is usable and no rate-limit slot was spent afterwards. |
| casts | `now as i64`, `(now * 1000.0) as i64`, `(stability * elapsed).round() as i64`: Rust float to int casts saturate and NaN is 0 (the clock is clamped / NaN-checked in `now()`); `b as u8` runs 0..=255; `band()` clamps before `as u8`; `created_at as i64` of 8 stored bytes wraps harmlessly. |
| i64 arithmetic with the clock | `gists`: `now as i64 - round(S * elapsed)` can only overflow when members and clock both sit within `64 * max_stability` of `i64::MIN`; `c2` tries clocks of -9.223e18 to -1e19 and does not panic (the encoded `last_recall` stays near the members' own). `c1` runs 16 hostile readings (NaN, +-inf, +-MAX, +-1e18, +-1e30, 9.3e18, i64::MIN as f64) and back-and-forth jumps through every operation: no panic, `R` always in [0,1], `records <= capacity + 1`. |
| division | `1 - cosine_floor > 0` (validated `< 1`), stability `>= 1` after `clamp_stability`, `max_cues >= 1`, constants. |
| `with_capacity` | `texts.len()`, `ids.len()` (limited by the rows a view returned), `k.min(best.len())`. No external size reaches it unclamped. |
| `unsafe` | none in `src/` (grep). `bench/src/bench/perf.rs` has two `unsafe` (a Windows `GetProcessMemoryInfo` FFI, not part of the published crate: `exclude = ["bench/"]`). The test file has one `unsafe impl GlobalAlloc` (counting allocator). |

## Hostile environment matrix (all pass unless listed above)

* **Storage** (`Hostile` wrapper over the real `Store`): count `0`, `1`, `usize::MAX / 2`, `usize::MAX` for stats/dream/recall (`g_s13`: only `victims` is arithmetic, H1); `near` returning more than `k`, in reverse order, with cosines NaN / +-inf / +-5 / +-1e300 (`g_s8_s9`: no panic, at most `inject_n` injected, scores in [0,1], clusters at most `dream_max_members`); writes silently dropped for 30 rounds of remember/recall/dream/reindex/cite (`g_s10`: terminates in well under a second); `clear()` failing (`g_s11`: engine usable, a second `reset` works); a store that panics in `write` or `find` (`g_s12`: engine usable, the panicked write did not consume a rate-limit slot, no half state); store that ignores `limit` or the view (H2); bottomless `near` (H4).
* **Embedder**: `dimension()` changing between calls (E1); `model_id` changing at every call (`g_e2`); `model_id` empty (now fine), containing NUL, `》`, a newline, `;`, a bidi override, Japanese, 100 kB (`g_e4`: recall, reindex, dream, forget all work); dimension 1 000 000 with four traces (`g_e5`: works, about 4 MB per vector, no ceiling on `dimension()`: the embedder owns it); NaN / +-inf / zero / `f32::MAX` / subnormal vectors (`g_e6`: treated as offline); the same vector for every text with 1 300 inserts (`g_e8`: IVF refuses to split, recall and dream stay bounded); a panic inside `embed_documents` (`g_e9`: nothing stored, slot not spent). Not tested because it is a contract, not a defect: `remember` takes the first of `embed_documents(&[text])` without checking the count is 1 (the other paths check).
* **Adjudicator**: 5 000 memories + 5 000 ids (verdict is keep, 10 s cap), a 1 MB memory (cut to `text_max`), ids of other clusters / nonexistent / repeated (only the cluster is absorbed, once), control, bidi, `《》` characters in memories (stripped), a panicking adjudicator (engine usable, the seed is met again) (`g_a1`-`g_a3`). **No timeout API**: a slow adjudicator, embedder or store blocks `dream` / `remember` for as long as it takes (the engine is synchronous behind `&mut self`); that is a documentation point, not a defect. A panicking adjudicator also drops the `reports` vector of the dream in progress, so verdicts already applied in that dream have no undo record (the engine cannot catch it without `catch_unwind`).
* **Clock**: `c1`, `c2` above; non-monotonic readings only delay the rate limiter (writes stamped more than a day ahead are discarded).
* **IdGen**: invalid, empty, newline, tab, `《》`, constant ids end in `Err` after 64 tries and store nothing (`g_i1`); a panicking generator spends no slot (`g_i2`); 1 MB ids (I3).
* **Config inside `validate()`** (`g_f1`, 13 corner configurations: capacity 1 and `usize::MAX`, `grace_period = f64::MAX`, `text_max = 1` and `usize::MAX`, `budget_chars`/`writes_per_day = usize::MAX`, `initial = max = spacing_gain = f64::MAX`, `max_stability = 1`, `theta_related = 1e-300`, `theta = cosine_floor = 1 - eps`, the four counts at 2^20, `writes_per_day = 0`, `min_score = relative_score = 1`): a full life (remember x6, recall, cite, dream, undo, reindex, clusters, forget) without a panic. Timezone names `""`, NUL, `../../../etc/passwd`, `UTC;+99:99`, `;;;;`, bidi, 1 MB (`g_f2`): no panic, stored field at most 64 characters.

## Resource bounds (`r0`-`r7`)

A counting `#[global_allocator]` measures live heap bytes (`g_r0` proves it sees a leak). Capacity 8, 20 000 operations of each kind, heap sampled at
operation 5 000 and at the end (tolerance 3 MiB; measured in parentheses):

| test | operations | growth |
|---|---|---|
| `r1` remember with eviction on every write (`write_times`) | 20 000 | +0 bytes |
| `r2` recall + cite + peek (`exposures`) | 20 000 each | +0 bytes |
| `r3` remember + forget + forget of an unknown id | 20 000 each | +0 bytes |
| `r4` dream (merge verdicts) + undo_all every other round + clusters (`cursor`) | 20 000 each | +1 696 bytes |
| `r5` reindex with an embedder that flaps offline/online (`scrub`) | 20 000 | +0 bytes |
| `r6` clock oscillating +-1e6 s per operation (rate limiter) | 20 000 | +0 bytes |
| `r7` every row's vector corrupted by the store: one `reindex` repairs at most 64 + 1 rows (the old `damaged` cap, the new scrub page) | 300 rows | bounded |

The live heap at the end is about 2.9 MB in every test: it is the 3.6 MB in-memory redb file, which plateaus. `write_times` is bounded by
`writes_per_day` (the limiter refuses at that length) and by one day of writes; `exposures` is replaced by every recall (at most `inject_n`);
`cursor` and `scrub` are single ids (so as long as an id: I3). Per-operation CPU is the cost to watch: `remember` with eviction is about 5 ms in
debug because `victims` opens 256 band windows (`r1` takes 100 s for 20 000).

## Dependencies

`Cargo.toml`: `jiff 0.2.38` (default features off; `std`, `tz-system`, `tzdb-bundle-platform`, `tzdb-zoneinfo`), `redb 2.6.4` (optional, feature `store`, default),
`serde_json 1.0.151` (`preserve_order` pulls `indexmap 2.14.2` + `hashbrown 0.17.1`; `float_roundtrip`), `ulid 3.0.0` (pulls `rand 0.10.3`, `chacha20`, `getrandom 0.4.3`
for one id generator: the heaviest dependency per line of use).
`cargo tree -p llm-long-term-memory -d --target all`: only `jiff-core 0.1.1` (listed twice: host proc-macro vs target, same version) and `syn 2.0.119` / `syn 3.0.6`
(build time only, `jiff-static` vs `serde_derive`). No duplicate at run time. The workspace (bench) has more duplicates (`bitflags 1/2`, `shlex 1/2`, `nom 7`, `itertools 0.13`,
llama.cpp bindings) that are not in the published crate. `Cargo.lock`: every crate is a recent release; nothing reads as yanked or old except
`redb 2.6.4` being a previous major and having the truncated-file panic of D4 (not checked against 3.x: no network). `cargo audit` is not installed; no advisory check was made.
Release profile: `lto = "thin"`, no `overflow-checks`, no `panic = "abort"`: debug panics of H1/D1 are silent wraps in release (H1 stops eviction, D1 corrupts a list size).

