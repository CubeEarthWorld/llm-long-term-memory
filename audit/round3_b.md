# Round 3, lane B: SPEC-vs-code conformance, public docs, bench, tests, dead code

Scope: SPEC.md §2-§8 against `src/`, rustdoc / README examples, the `ltm-bench` workspace member, `tests/support` and the
existing tests, clippy, dead code. No `src/`, existing test or doc was modified. New files: `tests/audit3_b.rs`,
this report.

Note: `src/engine.rs`, `src/store.rs` and `src/storage.rs` changed under me during the audit (another lane): `View::After`, `Engram::scrub`,
`Engram::sound()` + `VECTOR_DRIFT = 0.05` replaced the 1e-3 renormalise-and-`damaged`-list repair. Line numbers below are from the snapshot read
at the start (engine.rs 1022 lines; it is now ~1025, so later functions have moved a few lines; function names are given). Where the two
snapshots differ it is stated. My tests were re-run on the new code, same results.

## 1. Result summary

Failing tests (`CARGO_TARGET_DIR=C:/t/r3b cargo test --test audit3_b -j 3`: 5 pass, 2 fail):

| test | cause |
|---|---|
| `band_window_total_reads_match_spec_section_3` | SPEC §3 says one band's reads total "about 1.4x the 128 cap". Window growth is 31 -> 124 -> 128: **283 rows = 2.21x** (excess = 23). The final regrow step adds 4 rows but re-reads 128. Worst case over all excess is exactly 283. **SPEC is wrong** (the code is fine and bounded; optionally clamp growth so the last step is skipped when `4*limit` is within a few rows of the cap). |
| `readmes_name_only_methods_that_exist` | README.md and README.ja.md say Config has "JSON round trip with `to_json` / `merge_json`"; no such method exists anywhere (`grep -rn to_json src bench tests`: none). README.zh.md is clean. **Doc wrong.** |

Passing probes that document behaviour (so the matrix claims below are machine-backed): `dead_band_is_not_widened_and_first_window_is_excess_plus_8`
(SPEC §3 first window `excess+8`, DEAD stops growth), `window_larger_than_128_is_read_once`, `readme_quickstart_compiles_and_runs` (the README
snippet, verbatim, compiles against the public API and runs), `ids_do_not_roll_back_past_the_48_bit_clamp_in_one_process` (SPEC §8 "ids roll
back" is false in-process), `clusters_preview_lists_more_clusters_than_the_dream_would_judge` (rustdoc of `Engram::clusters` is loose).

No code defect (engine behaviour contradicting SPEC) was found in this lane beyond those listed; every other mismatch is SPEC or doc text.

## 2. SPEC statements that are false or stale (proposed exact edits)

1. **§3, window reads.** "読み出しの総量は上限の約 1.4 倍まで" -> "読み出しの総量は、最悪で上限の約 2.2 倍（excess = 23 のとき 31 + 124 + 128 = 283 件）。excess = 1 では 9 + 36 + 128 = 173 件（1.35 倍）、excess ≥ 120 では最初の 1 回だけ".
2. **§2, last paragraph of the Store text.** "壊れた行やエントリはエラーになり、パニックしない" -> a row whose JSON cannot be decoded is **not** an error: `Store::rows` returns it as an unusable row (empty text, `store.rs` `unreadable()`), which the engine counts, evicts first and can `forget`; only damaged similarity-list/centroid entries return `Err("damaged ...")`. Edit: "復号できない行は使えない行（本文が空）として返り、壊れた索引エントリはエラーになる。どちらもパニックしない".
3. **§2, engine state sentence and §8 first sentence** ("設定・直近 1 日の書込み時刻・直前の recall の露出・夢の巡回カーソル") omit: the repair state (snapshot A: `damaged`, <= 64 ids; current: `scrub`, 1 id), the clock, the id generator and the timezone. Add "（と reindex の巡回位置 1 個）".
4. **§2, views table.** New `View::After(id)` (every trace after an id, id order; used by `reindex`'s scrub) is not listed. Add it ("`After(id)`: そのid より後の全痕跡、id 順").
5. **§2, vector repair sentence** ("|‖v‖²−1| > 1e-3 ... 再正規化して使い ... コサインも再計算する ... 直せないものは未索引扱い"). Snapshot A matched it (`unit()` 1e-3, `renormalized`, cosine recomputed). The code now in the tree (`sound()`, `VECTOR_DRIFT = 0.05`, "the engine neither rescales nor recomputes", scrub embeds unsound rows again) contradicts it. Whichever lands, SPEC §2 and §6/§7 ("全ベクトルは単位ノルム") must be made to match; the new text should say 0.05 and that the store's cosine is used.
6. **§4 recall step 3, and README x3.** Pack line is `[unix tz] text　《id:…》` (U+3000 before the opener, `id:` prefix; `select()` format string), not "`[unix tz] text 《id》`". Edit SPEC §4 and the READMEs (`README.md` lines 24-25 table rows, `README.ja.md`, `README.zh.md`): write `《id:…》`.
7. **§5 first line.** "未索引の痕跡を新しい順に 64 × budget 件まで再埋め込み" is 64 x max(1, budget) in code (`dream`: `REINDEX_BATCH * left.max(1)`), so budget 0 still re-embeds 64. Edit: "64 × max(1, budget)".
8. **§6 last paragraph.** "テストのトークン重なり埋め込みは thetaRelated = 0.75 を使う": only `tests/longevity.rs` does; every other test uses the default 0.55 with `cosine_floor = 0`. Edit: "長期シミュレーション（`tests/longevity.rs`）は ... 0.75".
9. **§6 has no ranges.** `Config::validate` enforces (all reals finite): max_stability >= 1; spacing_gain, grace_period, mmr_lambda >= 0; min_score in [0,1]; gist_min_cosine in [-1,1]; max_cues >= 1; capacity >= 1; 0 < initial_stability <= max_stability; cosine_floor in [0,1); alpha, relative_score in [0,1]; theta_related in (0,1); dream_max_members >= 2; inject_n, budget_chars, text_max >= 1; inject_n, max_cues, dream_max_members, dream_budget <= 2^20 (`COUNT_MAX`, `config.rs`). Add a "範囲" column or sentence.
10. **§8 "全件の走査をする操作はない（唯一の例外は ... 移行）".** Second exception: `Tx::purge` (`store.rs`) scans every index to remove an undecodable row ("taken only to heal damage"). Also `count(view)` for any view but `All` is O(matches) in `Store` (`count()` scans): `victims` calls `count(View::Young)` on every at-capacity insert (O(#traces younger than the grace period), up to ~capacity right after a bulk import) and `stats().labile` is O(#labile). "各操作は O(log N)" holds only for `count(All)`. Either add the caveat or make `victims` ask `find(Young, capacity/10 + 1)` (reads rows, so a `Storage::count_up_to` would be the right seam).
11. **§8 query table, `remember` row** is incomplete: it also does `find(Text)` (rehearsal probe, up to 8 rows), `find(Stale, 1)` (whenever it would be born consolidated), `find(Id)` per candidate id (`new_id`). `recall` also does `count(All)`. Minor.
12. **§8 "id は 48bit ... それ以後は id が巻き戻る".** In one process ids never roll back: the timestamp is clamped to 2^48-1 (`id.rs` `ULID_MAX_MILLIS`) and `ulid::Generator` increments the previous ULID whenever the time is not greater (test `ids_do_not_roll_back_...`). Rollback only exists across a restart (fresh generator, random bits). Edit: "それ以後は時刻部が頭打ちになり、id は乱数部の増分で整列を保つ（再起動をまたぐと保証されない）".
13. **§8 "3000 年シミュレーション ... 確認している".** True only when run: `tests/longevity.rs::three_thousand_years` is `#[ignore]` (`--release --ignored`). Scale is right (365k remembers, ~36k dreams).
14. **Header line 5** "全件の順位付けなしに v2.1 と同じ痕跡を選ぶ" is conditional on the per-band cap (§3 itself says capped bands are approximate). Add "（帯の上限に達しない限り）".
15. **§2 id sentence vs §7**: §2 lists only 《》/control/bidi, §7 also forbids whitespace other than U+0020 (code: `valid_id`). Consistent, but `IdGen` rustdoc (below) lists only 《》.

## 3. Conformance matrix

Status: **OK** implemented as written; **DIFF** implemented differently (verdict: S = SPEC wrong, C = code wrong, D = doc wrong); **NOT** not implemented; **UNT** untestable (design/prose/LLM). Citations are function names (+ line in the snapshot read at the start).

### §2 Data model and store

| id | sentence (short) | status | src | note |
|---|---|---|---|---|
| S2.1 | default id = monotonic ULID, sorts in creation order even within a ms | OK | `id.rs:16-23`, unit test | |
| S2.2 | `IdGen` ids: unique, ordered, non-empty, no 《》/control/bidi; bad ids re-drawn; stored bad id = unusable row | OK | `new_id` 367-375, `valid_id` text.rs:36, `checked` | valid_id also rejects non-space whitespace (item 15) |
| S2.3 | id order = insertion order; ties and FIFO seeds follow it | OK | all views ordered by id | |
| S2.4 | text <= 170, no 《》/control/bidi; "self-contained, no pronouns" | OK / UNT | `clean_text` text.rs:53 | pronoun rule is the LLM prompt's |
| S2.5 | created_at, last_recall 64-bit seconds; tz `'IANA;+HH:MM'`; stability seconds; cue <= 170; model_id; unit vector | OK | `trace.rs`, `remember` 449-460 | |
| S2.6 | engine state = config, last-day write times, last exposure, cursor; bounded | DIFF (S) | `Engram` fields 136-153 | also clock, ids, timezone, `damaged`/`scrub` (item 3) |
| S2.7 | open reads nothing, O(1) | OK | `with_clock` | only `validate()` |
| S2.8 | store contract = 5 operations; view list | DIFF (S) | `storage.rs` | `View::After` added (item 4) |
| S2.9 | views `All/Id/Text/Labile/Stale/Band/Young` semantics and order | OK | `store.rs` `scan` 230-296; rustdoc of each variant checked true | Stale = other models, newest id first, model-major |
| S2.10 | `near`: cos >= min_cos, top k, cos desc then id; `stated_by` applied inside | OK | `store.rs` near 610-658 (order 643, filter 631) | |
| S2.11 | engine doubles k until k usable rows or store exhausted | OK | `Engram::near` 290-316 | |
| S2.12 | `write` = one atomic commit; Insert/State/Index/Remove; non-finite stability/vector rejected, nothing changed | OK | `Store::write` 662, `put_state` 496 | only the bundled Store polices it |
| S2.13 | `clear()` | OK | store.rs 690 | |
| S2.14 | each view answered by one index range ("one SQL statement") | DIFF (S, minor) | `scan` Band 277-285, Stale 266-276, `count` 596-606 | Band post-filters `created_by` while scanning; Stale = two ranges; non-All `count` is O(matches) (item 10) |
| S2.15 | Store = one redb file, B-tree per view | OK | tables ROWS..BAND..CREATED | |
| S2.16 | IVF: nearest lists first until 8192 vectors | OK | `PROBE_VECTORS` 26, loop 619-623 | list crossing 8192 is read whole |
| S2.17 | list > 512 splits by 2-means; indivisible list retried only at each doubling; split cost = list size | OK | `list` 393-422 (`over % LIST_MAX`, `is_power_of_two`), `split` 442 | splits at sizes 513, 1025, 2049 ... |
| S2.18 | `State` write does not touch vectors | OK | `apply` 555-559 | |
| S2.19 | RAM = centroids + capped page cache | OK | `CACHE_BYTES` 1 GiB, `lists` cache | |
| S2.20 | exact up to 8192 vectors/model, approximate beyond | OK | near | |
| S2.21 | broken rows / entries are errors, never panics | DIFF (S) | `rows` 218, `unreadable` 101 | row -> unusable row, entry -> Err (item 2) |
| S2.22 | other DBs implement `Storage` | OK | trait | bench `perf.rs` and tests do (`SharedStore`) |
| S2.23 | rows validated on read: id, text/cue/tz cleaned (tz keeps offset), stability in [1, S_max] | OK | `checked` 246-268, `clean_tz` | |
| S2.24 | last_recall/created_at are never clamped; future values give dt = 0; future-stamped row counts as young; dream bound keeps the raw value | OK | `checked` (`..t`), `retrievability` 210, `victims` cutoff, `candidates` | |
| S2.25 | non-finite / non-unit (1e-3) vector: renormalise + recompute cosine, else unindexed and re-embedded by `reindex` | DIFF (in flux) | snapshot A: `unit`, `renormalized`, `note_damaged`; now `sound`, `VECTOR_DRIFT 0.05`, scrub | SPEC must follow whichever lands (item 5) |
| S2.26 | cosine never > 1; non-finite cosine is no match | OK | `Engram::near` filter + `.min(1.0)` | |
| S2.27 | unusable rows are counted, evicted first | OK | `victims` `ranked` -> NEG_INFINITY; `count(All)` | |
| S2.28 | foreign-model rows are unindexed; re-embedded newest first at dream start and by `reindex(limit)` | OK (DIFF at budget 0) | `reindex` 386-407, `dream` 782 | item 7 |
| S2.29 | embedding failure keeps text; searches stay single-model; open never needs the embedder | OK | `remember` 435, `select` 568 | |

### §3 Forgetting curve

| id | sentence | status | src | note |
|---|---|---|---|---|
| S3.1 | R = 2^(-max(0,dt)/S) | OK | `retrievability` 210 | |
| S3.2 | strength = S x R | OK | `strength` 215 | |
| S3.3 | band = floor(2 log2 S) | OK | `trace.rs:33-37` | extra clamp 0..255 and NaN -> 0, unreachable below S = 2^127 |
| S3.4 | a = max(0,(cos-floor)/(1-floor)) | OK | `activation` 225 | cos capped at 1 |
| S3.5 | recall update min(S(1+gain a (1-R)), S_max), last_recall = now | OK | `grown`/`retrieved` 231-238 | |
| S3.6 | new trace: stability = clamp(S0 x salience, 1, S_max), last_recall = created_at = now | OK | `remember` 448-456 | NaN salience -> 1 |
| S3.7 | defaults S0 1 d, gain 3, S_max 10 y, salience [0,10] (default 1), floor 0.4 | OK | `config.rs` Default | "salience default 1" is the caller's: `remember` has no default |
| S3.8-10 | rehearsal ~0 gain at R~1; x(1+gain) at R~0, a~1; weak cue gives 30% | OK | formula; tests `engine.rs` | |
| S3.11 | salience is the only exception mechanism | UNT | | |
| S3.12 | eviction rank = log2 S - dt/S, min first | OK | `victims` `rank` 485 | |
| S3.13 | band rationale: within a band the longest-unrecalled is nearly weakest, order time-invariant | OK / UNT | `trace.rs` doc, `tests/audit_lane1.rs l1_4` | |
| S3.14 | store needs only a `(band, last_recall)` index | OK | BAND table | |
| S3.15 | band head is only approximately weakest (S differs by sqrt 2) | OK | | |
| S3.16 | read each band from its head `excess + 8` rows, widen x4 until a lower bound proves no unread row ranks lower | OK | `victims` 496-526 | test `dead_band_..._excess_plus_8` |
| S3.17 | unread rows have S >= band floor and dt <= last row's, so rank >= log2 S_lo - dt/S_lo | OK | 515-516 (`(1-1e-9)` fudge, `min(S_max)`) | |
| S3.18 | stop when the excess-th rank <= -64 | OK | `DEAD`, 510 | |
| S3.19 | order among dead traces is immaterial | OK | | |
| S3.20 | one band window <= 128 rows (or `excess + 8` if larger) | OK | `BAND_CAP`, `cap` 496 | test `window_larger_than_128_is_read_once`; `reads.iter().all(<=128)` |
| S3.21 | total reads of a band "about 1.4x the cap" | **DIFF (S)** | 502-525 | **283 = 2.21x worst**; failing test (item 1) |
| S3.22 | a band that hits the cap is approximate | OK | | |
| S3.23 | within usable range result equals full comparison | OK (conditional on S3.22) | | |
| S3.24 | unusable rows hidden outside the window die when they appear; capacity protected by `count(All)` | OK | `ranked`, `total` 479 | |
| S3.25 | S <= S_max, so no memory is immortal | OK | `clamp_stability` | |

### §4 Wake phase

| id | sentence | status | src | note |
|---|---|---|---|---|
| S4.1 | clean (strip 《》, controls, bidi), collapse spaces, cut to 170 at a sentence boundary (cue too); empty -> reject | OK | `clean_text`, `shorten`; `remember` 416-421 | boundary preference only in the second half |
| S4.2 | rate limit `writesPerDay`; rehearsal not counted; a backward clock reopens the window | OK | `remember` 429-434 | `retain(t >= day_ago && t <= now)` |
| S4.3 | existing usable same-text trace (first of `find(Text)`) -> rehearsal with a = 1, no new row | OK | 422-428, `TEXT_PROBE` | |
| S4.4 | embed text as document; q = cue as query, else own vector | OK | `cue_vectors` 683-707 | |
| S4.5 | nearest 1 older trace via `near(q, theta, 1)`; insert; consolidated iff nothing reached and self + all traces indexed and cue embedded | OK | 437-447 | snapshot A's "all indexed" ignored vectors repaired-away (store `Stale` cannot see them); current scrub reduces that |
| S4.6 | write never overwrites/deletes/rewrites existing traces | OK | `remember` writes Insert + eviction Removes only | |
| S4.7 | over capacity evict min-strength trace past grace (3 d), candidates = heads of ALL bands, bad rows first | OK | `victims` 477-535, bands `0..=255` | |
| S4.8 | young traces protected (incl. future-stamped) unless > capacity/10 (flood) or too few older rows | OK | 483-529 | |
| S4.9 | recall: cues = lines/sentences, merged beyond `maxCues`, nothing dropped; cos = max over cues | OK | `cues` text.rs:76, `select` 567 | |
| S4.10 | score = a (alpha + (1-alpha) R) | OK | 592 | |
| S4.11 | keep score >= max(minScore, relative x best) | OK | 596-599 | |
| S4.12 | MMR (lambda 0.3) over id-ordered pool, injectN, ties by id | OK | `mmr` 621-638 | |
| S4.13 | pack `[unix tz] text 《id》`, <= budgetChars, first line always | DIFF (S, D) | 607-613 | `《id:…》` and U+3000 (item 6); also skips an over-budget line but keeps trying later ones |
| S4.14 | candidate collection exact: k = 4 injectN, widen x4 until `a(upper bound) < floor` | OK | 575-602 | |
| S4.15 | injection = exposure: update at a/2, State write only | OK | `recall` 544 | |
| S4.16 | cite: ids in reply vs last recall, raise stability to a = 1 (multiply the difference); default is the half update | OK | `cite` 643-665 | |
| S4.17 | forget by id only, also unusable rows | OK | `forget` 669-674 | |
| S4.18 | no maintenance call; `reindex` optional; capacity held at each insert; clock defence = max(0,dt) + read clamp | OK | | |

### §5 Sleep phase

| id | sentence | status | src | note |
|---|---|---|---|---|
| S5.1 | re-embed unindexed traces newest first, up to 64 x budget; failure leaves them unindexed | OK (budget 0: 64) | `dream` 782 | item 7 |
| S5.2 | seeds = labile + indexed, ids after the cursor FIFO, wrap, only usable rows, <= 8 x budget, paged | OK | `seeds` 724-756, `SEED_PAGE` | |
| S5.3 | cursor advances per examined seed; host keeps it via `dream_cursor`/`set_dream_cursor` | OK | `dream` 796, accessors 177-184 | `reset` clears it |
| S5.4 | undecided seeds return after one cycle; none starves the rest; unusable head rows do not hide seeds | OK | `seeds` | tests `audit_lane3`, `audit2_*` |
| S5.5 | all cues embedded in one call, empty cue skipped, duplicates once; call failure -> own vector; wrong dimension -> own vector for that cue | OK | `cue_vectors` | |
| S5.6 | skip a seed that is gone or already consolidated | OK | `dream` 797-807 | |
| S5.7 | candidates = `near(q, theta, dreamMaxMembers, stated_by = seed.created_at)` minus seed, <= members-1; same second counts as past | OK | `candidates` 712-715 | |
| S5.8 | clusters may overlap | OK | | |
| S5.9 | no candidates -> settled in one end-of-dream write, no LLM, no budget; not settled if the cue fell back to the own vector | OK | 810-813, 825-829 | |
| S5.10 | budget -= 1 per LLM call; <= 1 call per seed | OK | 815 | |
| S5.11 | keep only on explicit `"action":"keep"`; empty/unparsable/`{"memories":[]}` is a failure | OK | `DreamDecision::parse` dream.rs:121-146 | the engine reads `memories == []` as keep: hosts must map `parse() == None` to `Err` (bench `claude.rs`/`llm.rs` do) |
| S5.12 | absorbed = seed + named candidates; unnamed untouched | OK | `verdict` 919-921 | |
| S5.13 | texts cleaned/deduped; all empty after cleaning = failure | OK | 908-918 | |
| S5.14 | more texts than absorbed -> keep; any text with cos < gistMinCosine to every absorbed -> keep | OK | 922, `gists` 942-947 | |
| S5.15 | one atomic write: keep marks only the seed; replace removes absorbed, inserts gists consolidated | OK | `adjudicate` 888-895 | |
| S5.16 | gist stability = min(max S + sum of the others' strength, S_max); R = min(1, sum strength / S); last_recall = now - S min(64, -log2 R); created_at/tz/cue from the newest absorbed | OK | `gists` 948-977 | |
| S5.17 | LLM/embedder/store failure changes nothing, seed stays labile; after an applied verdict a failure ends the dream as an Error report; failing store stops the dream | OK | `dream`, `adjudicate` | `Err` only before the first verdict |
| S5.18 | cue pulls the past; no cue = plain neighbour search | OK | | |
| S5.19 | direction: candidates never newer than the seed | OK | `stated_by` | |
| S5.20 | `clusters(budget)` = preview of the same seeds and candidates, writes nothing | OK (D) | `clusters` 760-771 | rustdoc says "would hand to the LLM" but returns the cluster of all 8 x budget seeds (test) and does not re-embed stale rows first |
| S5.21 | `undo_all` newest first | OK | 852-854 | |
| S5.22 | no whole-store backup; `DreamReport` is the undo record; undo of replace removes gists and restores absorbed, keep re-labilises | OK | `undo` 859-875 | |
| S5.23 | blast radius of one verdict <= 8 rows | OK | `dream_max_members` | |

### §6 Parameters

| id | sentence | status | src | note |
|---|---|---|---|---|
| S6.1 | 19 parameters with the listed defaults | OK | `config.rs` (19 fields; every default checked against the table) | |
| S6.2 | only cosineFloor, thetaRelated, gistMinCosine are model-dependent | OK / UNT | | |
| S6.3 | calibration recipe | UNT | | |
| S6.4 | token-overlap test embedder uses theta 0.75 | DIFF (S) | `tests/longevity.rs:16` only | item 8 |
| S6.5 | change dimension -> change model_id; same id different dimension is not re-indexed (dims mismatch are skipped) | OK | `near` length filter | |
| S6.6 | (not in SPEC) parameter ranges, `COUNT_MAX` 2^20 | NOT in SPEC | `config.rs` `validate` | item 9 |

### §7 Invariants

| id | sentence | status | src |
|---|---|---|---|
| S7.1 | 0 <= R <= 1, 1 <= S <= S_max at create/update/read | OK | `clamp_stability`, `checked` |
| S7.2 | dt >= 0; unit vectors; single-model search | OK (unit vectors only for rows the engine wrote; S2.25) | |
| S7.3 | count <= capacity after each insert, also dream; undo'd rows evicted at the next write | OK | `victims` excess = total + 1 - capacity |
| S7.4 | only eviction, `forget` and dream replace remove | OK | |
| S7.5 | text and id rules | OK | `clean_text`, `valid_id` |
| S7.6 | every applied verdict is reported | OK | `dream` |
| S7.7 | traces change only when named by a newer one; presented candidates unchanged | OK | `adjudicate` |
| S7.8 | <= 1 LLM call per seed; silence never consolidates | OK | |

### §8 Scale

| id | sentence | status | src | note |
|---|---|---|---|---|
| S8.1 | all state bounded (count <= capacity, S <= S_max, R in [0,1]); engine state independent of N | OK / DIFF (S2.6) | | `write_times` bounded by `writes_per_day`, exposures by `inject_n` |
| S8.2 | no whole-store pass except the one-time migration | DIFF (S) | `Tx::purge`, `count` | item 10 |
| S8.3 | per-operation store query table | DIFF (S, minor) | | item 11 |
| S8.4 | each operation O(log N) with an indexed store | DIFF (S) | `count(view)` | item 10 |
| S8.5 | embedding calls batched for recall, clusters, dream cues | OK | | |
| S8.6 | cosine = f32 multiply-add over 16 lanes summed in f64 | OK | `vector.rs` | |
| S8.7 | ids 48-bit ms (year 10,889), roll back afterwards | DIFF (S) | `id.rs` | item 12; clamp itself is right |
| S8.8 | local time formatting is pure arithmetic beyond year 9999 | OK | `time.rs` | |
| S8.9 | old-model rows healed newest first when the model is available | OK | `reindex` | |
| S8.10 | no LLM -> consolidation stops, writes/recall/reinforce/forget continue | OK | | |
| S8.11 | saturated store: new memory survives the grace period, then the longest-unrecalled yields | OK | `victims` | |
| S8.12 | 3000-year simulation confirms the invariants | UNT (ignored test) | `tests/longevity.rs` | item 13 |

## 4. Public API documentation (rustdoc)

`cargo rustdoc --lib -- -W missing_docs -W rustdoc::broken_intra_doc_links -W rustdoc::private_intra_doc_links` and `cargo test --doc` (the
`lib.rs` example passes). Issues:

* **Missing docs** (29 items): `Remembered::Inserted{trace,cosine,evicted}`, `RateLimited{text}`; `Recalled.{trace,score,cosine,retrievability}`;
  `Recall.recalled`; `DreamAction::{Keep,Replace}`; `DreamReport.{action,error}`; `Stats.{records,labile}`; `View::Labile{model_id,after}`,
  `Stale{model_id}`, `Band{band,created_by}`, `Id/Text` payloads; `Hit.trace`; `Write::{State,Index}` fields. Proposed one-liners:
  `Stats.records` "Number of traces (including unusable rows).", `Stats.labile` "Labile traces indexed under the current model: what the dream can take (unusable ids included).",
  `Hit.trace` "The matching row, vector included.", `DreamAction::Keep` "The seed was marked consolidated; nothing else changed.", `DreamAction::Replace` "The seed and the named candidates were replaced by gists.".
* **Private links in public docs**: `Engram::retrievability` links `Self::clamp_stability` (private); `Store::near` doc links `PROBE_VECTORS` (private). Use plain code spans.
* **`Store` docs are invisible**: `mod store` is private, so the long `//!` module text (IVF, `LIST_MAX`, exactness) never reaches rustdoc, and `Store`'s own doc says "(see the module docs)". Move that text onto `pub struct Store` and make `pub const LIST_MAX` private (it is `pub` in a private module, unreachable and used only by that module doc).
* **`Engram::clusters`** doc: "The clusters the next [`Engram::dream`] would hand to the LLM" -> "The cluster of each of the next 8·`budget` seeds that reactivates something (seed first), without calling the LLM or re-indexing. A dream judges at most `budget` of them. Clusters may overlap."
* **`engine.rs` `BAND_DEPTH`** doc ("so its least recently recalled traces are its weakest but for ties"): false since the bound loop; it is only the first window's slack. Replace with "Rows read per band beyond `excess` in the first window; windows then grow (x4, up to `BAND_CAP`) until the bound in `victims` holds."
* **`IdGen`** doc (`id.rs:5-8`) lists only "no `《》`"; the engine also needs non-empty, no control/bidi characters and no whitespace but U+0020 (`valid_id`), and re-draws on failure (as the SPEC says). **`Trace.id`** doc ("ULID-shaped id (first 10 chars = millisecond timestamp ...)") is only true of the default generator; say so.
* **`Remembered::Inserted.evicted`**: "the last trace forgotten" is `trace` itself whenever it is among the victims, else the strongest victim (`victims.iter().find(self).or(last())`); fine for excess 1, imprecise for a bulk excess.
* **`time::fmt_local`** says "(SPEC §2)"; the any-year claim is §8.
* **README.md / README.ja.md**: `to_json` / `merge_json` (item above). Exact edits: README.md "All 19 engine parameters live in `engram::Config` (JSON round trip with `to_json` / `merge_json`); see SPEC.md §6" -> "All 19 engine parameters are plain fields of `engram::Config` (`Config::default()`, checked by `Config::validate()`); see SPEC.md §6". README.ja.md "（`to_json` / `merge_json` で JSON と相互変換）" -> "（素のフィールドで、`Config::default()` と `validate()` があります）". README x3 `《id》` -> `《id:…》`. README "Project structure" lacks `bench/` (`ltm-bench`, unpublished) and `audit/`; add them if the tree is shown.
* README snippet verified (test `readme_quickstart_compiles_and_runs`). README performance table and "opens in milliseconds": UNT.

## 5. `bench/` (crate `ltm-bench`)

`cargo check --workspace --all-targets` with `CARGO_TARGET_DIR=C:/t/r3b` **cannot run here for an environmental reason**: `llama-cpp-sys-2`'s build script
needs `cmake`, which is not installed ("program not found"). Instead I copied `bench/` into a scratch crate with `llama.rs` replaced by a
stub `LlamaEmbedder` (the only module that touches llama-cpp-2; `lib.rs` uses `load` and `Embedder`/`Clone`), same engine path dependency, and built it
with `CARGO_TARGET_DIR=C:/t/r3b_bench`: `cargo check --all-targets` is **clean with zero warnings against both the original and the moved source**, and `cargo test`
passes (20 lib + 11 `app.rs` tests; only llama is stubbed). Engine API used by bench, verified by compilation: `Engram::{open,with_clock,stats,clusters,dream,
strength,store,remember,recall,...}`, `Stats{records,labile}` (constructed literally in `session.rs:193`: adding a `Stats` field would break it), `View::All`, `Storage`/`Write`/`Trace` (`perf.rs`), `DreamAction`, `ulid_ids`, `relaxed_json`, `fmt_local`. No bench code uses `View::Labile`, `Engram::checked` (private), `Config::validate` ranges, `json_texts` or the new `View::After`; none of the changed signatures affects it. Behavioural note: `supersede.rs` measures capture with `m.clusters(Some(pairs.len()))`, which relies on `clusters` returning every cluster of the next 8 x budget seeds; if `clusters` is ever truncated to `budget` clusters to match its rustdoc, that benchmark changes.

## 6. Existing tests, support, clippy

* `cargo clippy --all-targets` (CARGO_TARGET_DIR=C:/t/r3b): **`src/` has zero warnings**; 19 warnings, all in `tests/audit*`: `audit2_a.rs:268`, `audit2_c.rs:351,380` (`manual_is_multiple_of`), `audit2_b.rs:23` (`should_implement_trait`: method `next` on the test RNG), `audit2_b.rs:749` (`single_match`), `audit2_d.rs:384` (`needless_update`), `:596` x8 (`neg_cmp_op_on_partial_ord` in the `ck!` macro), `:1112` (`unnecessary_map_or`), `:1210` (`manual_range_patterns`), `:1537` (`collapsible_if`), `audit_lane2.rs:441` (`explicit_counter_loop`). Cosmetic; none hides a bug (the `!(a >= b)` in `ck!` is intended NaN-failing).
* `cargo check --no-default-features`: clean (no warnings).
* **Stale comments** (Dart or removed mechanisms): `tests/support/mod.rs:1-3` ("bit-for-bit identical to the Dart package's") and `:244` ("as the Dart traces write them"); `src/text.rs:3` ("shared with the Dart engine"; Dart alignment was dropped on 2026-10-10); directory name `tests/equivalence` (Dart equivalence, now a same-algorithm golden). `tests/audit_lane3.rs:32,50` still describe "`8·budget + |failed|`" / "goes to `failed` (capacity 8·dream_budget)" under "BUG (L3-1)" headings, and the test is named `bug_fallback_deferred_seeds_starve_later_seeds` (`failed`/`defer` no longer exist; the cursor cycle replaced them); `audit_lane1.rs:266`, `audit_lane2.rs:155,225` talk of the `excess + BAND_DEPTH` window as if it were the whole read (the window now widens). Suggested rewording: "Regression (fixed): ..." and "the first window (`excess + 8`) is not enough; the bound loop must widen it". `tests/config.rs` (10 lines) does not test any `validate` range other than two, including `COUNT_MAX`.
* `tests/support`: every pub helper is used; the blanket `#![allow(dead_code)]` hides per-binary unused ones (normal for a shared module).
* Baseline: full `cargo test` ran green through `audit2_d` (the `audit3_a` binary stopped the run with its own new failing tests: that is lane A's work, not a regression).

## 7. Dead code / leftovers (as of the snapshot)

None flagged by rustc or clippy. By inspection:
* `BAND_DEPTH` is live (first window); only its doc is stale. `BAND_CAP`, `DEAD`, `TEXT_PROBE`, `SEED_PAGE`, `TZ_MAX` live. `Hit.{cos,trace}` live (`Hit.trace` undocumented). `normalized` live (`vector()` and `Store::split`). `Window.{limit,more,last_dt,rows}` live. Snapshot A `damaged: Mutex<Vec<String>>` (a `Mutex` in an `Engram` that cannot be `Sync` anyway because `Box<dyn Storage>` is only `Send`) is gone in the current tree (replaced by `scrub: String`).
* `store::LIST_MAX` is `pub` in a private module (unreachable; make it private). `text::shorten`, `time::parse_offset`, `json::json_texts`, `text::valid_id` are `pub` in private modules and only crate-internal: harmless, but not API.
* `ManualClock` derives `Default` (reads 0.0); fine.

## 8. What to run

`CARGO_TARGET_DIR=C:/t/r3b cargo test --test audit3_b -j 3` -> 5 passed, 2 failed (expected, see section 1).
