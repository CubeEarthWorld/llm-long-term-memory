# Round 2, lane C: peripheral modules and boundary validation

Files: `audit/lean/R2C.lean` (compiles with `~/.elan/bin/lean.exe`, no `sorry`; one `native_decide` for the
146 097-day calendar era), `tests/audit2_c.rs` (22 tests).
Run: `CARGO_TARGET_DIR=C:/t/r2c cargo test --test audit2_c -j 3` (about 80 s in debug; `c20` is the long one).
Result: 22 tests, 14 pass, 8 fail. 7 failures are defects (root causes below), `note_c13` is a design note.
`src/` was not modified.

## Real defects (tests fail)

| id | sev | test | root cause | repro |
|---|---|---|---|---|
| C1 | med | `c1_prompt_does_not_resubstitute_inside_member_text` | `DreamRequest::prompt` does `.replace("{listing}", listing).replace("{current_time}", time)`: the second replace rescans the inserted listing, so any stored text (or id / timezone) containing the literal `{current_time}` is rewritten to the current time before the LLM sees it. The LLM then reads, and may copy into a gist, a fact that was never stored. `{listing}` in text is inert (it is replaced first). Fix: single pass over the template (or substitute `{current_time}` first). Lean `second_replace_concrete`, `single_pass_keeps_listing`, `replaceTok_hit`. | member text `"旅行は {current_time} に予定"`; `prompt()` no longer contains `listing()`. |
| C3 | med | `c3_ulid_ids_never_panic_on_any_millis`, `c8_absurd_clock_values_do_not_panic` | `ulid_ids` clamps negative ms but not large ones: `UNIX_EPOCH + Duration::from_millis(ms as u64)` panics when the instant is not representable. On Windows (FILETIME, 100 ns ticks since 1601) the limit is 910 692 730 085 477 ms (year ~30 800); the engine computes `(now * 1000.0) as i64`, which saturates to `i64::MAX` for any larger clock including `+inf` and `f64::MAX`. Every `remember` and every gist (`new_id`) therefore panics at a clock of 1e12 s, 1e15 s, 1e30 s, `+inf`. SPEC §8 promises "時計故障" survival and ids that merely wrap after year 10 889. NaN, `-inf`, `f64::MIN`, negative and 2e11 s are fine. Fix: `checked_add` / clamp ms to 2^48-1 (the ULID field) before building the `SystemTime`. Lean `win_max_millis`, `ulid_overflow_on_windows`, `ulid_48_bit_years`. Platform-dependent (Linux `SystemTime` holds i64 seconds, no panic). | `ulid_ids()(1_000_000_000_000_000, "")`; `ManualClock::new(f64::INFINITY)` + `remember`. |
| C6 | low | `c6_out_of_range_offsets_are_malformed` | `parse_offset` accepts any two digits for hours and minutes (`+99:99`, `+09:75`, `-30:00`), contrary to its "0 when malformed" doc; `fmt_local` then shifts the time by the made-up offset and prints a new one (`+100:39`). A stored or hand-written tz field is only cleaned, not range-checked (`clean_tz`). Fix: accept `h <= 23` (or <= 14), `m <= 59`. Lean `out_of_range_offsets_are_rewritten`, `offset_roundtrip` (exact for real offsets). | `fmt_local(0, "X;+99:99")` = `1970-01-05 04:39 +100:39`. |
| C10 | low-med | `c10_non_string_memories_are_not_texts` | `json_texts` stringifies every non-null scalar and any JSON value: `"memories":[1]`, `[true]`, `[["x"]]`, `[{"text":{"a":1}}]`, `[{"text":5}]` all parse as `replace` with texts `"1"`, `"true"`, `["x"]`, `{"a":1}`, `"5"`. SPEC §5: a replace must carry a text; a numeric/structural item is not one, so a confused answer should be a failure (retried), not a gist. The §5 cosine guard (>= gistMinCosine against an absorbed member) is the only remaining barrier, and it is weak with a token-overlap or low-threshold embedder. Fix: take only `String` and `{"text": String}`. | `DreamDecision::parse(r#"{"action":"replace","memories":[1]}"#)` is `Some`. |
| C17 | low | `c17_normalized_is_unit_for_any_finite_nonzero_magnitude` | `normalized` takes the norm from `dot` = f32 lanes, so `x*x` overflows f32 for `|x| >= ~1.8e19` (returned as the zero vector) and underflows for `|x| <= ~3e-23` (returned unnormalised). `embed.rs` allows unnormalised vectors; the engine then rejects them as non-unit (treated as offline embedder, never indexed). Realistic embedders are unaffected. Fix: accumulate the norm in f64 (or rescale by max abs). Fails at scales 1e-30, 1e-24, 1e20, 1e30; 1e-10..1e18 are fine. | `normalized(&[1e20; 64])` is all zeros. |
| C21 | low | `c21_huge_memory_list_is_not_quadratic` | `Engram::verdict` de-duplicates the adjudicator's texts with `Vec::contains` (O(n^2)) before it notices `texts.len() > absorbed.len()` (= keep). 8 000 texts 0.6 s, 32 000 texts 8.2 s (debug). A runaway custom adjudicator, not a real LLM. Fix: cap `memories` at `dream_max_members` before cleaning, or use a `HashSet`. | adjudicator closure returning 32 000 distinct memories. |

Structural area shared by C1, C10, C21 (and the keep rule): the dream boundary in `dream.rs` / `json.rs` / `Engram::verdict`
trusts LLM- or adjudicator-shaped data too early (templating, stringification, unbounded list). C3 and C17 share the "numeric boundary
of an unchecked cast / accumulation" area: `as i64`, `Duration` addition and f32 accumulation at the extremes of valid input. C6 is the
same kind of gap for the stored tz field (malformed but accepted).

## Design note (test `note_c13_*` fails, not counted as a defect)

* `relaxed_json` takes the leftmost `{` or `[` through the last matching closer. Prose with brackets before the object
  (`判断 [理由は省略]: {"action":"keep"}`) therefore fails to parse and the seed is retried (never read as keep, so it is safe, but a
  model that always prefaces a bracket never settles). A fallback "retry from the next opener" would fix it.

## Confirmed fine (tests pass or analysis)

* **time.rs.** `fmt_local` agrees with jiff's civil calendar on 20 000 random instants over the whole jiff range for offsets
  +14:00, -12:00, +05:45, -03:30, +12:45 (`c5`); no panic at `i64::MIN/MAX` for any offset (`c4`), and the Lean file proves for all i64 that
  `civil_from_days` is the exact Gregorian calendar (`fromCivil_civil`, `civil_valid`, one exhaustive era + periodicity) and that no
  intermediate of `fmt_local` overflows i64 (`fmt_local_no_overflow`, via `sat_in_range`): the claim "pure arithmetic beyond year 9999" is
  true. DST gap / overlap boundaries (New York 2024-03-10 07:00Z, 2024-11-03 06:00Z), `Asia/Kathmandu +05:45`, `Pacific/Kiritimati +14:00`,
  `Etc/GMT+12 -12:00` and the 30-minute DST of `Australia/Lord_Howe` stamp the right frozen offset (`c7`). `tz_field` is safe for NaN / inf / huge
  `now` (the `as i64` casts saturate, the value is clamped to jiff's range). Cosmetic only: a historic LMT offset with seconds
  (Asia/Tokyo before 1888, +09:18:59) is frozen as `+09:18`, 59 s lost (Lean `lmt_seconds_are_dropped`); an unknown / mistyped zone name
  silently stamps `+00:00` forever (documented in `tz_field`, but `Engram::open` does not validate the name).
* **Prompt injection.** Member text cannot forge instruction lines or the `# 記憶` heading: it is JSON-escaped inside the listing (`c2b`);
  `{listing}` inside text is not re-substituted (`c2`). Only `{current_time}` leaks (C1).
* **Parsing.** Fences (with CRLF), prose around the object, case / whitespace in `"keep"`, truncated answers (including truncation right after a
  complete nested object), a non-string / array / missing `action`, a top-level array or string never turn into a verdict or a keep
  (`c11`, `c12`, `c14`, `c15`). Duplicate keys: the last wins (`serde_json`), and `{"action":"keep","action":"replace"}` has no texts so is a failure.
  `ids` that are not strings are stringified (numbers match numeric ids; `null` becomes the id `"null"`): lenient, harmless for ULIDs.
* **vector.rs.** All lengths 0..70 and 255..3073 (tails, empty, multiples of 16) and 3072 dims stay within 1e-5 of the f64 reference
  (`c16`; measured worst ~1.7e-6 for 3072 identical positive components, so the module doc "near 1e-7" is optimistic by ~10x for aligned
  vectors; harmless against the 1e-3 unit check). Zero, NaN, empty and denormal vectors never panic (`c18`); a zero vector is returned as is and the
  engine's `unit` check rejects it.
* **Config::validate.** 200 randomised valid-but-extreme configurations (capacity 1..usize::MAX, S_max 1..f64::MAX, gain up to f64::MAX,
  grace 0..f64::MAX, floor up to 1-eps, theta 1e-300..1-eps, inject_n / max_cues / dream_max_members / dream_budget up to 2^20, budget_chars /
  text_max / writes_per_day up to usize::MAX, alpha / relative_score / min_score at 0 and 1, salience NaN / negative / 1e9, clock jumps of 1e9 s)
  ran 30-step lives (remember, recall, cite, dream with keep / merge / failing adjudicators, undo_all, forget, reindex, clusters, peek) with no
  panic, no hang (60 s cap per case), `records <= capacity` after every insert, `R in [0,1]` and `S` clamped (`c20`). Lean `counts_cannot_overflow`:
  every product of a validated count (`4n`, `8n`, `64n`, four rounds of `k *= 4`) is far below usize::MAX.
  One theoretical gap (no test): the gist `last_recall = now_i - round(S*elapsed) as i64` overflows i64 only when the clock is below 1970 AND
  `max_stability > ~1.4e17` (Lean `gist_last_recall_overflows_for_negative_clock`; safe for any clock >= 0 and for the default configuration
  down to -9.2e18 s). Note `Vec::with_capacity(k)` in `Store::near` makes a recall with `inject_n = 2^20` reserve ~1 GB of address space
  regardless of the data size (it ran fine here).
* **lib.rs surface.** `Engram`, `Store` are `Send` (not `Sync`: the embedder and store are only `Send`); `ManualClock`, `DreamDecision`, `DreamRequest`
  are `Send + Sync` (`c22`); nothing in the docs claims more. `relaxed_json` exposes `serde_json::Value` in the public API (a semver coupling, not
  a defect). The prompt tells the LLM "170字以内" regardless of `text_max` (texts are cut to `text_max` afterwards).
