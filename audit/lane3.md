# Lane 3 — the dream (SPEC §5, §7)

Files: `audit/lean/Lane3.lean` (compiles with Lean 4.34.1 core, no `sorry`), `tests/audit_lane3.rs`
(copy in `audit/tests/`; Cargo only discovers `tests/`). Run:
`CARGO_TARGET_DIR=C:/t/l3 cargo test --test audit_lane3 -j 3` -> 17 pass, 5 fail (the 5 are the defects below).

## Verdicts

| Inv | Status | Evidence |
|---|---|---|
| L3-1 `|seeds| <= 8*budget` | proved in model | `seeds_bound` |
| L3-1 failed bounded queue | proved in model | `defer_bound`, `deferAll_bound` (cap = 8*cfg.dream_budget) |
| L3-1 failed/fresh disjoint | holds by construction (`partition`) | — |
| L3-1 no starvation | **REAL BUG (three variants)** | `seeds_within_want_plus_cap`, `unusable_head_starves`, witness `decide +kernel`; tests `bug_*` |
| L3-2 calls <= budget, seed judged once | proved | `calls_le_budget`, `calls_le_seeds`; tests `llm_calls_never_exceed_budget...` |
| L3-2 settled seeds free, fallback deferral free | proved/tested | `settled_seeds_are_free`; `seeds_settled_without_candidates_cost_nothing`, `fallback_deferral_costs_no_budget` |
| L3-3 direction, size, seed != candidate | proved | `cand_direction`, `cand_not_seed`, `cluster_size` (needs `dream_max_members >= 1`; validate enforces >= 2) |
| L3-3 gist inherits seed created_at | proved | `newest_is_seed`; test `a_gist_inherits_created_at...` |
| L3-4 replace+undo exact | proved | `undo_replace` (needs gist ids fresh = `new_id`); tests incl. strengthened/forgotten gist, keep+undo |
| L3-4 capacity | proved | `replace_no_growth`; after undo rows may exceed capacity, next `remember` evicts (test passes) |
| L3-4 undo order | **GAP (design)** | `gap_undo_in_dream_order_double_counts`; Lean counter-example |
| L3-4 split vs guard | **GAP (spec vs prompt)** | `gap_split_of_a_compound_seed_is_kept` |
| L3-5 untouched candidates, vanished/consolidated seeds skipped | tested | `unnamed_candidates_stay_untouched...`, `seeds_that_vanish_or_settle_meanwhile_are_skipped` |
| L3-5 settled seed absorbed later | **OK, no resurrection** | `Store::apply(State)` returns early when the row is absent (`state_on_removed_is_noop`; test `a_settled_seed_absorbed_by_a_later_cluster_is_not_resurrected`) |

## Real bugs

### B1. Unusable labile rows starve the dream (L3-1) — `bug_unusable_labile_rows_starve_the_queue`
`seeds()` calls `find(Labile, want + failed.len())` and drops unusable rows afterwards in `checked`. A labile row that can
never be used (empty text / invalid id, written by an older version or by hand) is never consolidated, so it sits at the head of the
FIFO forever. `8*budget` such rows (8 for budget 1, 40 by default) and the dream returns `[]` for ever; usable seeds behind them are never read.
Repro: budget 1; raw-insert 8 labile rows with `text = ""` (ids sorting first), then remember a similar pair; `dream(Some(1))` -> no report.
Fix idea: page the Labile view until `want` usable seeds are found (or skip unusable rows in the store view / evict them).

### B2. Failed-seed queue only protects a window of `want + cap` rows (L3-1) — `bug_fallback_deferred_seeds_starve_later_seeds`, `bug_failing_verdicts_starve_later_seeds`
Seeds are always inside the first `want + |failed|` labile rows and `|failed| <= cap` (`seeds_within_want_plus_cap`). Undecided seeds
(cue embedding offline -> fallback path, or verdict `Err`) are re-queued, but `failed` forgets its oldest entries, so with >= `8b + 8b`
(=16 for budget 1, 80 for the default 5) permanently undecidable seeds at the head, the rotation cycles among them and the seeds behind
(decidable ones) are never examined. SPEC §5 claims "決められない種が先頭に居座って後続を飢えさせることもない".
Repro 1: budget 1, query embedder offline, 16 cued seeds then an uncued pair: 20 dreams never replace the pair.
Repro 2: 16 distinct "poison" clusters whose verdict is always `Err`, then a good pair: 40 dreams never reach it.
Fix idea: keep a cursor (id of the last examined seed) instead of a failed list, or size the window by `want + cap` pages and rotate past it.
Related inconsistency: `defer` caps at `8*cfg.dream_budget` while `seeds` uses the budget passed to `dream(Some(b))`; with `b > cfg.dream_budget`
one dream can defer more seeds than `failed` holds (`bound_mismatch`), so some deferrals are forgotten inside the same dream.

## Gaps (not code defects strictly, worth a decision)

* G1 `undo` is exact only for a single report or reverse order. Undoing a dream's reports in dream order (the order returned) when a later
  report absorbed an earlier gist re-inserts the earlier originals next to the later gist (the same facts twice, no error). `undo_in_reverse_order_is_exact` passes. Document "undo newest first" or have `undo` refuse when a gist is missing.
* G2 Spec/prompt conflict: the prompt tells the LLM to split a compound seed into independent memories (ids empty, several memories), but
  `|texts| > |absorbed|` is a keep (the capacity guard), so the split is silently discarded and the seed consolidated unsplit.
* G3 SPEC §5 wording "いずれかの absorbed とのコサインが gistMinCosine 未満なら keep" is ambiguous; the code (and doc comment) use "unrelated to *every* absorbed" (max cosine), which is the sensible reading.
* G4 `undo` re-inserts a row the user has `forget`-ten since (forget -> undo resurrects). Intended per spec, but it defeats an explicit delete.
* Minor: stale ids in `failed` (seeds later absorbed or settled) are never pruned; they only consume cap (bounded by 8*cfg budget).

## Checked, no defect
Same-second facts are mutual candidates and each seed gets <= 1 call; gist stability/R conservation matches the formula
(`gist_strength_is_conserved_and_bounded`); `memories == []` with named ids is a keep that touches no candidate; replace naming only unknown ids rewrites the seed alone;
settle write after the loop cannot resurrect or error on removed ids; capacity after replace never grows.
