# LLM Long-Term Memory

A long-term memory engine for LLMs implementing **ENGRAM v2.2** ([`SPEC.md`](SPEC.md), Japanese): a memory *trace* model derived from the principles of human memory. The package is the Rust crate `engram`, pure Rust. The engine keeps no copy of the memory: it asks a storage for what it needs, so opening costs nothing and every operation is a few index queries. The bundled `Store` keeps everything in one file (redb) with a self-refining similarity index; any database or vector index can take its place through the `Storage` trait. You bring the embedding model and the LLM. No networking, no settings; dependencies: `serde_json`, `jiff`, `ulid` and, for the bundled store, `redb`.

日本語版は [README.ja.md](README.ja.md)、中文版は [README.zh.md](README.zh.md)。

---

## The model in one screen

A memory is a **trace** with two numbers — when it was last recalled and how stable it is (a half-life) — plus one flag (`consolidated`).

```text
R(now)   = 2^(−max(0, now − last_recall) / stability)          retrievability ∈ [0,1]
strength = stability · R                                       total remaining retrievability
a        = max(0, (cos − cosine_floor) / (1 − cosine_floor))   cue activation
recall   : stability ← min(stability · (1 + gain·a·(1−R)), S_max);  last_recall ← now
new      : stability = clamp(S0 · salience, 1 s, S_max)
```

| verb | what happens |
|---|---|
| `remember(text, salience, cue)` | exact duplicate → rehearsal; otherwise insert, **never overwrite**. A trace whose `cue` reaches an older neighbour (cos ≥ θ_related) is born *labile*; no existing trace is touched. Over `capacity`, the lowest-strength trace outside the 3-day grace period is forgotten (found from per-stability-band indexes, not by ranking everything). |
| `recall(query)` | multi-cue cosine → `score = a·(α + (1−α)·R)` → absolute + relative cut → MMR → `[unix tz] text 《id》` pack ≤ 1024 chars. Injection is exposure: half-activation strengthening. `peek(query)` is the same without side effects. |
| `cite(reply)` | the `《id》`s the LLM quoted are strengthened as *used* (full activation). |
| `forget(id)` | id-only physical delete. |
| `dream(adjudicator, budget)` | labile traces (first in, first out; ≤ 8·budget) each seed a cluster of the older traces their `cue` reactivates (cos ≥ θ_related, ≤ 8); your LLM answers **keep** or **replace**. Gists inherit the strongest member's stability plus the *live* evidence of the others; unrelated outputs are rejected as confabulation. Each report is its own undo record (`undo(report)`). |

No tiers, no counters, no rings, no maintenance call, no snapshot copies. Nothing is loaded at open and nothing touches every trace: each operation is a few store queries (O(log N) with an indexed store), and nothing depends on elapsed time; a 3000-virtual-year simulation is part of the test suite.

## Using the library

```toml
[dependencies]
llm-long-term-memory = { git = "https://github.com/CubeEarthWorld/llm-long-term-memory" }
```

```rust
use engram::{Config, DreamDecision, DreamRequest, Engram, Store};

let mut memory = Engram::open(Box::new(Store::open("memory.redb".as_ref())?), Box::new(my_embedder), Config::default(), "Asia/Tokyo")?;
memory.remember("The user moved to Osaka in April 2026.", 1.0, "Where does the user live?")?;
let pack = memory.recall("where do I live now?")?.pack_text;      // inject into your prompt
memory.cite(&llm_reply)?;                                           // strengthen what was used
memory.dream(&mut |req: &DreamRequest| {                            // any LLM in JSON mode
    DreamDecision::parse(&my_llm(engram::DREAM_SYSTEM_PROMPT, &req.prompt())).ok_or("unparsable".into())
}, None)?;
```

The seams:

| trait / type | what you provide | in the crate |
|---|---|---|
| `Embedder` | text → vectors (`model_id`, `dimension`, document and query prompts) | — |
| `Storage` | persistence and lookup: `find(view)` (by id / text, labile queue, eviction order, …), `count`, `near` (similarity search), atomic `write`, `clear` | `Store`: one redb file + an IVF similarity index (default feature `store`) |
| `Adjudicator` | the dream verdict; any `FnMut(&DreamRequest) -> Result<DreamDecision>` | `DreamDecision::parse` for lenient JSON |
| `Clock` | Unix seconds | `system_clock()`, `ManualClock` |
| `IdGen` | trace ids (unique, creation-ordered) | `ulid_ids()` — monotonic ULIDs (default) |

`Store` opens in milliseconds whatever its size: rows and every view are B-trees in one file, and vectors sit in lists, one per centroid; a search reads the 8 lists nearest the query, and a list that passes 512 vectors splits in two, so the index refines itself without rebuilds. Only the centroids and a bounded page cache live in RAM. Search is approximate once a model has more than one list; to use another database or vector index (PostgreSQL + pgvector, Qdrant, LanceDB, …) implement `Storage` — every view is one index range — and build with `default-features = false` to drop `redb`. An embedder that fails is treated as offline: writes are kept unindexed and re-embedded, newest first, at the next dream (or `reindex(limit)`), as is every trace after a model switch (a new `model_id`).

## Parameters

All 19 engine parameters live in `engram::Config` (JSON round trip with `to_json` / `merge_json`); see [`SPEC.md`](SPEC.md) §6. Three depend on the embedding model's cosine distribution and are pre-set for EmbeddingGemma: `cosine_floor = 0.4`, `theta_related = 0.55`, `gist_min_cosine = 0.5`.

## Performance

The bundled `Store` on disk, one machine (8 cores / 16 threads), CPU build, embeddings precomputed. 768 dimensions: medians of 5 runs at 10k, one run at 100k. 128 dimensions up to a million.

| operation | 10k × 768 | 100k × 768 | 1M × 128 |
|---|---|---|---|
| remember (at capacity, with eviction) | 11 ms | 15 ms | 17 ms |
| recall (one cue) | 9 ms | 11 ms | 11 ms |
| dream, 5 verdicts | 36 ms | 43 ms | 56 ms |
| open | 4 ms | 7 ms | 7 ms |
| process memory | 70 MB | 0.5 GB | 1.3 GB |

Opening reads nothing, a dream copies nothing, and the cost of an operation grows with the lists it reads rather than the store size; memory is bounded by a 1 GiB page cache. Cosines are float32 multiply-adds over 16 SIMD lanes, summed in f64.

## Project structure

```
├── src/                    # the package (crate engram)
│   ├── engine.rs           #   remember / recall / peek / cite / forget / dream
│   ├── storage.rs store.rs                        # the Storage seam and the bundled Store (redb + IVF)
│   ├── embed.rs dream.rs time.rs id.rs             # the other seams: Embedder, Adjudicator, Clock, IdGen
│   └── config.rs trace.rs text.rs vector.rs json.rs
├── tests/                  # tests (in-memory store, deterministic fakes)
└── SPEC.md                 # the specification
```

## License

[MIT](LICENSE)
