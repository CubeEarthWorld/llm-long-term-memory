# LLM Long-Term Memory

A long-term memory engine for LLMs implementing **ENGRAM v2.1** ([`SPEC.md`](SPEC.md), Japanese): a memory *trace* model derived from the principles of human memory. The package is the Rust crate `engram` — only the algorithm. You bring the embedding model, the LLM and the database through four small traits; the crate itself has no networking, no settings and three dependencies (`serde_json`, `jiff`, `ulid`).

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
| `remember(text, salience, cue)` | exact duplicate → rehearsal; otherwise insert, **never overwrite**. A trace whose `cue` reaches an older neighbour (cos ≥ θ_related) is born *labile*; no existing trace is touched. Over `capacity`, the lowest-strength trace outside the 3-day grace period is forgotten. |
| `recall(query)` | multi-cue cosine → `score = a·(α + (1−α)·R)` → absolute + relative cut → MMR → `[unix tz] text 《id》` pack ≤ 1024 chars. Injection is exposure: half-activation strengthening. `peek(query)` is the same without side effects. |
| `cite(reply)` | the `《id》`s the LLM quoted are strengthened as *used* (full activation). |
| `forget(id)` | id-only physical delete. |
| `dream(adjudicator, budget)` | labile traces (first in, first out; ≤ 8·budget) each seed a cluster of the older traces their `cue` reactivates (cos ≥ θ_related, ≤ 8); your LLM answers **keep** or **replace**. Gists inherit the strongest member's stability plus the *live* evidence of the others; unrelated outputs are rejected as confabulation. |

No tiers, no counters, no rings, no maintenance call. Every operation is at most one linear pass over the store, and nothing depends on elapsed time; a 3000-virtual-year simulation is part of the test suite.

## Using the library

```toml
[dependencies]
llm-long-term-memory = { git = "https://github.com/CubeEarthWorld/llm-long-term-memory" }
```

```rust
use engram::{Config, DreamDecision, DreamRequest, Engram, MemoryStore};

let mut memory = Engram::open(Box::new(MemoryStore::default()), Box::new(my_embedder), Config::default(), "Asia/Tokyo")?;
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
| `Store` | durable rows: `load_all`, atomic `write`, `clear`, optional `backup` | `MemoryStore` (RAM only) |
| `Adjudicator` | the dream verdict; any `FnMut(&DreamRequest) -> Result<DreamDecision>` | `DreamDecision::parse` for lenient JSON |
| `Clock` | Unix seconds | `system_clock()`, `ManualClock` |
| `IdGen` | trace ids (unique, creation-ordered) | `ulid_ids()` — monotonic ULIDs (default) |

An embedder that fails is treated as offline: writes are kept unindexed and indexed once it answers again. Switching the model (a new `model_id`) re-embeds every trace from its text.

## Parameters

All 19 engine parameters live in `engram::Config` (JSON round trip with `to_json` / `merge_json`); see [`SPEC.md`](SPEC.md) §6. Three depend on the embedding model's cosine distribution and are pre-set for EmbeddingGemma: `cosine_floor = 0.4`, `theta_related = 0.55`, `gist_min_cosine = 0.5`.

## Performance

10,000 traces × 768 dimensions on one machine (8 cores / 16 threads, RTX 5070 Ti), with a SQLite store on disk; medians of 5 runs.

| operation | time |
|---|---|
| remember (average while filling / at capacity, with eviction) | 1.1 ms / 3.4 ms |
| recall (one cue) | 2.4 ms |
| cluster preview (40 seeds) | 25 ms |
| dream, 5 verdicts (mostly the pre-dream snapshot of the database) | ~270 ms |
| load 10k traces | ~100 ms |
| embedding one text: GPU / CPU | 1.3 ms / 22 ms |
| process memory with 10k traces | 83 MB |

Cosines are float32 multiply-adds over 16 SIMD lanes, summed in f64.

## Project structure

```
├── src/                    # the package (crate engram)
│   ├── engine.rs           #   remember / recall / peek / cite / forget / dream
│   ├── store.rs embed.rs dream.rs time.rs id.rs   # the seams: Store, Embedder, Adjudicator, Clock, IdGen
│   └── config.rs trace.rs text.rs vector.rs json.rs
├── tests/                  # tests (in-memory store, deterministic fakes)
└── SPEC.md                 # the specification
```

## License

[MIT](LICENSE)
