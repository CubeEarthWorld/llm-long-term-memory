# LLM Long-Term Memory

A long-term memory layer for LLMs implementing **ENGRAM v2.1** ([`SPEC.md`](SPEC.md), Japanese): a memory *trace* model derived from the principles of human memory, running on a **local embedding model** (EmbeddingGemma as a GGUF via llama.cpp, on the GPU when CUDA is available) and a **single-file SQLite DB**. Written in Rust: a dependency-light library crate (`engram`) for embedding in agent harnesses, plus the `ltm` app (web UI, headless runner, benchmarks). The LLM generates only at three points: write, use (with citation), and the offline **dream** (consolidation).

The same algorithm ships as a Dart package ([`../long-term-memory`](../long-term-memory)). Both share the arithmetic (float32 vectors, cosines summed in f64 in dimension order), so a cross-language conformance scenario and a 220-step seeded workload reproduce the Dart traces **bit for bit**.

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
| `remember(text, salience, cue)` | exact duplicate → rehearsal; otherwise insert, **never overwrite**. A trace whose `cue` reaches a neighbour (cos ≥ θ_related) is born *labile*; no existing trace is touched. Over `capacity`, the lowest-strength trace outside the 3-day grace period is forgotten. |
| `recall(query)` | multi-cue cosine → `score = a·(α + (1−α)·R)` → absolute + relative cut → MMR → `[unix tz] text 《id》` pack ≤ 1024 chars. Injection is exposure: half-activation strengthening. `peek(query)` is the same without side effects. |
| `cite(reply)` | the `《id》`s the LLM quoted are strengthened as *used* (full activation). |
| `forget(id)` | id-only physical delete. |
| `dream(adjudicator, budget)` | labile traces (first in, first out; ≤ 8·budget) each seed a cluster of the older traces their `cue` reactivates (cos ≥ θ_related, ≤ 8); your LLM answers **keep** or **replace**. Gists inherit the strongest member's stability plus the *live* evidence of the others; unrelated outputs are rejected as confabulation. |

No tiers, no counters, no rings, no maintenance call. Every operation is at most one linear pass over the store (id and text lookups are O(1), eviction and neighbour selection are partial selections, not sorts), and nothing depends on elapsed time; a 3000-virtual-year simulation is part of the test suite.

## Quick start

Requirements: [Rust](https://rustup.rs) and a C/C++ toolchain with CMake and libclang (on Windows: Visual Studio Build Tools with the C++ workload, which ships both). With the CUDA toolkit installed, the build offloads the embedding model to the GPU automatically.

```bash
mkdir secrets && copy .env.example secrets\.env       # LLM: Claude via Claude Code by default (no key); see below
# put embeddinggemma-300m-qat-Q4_0.gguf into ./model  (https://ai.google.dev/gemma/docs/embeddinggemma)
start.bat            # Windows  → http://localhost:8501
./start.sh           # macOS / Linux
```

CLI (after `build.bat` / `build.sh`):

```bash
target/release/ltm run --dream 5 --inspect     # replay the seed scenario, dream with 5 LLM calls, dump the store
target/release/ltm run --no-seed --say "I live in Kyoto"
cargo test                                     # deterministic suite (no model or key) incl. the Dart conformance scenario
cargo test --release --test longevity -- --ignored   # the 3000-virtual-year simulation (~15 s)
```

Builds use at most 80% of the CPU cores (`.cargo/config.toml`, `build.bat`), and so do llama.cpp's CPU threads and the benchmark workers.

## Using the library

```rust
use engram::{Config, Engram, SqliteStore, LlamaEmbedder, DreamDecision};
let embedder = LlamaEmbedder::load("model/embeddinggemma-300m-qat-Q4_0.gguf".as_ref(),
                                   "task: search result | query: ", "title: none | text: ")?;
let mut memory = Engram::open(Box::new(SqliteStore::open("memory.db".as_ref())?), Box::new(embedder),
                              Config::default(), "Asia/Tokyo")?;
memory.remember("The user moved to Osaka in April 2026.", 1.0, "Where does the user live?")?;
let pack = memory.recall("where do I live now?")?.pack_text;      // inject into your prompt
memory.cite(&llm_reply)?;                                           // strengthen what was used
memory.dream(&mut |req: &engram::DreamRequest| {                    // any LLM in JSON mode
    DreamDecision::parse(&my_llm(&req.prompt())).ok_or("unparsable".into())
}, None)?;
```

The seams are traits: `Store` (four operations; `SqliteStore`, `MemoryStore`), `Embedder` (`LlamaEmbedder`, `HttpEmbedder` for any OpenAI-compatible `/embeddings` endpoint) and `Adjudicator` (any closure). Without the `app` feature the crate depends only on `serde_json` and `jiff`: `default-features = false, features = ["sqlite"]`.

## The LLM: Claude through Claude Code

The default provider is `claude-code`: Claude Haiku 5.5 through the Claude Code CLI (`claude -p`), no API key in the app. Every answer is cached under a hash of (model, system prompt, prompt) in `LLM_CACHE` (default `data/llm_cache.json`), so a replayed scenario asks the model once and then reuses the same data. Prompts the CLI cannot answer (not installed / not signed in) go to `<cache>.pending.json`; answer them with any Claude client and merge them into the cache. `ltm run --epoch <unix>` pins the virtual clock and the trace ids, so a rerun sends exactly the same prompts: `bench/llm_cache.json` holds the Haiku answers for `ltm run --epoch 1780000000 --dream 5` and `ltm bench supersede --run` (`LLM_CACHE=bench/llm_cache.json`). `deepseek` (any OpenAI-compatible endpoint) and `gemini` remain available.

## What a turn does

```
recall(utterance) → LLM (system prompt + memory pack + save_memory / delete_memory tools) → cite(reply)
```

The LLM saves durable facts as pronoun-free propositions with absolute dates, an optional `salience` (1–10, emotional weight) and a required `cue`: the question this fact would later be asked with. The dream searches the past with that cue, so an update reaches the version it supersedes even when the two sentences are textually far apart. If a turn saved nothing, one extraction call proposes propositions through the same path.

## Storage

One table: `memory(id, text, created_at, tz, last_recall, stability, consolidated, model_id, vector, cue)` plus the UI's turn log. The whole store lives in RAM (≈80 MB process at 10k traces × 768 dims); SQLite only persists, and every engine operation is one transaction applied to RAM only after it commits. A rotating ring of 8 snapshots is written before every dream. Switching the embedding model re-embeds every trace from its text.

## Parameters

All 19 engine parameters live in `engram::Config` and are editable in the UI; see [`SPEC.md`](SPEC.md) §6. Three depend on the embedding model's cosine distribution and are pre-set for EmbeddingGemma: `cosine_floor = 0.4`, `theta_related = 0.55`, `gist_min_cosine = 0.5` (`ltm bench generality` calibrates them for another model).

## Benchmarks

```bash
ltm bench cache                 # embed bench/corpus once with the configured model
ltm bench sweep && ltm bench sweep --report     # forgetting: 155 runs (engram vs FIFO / LRU / random / none)
ltm bench supersede --analyze   # overwrite forgetting: cue capture (add --run for a live LLM dream)
ltm bench generality --dims 768,512,256,128     # other models / dimensions
ltm bench perf --n 10000        # latency and memory of every operation
```

## Performance

Measured with 10,000 traces × 768 dimensions on one machine (8 cores / 16 threads, RTX 5070 Ti), SQLite on disk.

| operation | time |
|---|---|
| remember (average while filling / at capacity, with eviction) | 2.0 ms / 4.3 ms |
| recall (one cue) | 3.8 ms |
| cluster preview (40 seeds) | 39 ms |
| dream, 5 verdicts (mostly the pre-dream snapshot) | ~240 ms |
| load 10k traces | 73 ms |
| embedding one text: GPU / CPU | 1.3 ms / 22 ms |
| process memory with 10k traces | 85 MB |
| 3000-virtual-year simulation | 15 s |

## Project structure

```
├── src/lib.rs, engine.rs   # the ENGRAM v2.1 engine (remember / recall / peek / cite / forget / dream)
├── src/{trace,config,text,time,id,vector,json,dream}.rs   # trace, parameters, hygiene, time, ids, cosine, prompts
├── src/{store,sqlite}.rs   # Store trait, MemoryStore, SqliteStore (+ snapshot ring)
├── src/embed.rs            # Embedder trait, llama.cpp (CPU / CUDA) and HTTP embedders
├── src/app/                # settings, LLM client, turn runner, session, web server, CLI, bench/
├── frontend/               # no-build React UI (compiled into the binary)
├── tests/                  # integration tests + deterministic fakes (conformance reads ../long-term-memory/test/conformance)
└── SPEC.md                 # the specification
```

## License

[MIT](LICENSE)
