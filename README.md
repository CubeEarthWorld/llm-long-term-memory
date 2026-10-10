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

One table: `memory(id, text, created_at, tz, last_recall, stability, consolidated, model_id, vector, cue)` plus the UI's turn log (the schema is unchanged from the Python version, so existing databases open as they are). The whole store lives in RAM (≈80 MB process at 10k traces × 768 dims); SQLite only persists, and every engine operation is one transaction applied to RAM only after it commits. A rotating ring of 8 snapshots is written before every dream. Switching the embedding model re-embeds every trace from its text.

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

## Rust 1.0.0 vs the Python 1.0.1 it replaces

Measured on one machine (Ryzen-class 8C/16T, RTX 5070 Ti) with identical inputs.

| | Python 1.0.1 | Rust 1.0.0 |
|---|---|---|
| remember, 10k traces × 768 d (avg while filling / at capacity) | 10.8 ms / 24.8 ms | **2.0 ms / 4.3 ms** |
| recall / cluster preview / dream(5) at 10k | 5.3 / 82.5 / 246 ms | **3.8 / 39.0 / ~240 ms** (dream = the pre-dream DB snapshot) |
| load 10k traces | 162 ms | **73 ms** |
| process memory at 10k (engine share) | 144 MB (113 MB) | **85 MB (76 MB)** |
| embedding, per text | 21 ms (CPU) | 22 ms (CPU, 12 threads) / **1.3 ms (GPU)** |
| one forgetting-benchmark run / 3000-virtual-year test | 12 s / minutes | **2.7 s / 15 s** |
| 155-run forgetting sweep (same vectors) | — | 2 m 47 s; every per-configuration median identical to Python's, per-run metrics within 0.0001 (decimal rounding) |
| seeded 220-step workload | float32 BLAS cosines | identical decisions; numbers within 8.3e-8 relative; **bit-identical to the Dart package** |
| embedding vectors vs llama-cpp-python | — | cosine ≥ 0.9993 (median 0.99993) |
| cue capture of 120 updates (θ = 0.55) | 91 / 120 | 92 / 120 |
| overwrite forgetting, live dream | stale 0.033 (DeepSeek) | stale 0.033 (Claude Haiku 5.5, recorded in `bench/llm_cache.json`) |

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
