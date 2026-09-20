# LLM Long-Term Memory

A long-term memory layer for LLMs implementing **ENGRAM v2** ([`SPEC.md`](SPEC.md), Japanese): a memory *trace* model derived from the principles of human memory, running on a **local embedding model** (EmbeddingGemma as a GGUF via llama.cpp) and a **single-file SQLite DB**. The LLM generates only at three points: write, use (with citation), and the offline **dream** (consolidation).

> Generation only at the moment of verbalization. All judgement is distance. All forgetting is arithmetic. All consolidation happens inside the dream.

The same algorithm ships as a Dart package ([`../long-term-memory`](../long-term-memory)); a cross-language conformance test replays one scripted scenario in both and requires identical traces.

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
| `recall(query)` | multi-cue cosine → `score = a·(α + (1−α)·R)` → absolute + relative cut → MMR → `[unix tz] text 《id》` pack ≤ 1024 chars. Injection is exposure: half-activation strengthening. |
| `cite(reply)` | the `《id》`s the LLM quoted are strengthened as *used* (full activation). |
| `forget(id)` | id-only physical delete. |
| `dream(budget)` | labile traces (most stable first) each seed a cluster of the older traces their `cue` reactivates (cos ≥ θ_related, ≤ 8); your LLM answers **keep** or **replace** (the ids it supersedes + the gist texts). Gists inherit the strongest member's stability plus the *live* evidence of the others; unrelated outputs are rejected as confabulation. A settled store makes no LLM calls. |

No tiers, no counters, no rings, no maintenance call. Everything is bounded, so cost does not depend on elapsed time; a 3000-virtual-year simulation is part of the test suite.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # or source .venv/bin/activate
pip install -r requirements.txt
mkdir secrets && copy .env.example secrets\.env       # fill in DEEPSEEK_API_KEY (or GEMINI_API_KEY)
# put embeddinggemma-300m-qat-Q4_0.gguf into ./model  (https://ai.google.dev/gemma/docs/embeddinggemma)
start.bat            # Windows  → http://localhost:8501
./start.sh           # macOS / Linux
```

CLI:

```bash
python cli.py --seed --dream 5 --inspect     # replay the seed scenario, dream with 5 LLM calls, dump the store
python cli.py --say "I live in Kyoto"
python -m pytest                             # deterministic suite (no model or key needed) incl. the Dart conformance scenario
python -m pytest -m slow                     # the 3000-virtual-year simulation (minutes)
```

## What a turn does

```
recall(utterance) → LLM (system prompt + memory pack + save_memory / delete_memory tools) → cite(reply)
```

The LLM saves durable facts as pronoun-free propositions with absolute dates and an optional `salience` (1–10, emotional weight). Every save also carries a required `cue`: the question this fact would later be asked with. The cue is what the dream phase searches the past with, so an update can reach the version it supersedes even when the two sentences are textually far apart. The LLM never re-saves facts it was just shown, and quotes the `《id》` of memories it used so the engine can strengthen them. If a turn saved nothing, one extraction call proposes propositions through the same path.

## Storage

One table: `memory(id, text, created_at, tz, last_recall, stability, consolidated, model_id, vector, cue)` plus the UI's turn log. The whole store lives in RAM (≈35 MB at 10k traces × 768 dims); SQLite only persists. A rotating ring of 8 snapshots is written before every dream. Switching the embedding model re-embeds every trace from its text (text is canonical; vectors are an index).

## Parameters

All 19 engine parameters live in `LongTermMemoryConfig` (`config.py`) and are editable in the UI; see [`SPEC.md`](SPEC.md) §6. Three of them depend on the embedding model's cosine distribution and are pre-set for EmbeddingGemma: `cosine_floor = 0.4`, `theta_related = 0.55`, `gist_min_cosine = 0.5`.

## Project structure

```
├── memory/engine.py      # the ENGRAM v2 engine (remember / recall / cite / forget / dream)
├── memory/model.py       # the Memory trace
├── memory/util.py        # ids, text hygiene, cues, local time (any year)
├── core/embedding.py     # EmbeddingGemma GGUF via llama.cpp (offline)
├── core/storage.py       # SQLite substrate + snapshot ring
├── core/llm_client.py    # DeepSeek / Gemini: converse (tools + citation), extraction, dream
├── core/turn.py          # per-turn runner (+ core/metrics.py)
├── core/session.py       # app session: assembly, turn log, seed replay, dream
├── core/seed.py          # built-in seed scenario + seed CSV
├── server.py / jobs.py / frontend  # FastAPI routes, job state, no-build React UI
├── cli.py                # headless runner
├── tests/                # pytest suite + deterministic fakes (conformance test reads ../long-term-memory/test/conformance)
└── SPEC.md               # the specification
```

## License

[MIT](LICENSE)
