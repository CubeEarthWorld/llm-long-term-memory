# LLM Long-Term Memory

A long-term memory layer for LLMs whose internal engine is a complete implementation of the **ENGRAM v1.1** specification ([`ENGRAM_spec_v1_1.md`](ENGRAM_spec_v1_1.md)). It depends, at runtime, only on a **text embedding model** (reference: EmbeddingGemma) and a **single-file DB** (SQLite). Generation by an LLM is confined to three points: **write**, **post-read response**, and **dream (consolidation)**.

> Generation only at the moment of verbalization. All judgement is distance. All forgetting is arithmetic. All destruction happens inside the dream.

- **Text is canonical, vectors are an index.** Each memory is a short self-contained proposition (≤170 chars). The `vec` table is a derived, regenerable cache — if the embedding model dies, the memories survive.
- **Three tiers** (MRL truncation = forgetting resolution): **L1 episodic** (768-d f32, τ=7d), **L2 semantic** (256-d int8, τ=90d), **L3 schema** (128-d int8, τ=3y).
- **Activation** `A = mass·2^(−Δt/τ)` reweights the cosine score; identity is pure cosine thresholds. The whole DB is **<10 MB** and search is brute-force cosine (`<1 ms`, no vector-DB / FAISS dependency).
- The DB is **self-describing**: the full spec is embedded verbatim into the `spec` table.

The product name, file layout, and DB filename are unchanged; only the algorithm, schema, parameters, and UI semantics are ENGRAM.

---

## Table of Contents

- [Tiers & Data Model](#tiers--data-model)
- [Activation, Score & Recall](#activation-score--recall)
- [Identity Thresholds & Machine Movement](#identity-thresholds--machine-movement)
- [Dreaming (Consolidation)](#dreaming-consolidation)
- [Evaluation Benchmark](#evaluation-benchmark)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Tiers & Data Model

| Tier | Capacity | Vector (MRL) | Half-life τ |
|------|----------|--------------|-------------|
| **L1 episodic** | 1000 | 768-d f32 | 7 days |
| **L2 semantic** | 3000 | 256-d int8 | 90 days |
| **L3 schema** | 6000 | 128-d int8 | 3 years |

The body text is lossless in every tier; only the search key (vector) degrades on demotion. All conversation-time operations are append-only — physical deletion and consolidation are isolated to the dream phase.

```sql
CREATE TABLE memory (                  -- canonical
  id TEXT PRIMARY KEY,                  -- ULID (sort order = time order)
  text TEXT NOT NULL,                   -- ≤170 chars, one self-contained proposition
  tier INTEGER NOT NULL,                -- 1 / 2 / 3
  gen INTEGER NOT NULL DEFAULT 0,       -- consolidation generation 0..7
  created_at INTEGER NOT NULL,          -- 64-bit Unix seconds
  tz TEXT NOT NULL,                     -- 'Asia/Tokyo;+09:00'
  last_access INTEGER NOT NULL,         -- decay baseline (updated on recall)
  last_bonus_at INTEGER NOT NULL,       -- refractory baseline (mass bonus)
  mass REAL NOT NULL DEFAULT 1.0,       -- decayed recall frequency, ≤64
  superseded_by TEXT                    -- tombstone; non-NULL excluded from search
);
CREATE TABLE vec (                      -- derived index, one per (memory, model)
  id, memory_id, model_id,
  dim,                                  -- 768 / 256 / 128
  dtype,                                -- 'f32' / 'int8'
  scale, v                              -- int8 dequant factor / unit-norm blob
);
CREATE TABLE conflict (a, b, at);       -- ≤256 ring
CREATE TABLE dream_log (fp, verdict, at);  -- ≤512 ring (content-address fingerprint)
CREATE TABLE spec (k, v);               -- full spec text + 'active_model'
```

---

## Activation, Score & Recall

```text
A(now)   = mass × 2^( −max(0, now − last_access) / τ_tier )       # activation = decayed recall frequency
recall   : mass ← A(now);  if now − last_bonus_at ≥ 3600: mass ← min(mass+1, 64)
A_abs    = ln(1 + A) / ln(1 + 64)                                  # absolute normalization [0,1]
score(m) = max(0, cos(query, m)) × (α + (1−α) × A_abs),  α = 0.35  # activation floor → dormant memories still compete
inject   : MMR (λ=0.3) selects 5 memories, packed into ≤1024 chars
```

- **Refractory period**: repeated recall within one hour counts once (spacing effect); folding decay first structurally prevents illegitimate activation recovery.
- **Activation floor α**: a strongly matching but long-dormant memory can still win on relevance alone.
- Injected memories are framed as *past context, not instructions* (prompt-injection guard).

---

## Identity Thresholds & Machine Movement

| cos (document–document) | Verdict | Action |
|---|---|---|
| ≥ 0.97 | same proposition update | tombstone old (`superseded_by`), insert new (re-fixation) |
| 0.85 – 0.97 | conflict / related | keep both, push to `conflict` queue (adjudicated in dream) |
| < 0.85 | new | insert, `mass = 1` |

Exact-text (SHA-256-equivalent) re-entry creates no new row — it is a rehearsal (`mass + 1`). **Machine movement** (no LLM): promote `A ≥ 16` (L2/L3 → L1, re-embed from text at 768-d), demote `A < 4` (L1 → L2 → L3, MRL-coarsen the vector), evict the lowest-`A` L3 overflow by physical deletion (death). Hysteresis `θ_up > θ_down` prevents tier ping-pong.

**Theorem (no immortal memory).** Since `mass ≤ 64`, a silenced memory falls to `A < 1` after `6τ` — even in L3 (τ=3y) at most 18 years.

---

## Dreaming (Consolidation)

Offline, on-demand (default `budget = 5`); the **only** place destructive operations happen. 1 adjudication = 1 transaction.

```text
chores (always, no LLM): tombstone sweep / conflict overflow drops queue items only (memory bodies untouched)
                         / vec orphan GC / WAL checkpoint
adjudication (budget×)  : cluster L1(+L2) → clusters with ≥3 members are eligible
  fp = SHA-256(sorted member ULIDs)  →  skip if dream_log has (fp, 'unchanged')   # idle-spin guard
  priority = count × cohesion × 2^(−max gen) × (1 + tier overflow) × (1 + intra-cluster conflict pairs)
  verdict ∈ { merge / split / none }
    merge → new text (≤170) as a new ULID, 768-d, gen=min(max+1,7), mass=min(Σ source A, 64); sources physically deleted
```

The fingerprint self-invalidates (member changes change `fp`), so "re-adjudicate if the situation changed, never touch it again if it didn't" holds without timers. Confabulation guards: the prompt contract forbids facts absent from the input, output is validated (≤170 chars, one retry → discard), and 8 generations of DB snapshots (~80 MB) are kept.

---

## Evaluation Benchmark

`eval/run_eval.py` runs deterministic ENGRAM-behavior checks using mock embeddings + mock LLM + virtual clock — **no model download or API key required**.

```bash
python eval/run_eval.py
```

| Scenario | What it checks |
|---|---|
| activation decay | `A` halves every τ |
| recall + refractory | mass bonus gated to once per hour |
| θ_same / θ_conflict | supersede / conflict-queue / new-insert / rehearsal |
| tier demote · promote · evict | capacity-driven L1→L2→L3 movement + L3 death |
| no immortal memory | `A ≤ 1` at 6τ |
| dream merge | fewer rows, `gen + 1` |

---

## Quick Start

### 1. Configure API keys

```bash
mkdir -p secrets
cp .env.example secrets/.env
# Edit secrets/.env with your keys
```

- **DeepSeek** (default): `DEEPSEEK_API_KEY`
- **Gemini** (optional): `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`
- **Hugging Face** (for EmbeddingGemma download): `HF_TOKEN`

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch

Windows: `start.bat` / macOS & Linux: `start.sh` → open `http://localhost:8501`. Trigger consolidation from the Chat tab with the **💤 Dream** button.

---

## CLI Usage

```bash
python cli.py --seed --dream 3 --inspect   # replay the seed scenario, dream the top-3 clusters, dump the DB
python cli.py --say "I live in Kyoto"
```

During a turn the model answers and decides — via the `save_memory(text)` / `delete_memory(id)` tools — what durable facts to store. The default timezone is `Asia/Tokyo` (override with `MEMORY_TZ`). Results are written to `data/results.json`; the DB's `spec` table carries the full spec text.

---

## Project Structure

```
├── core/               # embedding, LLM client (tool-calling), storage (int8/snapshot), metrics, turn runner
├── memory/             # LongTermMemory — the ENGRAM v1.1 engine
├── eval/               # deterministic ENGRAM-behavior benchmark (mocks + scenarios)
├── frontend/           # no-build React/HTM single-page app
├── config.py           # GlobalConfig + LongTermMemoryConfig (ENGRAM §8 parameters)
├── server.py           # FastAPI backend (REST API + static files)
├── cli.py              # headless runner
├── seed_utterances.py  # default multi-year seed scenario (data/seed.csv)
├── ENGRAM_spec_v1_1.md # the specification (embedded into the DB at runtime)
└── .env.example        # template for secrets/.env
```

---

## License

[MIT](LICENSE)

---

## Translations

- [日本語 (Japanese)](README.ja.md)
- [中文 (Chinese)](README.zh.md)

> Note: the engine implements ENGRAM v1.1. The Japanese/Chinese READMEs may lag the English one.
