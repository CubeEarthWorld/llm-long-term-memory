# LLM Long-Term Memory

A lightweight, human-memory-inspired long-term memory layer for LLMs.  
LLM calls happen **only at write-time (extraction)** and during **Dreaming (consolidation)**.  
Recall, scoring, clustering, and archive management are performed locally using embeddings and numerical computation — no vector DB or FAISS required.

- **Two-tier SQLite storage**: **active** (hot, ≤1000 records, 768-d vectors) and **archive** (cold, ≤5000 records, 256-d coarse vectors only). Total capacity is bounded (~tens of MB).
- **Full 768-d embeddings** for search. **256-d coarse vectors** are derived on-the-fly for clustering, diversity tuning, spreading activation, and archive matching.
- Retrieved memory packs are capped at **1024 characters**.
- Each memory separates three axes inspired by human memory: **`w`** (importance), **`confidence`** (reliability), and **`S`** (stability).
- Every memory carries an **IANA timezone** and is passed to the LLM during recall and dreaming.

---

## Table of Contents

- [Data Structures](#data-structures)
- [Retention, Recall & Forgetting](#retention-recall--forgetting)
- [Dreaming (Memory Consolidation)](#dreaming-memory-consolidation)
- [Evaluation Benchmark](#evaluation-benchmark)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Data Structures

```sql
CREATE TABLE memories (              -- active (hot)
  id TEXT PRIMARY KEY,               -- uuid7
  text TEXT, v_full BLOB,            -- 768-d (coarse derived on-the-fly)
  w REAL, provenance TEXT, confidence REAL,  -- importance / 'user'|'inferred' / reliability
  freq REAL, stability REAL,         -- cumulative access count / stability S (seconds)
  accessed_at_unix REAL,             -- last access → baseline for decay
  updated_at_unix REAL,              -- content version timestamp → used in recall / dream
  timezone TEXT, cluster_id TEXT,
  source_ids TEXT, summary_of TEXT, dream_action TEXT, created_by_dream_id TEXT  -- dream lineage
);

CREATE TABLE archive (               -- cold (savings)
  id TEXT PRIMARY KEY,
  text TEXT, v_coarse BLOB,          -- 256-d coarse only (for reappearance matching)
  w REAL, provenance TEXT, confidence REAL,
  last_r REAL, archived_at_unix REAL, text_hash TEXT, timezone TEXT
);

CREATE TABLE clusters (
  id TEXT PRIMARY KEY,
  last_dreaming_unix REAL            -- cluster creation UNIX; updated on every dream
);
```

Centroid / size / medoid are derived from cluster members on demand (not stored).

---

## Retention, Recall & Forgetting

Stability `S` (seconds) and retrievability `r` (FSRS-style) are kept separate. Forgetting follows a **power law**; every successful recall grows `S` (more when recalled near forgetting — spacing / testing effect, Jost’s law).

```text
S0           = stab_base * (1 + kappa*w) * (labile_frac + (1-labile_frac)*confidence)
r(t)         = (1 + (now - accessed_at) / S) ^ (-forget_beta)        # power-law forgetting curve
Reinforcement: S = S * (1 + stab_growth_c*(1-r)); freq += reinforce_inc; accessed_at = now
               (S and freq are clipped by max_stability_seconds / max_freq to prevent unbounded growth)
score        = alpha*cos + beta*r + delta*w + eta*log1p(freq) + zeta*confidence  (+ spreading activation)
Recall gate  : gate_w_cos*cos + gate_w_r*r (+noise) >= gate_theta    # functional forgetting
```

Forgetting is two-stage and capacity-bounded:

1. **active → archive**: When `r` falls below `r_archive_floor` (after a grace period), the memory is moved to the archive — inaccessible to normal recall, but retained as *savings*. High-`w` + high-`confidence` memories are protected down to `r_hard_floor`. When capacity is exceeded, the lowest-`r` memories are evicted first.
2. **archive → permanent deletion**: When the archive exceeds `archive_cap`, the lowest `last_r` entries are deleted permanently.
3. **savings (reappearance)**: When the same or similar text reappears (coarse cosine ≥ `tau_savings`, or text_hash match), it is restored to active memory with a stability head-start (`× savings_gain`) — relearning savings.
4. **interference**: Recalled memories slightly decay the stability of unrehearsed competitors in the same cluster (retrieval-induced forgetting).

---

## Dreaming (Memory Consolidation)

A sleep-analogy consolidation process. Run on-demand; the LLM processes high-priority clusters and chooses to **merge**, **split**, or do **nothing**.

- **Priority** (computed) = `w_size*log(1+size) + w_spread*(spread/norm) + w_age*(since_dream/norm) + w_disp*dispersion`
  - `spread` = time span of `updated_at_unix` within the cluster; `since_dream` = elapsed since `last_dreaming_unix`.
  - Clusters smaller than `dream_min_size` or within `dream_min_interval` cooldown are skipped.
- The LLM receives member content, importance, **timezone-aware timestamps**, and retrievability `r`. Output memories replace the inputs.
  - **merge** = reduce and gist-ify related memories (episodic → semantic).
  - **split** = separate multiple facts packed into one memory.
  - Stale content is updated with current time context (e.g. “planned trip in July” → “trip taken in July 2026”). Conflicts are resolved in favor of newer timestamps.
- Merged memory **stability does not exceed the best source** (`S = max(member S)`). Lineage is preserved via `source_ids` / `created_by_dream_id`.

---

## Evaluation Benchmark

`eval/run_eval.py` runs deterministic fidelity benchmarks using mock embeddings + mock LLM + virtual clock — **no model download or API key required**.

```bash
python eval/run_eval.py
```

It validates 8 human-memory behaviors:

| Scenario | What it checks |
|---|---|
| spacing > massed | Spaced recall grows stability more than massed recall |
| forgetting power-law tail | Retrievability decreases over time but keeps a heavy tail |
| recall gate | Functional forgetting blocks weak cues to decayed memories |
| stability growth | Stability increases on every successful retrieval |
| archive → savings restore | Archived memories restore with a stability head-start |
| archive bounded | Archive respects `archive_cap` (total capacity is bounded) |
| 3-axis protection | High-confidence, high-importance memories resist eviction |
| spreading activation | Cluster siblings receive a spreading-activation boost |

---

## Quick Start

### 1. Configure API keys

Copy the example file and fill in your real keys. The `secrets/` directory is git-ignored so keys never leak into version control.

```bash
mkdir -p secrets
cp .env.example secrets/.env
# Edit secrets/.env with your keys
```

- **DeepSeek** (default): set `DEEPSEEK_API_KEY`
- **Gemini** (optional): set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`
- **Hugging Face** (for EmbeddingGemma download): set `HF_TOKEN`

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch

Windows: `start.bat` / macOS & Linux: `start.sh` → open `http://localhost:8501` in your browser.  
You can trigger consolidation from the Chat tab with the “💤 Dream” button.

---

## CLI Usage

```bash
python cli.py --say "I live in Kyoto"
python cli.py --dream            # run 1 dreaming pass after seeding
python cli.py --dream 3 --inspect  # dream top-3 clusters and dump the DB
```

The default timezone is `Asia/Tokyo` (override with `MEMORY_TZ` env var).  
Results are written to `data/results.json`.

---

## Project Structure

```
├── core/               # embedding, LLM client, storage, metrics, base protocol
├── memory/             # LongTermMemory implementation (retrieval, forgetting, dreaming)
├── eval/               # deterministic fidelity benchmark (mocks + scenarios)
├── frontend/           # no-build React/HTM single-page app
├── config.py           # GlobalConfig + LongTermMemoryConfig dataclasses
├── server.py           # FastAPI backend (REST API + static files)
├── cli.py              # headless runner
├── seed_utterances.py  # default multi-year seed scenario
├── requirements.txt
└── .env.example        # template for secrets/.env
```

---

## License

[MIT](LICENSE)

---

## Translations

- [日本語 (Japanese)](README.ja.md)
- [中文 (Chinese)](README.zh.md)
