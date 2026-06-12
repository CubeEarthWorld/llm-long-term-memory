# Changelog

## 0.1.0

Initial release: pure-Dart port of the ENGRAM v1.1 long-term memory engine.

- 3-tier memory (L1 episodic / L2 semantic / L3 schema) with MRL vector
  truncation and int8 quantisation.
- Activation decay `A = mass·2^(−Δt/τ)`, refractory-gated reinforcement.
- Distance-only identity (supersede ≥0.97, conflict band 0.85–0.97).
- Retrieval with activation-floored scoring, progressive thresholds, MMR
  selection and a budgeted injection pack.
- Per-turn maintenance: tombstone sweeps, capacity demotion/eviction.
- Dream-phase consolidation through an app-supplied LLM callback
  (spherical k-means clustering, priority ranking, fingerprint cache,
  transactional merge/split).
- Injected interfaces: `Embedder`, `MemoryStore`, `DreamAdjudicator`;
  bundled `InMemoryStore`; SQLite adapter in `example/sqlite_adapter`.
- Unix-time + timezone inputs on every entry point, injectable clock,
  clock-rollback sanitisation.
- Japanese/English prompt templates and lenient JSON parsers.
