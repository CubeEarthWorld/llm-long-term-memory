# long_term_memory

A portable, dependency-light **long-term memory engine for LLM applications**, written in pure Dart and usable from any Flutter app (all platforms) or Dart server/CLI.

It is a faithful port of the ENGRAM v1.1 memory system in this repository. The whole design compresses to four sentences:

> Generation only at the moment of verbalization. All judgement is distance.
> All forgetting is arithmetic. All destruction happens inside the dream.

**The package contains only the algorithm.** The LLM, the embedding model and the database are *not* included — you inject them through three small interfaces, so it composes with anything: [firebase_ai](https://pub.dev/packages/firebase_ai), [llamadart](https://pub.dev/documentation/llamadart/latest/), [sqlite3](https://pub.dev/documentation/sqlite3/latest/), drift, Isar, Hive, REST endpoints, on-device ONNX models, …

日本語版は [README.ja.md](README.ja.md) を参照してください。

---

## Features

- **3-tier memory** — L1 episodic (768-d float32, half-life 7 days), L2 semantic (256-d int8, 90 days), L3 schema (128-d int8, 3 years). Tiers store Matryoshka (MRL) truncations of one embedding: lower tiers keep the same vector at lower resolution, no re-embedding needed.
- **Activation decay** — every memory carries a mass; its activation `A = mass · 2^(−Δt/τ)` decays with the tier half-life. Recall reinforces mass (spacing effect with a 1-hour refractory gate). Forgetting is pure arithmetic.
- **Distance-only identity** — writes are judged by document–document cosine: `≥ 0.97` supersedes the old memory, `0.85–0.97` keeps both and queues the conflict, below inserts a new row. No LLM in the write path.
- **Relevance + recency retrieval** — score = `max(0, cos) · (α + (1−α)·Â)` with an activation floor so dormant-but-relevant memories still compete, plus MMR diversity selection and a character-budgeted, injection-safe memory pack.
- **Dream-phase consolidation** — offline clustering (spherical k-means) + LLM adjudication (merge / split / none) through a callback you implement with *your* LLM; generation caps, confabulation guards, mass carry-over and one-transaction-per-adjudication are enforced by the engine.
- **Tier movement & capacity** — automatic demotion (L1→L2→L3 via MRL truncation + int8 quantisation), promotion of hot memories back to L1 (re-embedding), tombstone sweeps, ring-buffered conflict/dream logs, hard row caps.
- **Time & timezone aware** — every call accepts an explicit 64-bit Unix time and a timezone (`IANA name + offset`, stored as `'Asia/Tokyo;+09:00'`); the clock is injectable, clock rollbacks are sanitised, and local-time formatting needs no tz database.
- **Bring-your-own everything** — `Embedder` (embedding model), `MemoryStore` (database) and `DreamAdjudicator` (LLM) interfaces; an `InMemoryStore` is bundled so the package works out of the box, and a complete SQLite adapter ships in [`example/sqlite_adapter`](example/sqlite_adapter).
- **Prompt kit included** — Japanese + English system prompts, `save_memory` / `delete_memory` tool specs, extraction-fallback and dream-consolidation templates, and lenient JSON parsers (`DreamDecision.parseJson`, `EngramPrompts.parseExtractedTexts`).
- Pure Dart, one runtime dependency (`package:crypto`), no codegen, fully deterministic test suite.

## Installation

Until published on pub.dev, depend on it by path or git:

```yaml
dependencies:
  long_term_memory:
    git:
      url: https://github.com/CubeEarthWorld/llm-long-term-memory
      path: long-term-memory
```

## Quick start

```dart
import 'package:long_term_memory/long_term_memory.dart';

Future<void> main() async {
  final memory = EngramMemory(
    store: InMemoryStore(),               // swap for your DB adapter
    embedder: myEmbedder,                 // your Embedder implementation
    defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
  );
  await memory.initialize();

  // WRITE — store one self-contained fact (e.g. from an LLM tool call).
  final saved = await memory.saveMemory('ユーザーは京都に住んでいる');
  print(saved.action); // SaveAction.inserted

  // READ — fetch relevant memories for the next prompt.
  final recall = await memory.retrieve('どこに住んでいるか覚えてる？');
  print(recall.packText); // [<unix> Asia/Tokyo;+09:00] ユーザーは京都に住んでいる　《id:...》

  // Per-turn upkeep (no LLM, no embedding — cheap).
  await memory.maintain();

  // Offline consolidation through YOUR LLM (run when the app is idle).
  await memory.dream(adjudicate: (request) async {
    final raw = await myLlm.generateJson(request.buildPrompt());
    return DreamDecision.parseJson(raw);
  });
}
```

A runnable zero-dependency version (toy embedder + in-memory store) is in [`example/main.dart`](example/main.dart):

```bash
dart run example/main.dart
```

## How it fits into a chat app

The engine never talks to your conversation LLM; you wire its four primitives around your own tool-calling loop:

```
user utterance
   │
   ├─► memory.retrieve(utterance)          → memory pack
   │
   ├─► your LLM call
   │     system  = EngramPrompts.conversationSystemPromptJa (or En)
   │     user    = EngramPrompts.buildUserMessage(
   │                 currentTime: memory.nowLocal(),
   │                 memoryPack: pack.packText,
   │                 userText: utterance)
   │     tools   = [EngramPrompts.saveMemoryToolSpec,
   │                EngramPrompts.deleteMemoryToolSpec]
   │
   ├─► on tool call "save_memory"   → memory.saveMemory(args['text'])
   ├─► on tool call "delete_memory" → memory.deleteMemory(args['id'])
   │
   └─► memory.maintain()
```

Optionally, when a turn saved nothing, run the **extraction fallback**: send `EngramPrompts.buildExtractionInstruction(...)` to your LLM, parse with `EngramPrompts.parseExtractedTexts(raw)`, and feed each string through `saveMemory`.

## Time and timezone inputs

Every entry point accepts the current Unix time (seconds) and a timezone alongside the text:

```dart
await memory.saveMemory(
  'ユーザーは2026-04-01にベルリンへ出張した',
  nowUnix: 1774000000,                                       // explicit 64-bit Unix seconds
  timezone: const MemoryTimezone('Europe/Berlin', Duration(hours: 2)),
);
await memory.retrieve('出張の予定', nowUnix: 1774100000);
```

- Omitted values fall back to the injected `clock` / `defaultTimezone`, then to the system clock / UTC.
- Timezones are stored per memory as `'IANA_name;+HH:MM'`; the explicit offset means local-time formatting (`memory.nowLocal()`, dream prompts) is pure arithmetic — no tz database required. Pure Dart cannot resolve IANA names itself, so supply the offset (e.g. `DateTime.now().timeZoneOffset`, or `package:timezone` for exact historical DST).
- A `clock` callback (`int Function()` returning Unix seconds) makes the engine fully deterministic for tests and simulations; timestamps later than "now" are clamped so a clock rollback can never mint immortal memories.

## The three interfaces

### 1. `Embedder` — your embedding model

```dart
abstract class Embedder {
  String get modelId;       // stamped on vectors; switching models self-heals
  int get dimension;        // must be ≥ config.dim1 (768 by default)
  Future<List<Float32List>> embedQueries(List<String> texts);
  Future<List<Float32List>> embedDocuments(List<String> texts);
}
```

Vectors need not be normalised — the engine normalises and MRL-truncates them. Use an MRL-trained model (e.g. EmbeddingGemma) for best L2/L3 quality. `CallbackEmbedder` lets you wire an SDK without declaring a class.

**firebase_ai sketch** (Gemini embeddings, asymmetric task types):

```dart
final model = FirebaseAI.googleAI().generativeModel(model: 'gemini-embedding-001');
final embedder = CallbackEmbedder(
  modelId: 'gemini-embedding-001',
  dimension: 768,
  onEmbedDocuments: (texts) => /* batchEmbedContents with taskType RETRIEVAL_DOCUMENT */,
  onEmbedQueries:   (texts) => /* batchEmbedContents with taskType RETRIEVAL_QUERY */,
);
```

**llamadart sketch** (local GGUF embedding model):

```dart
final llama = Llama(modelPath: 'embeddinggemma-300m.gguf', embedding: true);
final embedder = CallbackEmbedder(
  modelId: 'embeddinggemma-300m',
  dimension: 768,
  onEmbedDocuments: (texts) async => [
    for (final t in texts) Float32List.fromList(await llama.embed('title: none | text: $t')),
  ],
  onEmbedQueries: (texts) async => [
    for (final t in texts) Float32List.fromList(await llama.embed('task: search result | query: $t')),
  ],
);
```

### 2. `MemoryStore` — your database

A typed interface (no SQL leaks through) with five groups: memory rows, vectors, the conflict ring, the dream log and metadata. Only the primitives are abstract; bulk operations and hooks (`runInTransaction`, `backup`, `compact`) have sensible defaults you can override for efficiency.

- `InMemoryStore` (bundled): zero-setup reference implementation with `toJson`/`fromJson`.
- [`example/sqlite_adapter`](example/sqlite_adapter): complete `package:sqlite3` adapter with WAL, real transactions, a rotating snapshot ring and vacuum — copy it into your app, or use it as the template for drift/Isar/Hive/ObjectBox adapters.

> **Transactionality note**: each dream adjudication (delete cluster → insert replacements) runs inside `runInTransaction`. On stores without transactions a crash inside that window can lose memories; implement `runInTransaction` and `backup` if your data matters.

### 3. `DreamAdjudicator` — your LLM (dream phase only)

```dart
final reports = await memory.dream(adjudicate: (request) async {
  // request.members: id / text / gen / activation / localTime / timezone
  final raw = await myLlm.generateJson(request.buildPrompt());   // ja or en
  return DreamDecision.parseJson(raw);                            // never throws
});
```

The engine enforces every guard itself: replacement texts are shortened to 170 chars, empty ones dropped, generation is capped at 7, mass carries over (sum for merge, split evenly for split, ≤64), clusters with an unchanged membership and a previous `none` verdict are skipped, and a throwing callback leaves the cluster untouched and retryable.

## API reference (engine)

| Method | Purpose |
|---|---|
| `initialize()` | Open the store, validate the embedder, write self-description metadata. |
| `retrieve(query, {nowUnix, timezone})` | READ: chunked query embedding → scoring → thresholds → MMR → ≤1024-char pack; recalled memories get a recall update. Returns `RetrieveResult(packText, recalled)`. |
| `saveMemory(text, {nowUnix, timezone})` | WRITE: validation → exact-text reinforcement → identity scan → insert / supersede / conflict. Returns `SaveResult` (`inserted` / `updated` / `conflict` / `reinforced` / `rejected` / `rateLimited`). |
| `deleteMemory(id, {hard})` | DELETE by id only: tombstone (default) or physical. Returns `DeleteResult`. |
| `maintain({nowUnix})` | Per-turn upkeep: timestamp sanitisation, tombstone sweep, capacity demotion/eviction, periodic `compact()`. No LLM, no embedding. |
| `dream(adjudicate: …, {budget, nowUnix, timezone, force})` | Offline consolidation: backup → housekeeping → ≤budget LLM adjudications → promotion → capacity → compaction. Returns `List<DreamReport>`. |
| `stats()` / `totalRecords()` / `activationOf(id)` / `listMemories()` | Introspection. |
| `nowUnix()` / `nowLocal({nowUnix, timezone})` | Clock helpers (`'2026-06-12 09:30 +09:00'` for prompts). |
| `reset()` | Erase everything and reinstall metadata. |

All methods are internally serialized; an `EngramMemory` instance is safe to call from interleaving async code. The dream phase can take ~0.5–2 s on mobile at full capacity (pure-Dart k-means over ≤4000×128-d vectors) — run it when idle, or in an isolate with its own store handle.

## Configuration

All ENGRAM §8 parameters live in `EngramConfig` (const-constructible, `copyWith`, JSON round-trip). Defaults:

| Group | Parameter | Default | Meaning |
|---|---|---|---|
| Tiers | `cap1/cap2/cap3` | 1000 / 3000 / 6000 | rows per tier (tombstones included) |
| | `dim1/dim2/dim3` | 768 / 256 / 128 | MRL vector dims (f32 / int8 / int8) |
| | `tau1/tau2/tau3` | 7 d / 90 d / 3 y | activation half-lives |
| Activation | `mMax` | 64 | mass / activation cap |
| | `refractorySeconds` | 3600 | min interval between mass bonuses |
| Retrieval | `alpha` | 0.35 | activation floor in the score |
| | `injectN` | 5 | memories injected per retrieve |
| | `mmrLambda` | 0.3 | MMR diversity penalty |
| | `scoreThresholds` | [0.1, 0.2] | progressive filters (strictest first) |
| | `budgetChars` | 1024 | pack character budget |
| Identity | `thetaSame` | 0.97 | ≥ → supersede |
| | `thetaConflict` | 0.85 | conflict band lower bound |
| | `preciseMargin` | 0.03 | full-precision re-embed margin |
| Movement | `thetaUp` / `thetaDown` | 16 / 4 | promote / demote hysteresis |
| Text | `textMax` / `textHardMax` | 170 chars / 1024 bytes | soft shorten / hard reject |
| | `genMax` | 7 | consolidation generation cap |
| Dream | `dreamBudget` (`dreamBudgetHard`) | 5 (4096) | adjudications per call |
| | `clusterMin` / `clusterCohesionMin` | 3 / 0.5 | cluster eligibility |
| | `dreamMaxMembers` | 64 | members per adjudication |
| Rings | `conflictCap` / `dreamLogCap` | 256 / 512 | ring buffer sizes |
| Housekeeping | `tombstoneSweepPct` / `tombstoneSweepAge` | 10% / 7 d | sweep triggers |
| Limits | `writeRatePerDay` | 2000 | soft write rate |
| | `hardMemoryRows` | 16384 | hard row cap / loop guard |
| | `decayExpCap` | 65536 | decay underflow guard |
| | `compactEvery` | 20 | maintains per `compact()` |

## Storage footprint & performance

With default capacities (10,000 memories) the entire store is **< 10 MB**: L1 vectors 3.0 MB, L2 0.8 MB, L3 0.8 MB, text + metadata ~2 MB. Retrieval is a brute-force scan (≤10k dot products of ≤768 dims ≈ sub-10 ms on a modern phone) — no ANN index to maintain or corrupt.

## Testing your integration

The package's test suite shows the patterns: a deterministic token-overlap `FakeEmbedder` and a `VirtualClock` (see [`test/support/fakes.dart`](test/support/fakes.dart)) make every scenario — decay, refractory gating, supersede/conflict thresholds, dreams over virtual months — reproducible without any model or API key. Run:

```bash
dart test                          # package suite (63+ tests)
cd example/sqlite_adapter && dart test   # adapter suite
```

## Design notes & limitations

- **Conversation LLM is app-side by design** — the engine stays usable with any SDK, tool-calling convention or agent framework.
- **No IANA tz database** — offsets are caller-supplied and frozen per memory (the spec stores `name;+offset` precisely so formatting survives without tzdata). Use `package:timezone` in your app if you need exact historical DST.
- **Soft text length counts UTF-16 code units** (Dart `String.length`); identical to character counts for Japanese/CJK and ASCII.
- **Embedding-model switches self-heal**: old-model vectors are GC'd during dream housekeeping; promotion re-embeds with the active model.

## License

MIT — see [LICENSE](LICENSE).
