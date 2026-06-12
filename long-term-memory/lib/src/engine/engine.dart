/// The ENGRAM engine — a faithful Dart port of the reference Python
/// implementation, split into parts mirroring the original operation mixins
/// (write / read / maintenance / dream).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import '../config.dart';
import '../dream/adjudicator.dart';
import '../embedder.dart';
import '../models.dart';
import '../store/memory_store.dart';
import '../text_utils.dart';
import '../timezone.dart';
import '../ulid.dart';
import '../vector_math.dart';

part 'engine_base.dart';
part 'write_ops.dart';
part 'read_ops.dart';
part 'maintenance_ops.dart';
part 'dream_ops.dart';

/// The long-term memory engine (ENGRAM v1.1).
///
/// The whole system compresses to four sentences: *generation only at the
/// moment of verbalization; all judgement is distance; all forgetting is
/// arithmetic; all destruction happens inside the dream.*
///
/// - **Text is canonical, vectors are an index.** Memories are short
///   self-contained propositions (≤170 chars); cached vectors are derived,
///   regenerable data.
/// - **Three tiers**: L1 episodic (768-d f32, τ=7 d), L2 semantic (256-d
///   int8, τ=90 d), L3 schema (128-d int8, τ=3 y). MRL truncation is the
///   forgetting resolution.
/// - **Append-only during conversation**; physical deletion / consolidation
///   only inside [dream].
///
/// All external capabilities are injected: a [MemoryStore] (any database),
/// an [Embedder] (any embedding model), and — for [dream] only — a
/// [DreamAdjudicator] callback backed by any LLM. Every entry point accepts
/// an explicit Unix time (`nowUnix`, seconds) and a [MemoryTimezone], so
/// hosts fully control the clock; omitted values fall back to the injected
/// [clock] / `defaultTimezone`, then to the system clock / UTC.
///
/// Public methods are internally serialized (one pipeline at a time), so an
/// `EngramMemory` instance is safe to call from interleaving async code.
class EngramMemory extends _EngineBase
    with _WriteOps, _ReadOps, _MaintenanceOps, _DreamOps {
  /// Creates an engine over [store] and [embedder].
  ///
  /// [clock] returns the current Unix time in **seconds** (injectable for
  /// tests and simulations). [defaultTimezone] is stamped onto writes that
  /// don't pass one explicitly.
  EngramMemory({
    required super.store,
    required super.embedder,
    super.config,
    super.clock,
    super.defaultTimezone,
  });

  /// Opens the store, validates the embedder dimension and writes the
  /// self-description metadata. Call once before anything else.
  Future<void> initialize() => _serialize(() async {
        await store.initialize();
        if (embedder.dimension < config.dim1) {
          throw ArgumentError(
              'Embedder dimension ${embedder.dimension} is smaller than '
              'config.dim1 (${config.dim1}). Use an embedding model with at '
              'least ${config.dim1} dimensions or lower dim1/dim2/dim3.');
        }
        await _installMeta();
      });

  // ================================================================== //
  // conversation phase (append-only)
  // ================================================================== //

  /// READ (§5.2): retrieves up to `config.injectN` memories relevant to
  /// [query] and packs them for prompt injection.
  ///
  /// Pipeline: split the query into chunks → embed (query side) → score
  /// every live memory `max(0,cos)·(α+(1−α)·Â)` with per-tier MRL
  /// truncation → progressive score thresholds → MMR diversity selection →
  /// pack ≤ `config.budgetChars` chars. Each injected memory receives a
  /// recall update (decay fold + refractory mass bonus).
  ///
  /// [nowUnix] is the current Unix time in seconds; [timezone] is accepted
  /// for API symmetry with the write path (retrieval itself is
  /// timezone-independent — stored memories carry their own timezone).
  Future<RetrieveResult> retrieve(
    String query, {
    int? nowUnix,
    MemoryTimezone? timezone,
  }) =>
      _serialize(() => _retrieve(query, _resolveNow(nowUnix)));

  /// WRITE (§5.1): stores one self-contained proposition.
  ///
  /// Verdict by document–document cosine at full precision:
  /// `≥ thetaSame` → supersede (old row tombstoned, mass inherited),
  /// conflict band → both kept + queued for dream adjudication,
  /// below → new row. Exact-duplicate text reinforces mass without a new
  /// row. Returns what happened as a [SaveResult].
  ///
  /// [nowUnix] (Unix seconds) and [timezone] are stamped onto the stored
  /// memory (`'IANA;+HH:MM'`).
  Future<SaveResult> saveMemory(
    String text, {
    int? nowUnix,
    MemoryTimezone? timezone,
  }) =>
      _serialize(
          () => _saveMemory(text, _resolveNow(nowUnix), _tzField(timezone)));

  /// DELETE (§5.3): id-only deletion (I10). Soft (default) tombstones the
  /// row — excluded from search immediately, purged in the next dream;
  /// [hard] deletes physically right away.
  Future<DeleteResult> deleteMemory(
    String id, {
    bool hard = false,
    int? nowUnix,
  }) =>
      _serialize(() => _deleteMemory(id, hard: hard));

  /// Machine movement (§5.4) — call once per turn (or periodically).
  /// Sweeps tombstones and enforces tier capacities via demotion/eviction.
  /// No LLM, no embedding: cheap enough for every turn.
  Future<void> maintain({int? nowUnix}) =>
      _serialize(() => _maintain(_resolveNow(nowUnix)));

  // ================================================================== //
  // dream phase (offline, destructive)
  // ================================================================== //

  /// DREAM (§6): offline consolidation. The only place memories are
  /// physically destroyed or rewritten.
  ///
  /// Runs: store backup → housekeeping (tombstone sweep, ring trims,
  /// vector GC) → up to [budget] LLM adjudications via [adjudicate]
  /// (merge/split/none per cluster, each applied in one transaction) →
  /// promotion (A ≥ θ_up → re-embed → L1) → capacity enforcement →
  /// compaction. Safe to call any time the app is idle; clusters whose
  /// membership hasn't changed since a `none` verdict are skipped
  /// (set [force] to re-adjudicate them anyway).
  Future<List<DreamReport>> dream({
    required DreamAdjudicator adjudicate,
    int? budget,
    int? nowUnix,
    MemoryTimezone? timezone,
    bool force = false,
  }) =>
      _serialize(() => _dream(
            adjudicate,
            _resolveNow(nowUnix),
            _tzField(timezone),
            budget: budget,
            force: force,
          ));

  // ================================================================== //
  // introspection / lifecycle
  // ================================================================== //

  /// Row counts per tier plus ring lengths.
  Future<EngramStats> stats() => _serialize(() async => EngramStats(
        l1: await store.countMemories(tier: 1),
        l2: await store.countMemories(tier: 2),
        l3: await store.countMemories(tier: 3),
        tombstones:
            await store.countMemories(liveness: Liveness.tombstonesOnly),
        conflicts: await store.countConflicts(),
        dreamLog: await store.countDreamLog(),
      ));

  /// Number of live (non-tombstoned) memories.
  Future<int> totalRecords() =>
      _serialize(() => store.countMemories(liveness: Liveness.liveOnly));

  /// Current activation `A = mass·2^(−Δt/τ)` of memory [id], or `null` if
  /// it doesn't exist.
  Future<double?> activationOf(String id, {int? nowUnix}) =>
      _serialize(() async {
        final row = await store.getMemory(id);
        if (row == null) return null;
        return activationOfRecord(row, _resolveNow(nowUnix));
      });

  /// Dumps memory rows (for inspection / debugging UIs).
  Future<List<MemoryRecord>> listMemories({
    int? tier,
    bool includeTombstones = true,
  }) =>
      _serialize(() => store.listMemories(
            tier: tier,
            liveness: includeTombstones ? Liveness.all : Liveness.liveOnly,
          ));

  /// Erases everything in the store and reinstalls the metadata.
  Future<void> reset() => _serialize(() async {
        await store.clear();
        _writeTimes.clear();
        _docVecCache.clear();
        await _installMeta();
      });

  /// Current Unix time in seconds (explicit override → injected clock →
  /// wall clock).
  @override
  int nowUnix() => super.nowUnix();

  /// Current local datetime string (e.g. `'2026-06-12 09:30 +09:00'`) —
  /// the format expected by the bundled prompt templates.
  String nowLocal({int? nowUnix, MemoryTimezone? timezone}) =>
      formatLocal(_resolveNow(nowUnix), _tzField(timezone));

  Future<void> _installMeta() async {
    await store.putMeta('spec_version', 'ENGRAM v1.1');
    await store.putMeta('active_model', modelId);
    await store.putMeta(
        'four_sentences', '生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。破壊はすべて夢の中。');
  }
}
