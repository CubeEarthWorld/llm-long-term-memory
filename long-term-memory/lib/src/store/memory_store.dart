import 'dart:async';

import '../models.dart';

/// Row liveness filter for [MemoryStore] queries.
enum Liveness {
  /// Live and tombstoned rows.
  all,

  /// Only rows with `supersededBy == null`.
  liveOnly,

  /// Only tombstoned rows.
  tombstonesOnly,
}

/// A live memory joined with its cached vector — the retrieval hot path.
class TierEntry {
  /// Creates an entry.
  const TierEntry({required this.memory, required this.vector});

  /// The memory row.
  final MemoryRecord memory;

  /// Its vector under the active embedding model.
  final VectorRecord vector;
}

/// The storage interface — implemented by the application over any database.
///
/// The engine speaks only these typed operations; **no SQL or query language
/// leaks through**, so the interface is implementable over SQLite
/// (`package:sqlite3`, drift, sqflite), Isar, Hive, ObjectBox, a server, or
/// plain maps (see `InMemoryStore`). All ENGRAM semantics (tombstones,
/// capacity, rings, tier movement) live in the engine; an adapter carries
/// zero algorithm knowledge.
///
/// ### Implementing an adapter
///
/// Only the abstract members are required. The bulk operations
/// ([clampFutureTimestamps], [deleteTombstones], [deleteVectorsExceptModel],
/// [deleteOrphanVectors], [listTierEntries]) have default implementations in
/// terms of the primitives — override them when your backend can do the same
/// in one statement. The hooks ([runInTransaction], [backup], [compact],
/// [sizeInBytes]) default to no-ops.
///
/// **Transactionality**: the engine wraps each dream adjudication
/// (delete cluster + insert replacements) in [runInTransaction]. On a
/// non-transactional store a crash inside that window can lose members
/// without their replacement; implement [runInTransaction] and [backup] if
/// your data matters.
abstract class MemoryStore {
  /// Opens / migrates the backing storage. Called once by the engine's
  /// `initialize()`.
  Future<void> initialize() async {}

  /// Releases resources.
  Future<void> close() async {}

  // ------------------------------------------------------------------ //
  // memory rows
  // ------------------------------------------------------------------ //

  /// Inserts a new memory row. Ids are ULIDs and never collide.
  Future<void> insertMemory(MemoryRecord record);

  /// Returns the row with [id], or `null`.
  Future<MemoryRecord?> getMemory(String id);

  /// Returns a **live** row whose text equals [text] exactly, or `null`
  /// (exact-text rehearsal, §5.1).
  Future<MemoryRecord?> getLiveMemoryByText(String text);

  /// Lists rows, optionally filtered by [tier] (1–3) and [liveness].
  Future<List<MemoryRecord>> listMemories({
    int? tier,
    Liveness liveness = Liveness.all,
  });

  /// Counts rows under the same filters as [listMemories].
  Future<int> countMemories({
    int? tier,
    Liveness liveness = Liveness.all,
  }) async =>
      (await listMemories(tier: tier, liveness: liveness)).length;

  /// Replaces the row with `record.id` (full-row update).
  Future<void> updateMemory(MemoryRecord record);

  /// Physically deletes the row **and all its vectors** (cascade).
  Future<void> deleteMemory(String id);

  // ------------------------------------------------------------------ //
  // vectors
  // ------------------------------------------------------------------ //

  /// Inserts or replaces the vector for `(vector.memoryId, vector.modelId)` —
  /// at most one row per pair (I6).
  Future<void> upsertVector(VectorRecord vector);

  /// Returns the vector for the pair, or `null`.
  Future<VectorRecord?> getVector(String memoryId, String modelId);

  /// Lists vectors, optionally restricted to [modelId].
  Future<List<VectorRecord>> listVectors({String? modelId});

  // ------------------------------------------------------------------ //
  // conflict ring (I11)
  // ------------------------------------------------------------------ //

  /// Appends a conflict pair.
  Future<void> addConflict(ConflictRecord conflict);

  /// All conflict entries, oldest first.
  Future<List<ConflictRecord>> listConflicts();

  /// Conflict-ring length.
  Future<int> countConflicts() async => (await listConflicts()).length;

  /// Drops the oldest entries beyond [maxEntries].
  Future<void> trimConflicts(int maxEntries);

  // ------------------------------------------------------------------ //
  // dream log ring (I11, I12)
  // ------------------------------------------------------------------ //

  /// Returns the recorded verdict for [fingerprint], or `null`.
  Future<String?> getDreamVerdict(String fingerprint);

  /// Inserts or replaces the verdict for `record.fingerprint`.
  Future<void> putDreamVerdict(DreamLogRecord record);

  /// Dream-log length.
  Future<int> countDreamLog();

  /// Drops the oldest entries (by `at`) beyond [maxEntries].
  Future<void> trimDreamLog(int maxEntries);

  // ------------------------------------------------------------------ //
  // metadata
  // ------------------------------------------------------------------ //

  /// Returns the metadata value for [key], or `null`.
  Future<String?> getMeta(String key);

  /// Sets a metadata value (engine stores e.g. `spec_version`,
  /// `active_model`).
  Future<void> putMeta(String key, String value);

  /// Deletes everything (all memories, vectors, rings, metadata).
  Future<void> clear();

  // ------------------------------------------------------------------ //
  // bulk operations — default implementations; override for efficiency
  // ------------------------------------------------------------------ //

  /// Clamps any timestamp fields greater than [nowUnix] down to [nowUnix]
  /// (clock-rollback sanitisation, I2).
  Future<void> clampFutureTimestamps(int nowUnix) async {
    for (final m in await listMemories()) {
      if (m.lastAccess > nowUnix ||
          m.lastBonusAt > nowUnix ||
          m.createdAt > nowUnix) {
        await updateMemory(m.copyWith(
          lastAccess: m.lastAccess > nowUnix ? nowUnix : m.lastAccess,
          lastBonusAt: m.lastBonusAt > nowUnix ? nowUnix : m.lastBonusAt,
          createdAt: m.createdAt > nowUnix ? nowUnix : m.createdAt,
        ));
      }
    }
  }

  /// Physically deletes tombstoned rows, optionally restricted to [tier]
  /// and/or to rows whose `lastAccess` is before [lastAccessBefore].
  Future<void> deleteTombstones({int? tier, int? lastAccessBefore}) async {
    final rows =
        await listMemories(tier: tier, liveness: Liveness.tombstonesOnly);
    for (final m in rows) {
      if (lastAccessBefore == null || m.lastAccess < lastAccessBefore) {
        await deleteMemory(m.id);
      }
    }
  }

  /// Deletes all vectors not belonging to [modelId] (dream housekeeping GC
  /// after an embedding-model switch).
  Future<void> deleteVectorsExceptModel(String modelId) async {
    final stale = (await listVectors())
        .where((v) => v.modelId != modelId)
        .toList(growable: false);
    for (final v in stale) {
      await deleteVector(v.memoryId, v.modelId);
    }
  }

  /// Deletes a single vector row. Used by the default bulk implementations;
  /// adapters overriding all bulk methods may leave this unimplemented only
  /// if they also override the bulks.
  Future<void> deleteVector(String memoryId, String modelId);

  /// Deletes vectors whose memory row no longer exists.
  Future<void> deleteOrphanVectors() async {
    final ids = (await listMemories()).map((m) => m.id).toSet();
    final orphans = (await listVectors())
        .where((v) => !ids.contains(v.memoryId))
        .toList(growable: false);
    for (final v in orphans) {
      await deleteVector(v.memoryId, v.modelId);
    }
  }

  /// Live rows of [tier] joined with their [modelId] vectors — the retrieval
  /// hot path; override with a JOIN for large stores.
  Future<List<TierEntry>> listTierEntries(int tier, String modelId) async {
    final rows = await listMemories(tier: tier, liveness: Liveness.liveOnly);
    final out = <TierEntry>[];
    for (final m in rows) {
      final v = await getVector(m.id, modelId);
      if (v != null) out.add(TierEntry(memory: m, vector: v));
    }
    return out;
  }

  // ------------------------------------------------------------------ //
  // optional hooks
  // ------------------------------------------------------------------ //

  /// Runs [action] atomically if the backend supports transactions.
  /// Defaults to a plain passthrough.
  Future<T> runInTransaction<T>(Future<T> Function() action) => action();

  /// Takes a backup/snapshot before destructive dream work (e.g. a rotating
  /// file ring for SQLite). Defaults to a no-op.
  Future<void> backup() async {}

  /// Compaction hook (e.g. WAL checkpoint + incremental vacuum). Called
  /// periodically by `maintain()` and at the end of `dream()`.
  Future<void> compact() async {}

  /// Size of the backing storage in bytes, or -1 if unknown.
  Future<int> sizeInBytes() async => -1;
}
