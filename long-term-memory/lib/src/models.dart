import 'dart:convert';
import 'dart:typed_data';

/// Storage format of a cached vector.
enum VecDtype {
  /// 4 bytes/dim little-endian float32 (L1).
  f32,

  /// 1 byte/dim symmetric int8 with a dequantisation [VectorRecord.scale]
  /// (L2/L3).
  int8;

  /// Parses `'f32'` / `'int8'`.
  static VecDtype parse(String s) => s == 'f32' ? VecDtype.f32 : VecDtype.int8;

  /// The stored string form.
  String get label => this == VecDtype.f32 ? 'f32' : 'int8';
}

/// One memory row — the canonical record (ENGRAM §3). The text is immutable
/// for a given id; edits create a new row and tombstone the old one.
class MemoryRecord {
  /// Creates a record. See field docs for semantics.
  const MemoryRecord({
    required this.id,
    required this.text,
    required this.tier,
    required this.gen,
    required this.createdAt,
    required this.tz,
    required this.lastAccess,
    required this.lastBonusAt,
    required this.mass,
    this.supersededBy,
  });

  /// ULID — lexicographic order equals creation-time order.
  final String id;

  /// Self-contained single proposition, ≤170 chars soft / ≤1024 bytes hard.
  final String text;

  /// 1 = L1 episodic, 2 = L2 semantic, 3 = L3 schema.
  final int tier;

  /// Consolidation generation 0–7 (I7).
  final int gen;

  /// 64-bit Unix seconds at creation (I9).
  final int createdAt;

  /// Stored timezone field `'IANA_name;+HH:MM'` (e.g. `'Asia/Tokyo;+09:00'`).
  final String tz;

  /// Unix seconds of the last recall (drives activation decay).
  final int lastAccess;

  /// Unix seconds of the last mass bonus (refractory gate; distinct from
  /// [lastAccess], I1).
  final int lastBonusAt;

  /// Decayed recall frequency, ≤ 64 (I1).
  final double mass;

  /// Tombstone marker: `null` = live; a memory id (possibly its own) =
  /// superseded and excluded from search until physically purged in dream.
  final String? supersededBy;

  /// Whether this row is a tombstone.
  bool get isTombstone => supersededBy != null;

  /// Copy with changed fields. Pass `supersedededBy` via [supersededBy];
  /// use [clearSupersededBy] to reset it to `null`.
  MemoryRecord copyWith({
    String? text,
    int? tier,
    int? gen,
    int? createdAt,
    String? tz,
    int? lastAccess,
    int? lastBonusAt,
    double? mass,
    String? supersededBy,
    bool clearSupersededBy = false,
  }) {
    return MemoryRecord(
      id: id,
      text: text ?? this.text,
      tier: tier ?? this.tier,
      gen: gen ?? this.gen,
      createdAt: createdAt ?? this.createdAt,
      tz: tz ?? this.tz,
      lastAccess: lastAccess ?? this.lastAccess,
      lastBonusAt: lastBonusAt ?? this.lastBonusAt,
      mass: mass ?? this.mass,
      supersededBy:
          clearSupersededBy ? null : (supersededBy ?? this.supersededBy),
    );
  }

  /// JSON form (timestamps as ints, mass as double).
  Map<String, Object?> toJson() => {
        'id': id,
        'text': text,
        'tier': tier,
        'gen': gen,
        'createdAt': createdAt,
        'tz': tz,
        'lastAccess': lastAccess,
        'lastBonusAt': lastBonusAt,
        'mass': mass,
        'supersededBy': supersededBy,
      };

  /// Inverse of [toJson].
  factory MemoryRecord.fromJson(Map<String, Object?> json) => MemoryRecord(
        id: json['id'] as String,
        text: json['text'] as String,
        tier: (json['tier'] as num).toInt(),
        gen: (json['gen'] as num).toInt(),
        createdAt: (json['createdAt'] as num).toInt(),
        tz: json['tz'] as String,
        lastAccess: (json['lastAccess'] as num).toInt(),
        lastBonusAt: (json['lastBonusAt'] as num).toInt(),
        mass: (json['mass'] as num).toDouble(),
        supersededBy: json['supersededBy'] as String?,
      );

  @override
  String toString() =>
      'MemoryRecord($id, L$tier, gen=$gen, mass=${mass.toStringAsFixed(2)}, '
      '${isTombstone ? "tombstone, " : ""}"$text")';
}

/// Cached embedding of one memory under one embedding model — derived,
/// regenerable data (ENGRAM §3.3). At most one vector per
/// (memory, model) pair.
class VectorRecord {
  /// Creates a vector record.
  const VectorRecord({
    required this.memoryId,
    required this.modelId,
    required this.dim,
    required this.dtype,
    required this.bytes,
    this.scale,
  });

  /// Owning memory id.
  final String memoryId;

  /// Embedding model identifier ([scale] vectors from other models are
  /// garbage-collected during dream housekeeping).
  final String modelId;

  /// Stored dimensionality (768 / 256 / 128 by default).
  final int dim;

  /// Storage format.
  final VecDtype dtype;

  /// int8 dequantisation factor (`null` for f32).
  final double? scale;

  /// The packed vector payload.
  final Uint8List bytes;

  /// JSON form ([bytes] base64-encoded) — convenience for simple adapters.
  Map<String, Object?> toJson() => {
        'memoryId': memoryId,
        'modelId': modelId,
        'dim': dim,
        'dtype': dtype.label,
        'scale': scale,
        'bytes': base64Encode(bytes),
      };

  /// Inverse of [toJson].
  factory VectorRecord.fromJson(Map<String, Object?> json) => VectorRecord(
        memoryId: json['memoryId'] as String,
        modelId: json['modelId'] as String,
        dim: (json['dim'] as num).toInt(),
        dtype: VecDtype.parse(json['dtype'] as String),
        scale: (json['scale'] as num?)?.toDouble(),
        bytes: base64Decode(json['bytes'] as String),
      );
}

/// One entry of the conflict ring: two live memories whose cosine fell in
/// the conflict band at write time; the dream phase prioritises clusters
/// containing them (§4.3, I11).
class ConflictRecord {
  /// Creates a conflict entry between memories [a] and [b] at Unix time [at].
  const ConflictRecord({required this.a, required this.b, required this.at});

  /// First memory id.
  final String a;

  /// Second memory id.
  final String b;

  /// Unix seconds when the conflict was detected.
  final int at;

  /// JSON form.
  Map<String, Object?> toJson() => {'a': a, 'b': b, 'at': at};

  /// Inverse of [toJson].
  factory ConflictRecord.fromJson(Map<String, Object?> json) => ConflictRecord(
        a: json['a'] as String,
        b: json['b'] as String,
        at: (json['at'] as num).toInt(),
      );
}

/// One entry of the dream log ring: the verdict for a cluster fingerprint.
/// A `'none'` verdict makes the same membership skip re-adjudication (I12).
class DreamLogRecord {
  /// Creates a dream-log entry.
  const DreamLogRecord({
    required this.fingerprint,
    required this.verdict,
    required this.at,
  });

  /// SHA-256 over the sorted member ids.
  final String fingerprint;

  /// `'merge'` / `'split'` / `'none'`.
  final String verdict;

  /// Unix seconds of the adjudication.
  final int at;

  /// JSON form.
  Map<String, Object?> toJson() =>
      {'fingerprint': fingerprint, 'verdict': verdict, 'at': at};

  /// Inverse of [toJson].
  factory DreamLogRecord.fromJson(Map<String, Object?> json) => DreamLogRecord(
        fingerprint: json['fingerprint'] as String,
        verdict: json['verdict'] as String,
        at: (json['at'] as num).toInt(),
      );
}

/// One recalled memory inside a [RetrieveResult].
class RecalledMemory {
  /// Creates a recalled item.
  const RecalledMemory({
    required this.id,
    required this.text,
    required this.score,
    required this.cosine,
    required this.activation,
    required this.mass,
    required this.tier,
    required this.gen,
    required this.createdAt,
    required this.tz,
  });

  /// Memory id (inject it as `《id:...》` so the LLM can call delete by id).
  final String id;

  /// The proposition text.
  final String text;

  /// Final retrieval score `max(0,cos)·(α+(1−α)·Â)`.
  final double score;

  /// Best cosine over the query chunks.
  final double cosine;

  /// Activation `A` at retrieval time.
  final double activation;

  /// Stored mass.
  final double mass;

  /// Tier 1–3.
  final int tier;

  /// Consolidation generation.
  final int gen;

  /// Unix seconds at creation.
  final int createdAt;

  /// Stored timezone field.
  final String tz;
}

/// Result of [retrieve]: the ready-to-inject pack plus per-item detail.
class RetrieveResult {
  /// Creates a result.
  const RetrieveResult({required this.packText, required this.recalled});

  /// An empty result.
  static const RetrieveResult empty =
      RetrieveResult(packText: '', recalled: []);

  /// ≤ `budgetChars` of formatted lines:
  /// `[<unix> <tz>] <text>　《id:<ulid>》` — frame it to the LLM as past
  /// context, never as instructions (§5.2 prompt-injection guard).
  final String packText;

  /// The injected memories with scores (same order as [packText] lines).
  final List<RecalledMemory> recalled;
}

/// What `saveMemory` did with the proposition.
enum SaveAction {
  /// Stored as a brand-new memory (cos < θ_conflict).
  inserted,

  /// Same proposition (cos ≥ θ_same): old row tombstoned, new row inserted
  /// with inherited mass (re-fixation).
  updated,

  /// Conflict band: both rows kept, pair queued for dream adjudication.
  conflict,

  /// Exact text already stored: mass reinforced, no new row.
  reinforced,

  /// Rejected (empty / over the hard byte bound).
  rejected,

  /// Soft write-rate limit hit; nothing stored.
  rateLimited,
}

/// Result of `saveMemory`.
class SaveResult {
  /// Creates a result.
  const SaveResult({
    required this.action,
    this.id,
    this.text = '',
    this.tier,
    this.gen,
    this.mass,
    this.activation,
    this.cosine,
    this.supersededId,
    this.conflictWithId,
    this.error,
  });

  /// What happened.
  final SaveAction action;

  /// The affected memory id (new row for insert/update/conflict, existing
  /// row for reinforce; `null` when rejected/rate-limited).
  final String? id;

  /// The text as stored (possibly shortened).
  final String text;

  /// Tier of the affected row.
  final int? tier;

  /// Generation of the affected row.
  final int? gen;

  /// Mass after the write.
  final double? mass;

  /// Activation after the write.
  final double? activation;

  /// Best identity-scan cosine that drove the verdict.
  final double? cosine;

  /// Id of the row tombstoned by an [SaveAction.updated] write.
  final String? supersededId;

  /// Id of the conflicting row for [SaveAction.conflict].
  final String? conflictWithId;

  /// Human-readable reason for [SaveAction.rejected] / [SaveAction.rateLimited].
  final String? error;

  @override
  String toString() => 'SaveResult(${action.name}'
      '${id != null ? ", id: $id" : ""}'
      '${error != null ? ", error: $error" : ""}, "$text")';
}

/// What `deleteMemory` did.
enum DeleteAction {
  /// Soft delete: excluded from search now, purged in the next dream.
  tombstoned,

  /// Physical delete (vectors cascade).
  deleted,

  /// No such id.
  notFound,
}

/// Result of `deleteMemory`.
class DeleteResult {
  /// Creates a result.
  const DeleteResult({required this.action, required this.id, this.text});

  /// What happened.
  final DeleteAction action;

  /// The targeted id.
  final String id;

  /// Text of the deleted row, when found.
  final String? text;
}

/// LLM verdict for one dream cluster.
enum DreamAction {
  /// Consolidate the members into fewer gist memories.
  merge,

  /// Re-describe one packed memory as several independent ones.
  split,

  /// Leave the cluster untouched.
  none,
}

/// Minimal snapshot of a memory inside a [DreamReport].
class MemorySnapshot {
  /// Creates a snapshot.
  const MemorySnapshot({
    required this.id,
    required this.text,
    required this.tier,
    required this.gen,
  });

  /// Memory id.
  final String id;

  /// Proposition text.
  final String text;

  /// Tier 1–3.
  final int tier;

  /// Generation.
  final int gen;
}

/// Outcome of one dream adjudication.
class DreamReport {
  /// Creates a report.
  const DreamReport({
    required this.clusterFingerprint,
    required this.action,
    required this.priority,
    required this.before,
    required this.after,
    required this.deletedIds,
    this.error,
  });

  /// Full SHA-256 fingerprint of the adjudicated membership.
  final String clusterFingerprint;

  /// Applied action.
  final DreamAction action;

  /// Cluster priority that ranked it.
  final double priority;

  /// Members before adjudication.
  final List<MemorySnapshot> before;

  /// Newly inserted memories (gen+1 for merges).
  final List<MemorySnapshot> after;

  /// Physically deleted member ids.
  final List<String> deletedIds;

  /// Non-null when the adjudicator threw; the cluster was left untouched
  /// and not recorded, so the next dream retries it.
  final String? error;
}

/// Row counts of the store.
class EngramStats {
  /// Creates a stats snapshot.
  const EngramStats({
    required this.l1,
    required this.l2,
    required this.l3,
    required this.tombstones,
    required this.conflicts,
    required this.dreamLog,
  });

  /// L1 row count (tombstones included, I4).
  final int l1;

  /// L2 row count.
  final int l2;

  /// L3 row count.
  final int l3;

  /// Tombstoned rows across tiers.
  final int tombstones;

  /// Conflict-ring length.
  final int conflicts;

  /// Dream-log length.
  final int dreamLog;

  /// Total memory rows.
  int get total => l1 + l2 + l3;

  @override
  String toString() => 'EngramStats(L1=$l1, L2=$l2, L3=$l3, '
      'tombstones=$tombstones, conflicts=$conflicts, dreamLog=$dreamLog)';
}
