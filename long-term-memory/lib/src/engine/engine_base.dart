part of 'engine.dart';

/// Max entries of the per-id full-precision re-embed cache (§5.1).
const int _docVecCacheMax = 4096;

/// A live memory row paired with its decoded unit-norm vector.
class _Entry {
  _Entry(this.row, this.vec);
  final MemoryRecord row;
  final Float32List vec;
}

/// Best identity-scan hit.
class _ScanHit {
  _ScanHit(this.row, this.cos);
  final MemoryRecord row;
  final double cos;
}

/// Shared state and low-level vector/activation operations (port of the
/// reference `vector_ops` mixin plus the engine facade's clock handling).
abstract class _EngineBase {
  _EngineBase({
    required this.store,
    required this.embedder,
    this.config = const EngramConfig(),
    int Function()? clock,
    MemoryTimezone? defaultTimezone,
  })  : _clock = clock,
        _defaultTz = defaultTimezone {
    _ulid = UlidGenerator(millis: () => nowUnix() * 1000);
  }

  /// The injected storage backend.
  final MemoryStore store;

  /// The injected embedding model.
  final Embedder embedder;

  /// Active parameters.
  final EngramConfig config;

  final int Function()? _clock;
  final MemoryTimezone? _defaultTz;
  late final UlidGenerator _ulid;

  final Map<String, Float32List> _docVecCache = {};
  final List<int> _writeTimes = [];
  int _maintCount = 0;
  Future<void> _chain = Future<void>.value();

  /// The active embedding model id stamped onto vectors.
  String get modelId => embedder.modelId;

  /// Serializes public pipelines: the reference engine is single-threaded,
  /// and the multi-step save/retrieve/dream pipelines must not interleave.
  Future<T> _serialize<T>(Future<T> Function() action) {
    final run = _chain.then((_) => action());
    _chain = run.then<void>((_) {}, onError: (Object _) {});
    return run;
  }

  /// Current Unix time in seconds.
  int nowUnix() {
    final c = _clock;
    if (c != null) return c();
    return DateTime.now().millisecondsSinceEpoch ~/ 1000;
  }

  int _resolveNow(int? explicit) => explicit ?? nowUnix();

  MemoryTimezone _resolveTz(MemoryTimezone? tz) =>
      tz ?? _defaultTz ?? MemoryTimezone.utc;

  /// The stored `'IANA;+HH:MM'` field for a write at the resolved timezone.
  String _tzField(MemoryTimezone? tz) => _resolveTz(tz).storageField;

  /// Clamps future timestamps so a clock rollback cannot mint immortal
  /// mass (I2).
  Future<void> _sanitizeTimestamps(int now) => store.clampFutureTimestamps(now);

  // ================================================================== //
  // activation A (§4.1)
  // ================================================================== //

  /// `A(now) = mass · 2^(−max(0, now−lastAccess)/τ_tier)`; underflow → 0.
  double activationOfRecord(MemoryRecord row, int now) {
    final dt = math.max(0, now - row.lastAccess).toDouble();
    final exp = dt / config.tauOf(row.tier);
    if (exp > config.decayExpCap) return 0.0;
    return math.min(row.mass * math.pow(2.0, -exp).toDouble(), config.mMax);
  }

  /// Recall update with the §4.1 ordering: fold decay into mass first,
  /// then apply the refractory-gated bonus, then touch `lastAccess`.
  Future<MemoryRecord> _applyRecall(MemoryRecord row, int now) async {
    var mass = activationOfRecord(row, now);
    var lastBonus = row.lastBonusAt;
    if (now - lastBonus >= config.refractorySeconds) {
      mass = math.min(mass + 1.0, config.mMax);
      lastBonus = now;
    }
    final updated =
        row.copyWith(mass: mass, lastAccess: now, lastBonusAt: lastBonus);
    await store.updateMemory(updated);
    return updated;
  }

  /// Exact-text rehearsal (§5.1.2): definite +1, bypassing the refractory
  /// gate.
  Future<MemoryRecord> _reinforceWrite(MemoryRecord row, int now) async {
    final mass = math.min(activationOfRecord(row, now) + 1.0, config.mMax);
    final updated = row.copyWith(mass: mass, lastAccess: now, lastBonusAt: now);
    await store.updateMemory(updated);
    return updated;
  }

  // ================================================================== //
  // vectors (§3.3, §8 MRL + int8)
  // ================================================================== //

  /// Embeds [text] on the document side at full precision (normalised,
  /// truncated to `dim1`).
  Future<Float32List> _embedDoc(String text) async {
    final raw = await embedder.embedDocuments([text]);
    return truncateNormalize(l2Normalized(raw[0]), config.dim1);
  }

  /// Full-precision document vector for a stored row, cached per id (text
  /// is immutable for a given id). Cleared wholesale at the size cap.
  Future<Float32List> _docVecCached(MemoryRecord row) async {
    final cached = _docVecCache[row.id];
    if (cached != null) return cached;
    if (_docVecCache.length >= _docVecCacheMax) _docVecCache.clear();
    final v = await _embedDoc(row.text);
    _docVecCache[row.id] = v;
    return v;
  }

  /// Encodes and stores the per-tier vector for memory [mid]: L1 keeps
  /// float32 at `dim1`; L2/L3 are MRL-truncated and int8-quantised.
  Future<void> _storeVec(String mid, Float32List vFull, int tier) async {
    final dim = config.dimOf(tier);
    final vt = truncateNormalize(vFull, dim);
    if (tier == 1) {
      await store.upsertVector(VectorRecord(
        memoryId: mid,
        modelId: modelId,
        dim: dim,
        dtype: VecDtype.f32,
        bytes: packF32(vt),
      ));
    } else {
      final q = quantizeInt8(vt);
      await store.upsertVector(VectorRecord(
        memoryId: mid,
        modelId: modelId,
        dim: dim,
        dtype: VecDtype.int8,
        scale: q.scale,
        bytes: q.bytes,
      ));
    }
  }

  /// Decodes a stored vector into a unit-norm float32 array.
  Float32List _decodeVec(VectorRecord v) => l2Normalized(
        v.dtype == VecDtype.f32
            ? unpackF32(v.bytes)
            : dequantizeInt8(v.bytes, v.scale ?? 1.0),
      );

  /// Coarse common-resolution vector used for clustering and MMR
  /// (`dim3`, 128 by default).
  Float32List _coarse(Float32List vec) => truncateNormalize(vec, config.dim3);

  /// Live rows of [tier] with decoded unit-norm vectors.
  Future<List<_Entry>> _tierCandidates(int tier) async {
    final entries = await store.listTierEntries(tier, modelId);
    return [
      for (final e in entries) _Entry(e.memory, _decodeVec(e.vector)),
    ];
  }
}
