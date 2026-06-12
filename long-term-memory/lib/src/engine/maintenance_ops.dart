part of 'engine.dart';

/// Tier movement, capacity enforcement and per-turn maintenance (§5.4, I4).
mixin _MaintenanceOps on _EngineBase {
  /// Per-turn maintenance: tombstone sweep + capacity enforcement.
  /// No LLM, no embedding (stored vectors are reused for demotion).
  Future<void> _maintain(int now) async {
    await _sanitizeTimestamps(now);
    await _sweepTombstones(now);
    await _enforceCapacity(now);
    _maintCount += 1;
    if (_maintCount % config.compactEvery == 0) {
      await store.compact();
    }
  }

  /// A ≥ θ_up L2/L3 rows → re-embed text at full precision → L1
  /// (re-fixation; text is canonical). Dream-time only (needs embedding).
  Future<void> _promote(int now) async {
    for (final tier in const [2, 3]) {
      final rows =
          await store.listMemories(tier: tier, liveness: Liveness.liveOnly);
      for (final row in rows) {
        if (activationOfRecord(row, now) < config.thetaUp) continue;
        final v = await _docVecCached(row);
        await store.updateMemory(row.copyWith(tier: 1));
        await _storeVec(row.id, v, 1);
      }
    }
  }

  /// Keeps each tier within capacity: purge tombstones first, then demote
  /// (L1→L2→L3) or evict (L3 has nowhere to go — eviction is death).
  Future<void> _enforceCapacity(int now) async {
    await _shrinkTier(1, config.cap1, now, (id) => _demote(id, 2));
    await _shrinkTier(2, config.cap2, now, (id) => _demote(id, 3));
    await _shrinkTier(3, config.cap3, now, (id) => store.deleteMemory(id));
  }

  Future<void> _shrinkTier(
    int tier,
    int cap,
    int now,
    Future<void> Function(String id) onVictim,
  ) async {
    var guard = 0;
    while (await store.countMemories(tier: tier) > cap &&
        guard < config.hardMemoryRows) {
      guard += 1;
      final tombs = await store.listMemories(
          tier: tier, liveness: Liveness.tombstonesOnly);
      if (tombs.isNotEmpty) {
        await store.deleteMemory(tombs.first.id);
        continue;
      }
      final vid = await _victim(tier, now);
      if (vid == null) break;
      await onVictim(vid);
    }
  }

  /// Lowest-activation live row in [tier], preferring rows with A < θ_down.
  Future<String?> _victim(int tier, int now) async {
    final rows =
        await store.listMemories(tier: tier, liveness: Liveness.liveOnly);
    if (rows.isEmpty) return null;
    MemoryRecord? best;
    var bestA = double.infinity;
    var bestUnder = false;
    for (final r in rows) {
      final a = activationOfRecord(r, now);
      final under = a < config.thetaDown;
      // Under-threshold rows always beat over-threshold rows.
      if (best == null ||
          (under && !bestUnder) ||
          (under == bestUnder && a < bestA)) {
        best = r;
        bestA = a;
        bestUnder = under;
      }
    }
    return best?.id;
  }

  /// Demotes a memory by MRL-truncating its **stored** vector (no
  /// re-embedding) and requantising to int8 (§5.4).
  Future<void> _demote(String id, int toTier) async {
    final vr = await store.getVector(id, modelId);
    if (vr == null) return;
    final v = _decodeVec(vr);
    final vt = truncateNormalize(v, config.dimOf(toTier));
    final q = quantizeInt8(vt);
    await store.upsertVector(VectorRecord(
      memoryId: id,
      modelId: modelId,
      dim: config.dimOf(toTier),
      dtype: VecDtype.int8,
      scale: q.scale,
      bytes: q.bytes,
    ));
    final row = await store.getMemory(id);
    if (row != null) await store.updateMemory(row.copyWith(tier: toTier));
  }

  /// Purges tombstones older than the sweep age, plus any tier whose
  /// tombstones exceed the sweep percentage (§6 housekeeping).
  Future<void> _sweepTombstones(int now) async {
    final cutoff = now - config.tombstoneSweepAge.toInt();
    await store.deleteTombstones(lastAccessBefore: cutoff);
    for (var tier = 1; tier <= 3; tier++) {
      final total = await store.countMemories(tier: tier);
      if (total <= 0) continue;
      final tomb = await store.countMemories(
          tier: tier, liveness: Liveness.tombstonesOnly);
      if (tomb > config.tombstoneSweepPct * total) {
        await store.deleteTombstones(tier: tier);
      }
    }
  }
}
