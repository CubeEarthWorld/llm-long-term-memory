part of 'engine.dart';

/// WRITE/DELETE pipeline (§5.1, §5.3) — port of the reference `write_ops`.
mixin _WriteOps on _EngineBase {
  Future<SaveResult> _saveMemory(String text, int now, String tzField) async {
    text = text.trim();
    if (text.isEmpty) {
      return const SaveResult(
          action: SaveAction.rejected, error: 'empty text', text: '');
    }
    final nbytes = utf8.encode(text).length;
    if (nbytes > config.textHardMax) {
      return SaveResult(
        action: SaveAction.rejected,
        error: 'text too long (> ${config.textHardMax} bytes)',
        text: text.substring(0, math.min(60, text.length)),
      );
    }
    if (text.length > config.textMax) text = shorten(text, config.textMax);
    if (!_writeRateOk(now)) {
      return SaveResult(
          action: SaveAction.rateLimited,
          error: 'daily write-rate limit reached',
          text: text);
    }

    // 1) Exact-text rehearsal — reinforce, no new row.
    final dup = await store.getLiveMemoryByText(text);
    if (dup != null) {
      final updated = await _reinforceWrite(dup, now);
      return _saveEvent(SaveAction.reinforced, updated, now);
    }

    // 2) Embed (document side) and scan all tiers (§5.1.3).
    final v = await _embedDoc(text);
    final best = await _identityScan(v);
    _writeTimes.add(now);

    if (best != null && best.cos >= config.thetaSame) {
      // Same proposition: tombstone old, insert new with inherited mass.
      final carry =
          math.min(activationOfRecord(best.row, now) + 1.0, config.mMax);
      final row = await _insertWithVec(text, v, carry, now, tzField);
      await store.updateMemory(best.row.copyWith(supersededBy: row.id));
      return _saveEvent(SaveAction.updated, row, now,
          cosine: best.cos, supersededId: best.row.id);
    }

    if (best != null && best.cos >= config.thetaConflict) {
      // Conflict band: keep both, enqueue for dream adjudication.
      final row = await _insertWithVec(text, v, 1.0, now, tzField);
      await store
          .addConflict(ConflictRecord(a: best.row.id, b: row.id, at: now));
      await store.trimConflicts(config.conflictCap);
      return _saveEvent(SaveAction.conflict, row, now,
          cosine: best.cos, conflictWithId: best.row.id);
    }

    final row = await _insertWithVec(text, v, 1.0, now, tzField);
    return _saveEvent(SaveAction.inserted, row, now, cosine: best?.cos ?? 0.0);
  }

  /// Best `(row, cosine)` across all tiers; L2/L3 candidates near the
  /// conflict band are refined with a full-precision re-embed (§5.1).
  Future<_ScanHit?> _identityScan(Float32List vFull) async {
    final coarseBest = <String, (double, MemoryRecord, int)>{};
    for (var tier = 1; tier <= 3; tier++) {
      final entries = await _tierCandidates(tier);
      if (entries.isEmpty) continue;
      final qt = truncateNormalize(vFull, config.dimOf(tier));
      for (final e in entries) {
        final c = dot(e.vec, qt);
        final prev = coarseBest[e.row.id];
        if (prev == null || c > prev.$1) {
          coarseBest[e.row.id] = (c, e.row, tier);
        }
      }
    }
    _ScanHit? best;
    final band = config.thetaConflict - config.preciseMargin;
    for (final (c, row, tier) in coarseBest.values) {
      var cos = c;
      if (tier != 1 && c >= band) {
        // Text is canonical → re-embed at full precision.
        cos = dot(vFull, await _docVecCached(row));
      }
      if (best == null || cos > best.cos) best = _ScanHit(row, cos);
    }
    return best;
  }

  /// Inserts a new memory row plus its tier vector. [vFull] must already be
  /// the full-precision document embedding of [text].
  Future<MemoryRecord> _insertWithVec(
    String text,
    Float32List vFull,
    double mass,
    int now,
    String tzField, {
    int gen = 0,
    int tier = 1,
  }) async {
    final row = MemoryRecord(
      id: _ulid.next(),
      text: text,
      tier: tier,
      gen: gen,
      createdAt: now,
      tz: tzField,
      lastAccess: now,
      lastBonusAt: now,
      mass: mass,
    );
    await store.insertMemory(row);
    await _storeVec(row.id, vFull, tier);
    return row;
  }

  bool _writeRateOk(int now) {
    final dayAgo = now - 86400;
    _writeTimes.removeWhere((t) => t < dayAgo);
    return _writeTimes.length < config.writeRatePerDay;
  }

  SaveResult _saveEvent(
    SaveAction action,
    MemoryRecord row,
    int now, {
    double? cosine,
    String? supersededId,
    String? conflictWithId,
  }) =>
      SaveResult(
        action: action,
        id: row.id,
        text: row.text,
        tier: row.tier,
        gen: row.gen,
        mass: row.mass,
        activation: activationOfRecord(row, now),
        cosine: cosine,
        supersededId: supersededId,
        conflictWithId: conflictWithId,
      );

  // ================================================================== //
  // DELETE (§5.3)
  // ================================================================== //

  Future<DeleteResult> _deleteMemory(String id, {required bool hard}) async {
    final row = await store.getMemory(id);
    if (row == null) {
      return DeleteResult(action: DeleteAction.notFound, id: id);
    }
    if (hard) {
      await store.runInTransaction(() => store.deleteMemory(id));
      return DeleteResult(action: DeleteAction.deleted, id: id, text: row.text);
    }
    // Conversation-time delete = self-referential tombstone, purged in the
    // next dream.
    await store.updateMemory(row.copyWith(supersededBy: id));
    return DeleteResult(
        action: DeleteAction.tombstoned, id: id, text: row.text);
  }
}
