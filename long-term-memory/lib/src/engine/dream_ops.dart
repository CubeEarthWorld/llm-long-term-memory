part of 'engine.dart';

class _DreamCluster {
  _DreamCluster(this.fingerprint, this.members, this.priority);
  final String fingerprint;
  final List<MemoryRecord> members;
  final double priority;
}

/// DREAM consolidation (§6) — offline, the only destructive phase.
mixin _DreamOps on _EngineBase, _WriteOps, _MaintenanceOps {
  Future<List<DreamReport>> _dream(
    DreamAdjudicator adjudicate,
    int now,
    String tzField, {
    int? budget,
    bool force = false,
  }) async {
    await _sanitizeTimestamps(now);
    final b = (budget ?? config.dreamBudget).clamp(0, config.dreamBudgetHard);

    await store.backup();
    await _housekeeping(now);

    final reports = <DreamReport>[];
    final clusters = await _dreamCandidates(now, force: force);
    for (final cluster in clusters.take(b)) {
      reports.add(await _adjudicate(adjudicate, cluster, now, tzField));
    }

    await _promote(now);
    await _enforceCapacity(now);
    await store.compact();
    return reports;
  }

  /// Cheap, LLM-free chores: tombstone sweep, ring trims, vector GC.
  Future<void> _housekeeping(int now) async {
    await _sweepTombstones(now);
    await store.trimConflicts(config.conflictCap);
    await store.trimDreamLog(config.dreamLogCap);
    await store.deleteVectorsExceptModel(modelId);
    await store.deleteOrphanVectors();
    await store.compact();
  }

  /// Clusters live L1+L2 rows by coarse vector and ranks eligible clusters
  /// by priority `n·cohesion·2^(−maxGen)·(1+overflow)·(1+conflicts)`.
  Future<List<_DreamCluster>> _dreamCandidates(
    int now, {
    required bool force,
  }) async {
    final entries = <_Entry>[
      ...await _tierCandidates(1),
      ...await _tierCandidates(2),
    ];
    if (entries.length < config.clusterMin) return const [];
    final coarse = [for (final e in entries) _coarse(e.vec)];
    final k = math.max(1, (entries.length / 3).round());
    final labels = sphericalKMeans(coarse, k);

    final groups = <int, List<int>>{};
    for (var i = 0; i < entries.length; i++) {
      groups.putIfAbsent(labels[i], () => []).add(i);
    }

    final conflictPairs = <String>{
      for (final c in await store.listConflicts())
        (c.a.compareTo(c.b) <= 0) ? '${c.a}|${c.b}' : '${c.b}|${c.a}',
    };
    final l1Count = await store.countMemories(tier: 1);
    final l1Overflow =
        math.max(0, l1Count - config.cap1) / math.max(1, config.cap1);

    final out = <_DreamCluster>[];
    for (final indices in groups.values) {
      if (indices.length < config.clusterMin) continue;
      final rows = [for (final i in indices) entries[i].row];
      final ids = rows.map((r) => r.id).toList()..sort();
      final fp = clusterFingerprint(ids);
      if (!force && await store.getDreamVerdict(fp) == 'none') continue;
      final coh = cohesion([for (final i in indices) coarse[i]]);
      if (coh < config.clusterCohesionMin) continue;
      final maxGen = rows.map((r) => r.gen).reduce(math.max);
      final idSet = ids.toSet();
      var nConf = 0;
      for (final pair in conflictPairs) {
        final parts = pair.split('|');
        if (idSet.contains(parts[0]) && idSet.contains(parts[1])) nConf++;
      }
      final prio = rows.length *
          coh *
          math.pow(2.0, -maxGen) *
          (1.0 + l1Overflow) *
          (1.0 + nConf);
      out.add(_DreamCluster(fp, rows, prio.toDouble()));
    }
    out.sort((a, b) => b.priority.compareTo(a.priority));
    return out;
  }

  /// Adjudicates one cluster: ask the app's LLM callback, then apply the
  /// verdict — delete members and insert replacements in one transaction
  /// (1 adjudication = 1 transaction, I5). A throwing callback leaves the
  /// cluster untouched and unrecorded so the next dream retries it.
  Future<DreamReport> _adjudicate(
    DreamAdjudicator adjudicate,
    _DreamCluster cluster,
    int now,
    String tzField,
  ) async {
    // Re-fetch members: earlier adjudications may have consumed them.
    final memberRows = <MemoryRecord>[];
    for (final m in cluster.members) {
      final fresh = await store.getMemory(m.id);
      if (fresh != null && !fresh.isTombstone) memberRows.add(fresh);
    }
    final before = [
      for (final r in memberRows)
        MemorySnapshot(id: r.id, text: r.text, tier: r.tier, gen: r.gen),
    ];
    DreamReport noneReport({String? error}) => DreamReport(
          clusterFingerprint: cluster.fingerprint,
          action: DreamAction.none,
          priority: cluster.priority,
          before: before,
          after: const [],
          deletedIds: const [],
          error: error,
        );
    if (memberRows.length < config.clusterMin) return noneReport();

    final payload = [
      for (final r in memberRows.take(config.dreamMaxMembers))
        DreamMember(
          id: r.id,
          text: r.text,
          gen: r.gen,
          activation:
              double.parse(activationOfRecord(r, now).toStringAsFixed(2)),
          localTime: formatLocal(r.createdAt, r.tz),
          timezone: r.tz,
        ),
    ];
    final request = DreamClusterRequest(
      clusterFingerprint: cluster.fingerprint,
      currentLocalTime: formatLocal(now, tzField),
      members: payload,
    );

    DreamDecision decision;
    try {
      decision = await Future.sync(() => adjudicate(request));
    } catch (e) {
      return noneReport(error: e.toString());
    }

    final proposals = [
      for (final p in decision.memories)
        if (p.text.trim().isNotEmpty) p,
    ];
    if (decision.action == DreamAction.none || proposals.isEmpty) {
      await _recordDream(cluster.fingerprint, 'none', now);
      return noneReport();
    }

    var sumA = 0.0;
    var maxGen = 0;
    for (final r in memberRows) {
      sumA += activationOfRecord(r, now);
      maxGen = math.max(maxGen, r.gen);
    }
    final isMerge = decision.action != DreamAction.split;
    final gen = math.min(maxGen + (isMerge ? 1 : 0), config.genMax);
    final picks = proposals.take(config.dreamMaxMembers).toList();
    final perMass = isMerge
        ? math.min(sumA, config.mMax)
        : math.min(sumA / math.max(1, picks.length), config.mMax);

    // Pre-embed outside the transaction (the embedder is external/async).
    final texts = [
      for (final p in picks)
        p.text.trim().length > config.textMax
            ? shorten(p.text.trim(), config.textMax)
            : p.text.trim(),
    ];
    final vectors = <Float32List>[];
    for (final t in texts) {
      vectors.add(await _embedDoc(t));
    }

    final deleted = [for (final r in memberRows) r.id];
    final after = <MemorySnapshot>[];
    await store.runInTransaction(() async {
      for (final r in memberRows) {
        await store.deleteMemory(r.id);
        _docVecCache.remove(r.id);
      }
      for (var i = 0; i < texts.length; i++) {
        final row = await _insertWithVec(
            texts[i], vectors[i], perMass, now, tzField,
            gen: gen, tier: 1);
        after.add(MemorySnapshot(
            id: row.id, text: row.text, tier: row.tier, gen: row.gen));
      }
    });
    final actionName = isMerge ? 'merge' : 'split';
    await _recordDream(cluster.fingerprint, actionName, now);
    return DreamReport(
      clusterFingerprint: cluster.fingerprint,
      action: isMerge ? DreamAction.merge : DreamAction.split,
      priority: cluster.priority,
      before: before,
      after: after,
      deletedIds: deleted,
    );
  }

  Future<void> _recordDream(String fp, String verdict, int now) =>
      store.putDreamVerdict(
          DreamLogRecord(fingerprint: fp, verdict: verdict, at: now));
}
