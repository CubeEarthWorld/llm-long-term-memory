part of 'engine.dart';

class _Candidate {
  _Candidate(this.row, this.cos, this.mmr);
  final MemoryRecord row;
  double cos;
  final Float32List mmr;
  double activation = 0.0;
  double score = 0.0;
}

/// READ pipeline — retrieve / filter / MMR / pack (§4.2, §5.2).
mixin _ReadOps on _EngineBase {
  Future<RetrieveResult> _retrieve(String query, int now) async {
    await _sanitizeTimestamps(now);
    final chunks = splitParagraphs(query).take(16).toList();
    if (chunks.isEmpty) return RetrieveResult.empty;
    final rawQ = await embedder.embedQueries(chunks);
    final qFull = [
      for (final q in rawQ) truncateNormalize(l2Normalized(q), config.dim1),
    ];

    final cands = <String, _Candidate>{};
    for (var tier = 1; tier <= 3; tier++) {
      final entries = await _tierCandidates(tier);
      if (entries.isEmpty) continue;
      final qts = [
        for (final q in qFull) truncateNormalize(q, config.dimOf(tier)),
      ];
      for (final e in entries) {
        var best = double.negativeInfinity;
        for (final qt in qts) {
          final c = dot(e.vec, qt);
          if (c > best) best = c;
        }
        final prev = cands[e.row.id];
        if (prev == null || best > prev.cos) {
          cands[e.row.id] = _Candidate(e.row, best, _coarse(e.vec));
        }
      }
    }

    for (final c in cands.values) {
      final a = activationOfRecord(c.row, now);
      final aAbs = math.log(1.0 + a) / math.log(1.0 + config.mMax);
      c.activation = a;
      c.score =
          math.max(0.0, c.cos) * (config.alpha + (1.0 - config.alpha) * aAbs);
    }

    final ranked = _mmr(_filterByScore(cands.values.toList()));
    final (result, packedIds) = _pack(ranked);
    for (final mid in packedIds) {
      final row = await store.getMemory(mid);
      if (row != null) await _applyRecall(row, now);
    }
    return result;
  }

  /// Drops low-score memories, relaxing the threshold (strictest first)
  /// only when nothing remains. If even the loosest threshold matches
  /// nothing, inject nothing — unrelated memories would be spuriously
  /// reinforced by the recall update.
  List<_Candidate> _filterByScore(List<_Candidate> items) {
    final thresholds = [...config.scoreThresholds]
      ..sort((a, b) => b.compareTo(a));
    for (final threshold in thresholds) {
      final filtered =
          items.where((it) => it.score >= threshold).toList(growable: false);
      if (filtered.isNotEmpty) return filtered;
    }
    return const [];
  }

  /// MMR selection (§4.2): `next = argmax[score − λ·max cos(m, selected)]`,
  /// maintaining a running max-similarity per candidate.
  List<_Candidate> _mmr(List<_Candidate> items) {
    if (items.isEmpty) return const [];
    final lam = config.mmrLambda;
    final pool = [...items]..sort((a, b) => b.score.compareTo(a.score));
    final limited = pool.take(50).toList(growable: false);
    final maxSim = List<double>.filled(limited.length, 0.0);
    final alive = List<bool>.filled(limited.length, true);
    final selected = <_Candidate>[];
    while (selected.length < config.injectN) {
      var bestI = -1;
      var bestVal = double.negativeInfinity;
      for (var i = 0; i < limited.length; i++) {
        if (!alive[i]) continue;
        final val = limited[i].score - lam * maxSim[i];
        if (val > bestVal) {
          bestVal = val;
          bestI = i;
        }
      }
      if (bestI < 0) break;
      selected.add(limited[bestI]);
      alive[bestI] = false;
      for (var i = 0; i < limited.length; i++) {
        if (!alive[i]) continue;
        final sim = dot(limited[i].mmr, limited[bestI].mmr);
        if (sim > maxSim[i]) maxSim[i] = sim;
      }
    }
    return selected;
  }

  /// Formats injected memories within `budgetChars`, one line each:
  /// `[<unix> <tz>] <text>　《id:<ulid>》`.
  (RetrieveResult, List<String>) _pack(List<_Candidate> ranked) {
    var used = 0;
    final lines = <String>[];
    final recalled = <RecalledMemory>[];
    final ids = <String>[];
    for (final it in ranked) {
      final row = it.row;
      final line = '[${row.createdAt} ${row.tz}] ${row.text}　《id:${row.id}》\n';
      if (used + line.length > config.budgetChars) continue;
      lines.add(line);
      used += line.length;
      ids.add(row.id);
      recalled.add(RecalledMemory(
        id: row.id,
        text: row.text,
        score: it.score,
        cosine: it.cos,
        activation: it.activation,
        mass: row.mass,
        tier: row.tier,
        gen: row.gen,
        createdAt: row.createdAt,
        tz: row.tz,
      ));
      if (used >= config.budgetChars) break;
    }
    return (
      RetrieveResult(packText: lines.join(), recalled: recalled),
      ids,
    );
  }
}
