/// Deterministic test doubles ported from the reference implementation's
/// eval mocks: a token-overlap embedder (texts sharing words/CJK characters
/// are similar; unrelated texts are near-orthogonal) and a virtual clock.
library;

import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:long_term_memory/long_term_memory.dart';

/// Virtual clock: lets scenarios accelerate time (activation decay,
/// refractory spacing) at zero real cost.
class VirtualClock {
  VirtualClock([this.t = 1700000000]);

  int t;

  int call() => t;

  void advance(int seconds) => t += seconds;

  void advanceDays(num days) => t += (days * 86400).round();
}

/// Embedder whose vectors are sums of per-token pseudo-random vectors:
/// `cos ≈ shared / √(n1·n2)` for texts with overlapping token sets, and
/// identical token multisets give identical vectors.
class FakeEmbedder implements Embedder {
  FakeEmbedder({this.dimension = 768});

  @override
  final int dimension;

  @override
  String get modelId => 'fake/token-overlap';

  final Map<String, Float32List> _cache = {};

  static List<String> tokens(String text) {
    final lower = text.toLowerCase();
    final words =
        RegExp(r'[a-z0-9]+').allMatches(lower).map((m) => m.group(0)!).toList();
    final cjk = <String>[];
    for (final rune in lower.runes) {
      if ((rune >= 0x3040 && rune <= 0x9FFF) ||
          (rune >= 0xFF66 && rune <= 0xFF9D)) {
        cjk.add(String.fromCharCode(rune));
      }
    }
    return [...words, ...cjk];
  }

  Float32List _tokenVec(String tok) => _cache.putIfAbsent(tok, () {
        final digest = md5.convert(utf8.encode(tok)).toString();
        final seed = int.parse(digest.substring(0, 8), radix: 16);
        final rng = Random(seed);
        final v = Float32List(dimension);
        for (var i = 0; i < dimension; i++) {
          v[i] = rng.nextDouble() * 2 - 1;
        }
        return v;
      });

  Float32List _embed(String text) {
    final toks = tokens(text);
    final v = Float32List(dimension);
    if (toks.isEmpty) {
      for (var i = 0; i < dimension; i++) {
        v[i] = 1.0;
      }
    } else {
      for (final t in toks) {
        final tv = _tokenVec(t);
        for (var i = 0; i < dimension; i++) {
          v[i] += tv[i];
        }
      }
    }
    return l2Normalized(v);
  }

  @override
  Future<List<Float32List>> embedDocuments(List<String> texts) async =>
      [for (final t in texts) _embed(t)];

  @override
  Future<List<Float32List>> embedQueries(List<String> texts) async =>
      [for (final t in texts) _embed(t)];
}

/// Merge-to-gist adjudicator: joins all member texts into one memory.
DreamDecision mergeToGist(DreamClusterRequest request) {
  final gist = request.members.map((m) => m.text).join(' / ');
  return DreamDecision(
    action: DreamAction.merge,
    memories: [
      DreamProposal(
          text: gist.length > 160 ? gist.substring(0, 160) : gist,
          timezone: 'Asia/Tokyo'),
    ],
  );
}

/// Adjudicator that always answers "no change".
DreamDecision alwaysNone(DreamClusterRequest request) =>
    const DreamDecision.none();
