// Zero-dependency quick start: the bundled InMemoryStore plus a toy
// hash-based embedder. Replace the embedder with a real one (firebase_ai,
// llamadart, ONNX, a REST endpoint, ...) and the store with a database
// adapter (see example/sqlite_adapter) for production use.
//
// Run with:  dart run example/main.dart
import 'dart:math';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';

/// Toy deterministic embedder: token-hash vectors, so overlapping wording
/// is similar. Good enough to see the engine working without any model.
class ToyEmbedder implements Embedder {
  @override
  String get modelId => 'toy/hash-v1';

  @override
  int get dimension => 768;

  Float32List _embed(String text) {
    final v = Float32List(dimension);
    final tokens = <String>[
      ...RegExp(r'[a-z0-9]+').allMatches(text.toLowerCase()).map((m) => m[0]!),
      ...text.runes
          .where((r) => r >= 0x3040 && r <= 0x9FFF)
          .map(String.fromCharCode),
    ];
    for (final tok in tokens) {
      final rng = Random(tok.hashCode);
      for (var i = 0; i < dimension; i++) {
        v[i] += rng.nextDouble() * 2 - 1;
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

Future<void> main() async {
  const tokyo = MemoryTimezone('Asia/Tokyo', Duration(hours: 9));

  final memory = EngramMemory(
    store: InMemoryStore(),
    embedder: ToyEmbedder(),
    defaultTimezone: tokyo,
  );
  await memory.initialize();

  // -- WRITE: store self-contained propositions (e.g. from LLM tool calls).
  print('--- save ---');
  for (final fact in [
    'ユーザーは京都に住んでいる',
    'ユーザーは抹茶アイスクリームが好き',
    'ユーザーは毎週水曜にジムへ通っている',
  ]) {
    final r = await memory.saveMemory(fact);
    print('${r.action.name}: ${r.text} (id=${r.id})');
  }

  // Explicit Unix time + timezone for a backdated import:
  await memory.saveMemory(
    'ユーザーは2026-04-01にベルリンへ出張した',
    nowUnix: 1774000000,
    timezone: const MemoryTimezone('Europe/Berlin', Duration(hours: 2)),
  );

  // -- READ: retrieve relevant memories for the next LLM prompt.
  print('\n--- retrieve ---');
  final recall = await memory.retrieve('抹茶のスイーツでおすすめある？');
  print(recall.packText.trimRight());
  print(
      '(scores: ${recall.recalled.map((m) => m.score.toStringAsFixed(3)).join(', ')})');

  // The pack plugs straight into the bundled prompt templates:
  final userMessage = EngramPrompts.buildUserMessage(
    currentTime: memory.nowLocal(),
    memoryPack: recall.packText,
    userText: '抹茶のスイーツでおすすめある？',
  );
  print('\n--- prompt for your LLM (first 120 chars) ---');
  print(userMessage.substring(0, 120).replaceAll('\n', ' | '));

  // -- MAINTAIN: cheap per-turn housekeeping (no LLM, no embedding).
  await memory.maintain();

  // -- DREAM: offline consolidation through *your* LLM. Here a fake one.
  // A few overlapping notes form a cluster the dream phase can merge:
  await memory.saveMemory('ユーザーは月曜の朝にコーヒーを飲む習慣がある');
  await memory.saveMemory('ユーザーは火曜の朝にコーヒーを飲む習慣がある');
  await memory.saveMemory('ユーザーは週末の朝にコーヒーを飲む習慣がある');

  print('\n--- dream ---');
  final reports = await memory.dream(
    adjudicate: (request) async {
      // Real apps: send request.buildPrompt() to an LLM in JSON mode and
      // return DreamDecision.parseJson(rawResponse).
      final gist = request.members.map((m) => m.text).join(' / ');
      return DreamDecision(
        action: DreamAction.merge,
        memories: [DreamProposal(text: gist)],
      );
    },
  );
  for (final r in reports) {
    print('${r.action.name}: ${r.before.length} memories '
        '-> ${r.after.length} (priority ${r.priority.toStringAsFixed(2)})');
  }

  print('\n--- stats ---');
  print(await memory.stats());
}
