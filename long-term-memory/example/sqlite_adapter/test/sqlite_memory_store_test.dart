import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:sqlite_memory_store/sqlite_memory_store.dart';
import 'package:test/test.dart';

class _HashEmbedder implements Embedder {
  @override
  String get modelId => 'test/hash';
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

void main() {
  late Directory tmp;
  late SqliteMemoryStore store;

  setUp(() async {
    tmp = Directory.systemTemp.createTempSync('engram_sqlite_test');
    store = SqliteMemoryStore('${tmp.path}/memory.db');
    await store.initialize();
  });

  tearDown(() async {
    await store.close();
    tmp.deleteSync(recursive: true);
  });

  MemoryRecord mem(String id, {int tier = 1, String? supersededBy}) =>
      MemoryRecord(
        id: id,
        text: 'text of $id',
        tier: tier,
        gen: 0,
        createdAt: 100,
        tz: 'UTC;+00:00',
        lastAccess: 100,
        lastBonusAt: 100,
        mass: 1.0,
        supersededBy: supersededBy,
      );

  VectorRecord vec(String memoryId, {String modelId = 'm1'}) => VectorRecord(
        memoryId: memoryId,
        modelId: modelId,
        dim: 4,
        dtype: VecDtype.f32,
        bytes: packF32(Float32List.fromList([1, 0, 0, 0])),
      );

  test('store contract: CRUD, filters, rings, meta, bulk ops', () async {
    await store.insertMemory(mem('a'));
    await store.insertMemory(mem('b', tier: 2, supersededBy: 'b'));
    expect((await store.getMemory('a'))!.text, 'text of a');
    expect(await store.countMemories(liveness: Liveness.liveOnly), 1);
    expect((await store.getLiveMemoryByText('text of a'))!.id, 'a');
    expect(await store.getLiveMemoryByText('text of b'), isNull);

    await store.updateMemory(mem('a').copyWith(mass: 7.0, tier: 3));
    final updated = (await store.getMemory('a'))!;
    expect(updated.mass, 7.0);
    expect(updated.tier, 3);

    await store.upsertVector(vec('a'));
    await store.upsertVector(vec('a')); // upsert: one row
    expect((await store.listVectors()).length, 1);
    final entries = await store.listTierEntries(3, 'm1');
    expect(entries.single.memory.id, 'a');

    await store.deleteMemory('a');
    expect(await store.getVector('a', 'm1'), isNull); // cascade

    for (var i = 0; i < 5; i++) {
      await store.addConflict(ConflictRecord(a: 'a$i', b: 'b$i', at: i));
    }
    await store.trimConflicts(2);
    expect((await store.listConflicts()).map((c) => c.a), ['a3', 'a4']);

    await store.putDreamVerdict(
        const DreamLogRecord(fingerprint: 'f1', verdict: 'none', at: 1));
    await store.putDreamVerdict(
        const DreamLogRecord(fingerprint: 'f2', verdict: 'merge', at: 2));
    await store.trimDreamLog(1);
    expect(await store.getDreamVerdict('f1'), isNull);
    expect(await store.getDreamVerdict('f2'), 'merge');

    await store.putMeta('k', 'v');
    expect(await store.getMeta('k'), 'v');

    await store.insertMemory(mem('future').copyWith(lastAccess: 9999));
    await store.clampFutureTimestamps(500);
    expect((await store.getMemory('future'))!.lastAccess, 500);

    await store.insertMemory(mem('tomb', supersededBy: 'tomb'));
    await store.deleteTombstones();
    expect(await store.getMemory('tomb'), isNull);

    // Orphan vectors cannot exist here (FOREIGN KEY + cascade); GC of
    // foreign-model vectors still applies after an embedder switch.
    await store.upsertVector(vec('future', modelId: 'old'));
    await store.deleteOrphanVectors();
    await store.deleteVectorsExceptModel('m1');
    expect(await store.listVectors(), isEmpty);
  });

  test('runInTransaction rolls back on error', () async {
    await store.insertMemory(mem('keep'));
    await expectLater(
      store.runInTransaction(() async {
        await store.deleteMemory('keep');
        throw StateError('boom');
      }),
      throwsStateError,
    );
    expect(await store.getMemory('keep'), isNotNull);
  });

  test('backup keeps a rotating snapshot ring', () async {
    await store.insertMemory(mem('a'));
    await store.backup();
    await store.backup();
    final snaps = Directory('${tmp.path}/memory.db.snapshots')
        .listSync()
        .map((e) => e.uri.pathSegments.last)
        .toList()
      ..sort();
    expect(snaps, ['snap-0.db', 'snap-1.db']);
  });

  test('persists across reopen', () async {
    await store.insertMemory(mem('a'));
    await store.close();
    store = SqliteMemoryStore('${tmp.path}/memory.db');
    await store.initialize();
    expect((await store.getMemory('a'))!.text, 'text of a');
  });

  test('end-to-end engine scenario on SQLite', () async {
    var t = 1700000000;
    final memory = EngramMemory(
      store: store,
      embedder: _HashEmbedder(),
      clock: () => t,
      defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
    );
    await memory.initialize();

    final saved = await memory.saveMemory('ユーザーは京都に住んでいる');
    expect(saved.action, SaveAction.inserted);
    await memory.saveMemory('ユーザーは抹茶アイスクリームが好き');
    await memory.saveMemory('user enjoys morning coffee with milk');
    await memory.saveMemory('user enjoys evening coffee with sugar');
    await memory.saveMemory('user enjoys sunday coffee with friends');
    t += 3600;

    final recall = await memory.retrieve('京都について');
    expect(recall.recalled.map((m) => m.id), contains(saved.id));

    await memory.maintain();
    final reports = await memory.dream(adjudicate: (req) {
      final gist = req.members.map((m) => m.text).join(' / ');
      return DreamDecision(
        action: DreamAction.merge,
        memories: [DreamProposal(text: gist)],
      );
    });
    expect(reports.where((r) => r.action == DreamAction.merge), isNotEmpty);
    expect((await memory.stats()).total, greaterThan(0));
  });
}
