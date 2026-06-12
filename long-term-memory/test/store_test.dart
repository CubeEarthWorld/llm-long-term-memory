import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

MemoryRecord _mem(String id,
        {int tier = 1, String? supersededBy, int lastAccess = 100}) =>
    MemoryRecord(
      id: id,
      text: 'text of $id',
      tier: tier,
      gen: 0,
      createdAt: 100,
      tz: 'UTC;+00:00',
      lastAccess: lastAccess,
      lastBonusAt: 100,
      mass: 1.0,
      supersededBy: supersededBy,
    );

VectorRecord _vec(String memoryId, {String modelId = 'm1'}) => VectorRecord(
      memoryId: memoryId,
      modelId: modelId,
      dim: 4,
      dtype: VecDtype.f32,
      bytes: packF32(Float32List.fromList([1, 0, 0, 0])),
    );

void main() {
  late InMemoryStore store;
  setUp(() async {
    store = InMemoryStore();
    await store.initialize();
  });

  test('memory CRUD, liveness filters and counts', () async {
    await store.insertMemory(_mem('a'));
    await store.insertMemory(_mem('b', tier: 2, supersededBy: 'b'));
    expect((await store.getMemory('a'))!.text, 'text of a');
    expect(await store.getMemory('zzz'), isNull);
    expect(await store.countMemories(), 2);
    expect(await store.countMemories(liveness: Liveness.liveOnly), 1);
    expect(
        await store.countMemories(tier: 2, liveness: Liveness.tombstonesOnly),
        1);
    expect((await store.getLiveMemoryByText('text of a'))!.id, 'a');
    expect(await store.getLiveMemoryByText('text of b'), isNull);

    await store.updateMemory(_mem('a').copyWith(mass: 5.0));
    expect((await store.getMemory('a'))!.mass, 5.0);

    await store.deleteMemory('a');
    expect(await store.getMemory('a'), isNull);
  });

  test('vector upsert/get cascades on memory delete', () async {
    await store.insertMemory(_mem('a'));
    await store.upsertVector(_vec('a'));
    await store.upsertVector(_vec('a')); // upsert keeps one row
    expect((await store.listVectors()).length, 1);
    expect(await store.getVector('a', 'm1'), isNotNull);
    await store.deleteMemory('a');
    expect(await store.getVector('a', 'm1'), isNull);
  });

  test('vector GC helpers', () async {
    await store.insertMemory(_mem('a'));
    await store.upsertVector(_vec('a', modelId: 'old-model'));
    await store.upsertVector(_vec('a', modelId: 'm1'));
    await store.upsertVector(_vec('ghost', modelId: 'm1')); // orphan
    await store.deleteVectorsExceptModel('m1');
    expect((await store.listVectors()).every((v) => v.modelId == 'm1'), isTrue);
    await store.deleteOrphanVectors();
    expect((await store.listVectors()).single.memoryId, 'a');
  });

  test('conflict ring trims oldest', () async {
    for (var i = 0; i < 5; i++) {
      await store.addConflict(ConflictRecord(a: 'a$i', b: 'b$i', at: i));
    }
    await store.trimConflicts(3);
    final left = await store.listConflicts();
    expect(left.length, 3);
    expect(left.first.a, 'a2');
  });

  test('dream log upserts by fingerprint and trims by age', () async {
    await store.putDreamVerdict(
        const DreamLogRecord(fingerprint: 'f1', verdict: 'none', at: 1));
    await store.putDreamVerdict(
        const DreamLogRecord(fingerprint: 'f1', verdict: 'merge', at: 5));
    expect(await store.getDreamVerdict('f1'), 'merge');
    expect(await store.countDreamLog(), 1);
    await store.putDreamVerdict(
        const DreamLogRecord(fingerprint: 'f2', verdict: 'none', at: 2));
    await store.trimDreamLog(1);
    expect(await store.getDreamVerdict('f2'), isNull);
    expect(await store.getDreamVerdict('f1'), 'merge');
  });

  test('clampFutureTimestamps clamps rollback artefacts', () async {
    await store.insertMemory(_mem('future', lastAccess: 9999));
    await store.clampFutureTimestamps(500);
    expect((await store.getMemory('future'))!.lastAccess, 500);
  });

  test('deleteTombstones honours tier and age filters', () async {
    await store.insertMemory(_mem('t1', supersededBy: 't1', lastAccess: 10));
    await store
        .insertMemory(_mem('t2', tier: 2, supersededBy: 't2', lastAccess: 99));
    await store.deleteTombstones(lastAccessBefore: 50);
    expect(await store.getMemory('t1'), isNull);
    expect(await store.getMemory('t2'), isNotNull);
    await store.deleteTombstones(tier: 2);
    expect(await store.getMemory('t2'), isNull);
  });

  test('listTierEntries joins live rows with model vectors', () async {
    await store.insertMemory(_mem('a'));
    await store.insertMemory(_mem('dead', supersededBy: 'dead'));
    await store.insertMemory(_mem('noVec'));
    await store.upsertVector(_vec('a'));
    await store.upsertVector(_vec('dead'));
    final entries = await store.listTierEntries(1, 'm1');
    expect(entries.map((e) => e.memory.id), ['a']);
  });

  test('meta and clear', () async {
    await store.putMeta('k', 'v');
    expect(await store.getMeta('k'), 'v');
    await store.insertMemory(_mem('a'));
    await store.clear();
    expect(await store.getMeta('k'), isNull);
    expect(await store.countMemories(), 0);
  });

  test('toJson/fromJson round-trips', () async {
    await store.insertMemory(_mem('a'));
    await store.upsertVector(_vec('a'));
    await store.addConflict(const ConflictRecord(a: 'a', b: 'b', at: 1));
    await store.putDreamVerdict(
        const DreamLogRecord(fingerprint: 'f', verdict: 'none', at: 1));
    await store.putMeta('k', 'v');
    final restored = InMemoryStore.fromJson(store.toJson());
    expect((await restored.getMemory('a'))!.text, 'text of a');
    expect(await restored.getVector('a', 'm1'), isNotNull);
    expect(await restored.countConflicts(), 1);
    expect(await restored.getDreamVerdict('f'), 'none');
    expect(await restored.getMeta('k'), 'v');
  });
}
