import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

const tokyo = MemoryTimezone('Asia/Tokyo', Duration(hours: 9));

void main() {
  late VirtualClock clock;
  late InMemoryStore store;
  late EngramMemory memory;

  Future<void> build({EngramConfig config = const EngramConfig()}) async {
    clock = VirtualClock();
    store = InMemoryStore();
    memory = EngramMemory(
      store: store,
      embedder: FakeEmbedder(),
      config: config,
      clock: clock.call,
      defaultTimezone: tokyo,
    );
    await memory.initialize();
  }

  group('initialize', () {
    test('rejects an embedder smaller than dim1', () async {
      final m = EngramMemory(
        store: InMemoryStore(),
        embedder: FakeEmbedder(dimension: 128),
      );
      expect(m.initialize(), throwsArgumentError);
    });

    test('installs metadata', () async {
      await build();
      expect(await store.getMeta('spec_version'), 'ENGRAM v1.1');
      expect(await store.getMeta('active_model'), 'fake/token-overlap');
    });
  });

  group('saveMemory (§5.1)', () {
    setUp(build);

    test('inserts a new memory in L1 with the timezone field', () async {
      final r = await memory.saveMemory('ユーザーは京都に住んでいる');
      expect(r.action, SaveAction.inserted);
      expect(r.tier, 1);
      expect(r.mass, 1.0);
      final row = (await store.getMemory(r.id!))!;
      expect(row.tz, 'Asia/Tokyo;+09:00');
      expect(row.createdAt, clock.t);
      final vec = await store.getVector(r.id!, 'fake/token-overlap');
      expect(vec!.dtype, VecDtype.f32);
      expect(vec.dim, 768);
    });

    test('rejects empty and oversized text', () async {
      expect((await memory.saveMemory('  ')).action, SaveAction.rejected);
      expect((await memory.saveMemory('あ' * 1200)).action, SaveAction.rejected);
    });

    test('shortens text over the soft limit', () async {
      final r = await memory.saveMemory('${'あ' * 150}。${'い' * 100}');
      expect(r.action, SaveAction.inserted);
      expect(r.text.length, lessThanOrEqualTo(170));
    });

    test('exact duplicate text reinforces mass without a new row', () async {
      final first = await memory.saveMemory('ユーザーは抹茶アイスが好き');
      final again = await memory.saveMemory('ユーザーは抹茶アイスが好き');
      expect(again.action, SaveAction.reinforced);
      expect(again.id, first.id);
      expect(again.mass, closeTo(2.0, 1e-9));
      expect(await memory.totalRecords(), 1);
    });

    test('near-identical proposition supersedes the old row', () async {
      final old = await memory.saveMemory('ユーザーは京都に住んでいる');
      // Same token multiset, different surface order → cosine 1.0.
      final updated = await memory.saveMemory('京都にユーザーは住んでいる');
      expect(updated.action, SaveAction.updated);
      expect(updated.supersededId, old.id);
      expect(updated.mass, closeTo(2.0, 1e-9)); // inherited A + 1
      final oldRow = (await store.getMemory(old.id!))!;
      expect(oldRow.supersededBy, updated.id);
      expect(await memory.totalRecords(), 1);
    });

    test('conflict band keeps both rows and queues the pair', () async {
      const a = 'alpha beta gamma delta epsilon zeta eta theta iota kappa';
      const b = 'alpha beta gamma delta epsilon zeta eta theta iota lambda';
      final first = await memory.saveMemory(a);
      final second = await memory.saveMemory(b);
      expect(second.action, SaveAction.conflict);
      expect(second.conflictWithId, first.id);
      expect(await memory.totalRecords(), 2);
      final conflicts = await store.listConflicts();
      expect(conflicts.single.a, first.id);
      expect(conflicts.single.b, second.id);
    });

    test('write-rate limit blocks beyond the daily budget', () async {
      await build(config: const EngramConfig(writeRatePerDay: 2));
      await memory.saveMemory('fact one');
      await memory.saveMemory('fact two');
      final third = await memory.saveMemory('fact three');
      expect(third.action, SaveAction.rateLimited);
      clock.advanceDays(1.5);
      final later = await memory.saveMemory('fact three');
      expect(later.action, SaveAction.inserted);
    });
  });

  group('activation (§4.1)', () {
    setUp(build);

    test('decays with the tier half-life', () async {
      final r = await memory.saveMemory('ユーザーは犬を飼っている');
      expect(await memory.activationOf(r.id!), closeTo(1.0, 1e-9));
      clock.advanceDays(7); // one L1 half-life
      expect(await memory.activationOf(r.id!), closeTo(0.5, 1e-9));
      clock.advanceDays(7);
      expect(await memory.activationOf(r.id!), closeTo(0.25, 1e-9));
    });

    test('recall bonus is gated by the refractory period', () async {
      final r = await memory.saveMemory('ユーザーは犬を飼っている');
      clock.advance(7200);
      await memory.retrieve('犬を飼っている');
      final afterFirst = (await store.getMemory(r.id!))!.mass;
      expect(afterFirst, greaterThan(1.0)); // decay fold + bonus
      await memory.retrieve('犬を飼っている'); // within the refractory hour
      final afterSecond = (await store.getMemory(r.id!))!.mass;
      expect(afterSecond, closeTo(afterFirst, 1e-9));
      clock.advance(3601);
      await memory.retrieve('犬を飼っている');
      expect(
          (await store.getMemory(r.id!))!.mass, greaterThan(afterSecond + 0.5));
    });

    test('mass never exceeds mMax', () async {
      final r = await memory.saveMemory('ユーザーは犬を飼っている');
      for (var i = 0; i < 70; i++) {
        await memory.saveMemory('ユーザーは犬を飼っている'); // exact rehearsal
      }
      expect((await store.getMemory(r.id!))!.mass, lessThanOrEqualTo(64.0));
    });
  });

  group('retrieve (§5.2)', () {
    setUp(build);

    test('finds the relevant memory and formats the pack', () async {
      final saved = await memory.saveMemory('ユーザーは抹茶アイスクリームが好き');
      await memory.saveMemory('completely unrelated quantum physics talk');
      final result = await memory.retrieve('抹茶アイスクリームについて教えて');
      expect(result.recalled, isNotEmpty);
      expect(result.recalled.first.id, saved.id);
      expect(result.packText, contains('《id:${saved.id}》'));
      expect(result.packText, contains('Asia/Tokyo;+09:00'));
    });

    test('returns empty for empty query and when nothing scores', () async {
      expect((await memory.retrieve('')).recalled, isEmpty);
      await memory.saveMemory('ユーザーは抹茶アイスクリームが好き');
      final unrelated = await memory.retrieve('zzz qqq xxx yyy');
      expect(unrelated.recalled, isEmpty); // below every threshold
    });

    test('injects at most injectN memories', () async {
      await build(config: const EngramConfig(injectN: 2));
      for (var i = 0; i < 6; i++) {
        await memory.saveMemory('ユーザーは抹茶アイス味$iが好き');
      }
      final result = await memory.retrieve('抹茶アイス');
      expect(result.recalled.length, lessThanOrEqualTo(2));
    });

    test('packed memories receive a recall update', () async {
      final saved = await memory.saveMemory('ユーザーは抹茶アイスクリームが好き');
      clock.advance(7200);
      await memory.retrieve('抹茶アイスクリーム');
      final row = (await store.getMemory(saved.id!))!;
      expect(row.lastAccess, clock.t);
      expect(row.mass, greaterThan(1.0));
    });
  });

  group('deleteMemory (§5.3)', () {
    setUp(build);

    test('soft delete tombstones and hides from retrieval', () async {
      final saved = await memory.saveMemory('ユーザーは抹茶アイスクリームが好き');
      final del = await memory.deleteMemory(saved.id!);
      expect(del.action, DeleteAction.tombstoned);
      expect((await store.getMemory(saved.id!))!.supersededBy, saved.id);
      final result = await memory.retrieve('抹茶アイスクリーム');
      expect(result.recalled, isEmpty);
    });

    test('hard delete removes the row physically', () async {
      final saved = await memory.saveMemory('ユーザーは抹茶アイスクリームが好き');
      final del = await memory.deleteMemory(saved.id!, hard: true);
      expect(del.action, DeleteAction.deleted);
      expect(await store.getMemory(saved.id!), isNull);
      expect(
          await memory.deleteMemory('nope'),
          isA<DeleteResult>()
              .having((d) => d.action, 'action', DeleteAction.notFound));
    });
  });

  group('maintain (§5.4)', () {
    test('demotes L1 overflow to L2 with int8 vectors', () async {
      await build(config: const EngramConfig(cap1: 3, cap2: 10, cap3: 10));
      final ids = <String>[];
      for (var i = 0; i < 6; i++) {
        final words = ['fact', 'topic$i', 'detail$i', 'item$i'].join(' ');
        ids.add((await memory.saveMemory(words)).id!);
        clock.advance(4000);
      }
      await memory.maintain();
      final stats = await memory.stats();
      expect(stats.l1, lessThanOrEqualTo(3));
      expect(stats.l2, 6 - stats.l1);
      final demoted = await store.listMemories(tier: 2);
      for (final m in demoted) {
        final v = (await store.getVector(m.id, 'fake/token-overlap'))!;
        expect(v.dtype, VecDtype.int8);
        expect(v.dim, 256);
      }
    });

    test('L3 overflow is physical eviction of the lowest activation', () async {
      await build(config: const EngramConfig(cap1: 2, cap2: 2, cap3: 2));
      for (var i = 0; i < 8; i++) {
        await memory.saveMemory('unique$i standalone$i fact$i');
        clock.advance(4000);
      }
      await memory.maintain();
      final stats = await memory.stats();
      expect(stats.l1, lessThanOrEqualTo(2));
      expect(stats.l2, lessThanOrEqualTo(2));
      expect(stats.l3, lessThanOrEqualTo(2));
      expect(stats.total, lessThanOrEqualTo(6));
    });

    test('tombstones are swept (age or tier-percentage trigger)', () async {
      await build();
      final saved = await memory.saveMemory('ユーザーは犬を飼っている');
      await memory.deleteMemory(saved.id!);
      // With a near-empty tier the 10% trigger fires immediately; in larger
      // stores the 7-day age trigger guarantees the purge.
      await memory.maintain();
      expect(await store.getMemory(saved.id!), isNull);
      expect((await memory.stats()).tombstones, 0);
    });
  });

  group('explicit time and timezone inputs', () {
    setUp(build);

    test('saveMemory honours explicit nowUnix and timezone', () async {
      const explicitNow = 1800000000;
      const berlin = MemoryTimezone('Europe/Berlin', Duration(hours: 2));
      final r = await memory.saveMemory('ユーザーはベルリンで働いている',
          nowUnix: explicitNow, timezone: berlin);
      final row = (await store.getMemory(r.id!))!;
      expect(row.createdAt, explicitNow);
      expect(row.tz, 'Europe/Berlin;+02:00');
    });

    test('nowLocal formats with the resolved timezone', () {
      expect(memory.nowLocal(nowUnix: 1781181000), '2026-06-11 21:30 +09:00');
      expect(memory.nowLocal(nowUnix: 1781181000, timezone: MemoryTimezone.utc),
          '2026-06-11 12:30 +00:00');
    });

    test('clock rollback cannot mint immortal mass (I2)', () async {
      final r = await memory.saveMemory('ユーザーは犬を飼っている');
      clock.t -= 86400; // roll the clock back a day
      await memory.maintain();
      final row = (await store.getMemory(r.id!))!;
      expect(row.lastAccess, lessThanOrEqualTo(clock.t));
    });
  });

  group('lifecycle', () {
    setUp(build);

    test('stats and reset', () async {
      await memory.saveMemory('fact alpha beta');
      await memory.saveMemory('other gamma delta');
      expect((await memory.stats()).total, 2);
      await memory.reset();
      expect((await memory.stats()).total, 0);
      expect(await store.getMeta('spec_version'), 'ENGRAM v1.1');
    });
  });
}
