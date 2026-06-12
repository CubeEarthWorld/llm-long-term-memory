import 'dart:typed_data';

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

  /// Three memories sharing most tokens: one coherent cluster, pairwise
  /// cosine below the conflict band so they stay separate rows.
  Future<List<String>> saveCluster() async {
    final ids = <String>[];
    for (final text in const [
      'user likes hot coffee in the morning routine',
      'user likes iced coffee in the evening routine',
      'user likes black coffee on sunday routine',
    ]) {
      final r = await memory.saveMemory(text);
      expect(r.action, isNot(SaveAction.rejected));
      ids.add(r.id!);
      clock.advance(4000);
    }
    return ids;
  }

  group('dream (§6)', () {
    setUp(build);

    test('merge consolidates the cluster into one gist (gen+1)', () async {
      final ids = await saveCluster();
      final reports = await memory.dream(adjudicate: mergeToGist);
      final merged = reports.where((r) => r.action == DreamAction.merge);
      expect(merged, hasLength(1));
      final report = merged.single;
      expect(report.deletedIds.toSet(), ids.toSet());
      expect(report.after, hasLength(1));
      expect(report.after.single.gen, 1);
      for (final id in ids) {
        expect(await store.getMemory(id), isNull);
      }
      final gist = (await store.getMemory(report.after.single.id))!;
      expect(gist.tier, 1);
      // Mass carries the members' summed activation (≤ 64).
      expect(gist.mass, greaterThan(1.0));
      expect(gist.mass, lessThanOrEqualTo(64.0));
      expect(await store.getDreamVerdict(report.clusterFingerprint), 'merge');
    });

    test('split distributes mass across the new memories', () async {
      await saveCluster();
      final reports = await memory.dream(
        adjudicate: (req) => DreamDecision(
          action: DreamAction.split,
          memories: const [
            DreamProposal(text: 'user likes coffee'),
            DreamProposal(text: 'user has a morning routine'),
          ],
        ),
      );
      final split = reports.where((r) => r.action == DreamAction.split);
      expect(split, hasLength(1));
      expect(split.single.after, hasLength(2));
      // Split keeps gen (no +1).
      expect(split.single.after.every((s) => s.gen == 0), isTrue);
    });

    test('none verdict records the fingerprint and skips next time', () async {
      await saveCluster();
      final first = await memory.dream(adjudicate: alwaysNone);
      expect(first, isNotEmpty);
      expect(first.first.action, DreamAction.none);
      var called = 0;
      final second = await memory.dream(adjudicate: (req) {
        called++;
        return const DreamDecision.none();
      });
      expect(second, isEmpty); // fingerprint cache skipped the cluster
      expect(called, 0);
      // force re-adjudicates.
      final forced = await memory.dream(adjudicate: alwaysNone, force: true);
      expect(forced, isNotEmpty);
    });

    test('a throwing adjudicator leaves the cluster untouched and retryable',
        () async {
      final ids = await saveCluster();
      final reports =
          await memory.dream(adjudicate: (req) => throw StateError('LLM down'));
      expect(reports.single.action, DreamAction.none);
      expect(reports.single.error, contains('LLM down'));
      for (final id in ids) {
        expect(await store.getMemory(id), isNotNull);
      }
      expect(await store.getDreamVerdict(reports.single.clusterFingerprint),
          isNull);
      // Next dream retries and succeeds.
      final retry = await memory.dream(adjudicate: mergeToGist);
      expect(retry.single.action, DreamAction.merge);
    });

    test('budget zero performs housekeeping only', () async {
      await saveCluster();
      final reports = await memory.dream(adjudicate: mergeToGist, budget: 0);
      expect(reports, isEmpty);
      expect(await memory.totalRecords(), 3);
    });

    test('long replacement texts are shortened, empty ones dropped', () async {
      await saveCluster();
      final reports = await memory.dream(
        adjudicate: (req) => DreamDecision(
          action: DreamAction.merge,
          memories: [
            DreamProposal(text: '${'う' * 100}。${'え' * 200}'),
            const DreamProposal(text: '   '),
          ],
        ),
      );
      final merged = reports.where((r) => r.action == DreamAction.merge).single;
      expect(merged.after, hasLength(1));
      expect(merged.after.single.text.length, lessThanOrEqualTo(170));
    });

    test('promotion: hot L2 memories return to L1 with full vectors', () async {
      await build(config: const EngramConfig(cap1: 1));
      // Two memories; capacity 1 forces one demotion to L2.
      await memory.saveMemory('alpha topic one fact');
      clock.advance(4000);
      final hot = await memory.saveMemory('beta topic two fact');
      await memory.maintain();
      expect((await memory.stats()).l2, 1);
      // Reinforce whichever row is in L2 until A ≥ θ_up.
      final l2row =
          (await store.listMemories(tier: 2, liveness: Liveness.liveOnly))
              .single;
      for (var i = 0; i < 20; i++) {
        await memory.saveMemory(l2row.text); // exact rehearsal, +1 each
      }
      expect((await memory.activationOf(l2row.id))!, greaterThanOrEqualTo(16));
      await memory.dream(adjudicate: alwaysNone, budget: 0);
      final promoted = (await store.getMemory(l2row.id))!;
      expect(promoted.tier, 1);
      final vec = (await store.getVector(l2row.id, 'fake/token-overlap'))!;
      expect(vec.dtype, VecDtype.f32);
      expect(vec.dim, 768);
      expect(hot.id, isNotNull);
    });

    test('housekeeping purges tombstones and foreign-model vectors', () async {
      final saved = await memory.saveMemory('user lives in kyoto japan');
      await memory.deleteMemory(saved.id!);
      clock.advanceDays(8);
      await store.upsertVector(VectorRecord(
        memoryId: 'ghost',
        modelId: 'other/model',
        dim: 4,
        dtype: VecDtype.f32,
        bytes: packF32(Float32List.fromList([1, 0, 0, 0])),
      ));
      await memory.dream(adjudicate: alwaysNone, budget: 0);
      expect(await store.getMemory(saved.id!), isNull);
      expect(await store.listVectors(), isEmpty);
    });

    test('request carries local times and a ready-to-use prompt', () async {
      await saveCluster();
      DreamClusterRequest? seen;
      await memory.dream(adjudicate: (req) {
        seen = req;
        return const DreamDecision.none();
      });
      expect(seen, isNotNull);
      expect(seen!.members.length, 3);
      expect(seen!.members.first.timezone, 'Asia/Tokyo;+09:00');
      expect(seen!.members.first.localTime, contains('+09:00'));
      final prompt = seen!.buildPrompt();
      expect(prompt, contains('merge'));
      expect(prompt, contains(seen!.members.first.text));
      final promptEn = seen!.buildPrompt(locale: EngramLocale.en);
      expect(promptEn, contains('consolidation'));
    });
  });
}
