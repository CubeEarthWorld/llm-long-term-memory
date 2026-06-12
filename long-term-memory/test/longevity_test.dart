@Tags(['slow'])
library;

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

/// Compressed port of the reference longevity scenario: months of virtual
/// time, hundreds of writes, periodic maintenance and dreams — asserting
/// the invariants hold and that important facts stay retrievable while
/// noise fades.
void main() {
  test('months of use keep invariants and recall quality', () async {
    final clock = VirtualClock();
    final store = InMemoryStore();
    final memory = EngramMemory(
      store: store,
      embedder: FakeEmbedder(),
      config: const EngramConfig(cap1: 40, cap2: 60, cap3: 80),
      clock: clock.call,
      defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
    );
    await memory.initialize();

    const keyFact = 'ユーザーは抹茶アイスクリームが大好物';
    await memory.saveMemory(keyFact);

    // 90 virtual days: daily noise writes, the key fact rehearsed weekly,
    // a dream every 10 days.
    for (var day = 1; day <= 90; day++) {
      clock.advanceDays(1);
      await memory.saveMemory('day$day note topic$day detail${day % 7}');
      if (day % 7 == 0) {
        final r = await memory.retrieve('抹茶アイスクリームは好き？');
        expect(r.recalled.map((m) => m.text), contains(keyFact),
            reason: 'key fact must stay retrievable on day $day');
      }
      await memory.maintain();
      if (day % 10 == 0) {
        await memory.dream(adjudicate: mergeToGist, budget: 3);
      }
    }

    final stats = await memory.stats();
    // Invariants: capacities respected (I4), mass bounded (I1), gen ≤ 7
    // (I7), no dangling vectors (I6), rings bounded (I11).
    expect(stats.l1, lessThanOrEqualTo(40));
    expect(stats.l2, lessThanOrEqualTo(60));
    expect(stats.l3, lessThanOrEqualTo(80));
    expect(stats.conflicts, lessThanOrEqualTo(256));
    expect(stats.dreamLog, lessThanOrEqualTo(512));
    final rows = await store.listMemories();
    final ids = rows.map((m) => m.id).toSet();
    for (final m in rows) {
      expect(m.mass, lessThanOrEqualTo(64.0));
      expect(m.gen, lessThanOrEqualTo(7));
      expect(m.text.length, lessThanOrEqualTo(170));
    }
    for (final v in await store.listVectors()) {
      expect(ids, contains(v.memoryId));
    }

    // The rehearsed key fact outranks same-age noise.
    final final$ = await memory.retrieve('抹茶アイスクリーム');
    expect(final$.recalled.first.text, keyFact);
    // Weekly rehearsal equilibrium for L1 (τ=7 d): m ≈ m/2 + 1 → 2.
    final keyRow = (await store.getLiveMemoryByText(keyFact))!;
    expect(keyRow.mass, greaterThan(1.5));
  }, timeout: const Timeout(Duration(minutes: 3)));
}
