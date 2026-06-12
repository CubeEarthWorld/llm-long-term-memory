import 'dart:math';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

void main() {
  group('ULID', () {
    test('is 26 Crockford chars and sorts by time', () {
      var t = 1700000000000;
      final gen = UlidGenerator(millis: () => t, random: Random(0));
      final a = gen.next();
      t += 1000;
      final b = gen.next();
      expect(a, matches(RegExp(r'^[0-9A-HJKMNP-TV-Z]{26}$')));
      expect(a.compareTo(b), lessThan(0));
    });

    test('default generator produces unique ids', () {
      final seen = {for (var i = 0; i < 100; i++) ulid()};
      expect(seen.length, 100);
    });
  });

  group('MemoryTimezone', () {
    test('builds and parses the storage field', () {
      const tz = MemoryTimezone('Asia/Tokyo', Duration(hours: 9));
      expect(tz.storageField, 'Asia/Tokyo;+09:00');
      final back = MemoryTimezone.parse('Asia/Tokyo;+09:00');
      expect(back, tz);
      expect(MemoryTimezone.parse('Bad').offset, Duration.zero);
    });

    test('negative and fractional offsets', () {
      const nf =
          MemoryTimezone('America/St_Johns', Duration(hours: -3, minutes: -30));
      expect(nf.storageField, 'America/St_Johns;-03:30');
      expect(MemoryTimezone.parse(nf.storageField).offset,
          const Duration(hours: -3, minutes: -30));
    });

    test('formatLocal renders offset arithmetic', () {
      // 2026-06-11 12:30:00 UTC → 21:30 in +09:00.
      const unix = 1781181000;
      expect(formatLocal(unix, 'Asia/Tokyo;+09:00'), '2026-06-11 21:30 +09:00');
      expect(formatLocal(unix, 'UTC;+00:00'), '2026-06-11 12:30 +00:00');
      expect(formatLocal(0, 'UTC;+00:00'), '1970-01-01 00:00 +00:00');
    });
  });

  group('text utils', () {
    test('splitParagraphs splits paragraphs, lines, then sentences', () {
      expect(splitParagraphs('a\n\nb'), ['a', 'b']);
      expect(splitParagraphs('a\nb'), ['a', 'b']);
      final long = '今日はとても良く晴れています。明日は朝から雨が降りそうです。'
          '来週は友人と一緒に京都へ旅行に行く予定があります。新幹線の座席はもう予約しました。'
          'ホテルは京都駅前のところにしました。今からとても楽しみにしています。';
      expect(splitParagraphs(long).length, greaterThan(2));
    });

    test('shorten cuts at a sentence boundary', () {
      final text = '${'あ' * 100}。${'い' * 100}';
      final cut = shorten(text, 170);
      expect(cut.length, lessThanOrEqualTo(170));
      expect(cut.endsWith('。'), isTrue);
      expect(shorten('short', 170), 'short');
    });

    test('clusterFingerprint is order-independent', () {
      expect(clusterFingerprint(['b', 'a']), clusterFingerprint(['a', 'b']));
      expect(clusterFingerprint(['a']), isNot(clusterFingerprint(['b'])));
    });
  });

  group('EngramConfig', () {
    test('defaults match the spec', () {
      const c = EngramConfig();
      expect(c.cap1, 1000);
      expect(c.dimOf(2), 256);
      expect(c.tauOf(1), 7 * 86400);
      expect(c.thetaSame, 0.97);
      expect(c.mMax, 64.0);
    });

    test('JSON round-trip', () {
      const c = EngramConfig(cap1: 10, alpha: 0.5);
      final back = EngramConfig.fromJson(c.toJson());
      expect(back.cap1, 10);
      expect(back.alpha, 0.5);
      expect(back.cap2, c.cap2);
    });
  });

  group('parsing', () {
    test('DreamDecision.parseJson handles fences, prose and bad input', () {
      final fenced = DreamDecision.parseJson(
          '```json\n{"action":"merge","memories":[{"text":"x"}]}\n```');
      expect(fenced.action, DreamAction.merge);
      expect(fenced.memories.single.text, 'x');

      final prose = DreamDecision.parseJson(
          'Sure! Here is the result: {"action":"split","memories":'
          '[{"text":"a"},{"text":"b"}]} hope that helps');
      expect(prose.action, DreamAction.split);
      expect(prose.memories.length, 2);

      final inferred =
          DreamDecision.parseJson('{"action":"???","memories":[{"text":"y"}]}');
      expect(inferred.action, DreamAction.merge);

      expect(DreamDecision.parseJson('garbage').action, DreamAction.none);
      expect(DreamDecision.parseJson(null).action, DreamAction.none);
      expect(
          DreamDecision.parseJson('{"action":"merge","memories":'
                  '[{"text":"  "},{"no":"text"}]}')
              .memories,
          isEmpty);
    });

    test('parseExtractedTexts accepts dict, list and text-items', () {
      expect(EngramPrompts.parseExtractedTexts('{"memories":["a","b"]}'),
          ['a', 'b']);
      expect(
          EngramPrompts.parseExtractedTexts('["a", {"text":"b"}]'), ['a', 'b']);
      expect(EngramPrompts.parseExtractedTexts('not json'), isEmpty);
    });
  });
}
