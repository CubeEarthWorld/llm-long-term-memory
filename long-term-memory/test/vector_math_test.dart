import 'dart:math';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

Float32List _randomVec(int dim, int seed) {
  final rng = Random(seed);
  final v = Float32List(dim);
  for (var i = 0; i < dim; i++) {
    v[i] = rng.nextDouble() * 2 - 1;
  }
  return v;
}

void main() {
  test('l2Normalized produces unit norm and keeps zero vectors intact', () {
    final v = l2Normalized(_randomVec(768, 1));
    expect(sqrt(dot(v, v)), closeTo(1.0, 1e-6));
    final zero = Float32List(8);
    expect(l2Normalized(zero), equals(zero));
  });

  test('truncateNormalize keeps the prefix direction (MRL property)', () {
    final v = l2Normalized(_randomVec(768, 2));
    final t = truncateNormalize(v, 256);
    expect(t.length, 256);
    expect(sqrt(dot(t, t)), closeTo(1.0, 1e-6));
    // Direction of the prefix is preserved: cosine with the raw prefix is 1.
    final prefix = Float32List.sublistView(v, 0, 256);
    expect(cosine(t, prefix), closeTo(1.0, 1e-6));
  });

  test('f32 pack/unpack round-trips exactly', () {
    final v = l2Normalized(_randomVec(64, 3));
    expect(unpackF32(packF32(v)), equals(v));
  });

  test('int8 quantisation round-trips with cosine > 0.99', () {
    final v = l2Normalized(_randomVec(256, 4));
    final q = quantizeInt8(v);
    expect(q.bytes.length, 256);
    final back = l2Normalized(dequantizeInt8(q.bytes, q.scale));
    expect(cosine(v, back), greaterThan(0.99));
  });

  test('cohesion is 1.0 for singletons and high for near-identical vectors',
      () {
    final v = l2Normalized(_randomVec(128, 5));
    expect(cohesion([v]), 1.0);
    expect(cohesion([v, v, v]), closeTo(1.0, 1e-5));
    final w = l2Normalized(_randomVec(128, 6));
    expect(cohesion([v, w]), lessThan(0.5));
  });

  test('sphericalKMeans is deterministic and separates distinct groups', () {
    final a = l2Normalized(_randomVec(64, 10));
    final b = l2Normalized(_randomVec(64, 20));
    Float32List jitter(Float32List base, int seed) {
      final noise = _randomVec(64, seed);
      final out = Float32List(64);
      for (var i = 0; i < 64; i++) {
        out[i] = base[i] + 0.05 * noise[i];
      }
      return l2Normalized(out);
    }

    final rows = [
      jitter(a, 1),
      jitter(a, 2),
      jitter(a, 3),
      jitter(b, 4),
      jitter(b, 5),
      jitter(b, 6),
    ];
    final labels1 = sphericalKMeans(rows, 2);
    final labels2 = sphericalKMeans(rows, 2);
    expect(labels1, equals(labels2));
    expect(labels1.sublist(0, 3).toSet().length, 1);
    expect(labels1.sublist(3).toSet().length, 1);
    expect(labels1[0], isNot(labels1[3]));
  });
}
