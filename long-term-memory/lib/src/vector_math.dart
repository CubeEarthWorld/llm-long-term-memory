/// Pure-Dart vector math used by the engine: L2 normalisation, MRL
/// (Matryoshka) truncation, dot/cosine, int8 quantisation and spherical
/// k-means. All vectors handled by the engine are **unit-norm**
/// [Float32List]s (ENGRAM I3); accumulation happens in doubles.
library;

import 'dart:math';
import 'dart:typed_data';

/// Returns an L2-normalised copy of [v] (zero vectors are returned as-is).
Float32List l2Normalized(Float32List v) {
  var sum = 0.0;
  for (var i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  final n = sqrt(sum);
  final out = Float32List(v.length);
  if (n == 0.0) {
    out.setAll(0, v);
    return out;
  }
  for (var i = 0; i < v.length; i++) {
    out[i] = v[i] / n;
  }
  return out;
}

/// MRL truncation: keep the first [dim] components, then renormalise
/// (ENGRAM §2, §8). A 768-d Matryoshka embedding truncated to 256/128 keeps
/// its semantics at lower resolution — no re-embedding needed.
Float32List truncateNormalize(Float32List v, int dim) {
  if (v.length <= dim) return l2Normalized(v);
  return l2Normalized(Float32List.sublistView(v, 0, dim));
}

/// Dot product (== cosine for unit-norm inputs). Lengths may differ; the
/// shorter length is used.
double dot(Float32List a, Float32List b) {
  final n = a.length < b.length ? a.length : b.length;
  var sum = 0.0;
  for (var i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/// Cosine similarity robust to non-normalised inputs.
double cosine(Float32List a, Float32List b) {
  var na = 0.0, nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    na += a[i] * a[i];
  }
  for (var i = 0; i < b.length; i++) {
    nb += b[i] * b[i];
  }
  if (na == 0.0 || nb == 0.0) return 0.0;
  return dot(a, b) / (sqrt(na) * sqrt(nb));
}

/// Packs a float32 vector into little-endian bytes (4 bytes/dim, L1 format).
Uint8List packF32(Float32List v) {
  final data = ByteData(v.length * 4);
  for (var i = 0; i < v.length; i++) {
    data.setFloat32(i * 4, v[i], Endian.little);
  }
  return data.buffer.asUint8List();
}

/// Inverse of [packF32].
Float32List unpackF32(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  final out = Float32List(bytes.length ~/ 4);
  for (var i = 0; i < out.length; i++) {
    out[i] = data.getFloat32(i * 4, Endian.little);
  }
  return out;
}

/// Result of [quantizeInt8]: the signed-int8 bytes and the dequantisation
/// scale such that `value ≈ byte × scale`.
class QuantizedVector {
  const QuantizedVector(this.bytes, this.scale);

  /// Two's-complement int8 values stored byte-per-dim.
  final Uint8List bytes;

  /// Symmetric dequantisation factor.
  final double scale;
}

/// Symmetric int8 quantisation of a unit-norm vector (L2/L3 format, 1
/// byte/dim): `scale = max|v| / 127`, values clamped to ±127.
QuantizedVector quantizeInt8(Float32List v) {
  var peak = 0.0;
  for (var i = 0; i < v.length; i++) {
    final a = v[i].abs();
    if (a > peak) peak = a;
  }
  final scale = peak > 0 ? peak / 127.0 : 1.0;
  final bytes = Uint8List(v.length);
  for (var i = 0; i < v.length; i++) {
    var q = (v[i] / scale).round();
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    bytes[i] = q & 0xFF;
  }
  return QuantizedVector(bytes, scale);
}

/// Inverse of [quantizeInt8]. The caller must renormalise before computing
/// cosines (the engine always does).
Float32List dequantizeInt8(Uint8List bytes, double scale) {
  final out = Float32List(bytes.length);
  for (var i = 0; i < bytes.length; i++) {
    final b = bytes[i];
    out[i] = (b > 127 ? b - 256 : b) * scale;
  }
  return out;
}

/// Mean pairwise cosine of a set of unit-norm vectors, in (0, 1].
/// Returns 1.0 for fewer than two vectors (ENGRAM §6 cluster cohesion).
double cohesion(List<Float32List> vectors) {
  final n = vectors.length;
  if (n < 2) return 1.0;
  var total = 0.0;
  for (var i = 0; i < n; i++) {
    for (var j = i + 1; j < n; j++) {
      total += dot(vectors[i], vectors[j]);
    }
  }
  return max(0.0, 2.0 * total / (n * (n - 1)));
}

/// Tiny spherical k-means on unit-norm rows (cosine == dot). Deterministic
/// for a given [seed]. Used by the dream phase to find candidate clusters;
/// the clustering method itself is spec-free (ENGRAM §6).
List<int> sphericalKMeans(
  List<Float32List> rows,
  int k, {
  int iterations = 12,
  int seed = 0,
}) {
  final n = rows.length;
  if (n == 0) return const [];
  k = k.clamp(1, n);
  final dim = rows[0].length;
  final rng = Random(seed);

  // Sample k distinct starting rows.
  final indices = List<int>.generate(n, (i) => i);
  for (var i = 0; i < k; i++) {
    final j = i + rng.nextInt(n - i);
    final t = indices[i];
    indices[i] = indices[j];
    indices[j] = t;
  }
  final centroids = List<Float32List>.generate(
      k, (i) => Float32List.fromList(rows[indices[i]]));

  var labels = List<int>.filled(n, 0);
  for (var it = 0; it < iterations; it++) {
    final next = List<int>.filled(n, 0);
    for (var i = 0; i < n; i++) {
      var bestJ = 0;
      var bestSim = double.negativeInfinity;
      for (var j = 0; j < k; j++) {
        final sim = dot(rows[i], centroids[j]);
        if (sim > bestSim) {
          bestSim = sim;
          bestJ = j;
        }
      }
      next[i] = bestJ;
    }
    if (it > 0 && _listEquals(next, labels)) {
      labels = next;
      break;
    }
    labels = next;
    for (var j = 0; j < k; j++) {
      final mean = Float64List(dim);
      var count = 0;
      for (var i = 0; i < n; i++) {
        if (labels[i] != j) continue;
        count++;
        final r = rows[i];
        for (var d = 0; d < dim; d++) {
          mean[d] += r[d];
        }
      }
      if (count == 0) continue;
      final c = Float32List(dim);
      for (var d = 0; d < dim; d++) {
        c[d] = mean[d] / count;
      }
      centroids[j] = l2Normalized(c);
    }
  }
  return labels;
}

bool _listEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
