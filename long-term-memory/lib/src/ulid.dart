import 'dart:math';

/// Crockford base32 alphabet used by ULID (excludes I, L, O, U).
const String _crockford = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

/// Generates ULIDs (Universally Unique Lexicographically Sortable Identifiers).
///
/// A ULID is 26 Crockford-base32 characters encoding a 48-bit millisecond
/// Unix timestamp followed by 80 random bits. Because the timestamp occupies
/// the high bits, **lexicographic order equals time order** (ENGRAM §3.2),
/// which the engine relies on for cheap "oldest first" semantics.
///
/// The clock and the random source are injectable so tests (and engines
/// running on a virtual clock) produce deterministic, time-consistent ids.
class UlidGenerator {
  /// Creates a generator.
  ///
  /// [millis] returns the current Unix time in **milliseconds** (defaults to
  /// the wall clock). [random] defaults to a cryptographically secure RNG.
  UlidGenerator({int Function()? millis, Random? random})
      : _millis = millis ?? (() => DateTime.now().millisecondsSinceEpoch),
        _random = random ?? Random.secure();

  final int Function() _millis;
  final Random _random;

  /// Returns a new 26-character ULID string.
  String next() {
    var ts = _millis();
    if (ts < 0) ts = 0;
    final codes = List<int>.filled(26, 0);
    // Time part: 10 chars × 5 bits = 50 bits (top 2 bits zero for a 48-bit ts).
    for (var i = 9; i >= 0; i--) {
      codes[i] = _crockford.codeUnitAt(ts % 32);
      ts = ts ~/ 32;
    }
    // Random part: 16 chars × 5 bits = 80 random bits.
    for (var i = 10; i < 26; i++) {
      codes[i] = _crockford.codeUnitAt(_random.nextInt(32));
    }
    return String.fromCharCodes(codes);
  }
}

final UlidGenerator _defaultGenerator = UlidGenerator();

/// Returns a new ULID using the wall clock and a secure RNG.
String ulid() => _defaultGenerator.next();
