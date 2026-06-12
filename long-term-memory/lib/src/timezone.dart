/// Timezone value passed alongside every write: an IANA name plus a fixed
/// UTC offset.
///
/// ENGRAM stores the timezone of each memory as the string
/// `'IANA_name;+HH:MM'` (e.g. `'Asia/Tokyo;+09:00'`, spec §3.2): the IANA
/// name survives for tooling that has a tz database, while the explicit
/// offset keeps local-time formatting working with pure arithmetic — no
/// timezone database is required by this package.
///
/// Pure Dart has no IANA database, so the offset must be supplied by the
/// caller (e.g. `DateTime.now().timeZoneOffset`, or `package:timezone` if
/// the app needs exact historical DST).
class MemoryTimezone {
  /// Creates a timezone from an IANA [name] (e.g. `'Asia/Tokyo'`) and its
  /// current UTC [offset] (e.g. `Duration(hours: 9)`).
  const MemoryTimezone(this.name, this.offset);

  /// Builds a timezone from a [DateTime]'s platform-reported zone.
  ///
  /// Note: `DateTime.timeZoneName` is a platform abbreviation (e.g. `JST`),
  /// not always an IANA name. Prefer the main constructor when you know the
  /// IANA name.
  factory MemoryTimezone.fromDateTime(DateTime dateTime) =>
      MemoryTimezone(dateTime.timeZoneName, dateTime.timeZoneOffset);

  /// The device's current local timezone.
  factory MemoryTimezone.local() => MemoryTimezone.fromDateTime(DateTime.now());

  /// Parses the stored `'name;+HH:MM'` field back into a [MemoryTimezone].
  ///
  /// A missing or malformed offset part is treated as UTC.
  factory MemoryTimezone.parse(String storageField) {
    final parts = storageField.split(';');
    final name = parts.isNotEmpty && parts[0].isNotEmpty ? parts[0] : 'UTC';
    final offset = parts.length > 1
        ? (parseUtcOffset(parts[1]) ?? Duration.zero)
        : Duration.zero;
    return MemoryTimezone(name, offset);
  }

  /// UTC (offset zero).
  static const MemoryTimezone utc = MemoryTimezone('UTC', Duration.zero);

  /// IANA timezone name (or platform label).
  final String name;

  /// Fixed UTC offset at write time.
  final Duration offset;

  /// The offset as `'+HH:MM'` / `'-HH:MM'`.
  String get offsetString => formatUtcOffset(offset);

  /// The canonical stored form, e.g. `'Asia/Tokyo;+09:00'`.
  String get storageField => '$name;$offsetString';

  @override
  String toString() => storageField;

  @override
  bool operator ==(Object other) =>
      other is MemoryTimezone && other.name == name && other.offset == offset;

  @override
  int get hashCode => Object.hash(name, offset);
}

/// Formats a UTC offset as `'+HH:MM'` (e.g. `'+09:00'`, `'-03:30'`).
String formatUtcOffset(Duration offset) {
  final sign = offset.isNegative ? '-' : '+';
  final total = offset.inMinutes.abs();
  final h = (total ~/ 60).toString().padLeft(2, '0');
  final m = (total % 60).toString().padLeft(2, '0');
  return '$sign$h:$m';
}

/// Parses `'+HH:MM'` / `'-HHMM'` / `'+HH'` into a [Duration], or `null`.
Duration? parseUtcOffset(String text) {
  final m = RegExp(r'^([+-])(\d{1,2}):?(\d{2})?$').firstMatch(text.trim());
  if (m == null) return null;
  final sign = m.group(1) == '-' ? -1 : 1;
  final hours = int.parse(m.group(2)!);
  final minutes = int.parse(m.group(3) ?? '0');
  return Duration(minutes: sign * (hours * 60 + minutes));
}

/// Formats a Unix time as a local datetime string using a stored tz field.
///
/// Returns `'2026-06-11 21:30 +09:00'` for `formatLocal(unix, 'Asia/Tokyo;+09:00')`.
/// Only the explicit offset is used (pure arithmetic, ENGRAM §3.2) — no
/// timezone database lookup, so it works for any year including >9999.
String formatLocal(int unixSeconds, String tzField) {
  final parts = tzField.split(';');
  final offText = parts.length > 1 ? parts[1] : '+00:00';
  final offset = parseUtcOffset(offText) ?? Duration.zero;
  final local = unixSeconds + offset.inSeconds;
  var days = local ~/ 86400;
  var secondsOfDay = local % 86400;
  if (secondsOfDay < 0) {
    secondsOfDay += 86400;
    days -= 1;
  }
  // 719163 = proleptic Gregorian ordinal of 1970-01-01.
  final (year, month, day) = ymdFromOrdinal(719163 + days);
  final hour = secondsOfDay ~/ 3600;
  final minute = (secondsOfDay % 3600) ~/ 60;
  String two(int v) => v.toString().padLeft(2, '0');
  final y = year.toString().padLeft(4, '0');
  return '$y-${two(month)}-${two(day)} ${two(hour)}:${two(minute)} '
      '${formatUtcOffset(offset)}';
}

/// Converts a proleptic Gregorian ordinal (1 = 0001-01-01) to (year, month,
/// day). Works for any year — used so timestamps beyond Dart's `DateTime`
/// comfort zone still format (64-bit Unix seconds per spec I9).
(int, int, int) ymdFromOrdinal(int ordinal) {
  var n = ordinal - 1;
  final n400 = n ~/ 146097;
  n %= 146097;
  var n100 = n ~/ 36524;
  n %= 36524;
  if (n100 > 3) {
    n100 = 3;
    n += 36524;
  }
  final n4 = n ~/ 1461;
  n %= 1461;
  var n1 = n ~/ 365;
  n %= 365;
  if (n1 > 3) {
    n1 = 3;
    n += 365;
  }
  final year = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1;
  final isLeap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
  final daysInMonth = [
    31,
    isLeap ? 29 : 28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31
  ];
  var month = 1;
  for (final dim in daysInMonth) {
    if (n < dim) break;
    n -= dim;
    month += 1;
  }
  return (year, month, n + 1);
}
