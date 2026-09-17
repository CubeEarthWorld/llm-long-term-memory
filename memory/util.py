"""Pure helpers shared by the engine: ids, text hygiene, cues, local time."""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone as dt_timezone

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[misc,assignment]

_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(millis: int) -> str:
    """26-char ULID-shaped id: 10 chars = 50-bit ms timestamp (valid past year
    37,000), 16 chars = 80 random bits. Lexicographic order = time order."""
    ts = max(0, int(millis))
    head = []
    for _ in range(10):
        head.append(_CROCKFORD32[ts % 32])
        ts //= 32
    tail = [_CROCKFORD32[secrets.randbelow(32)] for _ in range(16)]
    return "".join(reversed(head)) + "".join(tail)


def shorten(text: str, max_chars: int) -> str:
    """Truncate to ``max_chars``, preferring a sentence/clause boundary in the second half."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("。", "．", ".", "、", " "):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.5:
            return cut[: idx + 1]
    return cut


def clean_text(text: str, max_chars: int) -> str:
    """Strip the pack delimiters ``《》``, collapse whitespace, trim, shorten."""
    return shorten(re.sub(r"\s+", " ", re.sub("[《》]", "", text or "")).strip(), max_chars)


_CUE_BREAK = re.compile(r"\n+|(?<=[。！？])|(?<=[.!?])\s+")


def cues(text: str, max_cues: int) -> list[str]:
    """Split a query into lines/sentences; >max_cues pieces are coalesced (nothing dropped)."""
    parts = [p.strip() for p in _CUE_BREAK.split(text) if p and p.strip()]
    n = len(parts)
    if n <= max_cues:
        return parts
    return [" ".join(parts[i * n // max_cues:(i + 1) * n // max_cues]) for i in range(max_cues)]


def tz_field(name: str, now: float) -> str:
    """'IANA;+HH:MM' for timezone ``name`` at ``now`` (offset frozen into the field)."""
    try:
        off = (datetime(1970, 1, 1, tzinfo=dt_timezone.utc) + timedelta(seconds=float(now))) \
            .astimezone(ZoneInfo(name)).strftime("%z") or "+0000"
        return f"{name};{off[:3]}:{off[3:]}"
    except Exception:
        return f"{name};+00:00"


def parse_offset(text: str) -> int:
    """'+HH:MM' / '-HHMM' / '+HH' → seconds, 0 if malformed."""
    m = re.match(r"^([+-])(\d{1,2}):?(\d{2})?$", (text or "").strip())
    if not m:
        return 0
    sign = -1 if m.group(1) == "-" else 1
    return sign * (int(m.group(2)) * 3600 + int(m.group(3) or 0) * 60)


def fmt_local(unix, tzfield: str) -> str:
    """'2026-06-11 21:30 +09:00' from Unix seconds and a stored 'IANA;+offset' field.

    Pure offset arithmetic (no tz database), valid for any year (SPEC §2)."""
    parts = str(tzfield).split(";")
    off = parts[1] if len(parts) > 1 else "+00:00"
    seconds = parse_offset(off)
    local = int(unix) + seconds
    days, sod = divmod(local, 86400)
    year, month, day = _ymd_from_ordinal(719163 + days)   # 719163 = ordinal of 1970-01-01
    sign = "-" if seconds < 0 else "+"
    a = abs(seconds)
    return f"{year:04d}-{month:02d}-{day:02d} {sod // 3600:02d}:{sod % 3600 // 60:02d} {sign}{a // 3600:02d}:{a % 3600 // 60:02d}"


def _ymd_from_ordinal(n: int) -> tuple[int, int, int]:
    """Proleptic Gregorian ordinal (1 = 0001-01-01) → (year, month, day), any year."""
    n -= 1
    n400, n = divmod(n, 146097)
    n100, n = divmod(n, 36524)
    if n100 > 3:
        n100, n = 3, n + 36524
    n4, n = divmod(n, 1461)
    n1, n = divmod(n, 365)
    if n1 > 3:
        n1, n = 3, n + 365
    year = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1
    leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    for month, dim in enumerate((31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), 1):
        if n < dim:
            return year, month, n + 1
        n -= dim
    return year, 12, 31  # pragma: no cover
