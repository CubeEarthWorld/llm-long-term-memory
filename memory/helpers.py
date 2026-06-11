"""Pure module-level helpers for the ENGRAM engine (no engine state).

Extracted from ``memory.long_term_memory`` so the engine class file stays
focused on stateful behaviour; everything here is a deterministic function of
its arguments (plus the clock/randomness inside ``ulid``).
"""
from __future__ import annotations

import hashlib
import re
import secrets
import time
from datetime import datetime, timedelta, timezone as dt_timezone

import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[misc,assignment]

from core.embedding import l2_normalize


def lam_mmr(score: float, lam: float, max_sim: float) -> float:
    return score - lam * max_sim


def _fp(ids: list[str]) -> str:
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()


def _cohesion(coarses: list[np.ndarray]) -> float:
    """Mean pairwise cosine of a cluster's coarse vectors (∈ (0,1])."""
    n = len(coarses)
    if n < 2:
        return 1.0
    M = np.stack(coarses)
    sims = M @ M.T
    total = float(sims.sum() - np.trace(sims))
    return max(0.0, total / (n * (n - 1)))


def _kmeans(X: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> np.ndarray:
    """Tiny spherical k-means on unit-norm rows (cosine = dot). Method is free per §6."""
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(iters):
        new = (X @ cent.T).argmax(axis=1)
        if it > 0 and np.array_equal(new, labels):
            labels = new
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                cent[j] = l2_normalize(members.mean(axis=0))
    return labels


def _split_paragraphs(text: str) -> list[str]:
    """Split a turn into chunks so a memory relevant to any part can surface (§5.2)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(blocks) <= 2 and len(text) > 80:
        sents = [s.strip() for s in re.split(r"(?<=[。！？…\n])", text) if s.strip()]
        if len(sents) > len(blocks):
            blocks = sents
    return blocks or []


def _utc_datetime(unix) -> datetime:
    """UNIX seconds → aware UTC datetime; OverflowError beyond year 9999.

    Built via epoch + timedelta (not ``fromtimestamp``) so it works for the
    full pre-9999 range on every platform.
    """
    utc_dt = datetime(1970, 1, 1, tzinfo=dt_timezone.utc) + timedelta(seconds=float(unix))
    if utc_dt.year > 9999:
        raise OverflowError("year > 9999")
    return utc_dt


def _fmt_local(unix, tzfield) -> str:
    """'2026-06-11 21:30 +09:00' from a UNIX time and a stored 'IANA;+offset' field (§5.2)."""
    parts = str(tzfield).split(";")
    name = parts[0]
    off = parts[1] if len(parts) > 1 else "+00:00"
    try:
        dt = _utc_datetime(unix).astimezone(ZoneInfo(name))
        off = dt.strftime("%z") or "+0000"
        return f"{dt:%Y-%m-%d %H:%M} {off[:3]}:{off[3:]}"
    except Exception:
        try:
            dt = _utc_datetime(unix).astimezone(dt_timezone.utc)
            return f"{dt:%Y-%m-%d %H:%M} {off}"
        except Exception:
            # Beyond datetime.MAXYEAR (9999) — format manually
            secs = int(float(unix))
            mins, sec = divmod(secs, 60)
            hrs, min_ = divmod(mins, 60)
            days, hr = divmod(hrs, 24)
            # Gregorian calendar: days since 1970-01-01
            year, month, day = _ymd_from_ordinal(719163 + days)  # 719163 = date(1970,1,1).toordinal()
            return f"{year:04d}-{month:02d}-{day:02d} {hr:02d}:{min_:02d} {off}"


def _ymd_from_ordinal(n: int) -> tuple[int, int, int]:
    """Convert proleptic Gregorian ordinal (1 = 0001-01-01) to (year, month, day).
    Based on the algorithm from `datetime._ymd2ord` reversed. Works for any year."""
    # Adjust for 0001-01-01 being ordinal 1
    n -= 1
    # 400-year cycles: 146097 days
    n400, n = divmod(n, 146097)
    y400 = n400 * 400
    # 100-year cycles within 400: 36524 days (except every 400th)
    n100, n = divmod(n, 36524)
    if n100 > 3:
        n100 = 3
        n += 36524
    y100 = n100 * 100
    # 4-year cycles: 1461 days (except every 100th)
    n4, n = divmod(n, 1461)
    y4 = n4 * 4
    # 1-year cycles: 365 days (except every 4th)
    n1, n = divmod(n, 365)
    if n1 > 3:
        n1 = 3
        n += 365
    y1 = n1
    year = y400 + y100 + y4 + y1 + 1
    # Month/day within the year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_month = [31, 28 + is_leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = 1
    for dim in days_in_month:
        if n < dim:
            break
        n -= dim
        month += 1
    day = n + 1
    return year, month, day


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("。", "．", ".", "、", " "):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.5:
            return cut[: idx + 1]
    return cut


# Crockford base32 alphabet (RFC 9562 / ULID): excludes I, L, O, U.
_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid_fallback() -> str:
    """Self-contained ULID: 48-bit ms timestamp + 80 random bits, 26 Crockford chars.

    Keeps the engine running even if the optional ``python-ulid`` package is not
    installed (it lives only here, in a hot path). Sort order still equals time
    order because the timestamp occupies the high bits (§3.2).
    """
    ts = int(time.time() * 1000) & ((1 << 48) - 1)
    val = (ts << 80) | secrets.randbits(80)          # 128-bit ULID value
    chars = []
    for _ in range(26):                               # 26 × 5 bits = 130 bits (top 2 are 0)
        chars.append(_CROCKFORD32[val & 0x1F])
        val >>= 5
    return "".join(reversed(chars))


def ulid() -> str:
    """ULID as string: first 48 bits = ms Unix time → sort order = time order (§3.2).

    Uses ``python-ulid`` when available, otherwise the self-contained fallback so a
    missing/optional dependency never hard-crashes the write/dream path.
    """
    try:
        from ulid import ULID
        return str(ULID())
    except Exception:
        return _ulid_fallback()
