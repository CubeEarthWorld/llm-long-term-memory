"""memory.long_term_memory module-level helpers (pure functions)."""
from __future__ import annotations

import datetime

import numpy as np
import pytest

from memory.long_term_memory import (
    _CROCKFORD32,
    _cohesion,
    _fmt_local,
    _fp,
    _kmeans,
    _shorten,
    _split_paragraphs,
    _ulid_fallback,
    _ymd_from_ordinal,
    lam_mmr,
    ulid,
)


# -- fingerprint -------------------------------------------------------- #
def test_fp_deterministic_and_order_sensitive():
    assert _fp(["a", "b"]) == _fp(["a", "b"])
    assert _fp(["a", "b"]) != _fp(["b", "a"])   # caller is responsible for sorting
    assert len(_fp(["x"])) == 64                 # sha-256 hexdigest


# -- cohesion ----------------------------------------------------------- #
def test_cohesion_single_member_is_one():
    assert _cohesion([np.array([1.0, 0.0])]) == 1.0


def test_cohesion_identical_vectors():
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert _cohesion([v, v, v]) == pytest.approx(1.0, abs=1e-6)


def test_cohesion_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert _cohesion([a, b]) == pytest.approx(0.0, abs=1e-6)


# -- k-means ------------------------------------------------------------ #
def test_kmeans_labels_in_range():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 8)).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = _kmeans(X, 4)
    assert len(labels) == 20
    assert labels.min() >= 0 and labels.max() < 4


def test_kmeans_k_larger_than_n_is_clamped():
    X = np.eye(3, dtype=np.float32)
    labels = _kmeans(X, 10)   # k > n
    assert len(labels) == 3


def test_kmeans_separates_two_clusters():
    a = np.tile([1.0, 0.0], (10, 1)).astype(np.float32)
    b = np.tile([0.0, 1.0], (10, 1)).astype(np.float32)
    X = np.vstack([a, b])
    labels = _kmeans(X, 2)
    # all of the first group share a label, distinct from the second group
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[10]


# -- split paragraphs --------------------------------------------------- #
def test_split_paragraphs_empty():
    assert _split_paragraphs("") == []
    assert _split_paragraphs("   \n  \n ") == []


def test_split_paragraphs_single_line():
    assert _split_paragraphs("hello") == ["hello"]


def test_split_paragraphs_blocks():
    assert _split_paragraphs("a\n\nb\n\nc") == ["a", "b", "c"]


def test_split_paragraphs_long_japanese_sentences():
    text = "猫を飼っています。" * 12  # > 80 chars, no blank lines
    out = _split_paragraphs(text)
    assert len(out) > 1   # falls back to sentence splitting on 。


# -- shorten ------------------------------------------------------------ #
def test_shorten_under_limit_unchanged():
    assert _shorten("short", 170) == "short"


def test_shorten_respects_max():
    s = "a" * 300
    assert len(_shorten(s, 170)) <= 170


def test_shorten_prefers_sentence_boundary():
    s = "これは最初の文です。" + "x" * 200
    out = _shorten(s, 170)
    assert len(out) <= 170


# -- lam_mmr ------------------------------------------------------------ #
def test_lam_mmr():
    assert lam_mmr(1.0, 0.3, 0.5) == pytest.approx(0.85)


# -- fmt_local ---------------------------------------------------------- #
def test_fmt_local_normal_time():
    # 2021-... a normal time renders 'YYYY-MM-DD HH:MM +09:00'
    out = _fmt_local(1_600_000_000, "Asia/Tokyo;+09:00")
    assert out.endswith("+09:00")
    assert out.startswith("2020-")


def test_fmt_local_matches_datetime_for_normal_range():
    ts = 1_700_000_000
    out = _fmt_local(ts, "UTC;+00:00")
    expected = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    assert out.startswith(expected.strftime("%Y-%m-%d %H:%M"))


def test_ymd_from_ordinal_matches_stdlib():
    for ordn in (1, 366, 719163, 730000, 1_000_000, 3_652_059):
        got = _ymd_from_ordinal(ordn)
        exp = datetime.date.fromordinal(ordn)
        assert got == (exp.year, exp.month, exp.day)


def test_fmt_local_beyond_year_9999_uses_correct_epoch():
    """The manual fallback path (year > 9999) must use the correct epoch ordinal.

    date(1970,1,1).toordinal() == 719163.  The Gregorian day-count for the
    rendered date must therefore equal _ymd_from_ordinal(719163 + days).
    Regression guards against the off-by-365 epoch constant (719528).
    """
    unix = 300_000_000_000  # ~ year 11477, forces the manual path
    out = _fmt_local(unix, "UTC;+00:00")
    days = int(unix) // 86400
    y, m, d = _ymd_from_ordinal(719163 + days)
    assert out.startswith(f"{y:04d}-{m:02d}-{d:02d}")


# -- ulid --------------------------------------------------------------- #
def test_ulid_format():
    u = ulid()
    assert isinstance(u, str)
    assert len(u) == 26   # Crockford base32 ULID


def test_ulid_time_ordering():
    import time
    a = ulid()
    time.sleep(0.005)
    b = ulid()
    assert b >= a   # first 48 bits are ms timestamp → lexicographic time order


# -- ulid fallback (resilience to a missing python-ulid package) -------- #
def test_ulid_fallback_format():
    u = _ulid_fallback()
    assert len(u) == 26
    assert all(c in _CROCKFORD32 for c in u)


def test_ulid_fallback_time_ordering():
    import time
    a = _ulid_fallback()
    time.sleep(0.005)
    b = _ulid_fallback()
    assert b >= a


def test_ulid_works_without_python_ulid(monkeypatch):
    """ulid() must not hard-crash if the optional python-ulid package is absent.

    Regression for: ModuleNotFoundError: No module named 'ulid'.
    """
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ulid":
            raise ModuleNotFoundError("No module named 'ulid'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    u = ulid()
    assert len(u) == 26
    assert all(c in _CROCKFORD32 for c in u)


def test_engine_inserts_without_python_ulid(monkeypatch, system):
    """A full save_memory path works even when python-ulid cannot be imported."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ulid":
            raise ModuleNotFoundError("No module named 'ulid'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ev = system.save_memory("memory saved without the ulid package present")
    assert ev["action"] == "inserted"
    assert system.total_records() == 1
