"""core.engine — seed duration parsing and CSV ingestion (extreme inputs)."""
from __future__ import annotations

import pytest

from core.engine import (
    _parse_duration,
    clean_seed_items,
    parse_seed_csv,
)

_DAY = 86400
_YEAR = 31536000


@pytest.mark.parametrize("spec,expected", [
    ("0", 0.0),
    ("", 0.0),
    (None, 0.0),
    ("30s", 30.0),
    ("30m", 30 * 60.0),
    ("12h", 12 * 3600.0),
    ("8d", 8 * _DAY),
    ("2w", 2 * 604800.0),
    ("5y", 5 * _YEAR),
    ("7", 7 * _DAY),          # bare number = days
    ("3.5d", 3.5 * _DAY),
    ("garbage", 0.0),
    ("d", 0.0),               # unit with no number
    ("  5y  ", 5 * _YEAR),    # surrounding whitespace
])
def test_parse_duration(spec, expected):
    assert _parse_duration(spec) == pytest.approx(expected)


def test_parse_duration_negative():
    # Negative advances are nonsensical but must not raise.
    assert _parse_duration("-5d") == pytest.approx(-5 * _DAY)


def test_parse_seed_csv_with_header():
    csv = "text,advance\nhello world,5d\n,skip-empty\nsecond,2h\n"
    items = parse_seed_csv(csv)
    assert [i["text"] for i in items] == ["hello world", "second"]
    assert items[0]["advance"] == "5d"


def test_parse_seed_csv_headerless():
    csv = "first line,3d\nsecond line,1h\n"
    items = parse_seed_csv(csv)
    assert len(items) == 2
    assert items[0]["text"] == "first line"
    assert items[0]["advance"] == "3d"


def test_parse_seed_csv_empty():
    assert parse_seed_csv("") == []
    assert parse_seed_csv("\n\n") == []


def test_clean_seed_items_filters_and_defaults():
    raw = [{"text": " keep "}, {"text": ""}, {"no_text": 1}, {"text": "x", "advance": " 2d "}]
    out = clean_seed_items(raw)
    assert [i["text"] for i in out] == ["keep", "x"]
    assert out[0]["advance"] == "0"   # default when missing
    assert out[1]["advance"] == "2d"


def test_clean_seed_items_none_safe():
    assert clean_seed_items(None) == []
    assert clean_seed_items([None]) == []
