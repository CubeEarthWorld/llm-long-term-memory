"""Helpers, storage and configuration."""
from __future__ import annotations

import numpy as np
import pytest

from config import Config, LongTermMemoryConfig
from core.storage import Store
from memory.model import Memory
from memory.util import clean_text, cues, fmt_local, parse_offset, shorten, ulid


def test_cues():
    assert cues("a。b！c", 8) == ["a。", "b！", "c"]
    assert cues("one. two? three!\nfour", 8) == ["one.", "two?", "three!", "four"]
    assert cues("1.5 kg is fine", 8) == ["1.5 kg is fine"]
    assert cues("a\nb\nc\nd\ne", 2) == ["a b", "c d e"]
    assert cues("  \n ", 8) == []


def test_shorten_and_clean():
    assert shorten("abcdefghij", 20) == "abcdefghij"
    assert shorten("これは文です。これは長い文です。おわり", 11) == "これは文です。"
    assert shorten("a" * 30, 10) == "a" * 10
    assert clean_text(" x 《y》\t z ", 100) == "x y z"


def test_fmt_local_any_year_and_offsets():
    assert fmt_local(1749641400, "Asia/Tokyo;+09:00") == "2025-06-11 20:30 +09:00"
    assert fmt_local(0, "UTC;-03:30") == "1969-12-31 20:30 -03:30"
    assert fmt_local(253402300800, "UTC;+00:00").startswith("10000-01-01 00:00")
    assert fmt_local(100000000000, "X;+00:00").startswith("5138-11-16")
    assert parse_offset("-0530") == -(5 * 3600 + 30 * 60) and parse_offset("garbage") == 0


def test_ulid_sorts_by_time_and_survives_50_bits():
    a, b = ulid(1700000000000), ulid(1700000000001)
    assert len(a) == 26 and a < b
    far = ulid(1 << 49)
    assert far > b and not far.startswith("0")


def test_store_round_trip_and_transaction(tmp_path):
    st = Store(str(tmp_path / "s.db"))
    m = Memory("a", "t", 1, "UTC;+00:00", 1, 2.0, True, "m", np.array([1, 0], dtype=np.float32))
    st.put(m); st.put(m.with_(id="b", consolidated=False))
    rows = st.load_all()
    assert [r.id for r in rows] == ["a", "b"] and rows[0].consolidated and rows[0].vector.tolist() == [1, 0]
    with pytest.raises(RuntimeError):
        with st.transaction():
            st.remove("a")
            raise RuntimeError("boom")
    assert st.count() == 2
    st.remove("a")
    assert [r.id for r in st.load_all()] == ["b"]
    for _ in range(3):
        st.backup()
    assert len(list((tmp_path / "snapshots").glob("snap_*.db"))) == 3
    st.close()


def test_config_round_trip_and_validation():
    c = Config.from_dict({"memory": {"capacity": "42", "alpha": 0.5}, "glob": {"tool_fallback": "false"}})
    assert c.memory.capacity == 42 and c.memory.alpha == 0.5 and c.glob.tool_fallback is False
    assert Config.from_dict(c.to_dict()).memory.capacity == 42
    with pytest.raises(ValueError):
        LongTermMemoryConfig(theta_related=1.5)
    with pytest.raises(ValueError):
        Config.from_dict({"memory": {"capacity": 0}})
