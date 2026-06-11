"""core.storage — vector codecs, cosine, and table-name validation."""
from __future__ import annotations

import numpy as np
import pytest

from core.storage import (
    Store,
    cosine,
    dequantize_int8,
    pack_vec,
    quantize_int8,
    unpack_vec,
    vec_label,
)


def test_pack_unpack_roundtrip():
    v = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    out = unpack_vec(pack_vec(v))
    assert np.allclose(out, v, atol=1e-6)


def test_pack_none():
    assert pack_vec(None) is None
    assert unpack_vec(None) is None


def test_quantize_dequantize_approx():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(256).astype(np.float32)
    v = v / np.linalg.norm(v)
    blob, scale = quantize_int8(v)
    assert 0 < scale <= 65536, "scale must stay within the §8.1 hard bound"
    deq = dequantize_int8(blob, scale)
    # int8 with a per-vector peak scale: error is bounded by half a quantum.
    assert np.max(np.abs(deq - v)) <= scale, "dequant error exceeds one quantum"


def test_quantize_zero_vector_is_safe():
    z = np.zeros(128, dtype=np.float32)
    blob, scale = quantize_int8(z)
    assert scale > 0  # no division-by-zero, no zero scale
    deq = dequantize_int8(blob, scale)
    assert np.all(deq == 0.0)


def test_cosine_zero_vector_returns_zero():
    a = np.zeros(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    assert cosine(a, b) == 0.0
    assert cosine(b, b) == pytest.approx(1.0, abs=1e-6)


def test_vec_label():
    assert vec_label(None) == ""
    assert "B vec" in vec_label(b"abcd")


def test_count_rejects_unknown_table(tmp_path):
    st = Store(str(tmp_path / "s.db"))
    with pytest.raises(ValueError):
        st.count("definitely_not_a_table")
    st.close()


def test_count_allows_turn_log(tmp_path):
    """turn_log is a real table created by the engine; count() must accept it.

    Regression: _VALID_TABLES originally omitted turn_log, so any count()/
    rows_as_dicts() on it raised ValueError.
    """
    st = Store(str(tmp_path / "s.db"))
    st.execscript(
        "CREATE TABLE IF NOT EXISTS turn_log ("
        "turn INTEGER PRIMARY KEY, utterance TEXT NOT NULL, note TEXT, "
        "timestamp REAL, system_json TEXT);"
    )
    assert st.count("turn_log") == 0
    st.close()


def test_db_size_bytes_nonnegative(tmp_path):
    st = Store(str(tmp_path / "s.db"))
    st.execscript("CREATE TABLE IF NOT EXISTS spec(k TEXT, v TEXT);")
    assert st.db_size_bytes() >= 0
    st.close()
