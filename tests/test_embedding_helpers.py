"""core.embedding — l2_normalize / truncate_normalize (MRL) edge cases (§3.3)."""
from __future__ import annotations

import numpy as np
import pytest

from core.embedding import l2_normalize, truncate_normalize


def test_l2_normalize_1d_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(v)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)


def test_l2_normalize_zero_vector_safe():
    z = np.zeros(5, dtype=np.float32)
    out = l2_normalize(z)
    assert np.all(out == 0.0)  # must not divide by zero


def test_l2_normalize_2d_rows_unit_norm():
    m = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(m)
    norms = np.linalg.norm(out, axis=1)
    assert norms[0] == pytest.approx(1.0, abs=1e-6)
    assert norms[1] == pytest.approx(0.0, abs=1e-6)  # zero row stays zero
    assert norms[2] == pytest.approx(1.0, abs=1e-6)


def test_truncate_normalize_keeps_prefix_and_renormalizes():
    v = np.arange(1, 769, dtype=np.float32)
    out = truncate_normalize(v, 256)
    assert out.shape[0] == 256
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)
    # MRL = prefix truncation: direction of the first 256 dims is preserved.
    assert np.allclose(out, l2_normalize(v[:256]), atol=1e-6)


def test_truncate_normalize_dim_larger_than_vector():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = truncate_normalize(v, 768)  # asking for more dims than exist
    assert out.shape[0] == 3
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)


def test_truncate_normalize_zero_vector_safe():
    z = np.zeros(768, dtype=np.float32)
    out = truncate_normalize(z, 128)
    assert out.shape[0] == 128
    assert np.all(out == 0.0)
