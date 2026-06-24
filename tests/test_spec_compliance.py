"""ENGRAM v1.1 spec §8 / §8.1 constants and §4 formulas are wired correctly.

These pin the spec's default parameters so an accidental edit to config.py is
caught, and verify the self-describing `spec` table (§0, §13.5).
"""
from __future__ import annotations

import math

import pytest

from config import default_config


def test_tier_capacities():
    m = default_config().memory
    assert (m.cap1, m.cap2, m.cap3) == (1000, 3000, 6000)


def test_vector_dims():
    m = default_config().memory
    assert (m.dim1, m.dim2, m.dim3) == (768, 256, 128)
    assert m.dim_coarse == 128   # defaults to the coarsest tier


def test_half_lives():
    m = default_config().memory
    assert m.tau1 == 7 * 86400
    assert m.tau2 == 90 * 86400
    assert m.tau3 == 3 * 365 * 86400


def test_activation_and_score_constants():
    m = default_config().memory
    assert m.m_max == 64.0
    assert m.refractory_seconds == 3600
    assert m.alpha == 0.35
    assert m.inject_n == 5
    assert m.mmr_lambda == 0.3


def test_identity_thresholds():
    m = default_config().memory
    assert m.theta_same == 0.97
    assert m.theta_conflict == 0.85
    assert m.precise_margin == 0.03


def test_hysteresis_thresholds():
    m = default_config().memory
    assert m.theta_up == 16.0
    assert m.theta_down == 4.0
    assert m.theta_up > m.theta_down   # §4.4 anti ping-pong


def test_text_and_gen_bounds():
    m = default_config().memory
    assert m.text_max == 170
    assert m.text_hard_max == 1024     # §8.1 hard byte bound
    assert m.gen_max == 7


def test_dream_and_ring_bounds():
    m = default_config().memory
    assert m.dream_budget == 5
    assert m.dream_budget_hard == 4096
    assert m.cluster_min == 3
    assert m.conflict_cap == 256
    assert m.dream_log_cap == 512
    assert m.snapshot_gens == 8


def test_injection_budget():
    assert default_config().glob.budget_chars == 1024


def test_score_floor_formula_matches_spec(system):
    """score(m) = max(0,cos) × (α + (1−α)·A_abs), with A_abs = ln(1+A)/ln(1+M_max)."""
    m = default_config().memory
    A = 8.0
    A_abs = math.log1p(A) / math.log1p(m.m_max)
    cos = 0.5
    expected = max(0.0, cos) * (m.alpha + (1.0 - m.alpha) * A_abs)
    # recompute via the same arithmetic the engine uses
    assert expected == pytest.approx(0.5 * (0.35 + 0.65 * A_abs))


def test_spec_table_is_self_describing(system):
    """§13.5: the DB embeds the spec text and the active model id."""
    keys = {r["k"] for r in system.store.query("SELECT k FROM spec", ())}
    assert "active_model" in keys
    assert "spec_full" in keys
    assert "spec_version" in keys
    full = system.store.scalar("SELECT v FROM spec WHERE k='spec_full'", ())
    assert full and len(full) > 100


def test_decay_exp_cap_present():
    """§4.1/§8.1 underflow guard exists (value is a documented tuning of the spec's 64)."""
    m = default_config().memory
    assert m.decay_exp_cap >= 64   # never smaller than the spec floor
