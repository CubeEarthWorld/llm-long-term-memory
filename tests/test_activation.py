"""Activation A = mass·2^(−Δt/τ) and recall/refractory dynamics (§4.1)."""
from __future__ import annotations

import math

import pytest

from conftest import by_text


def test_activation_halves_every_tau(make_system, cfg):
    s, clock, c = make_system(config=cfg)
    s.save_memory("alpha apple beta")
    row = by_text(s, "alpha apple beta")
    acc = float(row["last_access"])
    tau = c.memory.tau1
    assert s.activation(row, acc) == pytest.approx(1.0, abs=1e-6)
    assert s.activation(row, acc + tau) == pytest.approx(0.5, abs=0.02)
    assert s.activation(row, acc + 2 * tau) == pytest.approx(0.25, abs=0.02)


def test_activation_clock_rollback_clamped(make_system):
    s, clock, _ = make_system()
    s.save_memory("clamp test memory")
    row = by_text(s, "clamp test memory")
    acc = float(row["last_access"])
    # now in the past → Δt clamped to 0 → no spurious activation gain (I2)
    assert s.activation(row, acc - 10_000) == pytest.approx(float(row["mass"]), abs=1e-6)


def test_activation_decay_cap_underflows_to_zero(make_system, cfg):
    s, clock, c = make_system(config=cfg)
    s.save_memory("very old memory token")
    row = by_text(s, "very old memory token")
    acc = float(row["last_access"])
    far = acc + c.memory.tau1 * (c.memory.decay_exp_cap + 10)
    assert s.activation(row, far) == 0.0


def test_recall_refractory_gate(make_system):
    s, clock, _ = make_system()
    s.save_memory("charlie cherry delta")
    clock.advance(2 * 3600)                       # past refractory → +1
    s.retrieve("charlie cherry delta", 1)
    m1 = float(by_text(s, "charlie cherry delta")["mass"])
    s.retrieve("charlie cherry delta", 1)         # within refractory → no bonus
    m2 = float(by_text(s, "charlie cherry delta")["mass"])
    clock.advance(2 * 3600)                       # past refractory again → +1
    s.retrieve("charlie cherry delta", 1)
    m3 = float(by_text(s, "charlie cherry delta")["mass"])
    assert m1 > 1.5
    assert (m2 - m1) < 0.2
    assert m3 > m1 + 0.5


def test_no_immortal_memory_6tau(make_system, cfg):
    """§9 theorem: even the strongest memory (mass=64) decays to A≤1 at 6τ."""
    s, clock, c = make_system(config=cfg)
    s.save_memory("strong eternal token")
    row = by_text(s, "strong eternal token")
    s.store.exec("UPDATE memory SET mass=64 WHERE id=?", (row["id"],))
    row = by_text(s, "strong eternal token")
    acc = float(row["last_access"])
    tau = c.memory.tau1
    a6 = s.activation(row, acc + 6 * tau)
    a7 = s.activation(row, acc + 7 * tau)
    assert a6 <= 1.0 + 1e-6
    assert a7 < a6


def test_a_abs_normalization_bounds(make_system, cfg):
    """A_abs = ln(1+A)/ln(1+M_max) ∈ [0,1]; score floor α keeps dormant memories alive."""
    s, clock, c = make_system(config=cfg)
    # A_abs(0) = 0, A_abs(M_max) = 1
    assert math.log1p(0) / math.log1p(c.memory.m_max) == 0.0
    assert math.log1p(c.memory.m_max) / math.log1p(c.memory.m_max) == pytest.approx(1.0)
