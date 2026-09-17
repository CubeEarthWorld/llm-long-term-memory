"""Millennia of virtual time (SPEC §8): bounded state, no NaN/overflow, key facts
survive while noise is forgotten, clock faults and decade-long silences absorbed."""
from __future__ import annotations

import random

import pytest

from core.storage import Store
from eval.mocks import FakeEmbeddingProvider
from memory.util import fmt_local

_DAY = 86400


@pytest.mark.slow
def test_three_thousand_years(make_system, cfg):
    cfg.memory.capacity, cfg.memory.writes_per_day = 300, 50
    s, clock = make_system(config=cfg, provider=FakeEmbeddingProvider(dimension=32), store=Store(":memory:"))
    rng = random.Random(7)
    facts = ["user likes matcha ice cream", "user was born in kyoto japan", "user has a cat named tama"]
    for f in facts:
        s.remember(f, salience=2)
    start = clock.t
    writes = recalls = dreams = 0
    for d in range(0, 3000 * 365, 3):
        clock.advance(3 * _DAY)
        if rng.randrange(1000) == 0:
            clock.advance(10 * 365 * _DAY)                 # a decade of silence
        s.remember(f"note {rng.randrange(1 << 30)} topic{d % 97}")
        writes += 1
        if d % 7 < 3:
            f = rng.choice(facts)
            assert f in [c["text"] for c in s.recall(f)["recalled"]], f"day {d}"
            recalls += 1
        if d % 30 < 3:
            s.dream(budget=2)
            dreams += 1
        if d == 1500 * 365:                                # clock fault: +200 years, then back
            t = clock.t
            clock.t += 200 * 365 * _DAY
            s.recall(facts[0])
            clock.t = t
    rows = s.memories()
    now = clock.t
    assert len(rows) <= cfg.memory.capacity
    for m in rows:
        assert 1 <= m.stability <= cfg.memory.max_stability
        assert 1 <= len(m.text) <= cfg.memory.text_max
        r = s.retrievability(m, now)
        assert 0 <= r <= 1 and r == r
    for f in facts:
        top = s.recall(f)["recalled"][0]
        assert top["text"] == f and top["stability"] > 365 * _DAY
    assert int(fmt_local(now, "UTC;+00:00")[:4]) > 5000
    assert (now - start) // (365 * _DAY) >= 3000
    assert writes > 300000 and recalls > 100000 and dreams > 30000
    assert s.store.count() == len(rows)
