"""Regression checks for bulk eviction and the budgeted cluster preview."""
import math

import numpy as np
import pytest

from core.storage import Store
from fakes import BrokenEmbeddingProvider
from memory.model import Memory


@pytest.mark.parametrize("capacity", [1, 10, 40, 119, 120])
@pytest.mark.parametrize("mode", ["mixed", "young", "old"])
def test_bulk_eviction_matches_repeated_scan(make_system, cfg, capacity, mode):
    now = 1_700_000_000
    cfg.memory.capacity = capacity
    cfg.memory.grace_period = 100
    store = Store(":memory:")
    for i in range(120):
        young = mode == "young" or (mode == "mixed" and i % 3 != 0)
        store.put(Memory(str(i), f"row {i}", now - (10 if young else 200), "UTC;+00:00",
                         now - (i % 5) * 100, 2.0 ** (i % 8), True, "", np.zeros(0, dtype=np.float32)))
    s, _ = make_system(store=store, provider=BrokenEmbeddingProvider())
    rows = s.memories()
    # Independent reference: the original full scan after every removal.
    expected = []
    rows.append(Memory("new", "new row", now, "UTC;+00:00", now,
                       cfg.memory.initial_stability, False, "", np.zeros(0, dtype=np.float32)))
    while len(rows) > capacity:
        young = [m for m in rows if 0 <= now - m.created_at < 100]
        old = [m for m in rows if m not in young]
        eligible = old + young if not old or len(young) > capacity // 10 else old
        victim = min(eligible, key=lambda m: math.log2(m.stability) - max(0, now - m.last_recall) / m.stability)
        expected.append(victim.text)
        rows.remove(victim)
    removed = []
    remove = store.remove
    def record(mid):
        removed.append(s.memory(mid).text if s.memory(mid) else by_id[mid])
        remove(mid)
    by_id = {m.id: m.text for m in s.memories()}
    # _remove removes the RAM entry first; the fresh row is always strongest.
    store.remove = record
    result = s.remember("new row", now=now)
    assert removed == expected
    assert result["evicted"] == expected[-1]
    assert [m.text for m in s.memories()] == [m.text for m in rows]


def test_clusters_budget_previews_exactly_the_seeds_dream_scans(make_system):
    s, _ = make_system()
    v = s.provider.encode_document(["same vector"])[0]
    for i in range(100):
        s._put(Memory(str(i), f"row {i}", 1_700_000_000 - i % 7, "UTC;+00:00", 1_700_000_000,
                      86400.0 + i % 3, False, s.provider.model_id, np.asarray(v, dtype=np.float32)))
    seeds = lambda cs: [c[0].id for c in cs]  # noqa: E731
    everything = seeds(s.clusters())
    assert len(everything) == 100
    assert seeds(s.clusters(0)) == []
    assert seeds(s.clusters(2)) == everything[:16]
    assert seeds(s.clusters(100)) == everything
