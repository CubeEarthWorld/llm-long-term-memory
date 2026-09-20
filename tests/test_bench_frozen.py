"""ベンチの計測は状態を変えない（`frozen`）— 崩れると測定値が自分の副作用を測る。"""
from __future__ import annotations

from bench.forget_bench import frozen


def test_frozen_recall_leaves_no_trace(make_system):
    s, clock = make_system()
    mid = s.remember("ユーザーは抹茶味のアイスクリームが好き")["id"]
    clock.advance_days(2)
    before = (s.memory(mid).stability, s.memory(mid).last_recall, s.store.count())

    with frozen(s):
        assert [r["text"] for r in s.recall("抹茶アイス")["recalled"]]
    assert (s.memory(mid).stability, s.memory(mid).last_recall, s.store.count()) == before

    s.recall("抹茶アイス")                                   # 凍結解除後は本当に強化される
    assert s.memory(mid).stability > before[0]
