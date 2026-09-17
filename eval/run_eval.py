"""ENGRAM v2 behaviour benchmark — deterministic scenarios (fake embedder + fake
LLM + virtual clock) asserting the SPEC behaviours without any model or API key.

Usage:  python eval/run_eval.py        (exit code 0 = all pass)
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config
from core.storage import Store
from eval.mocks import FakeEmbeddingProvider, FakeLLM, VirtualClock
from memory.engine import LongTermMemory

_DAY = 24 * 60 * 60


def build(llm=None, **overrides):
    cfg = default_config()
    cfg.memory.cosine_floor = 0.0
    for k, v in overrides.items():
        setattr(cfg.memory, k, v)
    store = Store(os.path.join(tempfile.mkdtemp(prefix="engram_eval_"), "eval.db"))
    system = LongTermMemory(store, FakeEmbeddingProvider(), llm or FakeLLM(), cfg.memory, cfg.glob)
    clock = VirtualClock()
    system.set_clock(clock)
    return system, clock, cfg


def scn_forgetting_curve():
    s, clock, cfg = build()
    m = s.memory(s.remember("alpha apple beta")["id"])
    t0 = m.last_recall
    r = [s.retrievability(m, t0 + k * _DAY) for k in (0, 1, 2)]
    ok = abs(r[0] - 1) < 1e-9 and abs(r[1] - 0.5) < 1e-9 and abs(r[2] - 0.25) < 1e-9
    return ("R halves every stability (1 day)", ok, f"R(0,1d,2d)={[round(x, 3) for x in r]}")


def scn_spacing():
    s, clock, _ = build()
    massed, spaced = s.remember("massed alpha")["id"], s.remember("spaced beta")["id"]
    for _ in range(3):
        clock.advance(60)
        s.recall("massed alpha")
    clock.advance_days(3)
    s.recall("spaced beta")
    sm, ss = s.memory(massed).stability / _DAY, s.memory(spaced).stability / _DAY
    return ("spaced recall strengthens, massed does not", sm < 1.05 and ss > 2, f"S massed={sm:.2f}d spaced={ss:.2f}d")


def scn_rehearsal_and_related():
    s, _, _ = build()
    a = s.remember("cat dog bird fish")
    b = s.remember("cat dog bird fish")
    c = s.remember("cat dog bird fish!")
    ok = (a["action"] == "inserted" and b["action"] == "reinforced" and c["action"] == "inserted"
          and not c["consolidated"] and not s.memory(a["id"]).consolidated and s.total_records() == 2)
    return ("exact text = rehearsal; paraphrase = labile pair, nothing overwritten", ok,
            f"{a['action']}/{b['action']}/{c['action']} records={s.total_records()}")


def scn_capacity_and_flood():
    s, clock, _ = build(capacity=20, grace_period=_DAY)
    keep = s.remember("precious fact", salience=10)["id"]
    clock.advance_days(2)
    for i in range(60):
        s.remember(f"junk {i} x{i}")
    ok = s.total_records() == 20 and s.memory(keep) is not None
    return ("flood evicts junk, never the stable fact", ok, f"records={s.total_records()} kept={s.memory(keep) is not None}")


def scn_no_immortal():
    s, clock, cfg = build()
    mid = s.remember("rehearsed monthly")["id"]
    for _ in range(300):
        clock.advance_days(30)
        s.recall("rehearsed monthly")
    m = s.memory(mid)
    r = s.retrievability(m, clock.t + 100 * 365 * _DAY)
    return ("no immortal memory (S ≤ S_max, R → 0)", m.stability == cfg.memory.max_stability and r < 1e-3,
            f"S={m.stability / _DAY / 365:.1f}y R(+100y)={r:.4f}")


def scn_dream():
    s, _, _ = build()
    for t in ("trip kyoto plan schedule note", "trip kyoto hotel schedule note", "trip kyoto food schedule note"):
        s.remember(t)
    reports = s.dream()
    quiet = s.dream()
    ok = len(reports) == 1 and reports[0]["action"] == "replace" and s.total_records() == 1 and quiet == []
    return ("dream merges labile cluster into a gist, then goes quiet", ok,
            f"reports={len(reports)} records={s.total_records()} second dream={len(quiet)}")


def scn_correction():
    s, clock, _ = build(llm=FakeLLM("keep"))
    old = s.remember("user lives in tokyo city", salience=5)["id"]
    clock.advance_days(30)
    s.remember("user lives in osaka city")
    cluster = s.clusters()[0]
    ok = cluster[0].id == old and len(cluster) == 2
    return ("a correction is seeded by the old fact (reconsolidation)", ok, f"seed={cluster[0].text}")


SCENARIOS = [scn_forgetting_curve, scn_spacing, scn_rehearsal_and_related, scn_capacity_and_flood,
             scn_no_immortal, scn_dream, scn_correction]


def main() -> int:
    print("=== LLM Long-Term Memory (ENGRAM v2) — behaviour benchmark ===\n")
    passed = 0
    for scn in SCENARIOS:
        try:
            name, ok, detail = scn()
        except Exception as e:  # noqa: BLE001
            name, ok, detail = scn.__name__, False, f"EXCEPTION: {type(e).__name__}: {e}"
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<62} {detail}")
    print(f"\n{passed}/{len(SCENARIOS)} scenarios passed")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
