"""Memory-fidelity benchmark for the LLM Long-Term Memory.

Runs deterministic scenarios (mock embeddings + mock LLM + virtual clock) that
assert the human-memory behaviors the model is meant to reproduce:

  spacing>massed | power-law forgetting | recall gate | stability growth |
  archive+savings | bounded archive | 3-axis protection | spreading activation

Usage:  python eval/run_eval.py        (exit code 0 = all pass)
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config
from core.storage import Store
from memory.long_term_memory import LongTermMemory
from eval.mocks import FakeEmbeddingProvider, FakeLLM, VirtualClock

_DAY = 24 * 60 * 60


def build():
    cfg = default_config()
    tmp = tempfile.mkdtemp(prefix="lms_eval_")
    store = Store(os.path.join(tmp, "eval.db"))
    system = LongTermMemory(store, FakeEmbeddingProvider(cfg.glob.dim_full), FakeLLM(), cfg.memory, cfg.glob)
    system.reset()
    clock = VirtualClock()
    system.set_clock(clock)
    return system, system.llm, clock, cfg


def write_mem(system, llm, text, w=0.6, prov="user"):
    llm.default_w, llm.default_provenance = w, prov
    return system.write(1, text, "")


def by_text(system, text):
    return system.store.one("SELECT * FROM memories WHERE text=?", (text,))


# -- scenarios --------------------------------------------------------- #
def scn_spacing():
    system, llm, clock, _ = build()
    write_mem(system, llm, "alpha apple", 0.6, "user")
    write_mem(system, llm, "bravo banana", 0.6, "user")
    for _ in range(4):                       # massed: tiny gaps -> r~1 -> little growth
        clock.advance(60)
        system.retrieve("alpha apple", 1)
    for _ in range(4):                       # spaced: long gaps -> low r -> big growth
        clock.advance_days(20)
        system.retrieve("bravo banana", 1)
    sa = system._stability(by_text(system, "alpha apple")) / _DAY
    sb = system._stability(by_text(system, "bravo banana")) / _DAY
    return ("spacing > massed", sb > sa * 1.5, f"S_massed={sa:.1f}d  S_spaced={sb:.1f}d")


def scn_forgetting_curve():
    system, llm, clock, _ = build()
    write_mem(system, llm, "charlie cherry", 0.6, "user")
    row = by_text(system, "charlie cherry")
    acc = float(row["accessed_at_unix"])
    rs = [system.retrievability(row, acc + dt) for dt in (3600, _DAY, 7 * _DAY, 30 * _DAY)]
    decreasing = all(rs[i] > rs[i + 1] for i in range(len(rs) - 1))
    heavy_tail = rs[-1] > 0.1                 # power-law keeps a long tail
    ok = decreasing and heavy_tail and rs[0] < 1.0
    return ("forgetting power-law tail", ok, f"r(1h,1d,7d,30d)={[round(x, 3) for x in rs]}")


def scn_recall_gate():
    system, _, _, _ = build()
    strong = system._recall_gate(0.95, 1.0)            # fresh strong cue -> recalled
    weak_forgotten = system._recall_gate(0.31, 0.0)    # weak cue to forgotten -> blocked
    strong_forgotten = system._recall_gate(0.95, 0.0)  # strong cue still recovers it
    ok = strong and (not weak_forgotten) and strong_forgotten
    return ("recall gate (functional forgetting)", ok,
            f"strong={strong} weak_forgotten={weak_forgotten} strong_forgotten={strong_forgotten}")


def scn_stability_growth():
    system, llm, clock, _ = build()
    write_mem(system, llm, "echo elephant", 0.6, "user")
    s0 = system._stability(by_text(system, "echo elephant")) / _DAY
    clock.advance_days(10)
    system.retrieve("echo elephant", 1)
    s1 = system._stability(by_text(system, "echo elephant")) / _DAY
    return ("stability grows on recall", s1 > s0, f"S0={s0:.1f}d  S1={s1:.1f}d")


def scn_archive_savings():
    system, llm, clock, _ = build()
    write_mem(system, llm, "delta dolphin ocean", 0.5, "inferred")
    clock.advance_days(4000)
    system.maintain(1)
    archived = system.stats()["archive"] == 1 and system.stats()["memories"] == 0
    write_mem(system, llm, "delta dolphin ocean", 0.5, "inferred")  # reappearance
    restored_row = by_text(system, "delta dolphin ocean")
    restored = restored_row is not None and system.stats()["archive"] == 0
    fresh_S = system.initial_stability(0.5, system.memory.confidence_inferred)
    savings = restored and system._stability(restored_row) > fresh_S * 1.2
    ok = archived and restored and savings
    sr = system._stability(restored_row) / _DAY if restored_row else 0
    return ("archive -> savings restore", ok,
            f"archived={archived} restored={restored} S_restored={sr:.1f}d fresh={fresh_S / _DAY:.1f}d")


def scn_archive_bounded():
    system, llm, clock, _ = build()
    system.glob.archive_cap = 3
    for i in range(6):
        write_mem(system, llm, f"item{i} token{i} word{i}", 0.4, "inferred")
    clock.advance_days(5000)
    # With max_decay_per_turn, decay is gradual; call maintain enough times.
    for _ in range(2):
        system.maintain(1)
    n_arch, n_mem = system.stats()["archive"], system.stats()["memories"]
    return ("archive bounded (total capacity)", n_arch <= 3 and n_mem == 0,
            f"archive={n_arch} (cap 3)  memories={n_mem}")


def scn_protection():
    system, llm, clock, _ = build()
    write_mem(system, llm, "foxtrot guard core", 0.95, "user")      # protected: w>=0.9, conf>=0.85
    write_mem(system, llm, "golf weak trivial", 0.2, "inferred")    # not protected
    # Advance to ~8 years (within the 10-year protection cap)
    clock.advance_days(3000)
    system.maintain(1)
    prot = by_text(system, "foxtrot guard core")
    weak_gone = by_text(system, "golf weak trivial") is None
    r_prot = system.retrievability(prot, clock()) if prot else 0.0
    ok = prot is not None and weak_gone and r_prot >= system.memory.r_hard_floor
    return ("3-axis eviction protection", ok,
            f"protected_kept={prot is not None} r_prot={r_prot:.3f} weak_archived={weak_gone}")


def scn_spreading():
    system, llm, clock, _ = build()
    write_mem(system, llm, "hotel india travel plan kyoto", 0.6, "user")
    write_mem(system, llm, "hotel india travel plan osaka", 0.6, "user")
    a = by_text(system, "hotel india travel plan kyoto")
    b = by_text(system, "hotel india travel plan osaka")
    same_cluster = a["cluster_id"] == b["cluster_id"]
    res = system.retrieve("hotel india travel plan kyoto", 1)
    spreads = [it.extra.get("spread", 0.0) for it in res.recalled]
    ok = same_cluster and any(s > 0 for s in spreads)
    return ("spreading activation", ok, f"same_cluster={same_cluster} spreads={[round(s, 3) for s in spreads]}")


SCENARIOS = [
    scn_spacing,
    scn_forgetting_curve,
    scn_recall_gate,
    scn_stability_growth,
    scn_archive_savings,
    scn_archive_bounded,
    scn_protection,
    scn_spreading,
]


def main() -> int:
    print("=== LLM Long-Term Memory - memory-fidelity benchmark ===\n")
    passed = 0
    for scn in SCENARIOS:
        try:
            name, ok, detail = scn()
        except Exception as e:  # noqa: BLE001
            name, ok, detail = scn.__name__, False, f"EXCEPTION: {type(e).__name__}: {e}"
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")
    print(f"\n{passed}/{len(SCENARIOS)} scenarios passed")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
