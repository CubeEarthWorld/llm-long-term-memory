"""ENGRAM v1.1 behavior benchmark for the LLM Long-Term Memory.

Deterministic scenarios (mock embeddings + mock LLM + virtual clock) that assert
the spec behaviors:

  activation decay (§4.1) | recall bonus + refractory (§4.1) |
  identity thresholds θ_same / θ_conflict (§4.3) | tier demote/promote/evict (§5.4) |
  dream merge (§6) | immortality non-existence (§9)

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
from memory.long_term_memory import LongTermMemory

_DAY = 24 * 60 * 60


def build():
    cfg = default_config()
    tmp = tempfile.mkdtemp(prefix="engram_eval_")
    store = Store(os.path.join(tmp, "eval.db"))
    system = LongTermMemory(store, FakeEmbeddingProvider(cfg.glob.dim_full), FakeLLM(), cfg.memory, cfg.glob)
    system.reset()
    clock = VirtualClock()
    system.set_clock(clock)
    return system, clock, cfg


def by_text(system, text):
    return system.store.one("SELECT * FROM memory WHERE text=? AND superseded_by IS NULL", (text,))


def _tokens(n, prefix="t"):
    return " ".join(f"{prefix}{i}" for i in range(n))


# -- scenarios --------------------------------------------------------- #
def scn_activation_decay():
    system, clock, cfg = build()
    system.save_memory("alpha apple beta")
    row = by_text(system, "alpha apple beta")
    acc = float(row["last_access"])
    tau = cfg.memory.tau1
    a0 = system.activation(row, acc)
    a1 = system.activation(row, acc + tau)        # one half-life
    a2 = system.activation(row, acc + 2 * tau)    # two half-lives
    ok = abs(a0 - 1.0) < 1e-6 and abs(a1 - 0.5) < 0.02 and abs(a2 - 0.25) < 0.02
    return ("activation A halves every τ", ok, f"A(0,τ,2τ)=[{a0:.3f},{a1:.3f},{a2:.3f}]")


def scn_recall_refractory():
    system, clock, cfg = build()
    system.save_memory("charlie cherry delta")
    clock.advance(2 * 3600)                         # past refractory → bonus
    system.retrieve("charlie cherry delta", 1)
    m1 = float(by_text(system, "charlie cherry delta")["mass"])
    system.retrieve("charlie cherry delta", 1)      # within refractory → no extra bonus
    m2 = float(by_text(system, "charlie cherry delta")["mass"])
    clock.advance(2 * 3600)                          # past refractory again → bonus
    system.retrieve("charlie cherry delta", 1)
    m3 = float(by_text(system, "charlie cherry delta")["mass"])
    ok = m1 > 1.5 and (m2 - m1) < 0.2 and m3 > m1 + 0.5
    return ("recall bonus gated by refractory", ok, f"mass=[{m1:.2f},{m2:.2f},{m3:.2f}]")


def scn_dup_rehearsal():
    system, clock, cfg = build()
    system.save_memory(_tokens(8))
    clock.advance(2 * 3600)
    ev = system.save_memory(_tokens(8))             # exact text → rehearsal, no new row
    live = system.total_records()
    ok = ev["action"] == "reinforced" and live == 1 and float(by_text(system, _tokens(8))["mass"]) > 1.0
    return ("exact-text rehearsal (mass+1, no row)", ok,
            f"action={ev['action']} live={live} mass={by_text(system, _tokens(8))['mass']:.2f}")


def scn_supersede():
    system, clock, cfg = build()
    system.save_memory("cat dog bird")
    ev = system.save_memory("cat dog bird!")        # same tokens (≥0.97), different text → supersede
    live = system.total_records()
    tomb = system.stats()["tombstones"]
    ok = ev["action"] == "updated" and live == 1 and tomb == 1
    return ("θ_same → supersede (tombstone old)", ok, f"action={ev['action']} live={live} tomb={tomb}")


def scn_conflict():
    system, clock, cfg = build()
    base = _tokens(20)
    system.save_memory(base)
    other = _tokens(18) + " z0 z1"                  # 18/20 shared ≈ cos 0.9 → conflict band
    ev = system.save_memory(other)
    nconf = system.stats()["conflict"]
    ok = ev["action"] == "conflict" and nconf == 1 and system.total_records() == 2
    return ("θ_conflict band → conflict queue", ok, f"action={ev['action']} conflict={nconf}")


def scn_new_insert():
    system, clock, cfg = build()
    system.save_memory(_tokens(10, "a"))
    ev = system.save_memory(_tokens(10, "b"))       # disjoint tokens → cos ~0 → new
    ok = ev["action"] == "inserted" and system.total_records() == 2
    return ("<θ_conflict → new memory", ok, f"action={ev['action']} live={system.total_records()}")


def scn_demote():
    system, clock, cfg = build()
    system.memory.cap1 = 3
    for i in range(6):
        system.save_memory(_tokens(6, f"g{i}_"))     # 6 disjoint memories, all in L1
    system.maintain(1)                               # capacity enforcement demotes lowest A
    s = system.stats()
    ok = s["L1"] <= 3 and s["L2"] >= 3 and system.total_records() == 6
    return ("L1 overflow → demote to L2", ok, f"L1={s['L1']} L2={s['L2']} L3={s['L3']}")


def scn_promote():
    system, clock, cfg = build()
    system.memory.cap1 = 1
    system.save_memory("anchor keep one")
    system.save_memory(_tokens(6, "p"))
    system.maintain(1)                               # second memory demoted to L2
    demoted = system.store.one("SELECT * FROM memory WHERE tier=2 AND superseded_by IS NULL")
    system.store.exec("UPDATE memory SET mass=?, last_access=? WHERE id=?",
                      (20.0, int(system.now_unix()), demoted["id"]))   # A ≥ θ_up
    system._promote(system.now_unix())
    tier = system.store.scalar("SELECT tier FROM memory WHERE id=?", (demoted["id"],))
    ok = tier == 1
    return ("A ≥ θ_up → promote to L1", ok, f"promoted_tier={tier}")


def scn_eviction_death():
    system, clock, cfg = build()
    system.memory.cap1 = system.memory.cap2 = system.memory.cap3 = 1
    for i in range(4):
        system.save_memory(_tokens(5, f"d{i}_"))     # 4 disjoint memories, caps 1/1/1
    before = system.total_records()
    system.maintain(1)                               # cascade demote, L3 overflow → physical delete
    after = system.total_records()
    ok = before == 4 and after == 3
    return ("L3 overflow → eviction (death)", ok, f"live {before}→{after}")


def scn_immortality():
    system, clock, cfg = build()
    system.save_memory("strong eternal token")
    row = by_text(system, "strong eternal token")
    system.store.exec("UPDATE memory SET mass=64 WHERE id=?", (row["id"],))
    row = by_text(system, "strong eternal token")
    acc = float(row["last_access"])
    tau = cfg.memory.tau1
    a6 = system.activation(row, acc + 6 * tau)       # strongest mass falls to A=1 at 6τ (§9)
    a7 = system.activation(row, acc + 7 * tau)
    ok = a6 <= 1.0 + 1e-6 and a7 < a6
    return ("no immortal memory (6τ → A≤1)", ok, f"A(6τ)={a6:.3f} A(7τ)={a7:.3f}")


def scn_dream_merge():
    system, clock, cfg = build()
    for tail in ("plan", "hotel", "food"):
        system.save_memory(f"trip kyoto {tail} schedule note")   # share tokens → one cluster
    before = system.total_records()
    results = system.dream(budget=5, force=True)
    after = system.total_records()
    merged = [r for r in results if r["action"] == "merge"]
    new_gen = max((a["gen"] for r in merged for a in r["after"]), default=0) if merged else 0
    ok = before == 3 and after < before and bool(merged) and new_gen == 1
    return ("dream merge → fewer rows, gen+1", ok,
            f"live {before}→{after} merged={len(merged)} gen={new_gen}")


SCENARIOS = [
    scn_activation_decay,
    scn_recall_refractory,
    scn_dup_rehearsal,
    scn_supersede,
    scn_conflict,
    scn_new_insert,
    scn_demote,
    scn_promote,
    scn_eviction_death,
    scn_immortality,
    scn_dream_merge,
]


def main() -> int:
    print("=== LLM Long-Term Memory (ENGRAM v1.1) — behavior benchmark ===\n")
    passed = 0
    for scn in SCENARIOS:
        try:
            name, ok, detail = scn()
        except Exception as e:  # noqa: BLE001
            name, ok, detail = scn.__name__, False, f"EXCEPTION: {type(e).__name__}: {e}"
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
    print(f"\n{passed}/{len(SCENARIOS)} scenarios passed")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
