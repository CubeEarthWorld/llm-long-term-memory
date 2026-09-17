"""Headless CLI runner for the LLM Long-Term Memory prototype.

This script is the primary entry-point for non-interactive experiments:
  - Builds the engine (embedding + LLM + memory system)
  - Optionally replays the default seed utterances along a virtual timeline
  - Runs extra turns (--say), dreaming passes (--dream), and inspection (--inspect)
  - Writes per-turn metrics and final statistics to a JSON file

Console output is UTF-8 reconfigured for Windows compatibility.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import default_config
from core.engine import (
    DATA_DIR,
    SYSTEM_ID,
    SYSTEM_TITLE,
    build_engine,
    dispose_engine,
    run_dream,
    run_seed,
    run_turn,
)


def _print_turn(engine: dict, turn: int) -> None:
    """Pretty-print a single turn's metrics to the console."""
    rec = engine["recorder"]
    utt = next((r["utterance"] for r in engine["log"] if r["turn"] == turn), "")
    print("\n" + "=" * 78)
    print(f"TURN {turn}  {utt}")
    print("=" * 78)
    m = rec.for_turn(turn, SYSTEM_ID)
    if m is None:
        print("No metrics recorded.")
        return
    print(f"\n-- {SYSTEM_TITLE} --")
    print(f"  応答: {m.response.strip()[:400]}")
    print(f"  書込: {m.write_note}")
    print(f"  records={m.total_records}  pack={m.pack_chars}字/{m.pack_n}件  time={m.total_ms:.0f}ms (llm {m.llm_ms:.0f})")
    if not m.recalled:
        print("  想起なし")
        return
    print(f"  想起された記憶 ({len(m.recalled)}) / 引用 {len(m.cited)}:")
    for r in m.recalled:
        extra = " ".join(f"{k}={v}" for k, v in list(r.extra.items())[:5])
        print(f"     - [{r.score:.2f}] {r.text[:80]}  ({extra})")


def _print_dream(results: list) -> None:
    """Pretty-print dreaming (consolidation) results to the console."""
    print("\n" + "=" * 78)
    print(f"DREAM  ({len(results)} クラスタ処理)")
    print("=" * 78)
    if not results:
        print("  対象クラスタなし（不安定な記憶に近傍がない）")
        return
    for r in results:
        print(f"\n-- action={r['action']}" + (f"  error={r['error']}" if r.get('error') else ""))
        print(f"   統合元 ({len(r['before'])}):")
        for b in r["before"]:
            print(f"     - [R={b.get('R')}] {b['text'][:70]}")
        if r["after"]:
            print(f"   統合後 ({len(r['after'])}):")
            for a in r["after"]:
                print(f"     + [S={a.get('stability')}s] {a['text'][:70]}")
        else:
            print("   統合後: （維持）")


def _summary(engine: dict) -> None:
    """Print high-level invariants and final DB statistics."""
    print("\n" + "#" * 78 + "\n# サマリ\n" + "#" * 78)
    rows = engine["recorder"].rows()
    cfg = engine["cfg"]
    if rows:
        print(f"  全 pack <= {cfg.memory.budget_chars}字 : {all(r['pack_chars'] <= cfg.memory.budget_chars for r in rows)}")
        print(f"  全 records <= {cfg.memory.capacity}件 : {all(r['records'] <= cfg.memory.capacity for r in rows)}")
    s = engine["system"]
    print(f"  {SYSTEM_TITLE}: records={s.total_records()}  {s.stats()}  "
          f"vec={s.vector_mb():.3f}MB  db={s.db_size_bytes() / 1024:.1f}KB")


def _dump_json(engine: dict, path: str) -> None:
    """Export the full turn log + metrics + final DB snapshot to JSON."""
    rec = engine["recorder"]
    out = {"turns": [], "final": {}}
    for r in engine["log"]:
        t = r["turn"]
        m = rec.for_turn(t, SYSTEM_ID)
        d = m.to_detail_dict() if m else {}
        out["turns"].append({
            "turn": t, "utterance": r["utterance"], "note": r.get("note", ""),
            "system": {k: d[k] for k in ("response", "write_note", "records", "pack_chars", "times", "recalled") if k in d},
        })
    system = engine["system"]
    out["final"] = {"records": system.total_records(), "stats": system.stats(), "vector_mb": system.vector_mb()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


def main() -> int:
    """Parse CLI arguments, build the engine, run the pipeline, and dump results."""
    ap = argparse.ArgumentParser(description="LLM Long-Term Memory CLI runner")
    ap.add_argument("--reset", dest="reset", action="store_true", default=True)
    ap.add_argument("--no-reset", dest="reset", action="store_false")
    ap.add_argument("--seed", dest="seed", action="store_true", default=True)
    ap.add_argument("--no-seed", dest="seed", action="store_false")
    ap.add_argument("--say", action="append", default=[], help="run an extra turn with this text")
    ap.add_argument("--dream", nargs="?", type=int, const=5, default=0,
                    help="run a dreaming (consolidation) pass with this LLM budget after seeding")
    ap.add_argument("--provider", default=None, help="deepseek | gemini (override)")
    ap.add_argument("--inspect", action="store_true", help="dump the LLM Long-Term Memory DB tables")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "results.json"))
    args = ap.parse_args()

    cfg = default_config()
    if args.provider:
        cfg.glob.llm_provider = args.provider

    model = cfg.glob.deepseek_model if cfg.glob.llm_provider == "deepseek" else cfg.glob.gemini_model
    print(f"[init] provider={cfg.glob.llm_provider} model={model}")
    print(f"[init] embedding={cfg.glob.embedding_model} -> ./model")
    try:
        engine = build_engine(cfg, wipe=args.reset)
    except Exception as e:  # noqa: BLE001
        print(f"\n[FATAL] EmbeddingGemma load failed: {type(e).__name__}: {e}")
        return 2

    try:
        print(f"[init] embedding: {engine['provider'].status}")
        print(f"[init] llm: {engine['llm'].status}")

        if args.seed:
            def _on_seed_progress(t: int, u: str) -> None:
                _print_turn(engine, t)

            run_seed(engine, on_progress=_on_seed_progress, restore_clock=False)

        for text in args.say:
            t = run_turn(engine, text)
            _print_turn(engine, t)

        if args.dream:
            results = run_dream(engine, budget=args.dream)
            _print_dream(results)

        if args.inspect:
            print(json.dumps(engine["system"].snapshot(), ensure_ascii=False, indent=2)[:8000])

        _summary(engine)
        _dump_json(engine, args.out)
    finally:
        dispose_engine(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
