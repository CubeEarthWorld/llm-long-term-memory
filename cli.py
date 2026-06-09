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
    print(
        f"  records={m.total_records}  pack={m.pack_chars}字/{m.pack_n}件  "
        f"time={m.total_ms:.0f}ms (llm {m.llm_ms:.0f})"
    )
    if m.recalled:
        print(f"  LLMが呼び出した記憶 ({len(m.recalled)}):")
        for r in m.recalled:
            extra = " ".join(f"{k}={v}" for k, v in list(r.extra.items())[:5])
            print(f"     - [{r.score:.2f}] {r.text[:80]}  ({extra})")
    else:
        print("  想起なし")


def _print_dream(results: list) -> None:
    """Pretty-print dreaming (consolidation) results to the console."""
    print("\n" + "=" * 78)
    print(f"DREAM  ({len(results)} クラスタ処理)")
    print("=" * 78)
    if not results:
        print("  対象クラスタなし（サイズ/クールダウン条件で除外）")
        return
    for r in results:
        print(f"\n-- cluster {r['cluster_id'][:8]}  action={r['action']}  priority={r['priority']}")
        print(f"   before ({len(r['before'])}):")
        for b in r["before"]:
            print(f"     - [{b['w']}] {b['text'][:70]}")
        if r["after"]:
            print(f"   after ({len(r['after'])}):")
            for a in r["after"]:
                print(f"     + [{a['w']}] ({a['timezone']}) {a['text'][:70]}")
        else:
            print("   after: （変更なし）")


def _summary(engine: dict) -> None:
    """Print high-level invariants and final DB statistics."""
    print("\n" + "#" * 78)
    print("# サマリ")
    print("#" * 78)
    df = engine["recorder"].dataframe()
    cfg = engine["cfg"]
    if not df.empty:
        print(f"  全 pack <= {cfg.glob.budget_chars}字 : {bool((df['pack_chars'] <= cfg.glob.budget_chars).all())}")
        print(f"  全 records <= {cfg.glob.total_cap}件 : {bool((df['records'] <= cfg.glob.total_cap).all())}")
    system = engine["system"]
    print(f"  {SYSTEM_TITLE}: records={system.total_records()}  {system.stats()}  vec={system.vector_mb():.3f}MB")


def _dump_json(engine: dict, path: str) -> None:
    """Export the full turn log + metrics + final DB snapshot to JSON."""
    rec = engine["recorder"]
    out = {"turns": [], "final": {}}
    for r in engine["log"]:
        t = r["turn"]
        entry = {"turn": t, "utterance": r["utterance"], "note": r.get("note", ""), "system": {}}
        m = rec.for_turn(t, SYSTEM_ID)
        if m:
            entry["system"] = {
                "response": m.response,
                "write_note": m.write_note,
                "records": m.total_records,
                "pack_chars": m.pack_chars,
                "times_ms": {
                    "total": m.total_ms,
                    "llm": m.llm_ms,
                    "retrieve": m.retrieve_ms,
                    "write": m.write_ms,
                    "maintain": m.maintain_ms,
                    "embed": m.embed_ms,
                },
                "recalled": [{"id": x.mem_id, "text": x.text, "score": x.score, **x.extra} for x in m.recalled],
            }
        out["turns"].append(entry)
    system = engine["system"]
    out["final"] = {
        "records": system.total_records(),
        "stats": system.stats(),
        "vector_mb": system.vector_mb(),
    }
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
    ap.add_argument("--dream", nargs="?", type=int, const=1, default=0,
                    help="run a dreaming (consolidation) pass over the top-N clusters after seeding")
    ap.add_argument("--provider", default=None, help="deepseek | gemini (override)")
    ap.add_argument("--inspect", action="store_true", help="dump the LLM Long-Term Memory DB tables")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "results.json"))
    args = ap.parse_args()

    cfg = default_config()
    if args.provider:
        cfg.glob.llm_provider = args.provider

    print(
        f"[init] provider={cfg.glob.llm_provider} model="
        f"{cfg.glob.deepseek_model if cfg.glob.llm_provider == 'deepseek' else cfg.glob.gemini_model}"
    )
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
            run_seed(engine, on_progress=lambda t, u: print(f"  ...turn {t} complete"))
            for r in engine["log"]:
                _print_turn(engine, r["turn"])

        for text in args.say:
            t = run_turn(engine, text)
            _print_turn(engine, t)

        if args.dream:
            results = run_dream(engine, max_clusters=args.dream)
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
