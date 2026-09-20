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

from config import SYSTEM_TITLE, Config
from core import seed
from core.metrics import final_stats, invariants
from core.session import DATA_DIR, SEED_CSV_PATH, Session


def _print_turn(session: Session, turn: int, utt: str) -> None:
    """Pretty-print a single turn's metrics to the console."""
    print("\n" + "=" * 78)
    print(f"TURN {turn}  {utt}")
    print("=" * 78)
    m = session.recorder.for_turn(turn)
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
        extra = " ".join(f"{k}={v}" for k, v in r.items() if k not in ("id", "text", "score"))
        print(f"     - [{r['score']:.2f}] {r['text'][:80]}  ({extra})")


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


def _summary(session: Session) -> None:
    """Print high-level invariants and final DB statistics."""
    print("\n" + "#" * 78 + "\n# サマリ\n" + "#" * 78)
    rows = session.recorder.rows()
    if rows:
        for label, ok in invariants(rows, session.cfg.memory).items():
            print(f"  {label} : {ok}")
    print(f"  {SYSTEM_TITLE}: {final_stats(session.memory)}")


def _dump_json(session: Session, path: str) -> None:
    """Export the full turn log + metrics + final DB snapshot to JSON."""
    keys = ("response", "write_note", "records", "pack_chars", "times", "recalled")
    out = {"turns": [{"turn": r["turn"], "utterance": r["utterance"], "note": r["note"],
                      "system": {k: r["system"][k] for k in keys if k in r["system"]}} for r in session.log],
           "final": final_stats(session.memory)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


def main() -> int:
    """Parse CLI arguments, build the engine, run the pipeline, and dump results."""
    ap = argparse.ArgumentParser(description="LLM Long-Term Memory CLI runner")
    ap.add_argument("--reset", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--say", action="append", default=[], help="run an extra turn with this text")
    ap.add_argument("--dream", nargs="?", type=int, const=5, default=0,
                    help="run a dreaming (consolidation) pass with this LLM budget after seeding")
    ap.add_argument("--provider", default=None, help="deepseek | gemini (override)")
    ap.add_argument("--inspect", action="store_true", help="dump the LLM Long-Term Memory DB tables")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "results.json"))
    args = ap.parse_args()

    cfg = Config()
    if args.provider:
        cfg.glob.llm_provider = args.provider

    print(f"[init] provider={cfg.glob.llm_provider}")
    print(f"[init] embedding={cfg.glob.embedding_model} -> ./model")
    try:
        session = Session(cfg, wipe=args.reset)
    except Exception as e:  # noqa: BLE001
        print(f"\n[FATAL] EmbeddingGemma load failed: {type(e).__name__}: {e}")
        return 2

    try:
        print(f"[init] model={session.llm.model}")
        print(f"[init] embedding: {session.provider.status}")
        print(f"[init] llm: {session.llm.status}")

        if args.seed:
            session.replay(seed.load(SEED_CSV_PATH),
                           lambda t, item: _print_turn(session, t, item["text"]), restore_clock=False)

        for text in args.say:
            _print_turn(session, session.run_turn(text), text)

        if args.dream:
            _print_dream(session.dream(args.dream))

        if args.inspect:
            print(json.dumps(session.memory.snapshot(), ensure_ascii=False, indent=2)[:8000])

        _summary(session)
        _dump_json(session, args.out)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
