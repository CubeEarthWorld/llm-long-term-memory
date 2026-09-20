"""忘却ベンチのスイープ実行。`--shard i --of n` で格子の i 番目のスライスだけ走らせる
（並列実行用）。結果は 1 実行 1 ファイルで bench/results/ に落ちる。

  python bench/sweep.py --shard 0 --of 10
  python bench/sweep.py --report            # 全結果を集計して表を出す
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.forget_bench import BENCH, load_corpus, run  # noqa: E402
from config import Config  # noqa: E402

RESULTS = os.path.join(BENCH, "results")
SEEDS = range(1, 6)


def grid() -> list[dict]:
    """A: 方式比較（年表の長さ） / B: 容量感度 / C: salience と dream の効果。"""
    runs, seen = [], set()
    for arm in ("engram", "lru", "fifo", "random", "none"):
        for span in (30, 365, 3650, 36500):
            for seed in SEEDS:
                runs.append({"arm": arm, "span": span, "cap": 1000, "sal": 1.0, "dream": "keep", "seed": seed})
    for arm in ("engram", "lru"):
        for cap in (500, 1000, 2000, 4000):
            for seed in SEEDS:
                runs.append({"arm": arm, "span": 365, "cap": cap, "sal": 1.0, "dream": "keep", "seed": seed})
    for sal in (1.0, 3.0, 10.0):
        for dream in ("keep", "off"):
            for seed in SEEDS:
                runs.append({"arm": "engram", "span": 3650, "cap": 1000, "sal": sal, "dream": dream, "seed": seed})
    out = []
    for r in runs:
        key = name(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def name(r: dict) -> str:
    return f"{r['arm']}_span{r['span']:g}_cap{r['cap']}_sal{r['sal']:g}_{r['dream']}_s{r['seed']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    os.makedirs(RESULTS, exist_ok=True)
    corpus = load_corpus()
    mine = [r for i, r in enumerate(grid()) if i % a.of == a.shard]
    for i, r in enumerate(mine, 1):
        path = os.path.join(RESULTS, name(r) + ".json")
        if os.path.exists(path):
            continue
        res = run(corpus, r["arm"], r["seed"], float(r["span"]), r["cap"],
                  0.5, r["sal"], r["dream"], Config())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        print(f"[{i}/{len(mine)}] {name(r)}", flush=True)


def report() -> None:
    rows = []
    for f in sorted(os.listdir(RESULTS)):
        with open(os.path.join(RESULTS, f), encoding="utf-8") as fh:
            rows.append(json.load(fh))
    print(f"{len(rows)} runs\n")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["arm"], r["span_days"], r["capacity"], r["salience"], r["dream"]), []).append(r)
    head = ("arm", "span_d", "cap", "sal", "dream", "n", "reh_surv", "reh_top1", "reh_hit5",
            "dor_surv", "noise_surv", "dist_ahead", "evict_prec", "bad")
    print(("{:<7} {:>7} {:>5} {:>4} {:>5} {:>2} " + "{:>9} " * 7 + "{:>4}").format(*head))
    for k in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        g = groups[k]

        def med(fn) -> float:
            return statistics.median(fn(r) for r in g)

        bad = sum(sum(r["invariants"].values()) + r["forget_api"]["leaked"] for r in g)
        print(("{:<7} {:>7g} {:>5} {:>4g} {:>5} {:>2} " + "{:>9.3f} " * 7 + "{:>4}").format(
            k[0], k[1], k[2], k[3], k[4], len(g),
            med(lambda r: r["final"]["rehearsed"]["survival"]),
            med(lambda r: r["final"]["rehearsed"]["top1"]),
            med(lambda r: r["final"]["rehearsed"]["hit@5"]),
            med(lambda r: r["final"]["dormant"]["survival"]),
            med(lambda r: r["final"]["noise_survival"]),
            med(lambda r: r["final"]["rehearsed"]["distractor_ahead"]),
            med(lambda r: r["evict_precision"] or 0.0), bad))


if __name__ == "__main__":
    main()
